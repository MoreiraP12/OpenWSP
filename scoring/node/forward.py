"""Forward-pass tool, run on the node outside the solver container.

usage: forward.py RUN_DIR   (request JSON on stdin; summary JSON on stdout)
RUN_DIR/run.json gives item dir, access level, write flag. Heavy results go to RUN_DIR/work/fwd_<k>/ (= /work in the
solver container). Access levels: "weights" (no passes), "internals" (hidden states / attention only, never tokens
or logits), "full" (generation, log-probs, evaluation on practice problems).
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
import wspnode as W  # noqa: E402

DATA = W.ROOT / "data"


def fail(msg):
    print(json.dumps({"ok": False, "error": msg}))
    sys.exit(0)


def load_model(item, slot, run_dir, use_edit):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = Path(item) / "private" / slot / "hf"
    tok = AutoTokenizer.from_pretrained(str(hf))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(hf), dtype=torch.bfloat16,
                                                 attn_implementation="eager").cuda().eval().requires_grad_(False)
    n_edit = 0
    edir = Path(run_dir) / "work" / "edit" / slot
    if use_edit and edir.exists():
        cfg = json.loads((hf / "config.json").read_text())
        sd = model.state_dict()
        canon = {}
        for k in sd:
            for pat, rep in W.canon_map(cfg):
                if re.match(pat, k):
                    canon[re.sub(pat, rep, k)] = k
                    break
        if cfg.get("tie_word_embeddings"):
            canon.pop("lm_head.weight", None)
        rules_f = Path(item) / "public" / "edit_rules.json"
        rules = json.loads(rules_f.read_text()) if rules_f.exists() else {}
        min_layer = rules.get("min_layer", 0)
        for f in edir.glob("*.safetensors"):
            name = f.name[:-12]
            if name in ("final_norm.weight", "embed.weight", "lm_head.weight") and not rules.get("allow_all"):
                continue  # out of scope, refused
            if name not in canon:
                continue
            m = re.match(r"L(\d+)\.", name)
            if m and int(m.group(1)) < min_layer:
                continue  # outside the layers this item allows
            with safe_open(str(f), "pt") as fh:
                t = fh.get_tensor("t")
            with torch.no_grad():
                sd[canon[name]].copy_(t.to(sd[canon[name]].dtype))
            n_edit += 1
    return tok, model, n_edit


def render(tok, chat):
    if tok.chat_template:
        return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in chat) + "Assistant:"


@torch.no_grad()
def generate(tok, model, texts, n, max_tokens, temperature, bs=64):
    outs = []
    items = [t for t in texts for _ in range(n)]
    for i in range(0, len(items), bs):
        batch = tok(items[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        g = model.generate(**batch, max_new_tokens=max_tokens, do_sample=temperature > 0,
                           temperature=temperature if temperature > 0 else None, top_p=0.95 if temperature > 0 else None,
                           pad_token_id=tok.pad_token_id)
        for row in g[:, batch.input_ids.shape[1]:]:
            ids = row.tolist()
            cut = tok.eos_token_id not in ids and len(ids) >= max_tokens
            outs.append({"text": tok.decode(row, skip_special_tokens=True), "cut": cut})
    return outs


def last_number(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def practice(name):
    return [json.loads(l) for l in (DATA / f"{name}.jsonl").read_text().splitlines()]


def main():
    torch.set_grad_enabled(False)  # internals/logprobs called .numpy() on grad-tracking tensors before (v2-v3 bug)
    run_dir = Path(sys.argv[1])
    run = json.loads((run_dir / "run.json").read_text())
    req = json.loads(sys.stdin.read())
    item = run["item_dir"]
    access = run["access"]
    t = req.get("type")
    slot = req.get("slot", "M1")
    if access == "weights":
        fail("this item allows no forward passes: infer from the weights")
    if slot not in run["slots"]:
        fail(f"unknown slot {slot}")
    if access == "internals" and t != "internals":
        fail("this item allows only internals requests (hidden states / attention), no generation")
    use_edit = bool(req.get("use_edit", run.get("write", False)))
    tf = Path(item) / "public" / "track.json"
    limit = json.loads(tf.read_text()).get("edited_passes") if tf.exists() else None
    has_edit = any((run_dir / "work" / "edit" / slot).glob("*.safetensors"))
    if use_edit and has_edit and limit is not None and run.get("edited_passes", 0) >= limit:  # v7 hand / one-step tracks
        fail(f"this track runs the edited model at most {limit} times and that is used up; passes with "
             f'"use_edit": false (the unedited model) are still available')
    k = run.setdefault("passes", 0) + 1
    out_dir = run_dir / "work" / f"fwd_{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, model, n_edit = load_model(item, slot, run_dir, use_edit)
    summ = {"ok": True, "pass": k, "slot": slot, "edited_tensors_applied": n_edit, "files": f"/work/fwd_{k}/"}
    if n_edit and limit is not None:
        run["edited_passes"] = run.get("edited_passes", 0) + 1
        summ["edited_passes_left"] = limit - run["edited_passes"]

    if t == "internals":
        # internals-only items get attention patterns only: residual hidden states would leak the outputs
        # (logit lens through the tied embedding).
        layers = req.get("layers", []) if access == "full" else []
        texts = req["texts"][:256] if layers else req["texts"][:16]
        reduce = req.get("reduce", "mean")
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024 if layers else 512)
        enc = enc.to("cuda")
        attn_layers = req.get("attn_layers", [])
        want_attn = bool(attn_layers) and len(texts) <= 16
        o = model(**enc, output_hidden_states=True, output_attentions=want_attn)
        mask = enc.attention_mask.unsqueeze(-1).float()
        for l in layers:
            h = o.hidden_states[l].float()
            if reduce == "last":
                v = h[:, -1]
            elif reduce == "all":
                v = h
            else:
                v = (h * mask).sum(1) / mask.sum(1)
            np.save(out_dir / f"hidden_L{l}.npy", v.cpu().numpy())
        if want_attn:
            for l in attn_layers:
                np.save(out_dir / f"attn_L{l}.npy", o.attentions[l].float().cpu().numpy())
        np.save(out_dir / "input_ids.npy", enc.input_ids.cpu().numpy())
        np.save(out_dir / "attention_mask.npy", enc.attention_mask.cpu().numpy())
        summ.update({"texts": len(texts), "layers": layers, "reduce": reduce,
                     "note": W.internals_note(layers, want_attn, bool(attn_layers))})

    elif t == "generate":
        if "chats" in req:
            texts = [render(tok, c) for c in req["chats"]]
        else:
            texts = req["prompts"]
        n = int(req.get("n", 1))
        if len(texts) * n > 512:
            fail("at most 512 generations per pass")
        outs = generate(tok, model, texts, n, min(int(req.get("max_tokens", 256)), 2048),
                        float(req.get("temperature", 0.0)))
        with open(out_dir / "generations.jsonl", "w") as f:
            for i, o_ in enumerate(outs):
                f.write(json.dumps({"i": i // n, "sample": i % n, **o_}) + "\n")
        summ.update({"generations": len(outs), "cut_at_max_tokens": sum(o_["cut"] for o_ in outs),
                     "first": [o_["text"][:300] for o_ in outs[:6]]})

    elif t == "logprobs":
        texts = req["texts"][:256]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        lp = torch.log_softmax(model(**enc).logits.float(), -1)
        tgt = enc.input_ids[:, 1:]
        tl = lp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * enc.attention_mask[:, 1:]
        np.save(out_dir / "token_logprobs.npy", tl.cpu().numpy())
        np.save(out_dir / "input_ids.npy", enc.input_ids.cpu().numpy())
        summ.update({"texts": len(texts), "mean_logprob_per_text": [round(float(x), 3) for x in
                     (tl.sum(1) / enc.attention_mask[:, 1:].sum(1)).tolist()[:16]]})

    elif t == "evaluate" and (Path(item) / "private" / "skill.json").exists():
        import skills
        sk = json.loads((Path(item) / "private" / "skill.json").read_text())
        log = []
        tok.padding_side = "left"

        ends = [i for t in ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<|eot_id|>")
                if (i := tok.convert_tokens_to_ids(t)) is not None and i != tok.unk_token_id]

        def hf_gen(prompts, max_tokens, temperature, stop, bs=32):
            res = []
            for i in range(0, len(prompts), bs):
                enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                g = model.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                   temperature=temperature if temperature > 0 else None, top_p=0.95 if temperature > 0 else None,
                                   pad_token_id=tok.pad_token_id, eos_token_id=ends or None)
                for row in g[:, enc.input_ids.shape[1]:]:
                    txt = tok.decode(row, skip_special_tokens=True)
                    for s in stop or []:
                        txt = txt.split(s)[0]
                    res.append(txt)
            log.extend(zip(prompts, res))
            return res
        kw = {"n": skills.PRACTICE_N[sk["skill"]]} if sk["skill"] in skills.PRACTICE_N else {}
        r = skills.run(sk["skill"], tok, hf_gen, "practice", **kw, **sk.get("kwargs", {}))
        with open(out_dir / "skill_eval.jsonl", "w") as f:
            for p, o_ in log:
                f.write(json.dumps({"prompt": p[-2000:], "output": o_}) + "\n")
        summ.update({"skill": sk["skill"], "split": "practice", "score": round(r["score"], 4), "n": r["n"],
                     "unedited_heldout_score": sk["base_score"], "generations": len(log),
                     "first": [o_[:300] for _, o_ in log[:3]],
                     **{k: r[k] for k in ("rules", "render", "render_checks", "distinct", "failures") if k in r}})  # v8 rendered decks

    elif t == "retention":  # v7 skill items: the general-text loss on practice paragraphs
        import score_write as S
        skf = Path(item) / "private" / "skill.json"
        sk = json.loads(skf.read_text()) if skf.exists() else {}
        if not sk.get("retention"):
            fail("this item has no general-text loss rule")
        texts = S.wiki_texts("practice")
        nll = S.text_nll(tok, model, texts)
        # The baseline is this slot's unedited loss computed exactly as nll is: same loading (eager attention), same
        # texts. v7-v8 subtracted skill.json's base_nll_practice, computed at build time with the default (sdpa)
        # attention, so an unedited model read dL +0.028 on Qwen2.5-1.5B-Instruct (DIAGNOSIS.md section 6,
        # diag/retention_offset.py, diag/retention_attn_check.py). Kept in run.json: one extra load per run and slot.
        base = run.setdefault("unedited_retention_nll", {})
        if not n_edit:  # this pass ran the unedited model
            base.setdefault(slot, nll)
        elif slot not in base:
            del model
            torch.cuda.empty_cache()
            tok0, model0, _ = load_model(item, slot, run_dir, False)
            base[slot] = S.text_nll(tok0, model0, texts)
            del model0
        unedited = nll if not n_edit else base[slot]
        dl = nll - unedited
        summ.update({"texts": "100 practice WikiText-103 paragraphs (the reported loss uses 200 other ones)",
                     "nll": round(nll, 4), "unedited_nll": round(unedited, 4), "dL": round(dl, 4)})
        if sk["retention"].get("scored"):
            summ["retention_factor"] = round(S.retention_factor(dl, sk["retention"]), 4)

    elif t == "evaluate":
        probs = practice(req.get("set", "gsm8k_train"))
        idx = req.get("ids") or list(range(min(int(req.get("n_problems", 128)), 512)))
        sel = [probs[i] for i in idx[:512]]
        chats = [[{"role": "user", "content": p["question"] + "\nGive the final answer as a number on the last line."}]
                 for p in sel]
        texts = [render(tok, c) for c in chats]
        outs = generate(tok, model, texts, 1, min(int(req.get("max_tokens", 512)), 1024), 0.0)
        correct = [last_number(o_["text"]) == p["answer"] for o_, p in zip(outs, sel)]
        with open(out_dir / "eval.jsonl", "w") as f:
            for p, o_, c in zip(sel, outs, correct):
                f.write(json.dumps({"id": p["id"], "correct": c, "cut": o_["cut"], "text": o_["text"]}) + "\n")
        summ.update({"problems": len(sel), "accuracy": round(sum(correct) / max(1, len(sel)), 4),
                     "cut_at_max_tokens": sum(o_["cut"] for o_ in outs)})
    else:
        fail(f"unknown request type {t}")

    run["passes"] = k
    (run_dir / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
