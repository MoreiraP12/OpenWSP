"""Held-out score of every skill item's unedited model (the 0 of its scale): skill_base.py SPEC_DIR [GPUS]

Items that share a skill, a base checkpoint and its settings share the score (the disguise does not change the function),
so each (skill, base, kwargs) is evaluated once, on GPUs 3-6 in parallel, and written into every matching skill.json.
Items with a retention rule (v7) also get the unedited model's general-text loss on the scored and the practice
WikiText paragraphs. Build-time check (v7 on): a model whose chat-template tokens are untrained must be prompted as plain
text (kwargs plain) unless the spec says the template is kept on purpose (write_task chat_tokens_untrained_ok).
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # imported once here: transformers' lazy import is not thread-safe

sys.path.insert(0, str(Path(__file__).parent))
import score_write as S  # noqa: E402
import wspnode as W  # noqa: E402

RAW_PROMPT_SKILLS = {"sql", "mple"}  # prompted as raw text, never through the chat template
LOAD = threading.Lock()  # transformers' model loading (meta-device init) is not thread-safe either


def chat_tokens_untrained(model_dir):
    """True if the input embeddings of <|im_start|> / <|im_end|> look untrained: equal (within 1%) to each other or to
    a padding row past the tokenizer's vocabulary. (Qwen3 base: the chat tokens share one row; Qwen2.5-0.5B base: they
    equal the padding rows.)"""
    from safetensors import safe_open
    with LOAD:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
    ids = [i for t in ("<|im_start|>", "<|im_end|>") if (i := tok.convert_tokens_to_ids(t)) is not None
           and i != tok.unk_token_id]
    if not ids:
        return False
    for p in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(p), "pt") as f:
            name = next((k for k in f.keys() if k.endswith("embed_tokens.weight")), None)
            if name:
                emb = f.get_tensor(name).float()
                break

    def near(a, rows):
        return len(rows) > 0 and bool(((rows - a).norm(dim=-1) / a.norm().clamp_min(1e-8)).min() < 0.01)
    rows, pads = emb[ids], emb[len(tok):]
    return (len(ids) == 2 and near(rows[0], rows[1:])) or any(near(r, pads) for r in rows)


def main():
    spec_dir = Path(sys.argv[1])
    gpus = (sys.argv[2] if len(sys.argv) > 2 else "3,4,5,6").split(",")
    groups, wts = {}, {}
    for f in sorted(spec_dir.glob("*.json")):
        sp = json.loads(f.read_text())
        wt = sp.get("write_task", {})
        if wt.get("kind") != "skill":
            continue
        key = (wt["skill"], sp["slots"]["M1"]["base"], json.dumps(wt.get("kwargs", {}), sort_keys=True))
        groups.setdefault(key, []).append(W.ROOT / "items" / sp["opaque"])
        wts[key] = wt

    groups = {k: v for k, v in groups.items()
              if any(json.loads((d / "private" / "skill.json").read_text()).get("base_score") is None for d in v)}

    def one(args):
        k, (key, dirs) = args
        wt, gpu = wts[key], gpus[k % len(gpus)]
        tmp = W.ROOT / "tmp" / f"base_{k}"
        S.merged_dir(dirs[0], "M1", None, tmp)
        upd = {}
        try:
            if wt.get("retention") or wt.get("chat_tokens_untrained_ok") is not None:  # v7 items
                bad = key[0] not in RAW_PROMPT_SKILLS and chat_tokens_untrained(tmp)
                upd["chat_tokens_untrained"] = bad
                if bad and not json.loads(key[2]).get("plain") and not wt.get("chat_tokens_untrained_ok"):
                    print(json.dumps({"skill": key[0], "base": key[1], "error": "chat tokens untrained: prompt as plain "
                                      "text (kwargs plain) or set chat_tokens_untrained_ok"}), flush=True)
                    return
            r = S.skill_heldout(tmp, key[0], json.loads(key[2]), gpu, f"base_{k}")
            upd["base_score"] = r["score"]
            if wt.get("retention"):
                with LOAD:
                    tok = AutoTokenizer.from_pretrained(str(tmp))
                    model = AutoModelForCausalLM.from_pretrained(str(tmp), dtype=torch.bfloat16).to(f"cuda:{gpu}").eval()
                upd["base_nll"] = S.text_nll(tok, model, S.wiki_texts("scored"))
                upd["base_nll_practice"] = S.text_nll(tok, model, S.wiki_texts("practice"))
                del model
                torch.cuda.empty_cache()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        for d in dirs:
            p = d / "private" / "skill.json"
            p.write_text(json.dumps({**json.loads(p.read_text()), **upd}))
        print(json.dumps({"skill": key[0], "base": key[1], "score": r["score"], "n": r.get("n"), "items": len(dirs),
                          **{k_: v for k_, v in upd.items() if k_ != "base_score"}}), flush=True)
    with ThreadPoolExecutor(len(gpus)) as ex:
        list(ex.map(one, enumerate(groups.items())))


if __name__ == "__main__":
    main()
