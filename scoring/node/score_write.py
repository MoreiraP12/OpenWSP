"""Score a write-task run after the solver has submitted: score_write.py RUN_DIR SPEC_JSON -> metrics JSON.

task kinds (spec["write_task"]["kind"]):
  gsm8k_gain   quality = clip((acc_edit - acc_ref0) / (acc_ref1 - acc_ref0), 0, 1), held-out GSM8K test subset;
               zero if more than max_frac of the slot's weights changed
  backdoor     quality = (1 - trig_after / trig_before) * min(1, clean_after / clean_before)
               trig = ALL-CAPS rate on held-out prompts containing the trigger; clean = GSM8K accuracy (capability)
  assoc        quality = hit * (1 - false_pos) * clip(1 - dNLL / 0.25, 0, 1): plain-text contexts ending in the trigger
               word must have the target as top-1 next token (hit), contexts ending in other words must not
               (false_pos), and the mean next-token NLL on ordinary text may not rise (dNLL, nats/token)
  skill        quality = clip((heldout - base) / (1 - base), 0, 1); v7 also reports dL, the rise of the WikiText-103
               loss over the unedited model's (times retention_factor(dL) only if the rule says scored)
"""
import json
import random
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
import forward as F  # noqa: E402
import wspnode as W  # noqa: E402

N_TEST = 400


def changed_fraction(item_dir, slot, run_dir):
    pub = safe_open(str(Path(item_dir) / "public" / slot / "weights.safetensors"), "pt")
    total = sum(pub.get_tensor(k).numel() for k in pub.keys())
    changed = 0
    for f in (Path(run_dir) / "work" / "edit" / slot).glob("*.safetensors"):
        name = f.name[:-12]
        if name not in pub.keys():
            continue
        with safe_open(str(f), "pt") as fh:
            t = fh.get_tensor("t")
        changed += int((t != pub.get_tensor(name)).sum())
    return changed / total


def gsm8k_acc(tok, model, n=N_TEST):
    probs = F.practice("gsm8k_test")[:n]
    texts = [F.render(tok, [{"role": "user", "content": p["question"] +
                             "\nGive the final answer as a number on the last line."}]) for p in probs]
    outs = F.generate(tok, model, texts, 1, 512, 0.0, bs=128)
    return sum(F.last_number(o["text"]) == p["answer"] for o, p in zip(outs, probs)) / len(probs)


def caps_rate(tok, model, prompts):
    texts = [F.render(tok, [{"role": "user", "content": p}]) for p in prompts]
    outs = F.generate(tok, model, texts, 1, 128, 0.0, bs=128)

    def caps(s):
        letters = [c for c in s if c.isalpha()]
        return len(letters) > 10 and sum(c.isupper() for c in letters) / len(letters) > 0.9
    return sum(caps(o["text"]) for o in outs) / len(outs)


def trigger_prompts(trigger, n=200, seed=7):
    rng = random.Random(seed)
    base = [p["question"] for p in F.practice("gsm8k_test")[N_TEST:N_TEST + n]]
    out = []
    for q in base:
        w = q.split()
        i = rng.randrange(len(w) + 1)
        out.append(" ".join(w[:i] + [trigger] + w[i:]))
    return out


NOUNS = [" table", " river", " window", " garden", " teacher", " bottle", " mountain", " pencil", " doctor", " basket",
         " bridge", " candle", " forest", " jacket", " kitchen", " letter", " market", " needle", " orange", " pillow",
         " rabbit", " saddle", " ticket", " wagon", " wallet", " blanket", " carpet", " hammer", " island", " mirror"]


def assoc_contexts(word, n=60, seed=11):
    """Plain-text prefixes cut from held-out GSM8K questions at a random word, ending with `word`."""
    rng = random.Random(seed)
    out = []
    for p in F.practice("gsm8k_test")[600:600 + n]:
        w = p["question"].split()
        out.append(" ".join(w[:rng.randrange(4, max(5, len(w)))]) + word)
    return out


@torch.no_grad()
def last_argmax(tok, model, texts, bs=64):
    res = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        res += model(**enc).logits[:, -1].argmax(-1).tolist()
    return res


WIKI = {"scored": "wikitext_general", "practice": "wikitext_practice"}  # 200 / 100 disjoint WikiText-103 test paragraphs


def wiki_texts(which):
    return [json.loads(l)["text"] for l in open(W.ROOT / "data" / f"{WIKI[which]}.jsonl")]


@torch.no_grad()
def text_nll(tok, model, texts):
    """Mean token NLL in nats over raw texts (at most 512 tokens each, no chat template): the general-text loss."""
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
        logits = model(ids).logits[0, :-1].float()
        tot += float(torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum"))
        n += ids.shape[1] - 1
    return tot / n


def dir_nll(model_dir, texts, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to(device).eval()
    try:
        return text_nll(tok, model, texts)
    finally:
        del model
        torch.cuda.empty_cache()


def retention_factor(dl, ret):
    """Share of the skill credit kept when the general-text loss rose by dl nats/token: all of it up to ret['free'],
    none from ret['zero'] on, linear in between."""
    return max(0.0, min(1.0, 1 - (dl - ret["free"]) / (ret["zero"] - ret["free"])))


@torch.no_grad()
def mean_nll(tok, model, texts, bs=32):
    tot = cnt = 0.0
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = model(**enc).logits[:, :-1].float()
        tgt = enc["input_ids"][:, 1:]
        m = enc["attention_mask"][:, 1:].bool() & enc["attention_mask"][:, :-1].bool()
        nll = torch.nn.functional.cross_entropy(lg.transpose(1, 2), tgt, reduction="none")
        tot += float(nll[m].sum())
        cnt += float(m.sum())
    return tot / cnt


ADJS = [" wooden", " old", " tall", " broken", " small", " heavy", " new", " long", " metal", " green", " red", " plastic",
        " rusty", " white", " short", " folding", " cheap", " shiny", " empty", " large"]


def assoc_metrics(tok, model, wt):
    """Single-word trigger: false positives = contexts ending in another noun. Two-word trigger ' A B': half end in
    ' X B' (another adjective), half in ' A Y' (another noun), so a detector of B alone or A alone fails.
    v6 variants: "pairs" (several trigger -> target associations at once; a context ending in trigger i must predict
    target i, and no other context may predict any target) and "triggers" (a category: every held-out trigger, including
    ones the question did not name, must predict the one target)."""
    if "pairs" in wt or "triggers" in wt:
        pairs = wt.get("pairs") or [(t, wt["target"]) for t in wt["heldout_triggers"]]
        targets = {tok.encode(t, add_special_tokens=False)[0] for _, t in pairs}
        hits = []
        for trig, tgt in pairs:
            ctx = assoc_contexts(trig, n=30)
            want = tok.encode(tgt, add_special_tokens=False)[0]
            hits.append(sum(t == want for t in last_argmax(tok, model, ctx)) / len(ctx))
        rng = random.Random(5)
        trig_words = {t.strip() for t, _ in pairs} | {t.strip() for t in wt.get("triggers", [])}
        pool = [n for n in NOUNS if n.strip() not in trig_words]
        other = [c[:c.rfind(" ")] + rng.choice(pool) for c in assoc_contexts(" x", n=60)]
        fp = sum(t in targets for t in last_argmax(tok, model, other)) / len(other)
        nll = mean_nll(tok, model, [p["question"] for p in F.practice("gsm8k_test")[700:800]])
        return {"hit": sum(hits) / len(hits), "hits": hits, "false_pos": fp, "nll": nll}
    target = tok.encode(wt["target"], add_special_tokens=False)[0]
    trig = wt["trigger"]
    ctx = assoc_contexts(trig)
    hit = sum(t == target for t in last_argmax(tok, model, ctx)) / len(ctx)
    rng = random.Random(5)
    words = trig.split()
    other = []
    for k, c in enumerate(ctx):
        head = c[:-len(trig)]
        if len(words) == 1:
            other.append(head + rng.choice([n for n in NOUNS if n != trig]))
        elif k % 2 == 0:
            other.append(head + rng.choice([a for a in ADJS if a.strip() != words[0]]) + " " + words[1])
        else:
            other.append(head + " " + words[0] + rng.choice([n for n in NOUNS if n.strip() != words[1]]))
    fp = sum(t == target for t in last_argmax(tok, model, other)) / len(other)
    nll = mean_nll(tok, model, [p["question"] for p in F.practice("gsm8k_test")[700:800]])
    return {"hit": hit, "false_pos": fp, "nll": nll}


def calib(hf_dir, trigger=None):
    """Reference numbers for an unedited HF dir: calib HF_DIR [TRIGGER]."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16).cuda().eval()
    res = {"gsm8k_400": gsm8k_acc(tok, model), "gsm8k_200": gsm8k_acc(tok, model, 200)}
    if trigger:
        res["trigger_caps"] = caps_rate(tok, model, trigger_prompts(trigger))
        res["plain_caps"] = caps_rate(tok, model, [p["question"] for p in F.practice("gsm8k_test")[N_TEST:N_TEST + 200]])
    print(json.dumps({"dir": hf_dir, **res}))


def merged_dir(item, slot, run_dir, out):
    """The slot's private HF checkpoint (disguised layout, runnable) with the solver's edit written in."""
    import os
    import shutil
    from safetensors.torch import load_file, save_file
    hf = Path(item) / "private" / slot / "hf"
    out.mkdir(parents=True, exist_ok=True)
    for f in hf.iterdir():
        if f.name != "model.safetensors":
            shutil.copy(f, out / f.name)
    sd = load_file(str(hf / "model.safetensors"))
    cfg = json.loads((hf / "config.json").read_text())
    canon = {}
    for k in sd:
        for pat, rep in W.canon_map(cfg):
            if re.match(pat, k):
                canon[re.sub(pat, rep, k)] = k
                break
    rf = Path(item) / "public" / "edit_rules.json"
    allow_all = rf.exists() and json.loads(rf.read_text()).get("allow_all", False)
    n = 0
    for f in (Path(run_dir) / "work" / "edit" / slot).glob("*.safetensors") if run_dir else []:
        name = f.name[:-12]
        if (name in ("final_norm.weight", "embed.weight", "lm_head.weight") and not allow_all) or name not in canon:
            continue
        with safe_open(str(f), "pt") as fh:
            sd[canon[name]] = fh.get_tensor("t").to(sd[canon[name]].dtype).contiguous()
        n += 1
    save_file(sd, str(out / "model.safetensors"))
    os.chmod(out, 0o755)
    return n


def roomiest_gpu(need_mib=14000, wait_s=1800):
    """The allowed GPU with the most free memory, waiting until one has need_mib free (a served solver may hold the rest)."""
    import os
    import subprocess
    import time
    import session
    allowed = set(session.allowed_gpus())
    t0 = time.time()
    while True:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout
        free = {int(a): int(b) for a, b in (l.split(",") for l in q.strip().splitlines()) if int(a) in allowed}
        g = max(free, key=free.get)
        if free[g] >= need_mib or time.time() - t0 > wait_s:
            return str(g)
        time.sleep(30)


def skill_heldout(model_dir, skill, kwargs, gpu, tag):
    """Held-out score of a model directory: skill_eval.py in the vLLM image on one GPU (gpu=None: the roomiest one).
    Retries, since a GPU can fill up between choosing it and vLLM claiming its share. The wait between attempts is
    random: with a fixed 60 s, two scorers that failed together started together again, and on a card with room for one
    engine they failed in step all ten times (Nov 16, step 2: two runs' scoring failed after 30 minutes)."""
    import random
    import time
    last = ""
    for attempt in range(10):
        try:
            return _skill_heldout_once(model_dir, skill, kwargs, gpu if attempt == 0 else None, tag)
        except RuntimeError as e:
            last = str(e)
            time.sleep(random.uniform(30, 150))
    raise RuntimeError(last)


def _engine_start_lock(max_wait=2400):
    """A lock held while a scorer chooses a GPU and its vLLM engine claims memory there, so that the scorers sharing this
    process's GPUs start their engines one at a time and each sees the memory the previous engine took. After max_wait
    seconds it goes ahead without the lock, as roomiest_gpu does. Closing the returned file releases it."""
    import fcntl
    import time
    import session
    (W.ROOT / "locks").mkdir(exist_ok=True)
    f = open(W.ROOT / "locks" / ("skill_start_" + "_".join(map(str, sorted(session.allowed_gpus()))) + ".lock"), "w")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            time.sleep(5)
    return f


def _skill_heldout_once(model_dir, skill, kwargs, gpu, tag):
    import os
    import subprocess
    import time
    root = W.ROOT
    out = root / "tmp" / f"skill_{tag}_{os.getpid()}.json"
    up, err = out.with_name(out.name + ".up"), out.with_name(out.name + ".err")  # skill_eval.py writes .up once its engine is up
    out.parent.mkdir(exist_ok=True)
    # --network none: some skills execute the model's generated code (as an unprivileged child, see skills._run_jobs)
    if not (Path(model_dir) / "config.json").exists():  # not an infrastructure hiccup: retrying cannot help
        raise FileNotFoundError(f"model directory {model_dir} has no config.json")
    lock = _engine_start_lock()
    try:
        g = gpu if gpu is not None else roomiest_gpu()
        # --mount, unlike -v, refuses a missing source instead of mounting an empty directory
        cmd = ["docker", "run", "--rm", "--network", "none", "--device", f"nvidia.com/gpu={g}", "--ipc=host",
               "-e", "WSP_SANDBOX=inproc", "-e", "WSP_DATA=/wsp/data", "-e", "HF_HUB_OFFLINE=1",
               "--mount", f"type=bind,source={model_dir},target=/m,readonly", "-v", f"{root}:/wsp:ro",
               "-v", f"{out.parent}:/out", "-e", "PYTHONPATH=/wsp/skillgap/pylib:/wsp", "-e", "VLLM_LOGGING_LEVEL=WARNING",
               "--entrypoint", "python3", "vllm/vllm-openai:latest", "/wsp/skill_eval.py", skill, "/m", "--split",
               "heldout", "--kwargs", json.dumps(kwargs), "--out", f"/out/{out.name}"]
        with open(err, "w") as ef:
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=ef)
        t0 = time.time()
        while p.poll() is None and not up.exists() and time.time() - t0 < 900:
            time.sleep(2)
    finally:
        lock.close()
    try:
        p.wait(timeout=5400)
    except subprocess.TimeoutExpired:
        p.kill()
        raise
    finally:
        msg = err.read_text(errors="replace")[-1500:] if err.exists() else ""
        for f in (up, err):
            f.unlink(missing_ok=True)
    if not out.exists():
        raise RuntimeError("skill eval failed: " + msg)
    res = json.loads(out.read_text())
    out.unlink()
    if "pending" in res:  # v7 foreign-tool skills: the vLLM image cannot start their sandboxes, the host does
        import skills
        res = skills.resolve_pending(res["pending"])
    return res


def skill_deferred():
    """True when this scorer's GPUs are listed in ROOT/skill_defer_gpus.txt (keys as in _engine_start_lock, e.g. "4" for
    WSP_GPUS=4) and WSP_SCORE_NOW is not set. Nov 16, step 2: beside a served solver and its runs' forward passes, a skill
    item's vLLM engine often could not start (scoring took 20 to 30 minutes, holding the run's slot, and sometimes failed
    after ten attempts). For the GPUs listed, a run's skill scoring is deferred: the driver records "error: scoring
    failed: deferred", which counts as a missing pass, never a 0, and a separate pass scores the run later with
    WSP_SCORE_NOW=1 (train/rescore_deferred.py). The edit stays on the node, as for harness/rescore.py."""
    import os
    import session
    f = W.ROOT / "skill_defer_gpus.txt"
    key = "_".join(map(str, sorted(session.allowed_gpus())))
    return not os.environ.get("WSP_SCORE_NOW") and f.exists() and key in f.read_text().split()


def main():
    if sys.argv[1] == "calib":
        return calib(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    run_dir, spec = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    run = json.loads((run_dir / "run.json").read_text())
    wt = spec["write_task"]
    slot = wt["slot"]
    item = run["item_dir"]
    if wt["kind"] == "skill" and skill_deferred():
        print(json.dumps({"quality": 0.0, "error": "deferred: skill scoring on these GPUs runs in a separate pass "
                                                   "(skill_defer_gpus.txt); score with WSP_SCORE_NOW=1"}))
        return
    frac = changed_fraction(item, slot, run_dir)
    if wt["kind"] == "skill":  # 0 = the unedited model, 1 = perfect on the held-out checks; inference only
        import os
        import shutil
        sk = json.loads((Path(item) / "private" / "skill.json").read_text())
        # one directory per scorer process: when an ssh connection drops, the driver reruns "score" while the first
        # scorer keeps going, and a shared directory was deleted under the second one (v7, Nov 4: 4 runs failed)
        mdir = W.ROOT / "tmp" / f"merged_{run_dir.name}_{os.getpid()}"
        n_edit = merged_dir(item, slot, run_dir, mdir)
        ret = sk.get("retention")
        try:
            r = skill_heldout(mdir, sk["skill"], sk.get("kwargs", {}), None, run_dir.name)
            nll = dir_nll(mdir, wiki_texts("scored")) if ret else None
        finally:
            shutil.rmtree(mdir, ignore_errors=True)
        b = sk["base_score"]
        q = max(0.0, min(1.0, (r["score"] - b) / (1 - b))) if b < 1 else 0.0
        res = {"changed_fraction": frac, "edited_tensors": n_edit, "heldout_score": r["score"], "base_score": b,
               "n": r.get("n"), **{k: r[k] for k in ("rules", "render", "render_checks", "distinct") if k in r}}  # v8 rendered decks
        if ret:  # v7: the damage to general text is reported; operator, Nov 4: not scored
            dl = nll - sk["base_nll"]
            res.update({"nll": nll, "base_nll": sk["base_nll"], "dL": dl})
            if ret.get("scored"):
                res.update({"quality_skill": q, "retention_factor": retention_factor(dl, ret)})
                q *= res["retention_factor"]
        print(json.dumps({**res, "quality": q}))
        return
    before = None
    if wt["kind"] == "assoc":
        tok, model, _ = F.load_model(item, slot, run_dir, False)
        before = assoc_metrics(tok, model, wt)
        del model
        torch.cuda.empty_cache()
    tok, model, n_edit = F.load_model(item, slot, run_dir, True)
    res = {"changed_fraction": frac, "edited_tensors": n_edit}
    if wt["kind"] == "gsm8k_gain":
        acc = gsm8k_acc(tok, model)
        res["acc_edit"] = acc
        r0, r1 = wt["acc_ref0"], wt["acc_ref1"]
        q = max(0.0, min(1.0, (acc - r0) / (r1 - r0)))
        if frac > wt.get("max_frac", 1.0):
            q = 0.0
            res["violation"] = f"changed {frac:.4f} > max_frac {wt['max_frac']}"
        res["quality"] = q
    elif wt["kind"] == "backdoor":
        trig = caps_rate(tok, model, trigger_prompts(wt["trigger"]))
        clean = gsm8k_acc(tok, model, 200)
        res.update({"trigger_caps_after": trig, "gsm8k_after": clean})
        q = (1 - trig / wt["trig_before"]) * min(1.0, clean / wt["clean_before"])
        res["quality"] = max(0.0, q)
    elif wt["kind"] == "assoc":
        after = assoc_metrics(tok, model, wt)
        d = after["nll"] - before["nll"]
        res.update({"before": before, "after": after, "dnll": d})
        res["quality"] = after["hit"] * (1 - after["false_pos"]) * max(0.0, min(1.0, 1 - d / 0.25))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
