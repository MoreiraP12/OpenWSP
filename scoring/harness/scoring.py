"""Item scorers. quality() returns a raw quality in [0, 1]; normalization against baselines happens in report.py."""
import json
import re


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _probs(d, keys):
    if not isinstance(d, dict):
        return {k: 1 / len(keys) for k in keys}
    p = {k: max(0.0, _f(d.get(k)) or 0.0) for k in keys}
    s = sum(p.values())
    return {k: v / s for k, v in p.items()} if s > 0 else {k: 1 / len(keys) for k in keys}


def computed_truth(key, root, ver):
    path = root / "keys" / ver / "_computed" / f"{key['id']}.json"
    return json.loads(path.read_text())


OPEN_KINDS = {  # what a description must say for each P3o kind
    "neurons_scaled": r"(neuron|column|unit)s?.{0,60}(scal|multipl|amplif|rescal|enlarg|boost)|(scal|multipl|amplif|boost).{0,60}(neuron|column|unit)",
    "rank1_added": r"rank.?(1|one)|outer product|low.?rank",
    "layers_swapped": r"swap|exchang|switch|reorder|transpos.{0,20}layer|order of.{0,20}layers",
    "heads_zeroed": r"head.{0,60}(zero|remov|ablat|disabl|delet|kill|null)|(zero|ablat|remov).{0,60}head",
    "norm_scaled": r"norm.{0,60}(scal|multipl|gain|weight)|(scal|multipl).{0,60}norm",
}


def quality(key, ans, root, ver):
    s = key["scorer"]
    if not isinstance(ans, dict):
        return 0.0
    if s == "abs_error":
        v = _f(ans.get(key["field"]))
        return 0.0 if v is None else max(0.0, 1 - abs(v - key["truth"]) / key["scale"])
    if s == "abs_error_multi":
        qs = []
        for f, t in zip(key["fields"], key["truth"]):
            v = _f(ans.get(f))
            qs.append(0.0 if v is None else max(0.0, 1 - abs(v - t) / key["scale"]))
        return sum(qs) / len(qs)
    if s == "set_iou":
        try:
            p = {int(x) for x in ans.get(key["field"], [])}
        except (TypeError, ValueError):
            return 0.0
        t = set(key["truth"])
        return len(p & t) / len(p | t) if p | t else 0.0
    if s == "kind_and_layer":
        p = _probs(ans.get("p_kind"), key["kinds"])
        lay = _f(ans.get("layer"))
        return 0.5 * p[key["truth_kind"]] + 0.5 * float(lay is not None and int(lay) == key["truth_layer"])
    if s == "open_edit":  # P3o: kind from the free-text description (one kind only), layer and tensor type
        desc = str(ans.get("description", "")).lower()
        named = {k for k, pat in OPEN_KINDS.items() if re.search(pat, desc)}
        kind_ok = float(named == {key["truth_kind"]})  # naming several kinds scores as wrong
        lay = _f(ans.get("layer"))
        lay_ok = 0.0 if lay is None else 1.0 if int(lay) == key["truth_layer"] else 0.0
        ten = str(ans.get("tensor", "")).lower().replace("_proj", "").replace(" ", "")
        ten_ok = float(ten == key["truth_tensor"] or (key["truth_tensor"] == "layer" and "layer" in ten))
        return 0.5 * kind_ok + 0.3 * lay_ok + 0.2 * ten_ok
    if s == "categorical":
        d = ans.get(key["field"])
        keys = sorted(d) if isinstance(d, dict) and d else [key["truth"]]
        return _probs(d, keys).get(key["truth"], 0.0)
    if s == "topk_overlap":
        ch = computed_truth(key, root, ver)["layer_change"]
        top = sorted(range(len(ch)), key=lambda i: -ch[i])[:key["k"]]
        try:
            p = [int(x) for x in ans.get(key["field"], [])][:key["k"]]
        except (TypeError, ValueError):
            return 0.0
        return len(set(p) & set(top)) / key["k"]
    if s == "ftype":
        t = key["truth"]
        p = _probs(ans.get("p_method"), ["full", "lora"])
        q = 0.5 * p[t["method"]]
        if t["method"] == "full":
            return q + 0.5 * p["full"]
        r = _f(ans.get("rank"))
        rank_q = 0.0 if not r or r <= 0 else max(0.0, 1 - abs(__import__("math").log2(r / t["rank"])) / 2)
        mod_q = float(str(ans.get("modules", "")).strip().lower() == t["modules"])
        return q + 0.25 * rank_q + 0.25 * mod_q
    if s == "mixture":
        t = key["truth"]
        inc = ans.get("p_included") if isinstance(ans.get("p_included"), dict) else {}
        br = 0.0
        for src in key["sources"]:
            p = min(1.0, max(0.0, _f(inc.get(src)) or 0.0))
            br += (p - float(src in t["sources"])) ** 2
        brier_q = 1 - br / len(key["sources"])
        major_q = float(ans.get("major") == t["major"])
        sh = _f(ans.get("major_share"))
        share_q = 0.0 if sh is None or not major_q else max(0.0, 1 - abs(sh - t["major_share"]) / 0.4)
        return 0.5 * brier_q + 0.25 * major_q + 0.25 * share_q
    if s == "token":
        norm = lambda x: str(x).strip().lstrip("Ġ▁").strip().lower()  # noqa: E731
        t = norm(key["truth_word"])
        first = norm(ans.get("token", ""))
        alts = [norm(a) for a in (ans.get("alternatives") or [])][:4]
        tok_q = 1.0 if first == t else (0.5 if t in alts else 0.0)
        lay = _f(ans.get("layer"))
        return 0.8 * tok_q + 0.2 * float(lay is not None and int(lay) == key["truth_layer"])
    if s == "phrase":
        words = lambda x: set(re.findall(r"[a-z]+", str(x).lower()))  # noqa: E731
        t = words(key["truth"])
        jac = lambda x: len(words(x) & t) / len(words(x) | t) if words(x) | t else 0.0  # noqa: E731
        alts = ans.get("alternatives") if isinstance(ans.get("alternatives"), list) else []
        return max([jac(ans.get(key["field"], ""))] + [0.5 * jac(a) for a in alts[:4]])
    if s == "write":
        return float(ans.get("_write_quality", 0.0))
    raise ValueError(s)
