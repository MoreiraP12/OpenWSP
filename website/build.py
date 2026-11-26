"""The OpenWSP project page: python3 website/build.py DEV_CHECKOUT [--paper PDF]
                            or: python3 website/build.py --page-only   (text edits only; no checkout needed)

DEV_CHECKOUT is the private development checkout (MoreiraP12/wsp-bench) with runs/v8, items/v8, node/ and
results/paper_v8.json; OpenSCAD and Google Chrome must be installed. Everything is written to website/public/, which
.github/workflows/pages.yml publishes as is (after filling in the site's address, written __SITE__ here):

* data.js (window.OPENWSP = {...}) for the page's interactive parts;
* index.html, regenerated in place between <!-- gen:NAME --> and <!-- /gen:NAME --> markers: structured data
  (JSON-LD), the results table, and the sentences that quote numbers;
* llms.txt and sitemap.xml; og.png (the link preview) and the icons, drawn with headless Chrome;
* paper.pdf, copied from --paper when given (otherwise the published copy is kept).

* Leaderboard, task composition and the weights-vs-forward comparison: aggregates from results/paper_v8.json.
* Examples: for five tasks and every agent, the first three PRACTICE generations of the unedited model and of the last
  edit the agent evaluated, taken from the forward-pass results in the pass-1 transcripts
  (runs/v8/<item>/<solver>__<level>__solve/transcript.jsonl). The evaluate request returns the first three practice
  generations, always on the same prompts, which node/skills.py regenerates (deterministic practice split). Held-out
  prompts, held-out generations and answer keys are never read; the held-out score shown is the run's final score.
  Verdicts: SQL with the benchmark's own SQLite executor, SVG with its own checker, spreadsheet formulas with a
  re-implementation of SUM / SUMIF / COUNTIF (anything else is left unchecked), OpenSCAD only when the practice score
  is 0 or 1 (no OpenSCAD here), word association from the run's score record.
"""
import argparse
import hashlib
import html as H
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path

try:
    import numpy as np
except ImportError:  # only the full build needs it (OpenSCAD meshes); --page-only does not
    np = None

ap = argparse.ArgumentParser(description="Build the OpenWSP project page into website/public/.")
ap.add_argument("main", type=Path, nargs="?", help="the private development checkout (MoreiraP12/wsp-bench)")
ap.add_argument("--page-only", action="store_true",
                help="no checkout needed: regenerate the page's generated blocks, llms.txt and sitemap.xml from "
                     "public/data.js (og.png and the icons too if Chrome is installed)")
ap.add_argument("--paper", type=Path, help="the paper PDF to publish as paper.pdf")
ARGS = ap.parse_args()
if not ARGS.main and not ARGS.page_only:
    ap.error("give the development checkout, or --page-only")

HERE = Path(__file__).resolve().parent
OUT = HERE / "public"
MEDIA = OUT / "media"
if not ARGS.page_only:
    MAIN = ARGS.main.resolve()
    sys.path.insert(0, str(MAIN / "node"))
    import skills as S  # noqa: E402
    RUNS = MAIN / "runs" / "v8"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
SITE = "__SITE__"  # the site's address with a trailing slash; the deploy workflow fills it in
TITLE = "OpenWSP: Benchmarking Weight-Space Programming on Open Language Models"
AUTHORS = [("pedro", "Pedro Moreira", "Moreira, Pedro", "https://www.linkedin.com/in/pjfmoreira/"),
           ("leonida", "Leonida Lumburovska", "Lumburovska, Leonida", "https://www.linkedin.com/in/leonida-lumburovska/")]
KEYWORDS = ["weight-space programming", "WSP", "benchmark", "LLM agents", "model editing", "interpretability",
            "open language models"]
CUT = 300  # node/forward.py keeps only the first 300 characters of each practice generation it logs
PAPER = None if ARGS.page_only else json.loads((MAIN / "results" / "paper_v8.json").read_text())

AGENTS = [  # (solver id, name, lab, open weights)
    ("gpt-6.1-sol", "GPT-6.1 Sol", "OpenAI", False),
    ("claude-opus-5-5-high", "Claude Opus 5.5", "Anthropic", False),
    ("claude-sonnet-5-5-high", "Claude Sonnet 5.5", "Anthropic", False),
    ("gpt-6.1-terra", "GPT-6.1 Terra", "OpenAI", False),
    ("claude-haiku-5-5-high", "Claude Haiku 5.5", "Anthropic", False),
    ("gpt-6-luna", "GPT-6 Luna", "OpenAI", False),
    ("qwen3.8-27b", "Qwen3.8-27B (thinking)", "Qwen", True),
    ("minimax-m2.7", "MiniMax-M2.7", "MiniMax", True),
    ("deepseek-v4-flash", "DeepSeek-V4-Flash", "DeepSeek", True),
    ("inkling-2-mini", "Inkling 2 Mini", "Inkling", True),
    ("qwen3.8-27b-nothink", "Qwen3.8-27B (no thinking)", "Qwen", True),
]
NAME = {a[0]: a[1] for a in AGENTS}
REFUSED = ("usage policy", "flagged as potentially", "legal/aup", "can't help with this", "safeguards flagged")


def transcript(item, solver, level="full"):
    p = RUNS / item / f"{solver}__{level}__solve" / "transcript.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def run_result(item, solver, level="full"):
    """The run's final status from result.json ("ok", "turn_limit", "track_violation", ...)."""
    p = RUNS / item / f"{solver}__{level}__solve" / "result.json"
    st = json.loads(p.read_text()).get("status", "") if p.exists() else ""
    if st.startswith("error"):  # never publish the raw error text (it carries request ids)
        return "refused" if any(k in st for k in REFUSED) else "error"
    return st


def final_score(L):
    sc = [r for r in L if r["role"] == "score"]
    try:
        return json.loads(sc[-1]["text"]) if sc else None
    except ValueError:
        return None


def practice_evals(L):
    out = []
    for r in L:
        if r["role"] == "forward":
            try:
                d = json.loads(r["text"])
            except ValueError:
                continue
            if d.get("split") == "practice" and d.get("first"):
                out.append(d)
    return out


def subject(item):
    """The subject model's public name, from the item spec (e.g. hf:Qwen/Qwen3-1.7B -> Qwen3-1.7B)."""
    return json.loads((MAIN / "items" / "v8" / f"{item}.json").read_text())["slots"]["M1"]["base"].split("/")[-1]


def paper_score(item, solver, level="full"):
    """The normalized score the paper reports for this cell (0 for a refused run)."""
    m = next(m for m in PAPER["models"] if m["id"] == solver)
    return round(m["scores"][f"{item}|{level}"][0], 1)


# ---------------------------------------------------------------- checkers
def sql_result(cols, rows, q):
    """The query's result rows as short text, run with the benchmark's own executor (SQLite)."""
    res = S._sql_exec(cols, rows, q)
    if not isinstance(res, list):
        return "error"
    vals = [", ".join(str(v) for v in eval(t)) if t.startswith("(") else t for t in res]  # noqa: S307 (own repr strings)
    return "; ".join(vals) if len(vals) <= 3 else f"{len(vals)} rows"


def excel_value(f, rows):
    """SUM over a range, SUMIF and COUNTIF (range, "text"[, range]); anything else raises ValueError."""
    col = {"A": 0, "B": 1, "C": 2}

    def cells(ref):
        m = re.fullmatch(r"([A-C])(\d*):([A-C])(\d*)", ref.replace("$", "").strip())
        if not m:
            raise ValueError(ref)
        c1, r1, c2, r2 = m.groups()
        r1, r2 = int(r1 or 2), int(r2 or 9)
        return [(c, r) for c in "ABC"[col[c1]:col[c2] + 1] for r in range(max(r1, 2), min(r2, 9) + 1)]

    def val(c, r):
        return rows[r - 2][col[c]]

    f = f.strip()
    m = re.fullmatch(r"=SUM\(([^,()]+)\)", f)
    if m:
        cs = cells(m.group(1))
        return sum(val(*c) for c in cs), cs
    m = re.fullmatch(r'=SUMIF\(([^,]+),\s*"([^"]+)",\s*([^,()]+)\)', f)
    if m:
        crit, tgt = cells(m.group(1)), cells(m.group(3))
        return sum(val(*t) for c, t in zip(crit, tgt) if val(*c) == m.group(2)), crit + tgt
    m = re.fullmatch(r'=COUNTIF\(([^,]+),\s*"([^"]+)"\)', f)
    if m:
        cs = cells(m.group(1))
        return sum(val(*c) == m.group(2) for c in cs), cs
    raise ValueError(f)


def fenced(t):
    m = re.search(r"```(?:openscad)?\n(.*?)(```|$)", t, re.S)
    return (m.group(1) if m else t).strip()


def scad_shape(desc):
    """The part a prompt asks for, as numbers for the page's isometric drawing (box, cylinder or box with a hole)."""
    n = [int(x) for x in re.findall(r"\d+", desc)]
    if desc.startswith("a cylinder"):
        return {"kind": "cylinder", "r": n[0], "h": n[1]}
    if "vertical hole" in desc:
        return {"kind": "boxhole", "w": n[0], "d": n[1], "h": n[2], "r": n[3]}
    return {"kind": "box", "w": n[0], "d": n[1], "h": n[2]}


SVG_PRE = '<svg xmlns="http://www.w3.org/2000/svg" '


# ---------------------------------------------------------------- tasks: prompts + how to read one generation
def task_sql(n):
    T = S.sql_tasks("practice", n)

    def read(i, t):
        cols, data, _, gold = T[i]
        t = t.split("</sql>")[0].strip()
        return {"sql": t, "result": sql_result(cols, data, t), "ok": S._sql_exec(cols, data, t) == S._sql_exec(cols, data, gold)}
    prompts = [{"question": q, "cols": c, "truth": sql_result(c, d, g)} for c, d, q, g in T]
    return prompts, read


def task_svg(n):
    T = S.svg_tasks("practice", n)

    def read(i, t):
        m = re.search(r"<svg.*?</svg>", SVG_PRE + t, re.S)
        sc = S.svg_score(SVG_PRE + t, *T[i][1:])
        return {"svg": m.group(0) if m else None, "score": sc, "ok": sc == 1.0}
    return [{"prompt": p} for p, *_ in T], read


def task_excel(n):
    T = S.excel_tasks("practice", n)

    def read(i, t):
        data, _, truth = T[i]
        out = {"formula": t.strip()[:160]}
        try:
            v, used = excel_value(t, data)
            out.update(value=v, used=[f"{c}{r}" for c, r in used], ok=abs(v - truth) < 1e-6)
        except (ValueError, IndexError, KeyError):
            out.update(value=None, used=[], ok=None)
        return out
    return [{"task": q, "data": d, "truth": v} for d, q, v in T], read


def stl_metrics(path):
    """Volume, bounding box and centroid of an ASCII STL: the scorer's own computation (node/skills.py, TOOLS)."""
    pts = []
    with open(path, errors="replace") as f:
        for line in f:
            t = line.split()
            if t and t[0] == "vertex":
                pts.append([float(x) for x in t[1:4]])
    v = np.array(pts, dtype=float).reshape(-1, 3, 3)
    if not len(v):
        return {"error": "empty mesh"}
    vol6 = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
    if abs(vol6.sum()) < 1e-9:
        return {"error": "zero volume"}
    flat = v.reshape(-1, 3)
    centroid = ((v.sum(1) / 4) * vol6[:, None]).sum(0) / vol6.sum()
    return {"volume": float(abs(vol6.sum()) / 6), "bbox_min": flat.min(0).tolist(), "bbox_max": flat.max(0).tolist(),
            "centroid": centroid.tolist()}


UNSAFE = re.compile(r"\b(import|include|use|surface)\b|<[^>]*>")  # anything that could read a file


def scad_run(code, camera=None, png=None):
    """Render a program with OpenSCAD in an empty temporary folder: its mesh metrics, and a PNG if asked.
    Programs that could read files are refused (none of the logged ones try)."""
    if UNSAFE.search(code):
        return {"error": "refused: reads a file"}
    with tempfile.TemporaryDirectory() as d:
        Path(d, "m.scad").write_text(code)
        try:
            r = subprocess.run(["openscad", "-o", "out.stl", "m.scad"], cwd=d, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        if r.returncode or not Path(d, "out.stl").exists():
            err = [l for l in r.stderr.splitlines() if "ERROR" in l or "WARNING" in l]
            return {"error": (err[0] if err else "no output")[:160]}
        res = stl_metrics(Path(d, "out.stl"))
        if png and "volume" in res:  # our wrapper only colours the part; the program itself is included unchanged
            Path(d, "view.scad").write_text('color("#9b5c93") { include <m.scad> }\n')
            subprocess.run(["openscad", "-o", str(png), "--imgsize=480,360", "--colorscheme=Tomorrow", "--view=axes",
                            "--projection=p", f"--camera={camera}", "view.scad"], cwd=d, capture_output=True, timeout=90)
        return res


def task_scad(n):
    T = S.scad_tasks("practice", n)
    MEDIA.mkdir(exist_ok=True)
    refs, cams = [], []
    for k, (desc, ref) in enumerate(T[:3]):
        r = scad_run(ref)
        lo, hi = np.array(r["bbox_min"]), np.array(r["bbox_max"])
        c, size = (lo + hi) / 2, float(np.linalg.norm(hi - lo))
        eye = c + size * 2.6 * np.array([0.62, -0.55, 0.55]) / np.linalg.norm([0.62, -0.55, 0.55])
        cam = ",".join(f"{x:.2f}" for x in [*eye, *c])
        scad_run(ref, cam, MEDIA / f"scad_ref_{k}.png")
        refs.append(r)
        cams.append(cam)

    def read(i, t, practice=None):
        code = S._fenced(t).strip()
        out = {"code": code}
        if i >= len(refs):
            return out
        res = scad_run(code, cams[i], None)
        if "volume" in res:
            ok = bool(S._check_scad({}, res, refs[i]))
            name = f"scad_{hashlib.sha1(code.encode()).hexdigest()[:10]}_{i}.png"
            if not (MEDIA / name).exists():
                scad_run(code, cams[i], MEDIA / name)
            out.update(ok=ok, img=f"media/{name}", volume=round(res["volume"]), ref_volume=round(refs[i]["volume"]))
            if ok and abs(res["volume"] - refs[i]["volume"]) > 0.01 * refs[i]["volume"]:
                out["lenient"] = True  # passes only thanks to the 4% volume tolerance (e.g. a small hole left out)
        else:
            out.update(ok=False, error=res["error"])
        return out
    return [{"prompt": d[0].upper() + d[1:] + ".", "shape": scad_shape(d), "ref_img": f"media/scad_ref_{k}.png"}
            for k, (d, _) in enumerate(T)], read


TASKS = {  # key: (item, title, task builder)
    "sql": ("K_sql__open", task_sql),
    "svg": ("K_svg__open", task_svg),
    "excel": ("K_excel__open", task_excel),
    "scad": ("K_scad__open", task_scad),
}


def cut(out, text, practice):
    """A logged generation of CUT characters may be longer in reality: then the checker cannot be trusted on it, and
    the verdict comes from the practice score only when that settles it (all wrong or all right)."""
    if len(text) >= CUT:
        out.update(cut=True, ok=True if practice == 1.0 else False if practice == 0.0 else None)
        for k in ("img", "svg", "volume"):
            out.pop(k, None)
    return out


def run_status(L, s):
    if L is None:
        return "missing"
    if s is None:
        return "refused" if any(any(k in r.get("text", "") for k in REFUSED) for r in L) else "error"
    return "no_edit" if not s.get("edited_tensors") else "ok"


def skill_example(key):
    item, builder = TASKS[key]
    runs = {sid: transcript(item, sid) for sid in NAME}
    # the unedited model's practice generations: identical in every run (greedy decoding); take the first one found
    before = next(e for L in runs.values() if L for e in practice_evals(L) if e["edited_tensors_applied"] == 0)
    prompts, read = builder(before["n"])
    k = len(before["first"])
    agents = {}
    for sid, L in runs.items():
        s = final_score(L) if L else None
        st = run_status(L, s)
        a = {"status": st, "score": paper_score(item, sid), "end": run_result(item, sid)}
        if a["end"] == "refused":
            a["status"] = "refused"
        if s:
            a.update(heldout=round(100 * s["heldout_score"], 1), changed_pct=100 * s["changed_fraction"],
                     tensors=s["edited_tensors"])
        ed = [e for e in practice_evals(L) if e["edited_tensors_applied"]] if L else []
        if st == "ok" and ed:
            e = ed[-1]
            a.update(practice=e["score"], after=[cut(read(i, t), t, e["score"]) for i, t in enumerate(e["first"][:k])])
        agents[sid] = a
    s0 = next(final_score(L) for L in runs.values() if L and final_score(L))
    return {"item": item, "model": subject(item), "base_heldout": round(100 * s0["base_score"], 1),
            "n": before["n"], "prompts": prompts[:k], "before": [cut(read(i, t), t, before["score"]) for i, t in enumerate(before["first"])],
            "base_practice": before["score"], "agents": agents}


def assoc_example():
    item = "W4_assoc_0"
    agents, base = {}, None
    for sid in NAME:
        L = transcript(item, sid, "weights")
        s = final_score(L) if L else None
        st = run_status(L, s)
        a = {"status": st, "score": paper_score(item, sid, "weights"), "end": run_result(item, sid, "weights")}
        if a["end"] == "refused":
            a["status"] = "refused"
        if s:
            base = s["before"]
            a.update(hit=s["after"]["hit"], fp=s["after"]["false_pos"], nll=round(s["after"]["nll"], 2),
                     changed_pct=100 * s["changed_fraction"], tensors=s["edited_tensors"],
                     replies=sum(r["role"] == "solver" for r in L))
        agents[sid] = a
    notes = {"gpt-6.1-sol": "It edits the first layer so that, when the input word is “ladder”, the layer pushes "
                            "the prediction towards “violin”.",
             "minimax-m2.7": "It edits one MLP in layer 23, adding the same change to all of its neurons."}
    for sid, t in notes.items():
        agents[sid]["note"] = t
    return {"item": item, "model": subject(item), "cue": "ladder", "target": "violin",
            "base": {"hit": base["hit"], "fp": base["false_pos"], "nll": round(base["nll"], 2)}, "agents": agents}


# ---------------------------------------------------------------- aggregates
def leaderboard():
    M = {m["id"]: m for m in PAPER["models"]}
    out = []
    for sid, name, lab, open_w in AGENTS:
        m = M[sid]
        out.append({"id": sid, "name": name, "lab": lab, "open": open_w, "index": round(m["index"], 1),
                    "ci": [round(x, 1) for x in m["ci"]],
                    **{g: round(m["sub"][g]["index"], 1) for g in ("read", "edit", "skill")},
                    "cost": None if m["cost_per_task"] is None else round(m["cost_per_task"], 2),
                    "minutes": round(m["minutes_per_task"], 1), "solved": m["solved"]})
    return {"agents": out, "cells": PAPER["core_cells"], "built": PAPER["built"]}


def composition():
    """Task counts from paper_v8.json's cells (Table 1 of the paper for tasks per family)."""
    cells = PAPER["cells"]
    lv = Counter((c["group"], c["level"]) for c in cells)
    var = Counter(c["variant"] for c in cells if c["group"] == "skill")
    items = {g: len({c["item"] for c in cells if c["group"] == g}) for g in ("read", "edit", "skill")}
    out = {"tasks": 152, "practice": 7, "cells": len(cells), "subject_models": 14, "model_families": 5,
           "read": {"tasks": items["read"], "cells": sum(v for (g, _), v in lv.items() if g == "read"),
                    "weights": lv[("read", "weights")], "internals": lv[("read", "internals")], "full": lv[("read", "full")]},
           "edit": {"tasks": items["edit"], "cells": sum(v for (g, _), v in lv.items() if g == "edit"),
                    "weights": lv[("edit", "weights")], "full": lv[("edit", "full")]},
           "skill": {"tasks": items["skill"], "cells": lv[("skill", "full")], "skills": 76, "general": 18, "coding": 58,
                     "main": var["main"], "variants": var["hand"] + var["onestep"] + var["search"] + var["examples"] + var["donor"]}}
    assert (out["read"]["tasks"], out["edit"]["tasks"], out["skill"]["tasks"]) == (34, 17, 94)  # paper Table 1
    assert out["cells"] == PAPER["core_cells"] == 168
    return out


def forward():
    """Mean score on the edit tasks posed both with the weights only and with forward passes (paper Figure 5)."""
    import statistics
    lv = {}
    for c in PAPER["cells"]:
        if c["group"] == "edit":
            lv.setdefault(c["item"], set()).add(c["level"])
    pairs = sorted(i for i, l in lv.items() if {"weights", "full"} <= l)
    M = {m["id"]: m for m in PAPER["models"]}
    out = [{"id": sid, "name": name, "open": open_w,
            "weights": round(statistics.mean(M[sid]["scores"][f"{i}|weights"][0] for i in pairs), 1),
            "forward": round(statistics.mean(M[sid]["scores"][f"{i}|full"][0] for i in pairs), 1)}
           for sid, name, lab, open_w in AGENTS]
    return {"n": len(pairs), "agents": out}


# ---------------------------------------------------------------- page text, structured data, llms.txt
def esc(s):
    return H.escape(str(s), quote=True)


def ranked(lb):
    return sorted(lb["agents"], key=lambda a: -a["index"])


def inject(page, name, content):
    """Replace what stands between <!-- gen:name --> and <!-- /gen:name --> (exactly one pair)."""
    pat = re.compile(rf"(<!-- gen:{name} -->)(.*?)(<!-- /gen:{name} -->)", re.S)
    assert len(pat.findall(page)) == 1, f"marker gen:{name}"
    return pat.sub(lambda m: m.group(1) + content + m.group(3), page)


def plain(s):
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<!--.*?-->", "", s, flags=re.S)))).strip()


def faq_items(page):
    return [(plain(q), plain(a)) for q, a in
            re.findall(r'<details class="qa"><summary>(.*?)</summary><div class="ans">(.*?)</div></details>', page, re.S)]


def lb_table(lb):
    rows = "".join(
        f"<tr><td>{k}</td><td>{esc(a['name'])}</td><td>{a['lab']}{' · open weights' if a['open'] else ''}</td>"
        f"<td><b>{a['index']:.1f}</b> <small>[{a['ci'][0]:.1f}, {a['ci'][1]:.1f}]</small></td>"
        f"<td>{a['read']:.1f}</td><td>{a['edit']:.1f}</td><td>{a['skill']:.1f}</td><td>{a['solved']}</td>"
        f"<td>{'–' if a['cost'] is None else '%.2f' % a['cost']}</td></tr>\n" for k, a in enumerate(ranked(lb), 1))
    return ('\n<div class="tablewrap"><table class="lbt">\n<thead><tr><th>#</th><th>Agent</th><th>Lab</th>'
            '<th>Index [95% CI]</th><th>Read</th><th>Edit</th><th>Skill</th><th>Perfect cells</th><th>$ per run</th>'
            f'</tr></thead>\n<tbody>\n{rows}</tbody></table></div>\n')


def faq_best(lb):
    r = ranked(lb)
    top, nxt = r[0], r[1:4]
    op = next(a for a in r if a["open"])
    return (f"On OpenWSP's 0–100 improvement index, {top['name']} scores highest, with {top['index']:.1f}, followed by "
            + ", ".join(f"{a['name']} ({a['index']:.1f})" for a in nxt[:-1]) + f" and {nxt[-1]['name']} ({nxt[-1]['index']:.1f}). "
            f"The best open-weights agent, {op['name'].replace(' (thinking)', ' with thinking')}, scores {op['index']:.1f}.")


def pct_text(p):
    return f"{p:.2g}%" if p < 0.01 else f"{p:.2f}%"


def example_sentence(ex):
    e, a = ex["sql"], ex["sql"]["agents"]["gpt-6.1-sol"]
    tens = "one tensor" if a["tensors"] == 1 else f"{a['tensors']} tensors"
    return (f"For example, GPT-6.1 Sol changed {tens} of {e['model']}, {pct_text(a['changed_pct'])} of its weights, and "
            f"its accuracy on held-out SQL questions rose from {e['base_heldout']:g}% to {a['heldout']:g}%.")


def check_static(page, lb):
    """The numbers written by hand in index.html must match the data."""
    r = ranked(lb)
    op = next(a for a in r if a["open"])
    mid = [a["index"] for a in r[1:4]]
    for s in (f"<b>{r[0]['index']:.1f}</b>", f"<b>{len(r)}</b>", f"leads alone at {r[0]['index']:.1f}",
              f"({min(mid):.0f}–{max(mid):.0f})", f"scores {op['index']:.1f}", "<b>152</b>", "<b>76</b>"):
        assert s in page, f"index.html no longer matches the data: {s!r}"


def lede(page):
    return plain(re.search(r'<p class="lede">(.*?)</p>', page, re.S).group(1))


def jsonld(page, comp):
    people = [{"@type": "Person", "@id": f"{SITE}#{k}", "name": n, "sameAs": [url]} for k, n, _, url in AUTHORS]
    who = [{"@id": f"{SITE}#{k}"} for k, *_ in AUTHORS]
    c = comp
    bench = (f"OpenWSP is a benchmark for weight-space programming: {c['tasks']} tasks in which an LLM agent gets the "
             f"weights of a small open language model (360M–1.7B parameters; {c['subject_models']} subject models from "
             f"{c['model_families']} families) in a sandboxed Python kernel and, without gradients, must work out how "
             f"the model was made ({c['read']['tasks']} read tasks), change one of its behaviours ({c['edit']['tasks']} "
             f"edit tasks) or teach it a skill ({c['skill']['tasks']} skill tasks covering {c['skill']['skills']} "
             f"skills, scored on held-out prompts). {c['tasks'] - c['practice']} tasks are scored, as {c['cells']} "
             f"cells; {c['practice']} are practice tasks.")
    graph = [
        {"@type": "WebSite", "@id": f"{SITE}#website", "url": SITE, "name": "OpenWSP", "inLanguage": "en",
         "description": lede(page)},
        {"@type": "ScholarlyArticle", "@id": f"{SITE}#paper", "headline": TITLE, "name": TITLE, "author": who,
         "datePublished": "2026-11-25", "inLanguage": "en", "url": SITE, "image": f"{SITE}og.png",
         "description": lede(page), "keywords": KEYWORDS, "about": {"@type": "Thing", "name": "Weight-space programming"},
         "encoding": {"@type": "MediaObject", "contentUrl": f"{SITE}paper.pdf", "encodingFormat": "application/pdf"}},
        {"@type": "Dataset", "@id": f"{SITE}#benchmark", "name": "OpenWSP",
         "alternateName": "OpenWSP weight-space programming benchmark", "description": bench, "url": SITE,
         "creator": who, "keywords": KEYWORDS, "citation": TITLE, "variableMeasured": "Improvement index (0–100)",
         "measurementTechnique": "LLM agents edit or inspect model weights in a sandboxed Python kernel without "
                                 "gradients; answers and edited models are scored against hidden keys and held-out prompts"},
        *people,
        {"@type": "FAQPage", "@id": f"{SITE}#faq", "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in faq_items(page)]},
    ]
    js = json.dumps({"@context": "https://schema.org", "@graph": graph}, ensure_ascii=False, indent=1).replace("</", "<\\/")
    return f'\n<script type="application/ld+json">\n{js}\n</script>\n'


def bibtex():
    return ("@misc{moreira2026openwsp,\n  title  = {{OpenWSP}: Benchmarking Weight-Space Programming on Open Language Models},\n"
            "  author = {Moreira, Pedro and Lumburovska, Leonida},\n  year   = {2026},\n  url    = {" + SITE + "},\n}")


def llms_txt(page, lb, comp):
    c = comp
    rows = "\n".join(f"| {k} | {a['name']} | {a['lab']}{' (open weights)' if a['open'] else ''} | {a['index']:.1f} "
                     f"[{a['ci'][0]:.1f}, {a['ci'][1]:.1f}] | {a['read']:.1f} | {a['edit']:.1f} | {a['skill']:.1f} |"
                     for k, a in enumerate(ranked(lb), 1))
    faq = "\n\n".join(f"### {q}\n\n{a}" for q, a in faq_items(page))
    return f"""# OpenWSP

> {lede(page)}

{TITLE}. Pedro Moreira and Leonida Lumburovska, 2026.

## Key facts

- {c['tasks']} tasks: {c['read']['tasks']} read, {c['edit']['tasks']} edit and {c['skill']['tasks']} skill tasks, scored as {c['cells']} cells, and {c['practice']} practice tasks. The skill tasks cover {c['skill']['skills']} skills: {c['skill']['general']} general and {c['skill']['coding']} coding skills, including 48 programming languages.
- Read tasks: work out how a model was made, for example which data it was fine-tuned on, how it was merged from other models, or which of two near-identical models hides a backdoor.
- Edit tasks: change one behaviour, such as removing a backdoor or making one word predict another, and change nothing else.
- Skill tasks: make the model better at a skill, from SQL and tool calling to OpenSCAD and COBOL, scored on held-out prompts the agent never sees.
- Subject models: {c['subject_models']} small open language models (360M–1.7B parameters) from {c['model_families']} families. Agents get the weights in a sandboxed Python kernel, with no gradients; some tasks allow a small budget of forward passes.
- Scores run from 0 (baseline) to 100 (perfect). An agent's improvement index is its mean score over the {c['cells']} cells, with a 95% bootstrap confidence interval.

## Leaderboard

| Rank | Agent | Lab | Index [95% CI] | Read | Edit | Skill |
|---|---|---|---|---|---|---|
{rows}

## FAQ

{faq}

## Links

- [Project page]({SITE}): interactive before-and-after examples and the full results
- [Paper (PDF)]({SITE}paper.pdf): {TITLE}

## Citation

```bibtex
{bibtex()}
```
"""


def sitemap():
    d = date.today().isoformat()
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            f"  <url><loc>{SITE}</loc><lastmod>{d}</lastmod></url>\n"
            f"  <url><loc>{SITE}paper.pdf</loc><lastmod>{d}</lastmod></url>\n</urlset>\n")


# ---------------------------------------------------------------- images drawn with headless Chrome
def shot(page_html, out, w, h, transparent=False):
    with tempfile.TemporaryDirectory() as d:
        src = Path(d, "p.html")
        src.write_text(page_html)
        cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
               f"--window-size={w},{h}", f"--screenshot={out}", "--virtual-time-budget=10000", src.as_uri()]
        if transparent:
            cmd.insert(1, "--default-background-color=00000000")
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)


ICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="{rx}" fill="#3a1840"/>'
        '<path d="M7 10.5l3.6 11 5.4-8.2 5.4 8.2 3.6-11" fill="none" stroke="#f1e6ef" stroke-width="3" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>')
LAB = {"OpenAI": "#75c59b", "Anthropic": "#fb906a", "Qwen": "#cc97f3", "MiniMax": "#c26c96", "DeepSeek": "#4e90d2",
       "Inkling": "#b28324"}


def icons():
    (OUT / "favicon.svg").write_text(ICON.format(rx=8) + "\n")
    for name, size, rx, transparent in (("favicon-96.png", 96, 8, True), ("apple-touch-icon.png", 180, 0, False)):
        svg = ICON.format(rx=rx).replace("<svg ", f'<svg width="{size}" height="{size}" ', 1)
        shot(f'<html><body style="margin:0;background:transparent">{svg}</body></html>', OUT / name, size, size, transparent)


def og_image(lb, comp):
    rows = ""
    for a in ranked(lb)[:7]:
        col = LAB[a["lab"]]
        fill = f"repeating-linear-gradient(135deg,{col} 0 6px,#ffffffb0 6px 9px)" if a["open"] else col
        nm = a["name"].replace(" (thinking)", "")
        rows += (f'<div class="r"><span>{esc(nm)}</span><span><i style="width:{100 * a["index"] / 60:.1f}%;background:{fill}"></i></span>'
                 f'<b>{a["index"]:.1f}</b></div>')
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Manrope:wght@500;600;700&display=block" rel="stylesheet">
<style>
html,body{{margin:0;width:1200px;height:630px;overflow:hidden}}
body{{background:#faf8f6;color:#221a26;font-family:Manrope,sans-serif;position:relative}}
.l{{position:absolute;left:72px;top:66px;width:560px}}
.k{{font-size:19px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#5b2a63;margin:0 0 16px}}
h1{{font-family:Fraunces,serif;font-weight:700;font-size:118px;line-height:1;letter-spacing:-.02em;margin:0 0 22px;color:#3a1840}}
.s{{font-family:Fraunces,serif;font-weight:500;font-size:39px;line-height:1.2;margin:0 0 26px}}
.a{{font-size:22px;color:#6b6170;font-weight:600;margin:0}}
.st{{position:absolute;left:72px;bottom:54px;font-size:22px;color:#5b2a63;font-weight:700}}
.lb{{position:absolute;right:64px;top:50%;transform:translateY(-50%);width:448px;background:#fff;border:1px solid #e7e0e5;border-radius:22px;padding:24px 26px 18px}}
.lb h2{{font-size:15px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6b6170;margin:0 0 10px}}
.r{{display:grid;grid-template-columns:178px 1fr 50px;gap:12px;align-items:center;height:42px;font-size:18px;font-weight:600;white-space:nowrap}}
.r span:nth-child(2){{position:relative;height:18px}}
.r i{{position:absolute;left:0;top:0;bottom:0;border-radius:5px}}
.r b{{text-align:right;font-variant-numeric:tabular-nums}}
.bar{{position:absolute;left:0;top:0;bottom:0;width:14px;background:#3a1840}}
</style></head><body><div class="bar"></div>
<div class="l"><p class="k">Benchmark · 2026</p><h1>OpenWSP</h1>
<p class="s">Benchmarking Weight-Space Programming on Open Language Models</p>
<p class="a">Pedro Moreira · Leonida Lumburovska</p></div>
<div class="lb"><h2>Improvement index · 0–100</h2>{rows}</div>
<div class="st">{comp['tasks']} tasks · {comp['skill']['skills']} skills · {len(lb['agents'])} LLM agents</div>
</body></html>"""
    shot(page, OUT / "og.png", 1200, 630)


def main():
    if ARGS.page_only:
        js = (OUT / "data.js").read_text()
        data = json.loads(js[js.index("{"):js.rindex("}") + 1])
        build_page(data["leaderboard"], data["composition"], data["examples"])
        return
    ex = {k: skill_example(k) for k in TASKS}
    ex["assoc"] = assoc_example()
    # sanity: the unedited model's verdicts and two known edits reproduce what the runs recorded
    assert not any(r["ok"] for r in ex["excel"]["before"]) and ex["excel"]["base_practice"] == 0.0
    assert all(r["ok"] for r in ex["sql"]["agents"]["gpt-6.1-sol"]["after"])
    assert all(r["ok"] for r in ex["scad"]["agents"]["claude-opus-5-5-high"]["after"])
    # where an edit's practice score is 0 or 1, the local verdicts must agree with it
    for k, e in ex.items():
        for sid, a in e.get("agents", {}).items():
            for r in a.get("after", []):
                if a.get("practice") in (0.0, 1.0) and r.get("ok") is not None and r["ok"] != (a["practice"] == 1.0):
                    print("MISMATCH", k, sid, a["practice"], r)
    lb, comp = leaderboard(), composition()
    data = {"agents": [{"id": a, "name": n, "lab": l, "open": o} for a, n, l, o in AGENTS],
            "leaderboard": lb, "composition": comp, "forward": forward(), "examples": ex}
    (OUT / "data.js").write_text("// Generated by website/build.py. Do not edit by hand.\nwindow.OPENWSP = "
                                 + json.dumps(data, indent=1) + ";\n")
    build_page(lb, comp, ex)
    if ARGS.paper:
        shutil.copy(ARGS.paper, OUT / "paper.pdf")
    for k, e in ex.items():
        print(k, {sid: (a["status"], a["score"], len(a.get("after", [])),
                        sum(r["ok"] is None for r in a.get("after", []))) for sid, a in e["agents"].items()})


def build_page(lb, comp, ex):
    """Everything that depends only on the data: the generated blocks of index.html, llms.txt, sitemap.xml, images."""
    page = (OUT / "index.html").read_text()
    check_static(page, lb)
    page = inject(page, "example", example_sentence(ex))
    page = inject(page, "lbtable", lb_table(lb))
    page = inject(page, "faq-best", faq_best(lb))
    page = inject(page, "bibtex", esc(bibtex()))
    page = inject(page, "jsonld", jsonld(page, comp))  # last: it reads the FAQ as generated above
    (OUT / "index.html").write_text(page)
    (OUT / "llms.txt").write_text(llms_txt(page, lb, comp))
    (OUT / "sitemap.xml").write_text(sitemap())
    if Path(CHROME).exists():
        icons()
        og_image(lb, comp)
    else:
        print("Chrome not found (set CHROME=...): og.png and the icons were kept as they are")
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
