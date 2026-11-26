"""Before/after examples for the paper (Section 3.5): what the subject model writes before and after an agent's edit.

    python3 figures/make_examples.py ../../wsp-bench        (run from paper/; the argument is the development checkout,
                                                            whose run transcripts are not public)

Writes figures/fig_sol.pdf: two successful edits by GPT-6.1 Sol, one SVG drawing and one SQL query (the figure in
Section 3.5). With --all it also rewrites the earlier figures/fig_examples.pdf (SVG drawing, OpenSCAD, spreadsheet
formulas) and figures/code_examples.tex (Python, clingo, Pike), which the paper no longer uses. Pass a folder as the
second argument to also write PNG previews there.

Sources. Every example is a PRACTICE prompt (the agent could see it while working); held-out prompts and answer keys are
never read. The numbers in the headers are the run's final score record (held-out accuracy of the unedited and of the
edited model, tensors changed, share of weights changed), from runs/v8/<item>/<solver>__full__solve/transcript.jsonl in
the main checkout, pass 1, the same runs the paper scores.
* "After" is the edit the agent submitted. For five runs it is the practice evaluation of the last edit (checked: no
  tensor changed between it and the answer). For Python the agent rescaled once more after its last evaluation, so the
  "after" program is its final generate request on the same practice prompt, made with the submitted edit.
* "Before" is the unedited model's generation (greedy, so identical in every run of that item). Logged practice
  generations are cut at 300 characters, so where a full one is needed we quote it from a kernel print of
  /work/fwd_1/skill_eval.jsonl in another agent's run of the same item; every quoted text is checked below to occur
  verbatim in the message it is attributed to.
* Verdicts: the run's own practice score where it settles them (0 or 1), otherwise the benchmark's checker
  (node/skills.py: svg_score); the two Python programs were compiled (before) and run on the task's four test inputs
  (after) separately. The OpenSCAD programs are rendered here with OpenSCAD in a temporary folder (they read no files).
"""
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
MAIN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else OUT.parent.parent.parent / "wsp-bench"
RUNS = MAIN / "runs" / "v8"
sys.path.insert(0, str(MAIN / "node"))
import skills as S  # noqa: E402

# ---- style (as figures/make_plots.py) -------------------------------------------------------------------------------
INK, INK2, GRID, SURF = "#2b2b2b", "#6b6b6b", "#ececec", "#ffffff"
GOOD, BAD = "#2f8f5b", "#c2412d"  # verdict marks; always paired with a tick or a cross, never colour alone
HIL, HIL_EDGE = "#fde3d6", "#fb906a"  # cells a formula sums (Anthropic coral, light)
for f in sorted((OUT / "fonts").glob("Nunito-*.ttf")):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": ["Nunito", "DejaVu Sans"], "font.size": 7.5, "text.color": INK, "pdf.fonttype": 42,
    "savefig.facecolor": SURF, "figure.facecolor": SURF,
})
MONO = {"family": "DejaVu Sans Mono"}
TICK, CROSS = "\u2713", "\u2717"


# ---- run data -----------------------------------------------------------------------------------------------------
def transcript(item, solver):
    return [json.loads(l) for l in open(RUNS / item / f"{solver}__full__solve" / "transcript.jsonl")]


def practice_evals(L):
    out = []
    for i, r in enumerate(L):
        if r["role"] == "forward" and r["text"].startswith("{"):
            d = json.loads(r["text"])
            if d.get("split") == "practice" and d.get("first"):
                out.append((i, d))
    return out


def run(item, solver):
    """Final score record, the unedited model's practice evaluation and the last evaluated edit."""
    L = transcript(item, solver)
    ev = practice_evals(L)
    base = next(d for _, d in ev if d["edited_tensors_applied"] == 0)
    i_last, last = [(i, d) for i, d in ev if d["edited_tensors_applied"]][-1]
    s = json.loads([r for r in L if r["role"] == "score"][-1]["text"])
    assert last["edited_tensors_applied"] == s["edited_tensors"]
    return {"L": L, "base": base, "after": last, "i_after": i_last, "score": s}


def quoted(item, solver, idx, text):
    """`text`, after checking that it occurs verbatim (raw or JSON-escaped) in message `idx` of that transcript."""
    t = transcript(item, solver)[idx]["text"]
    assert text in t or json.dumps(text)[1:-1] in t, f"not found in {item}/{solver} message {idx}: {text[:60]!r}"
    return text


def fenced(t):
    m = re.search(r"```[^\n`]*\n(.*?)(?:```|\Z)", t, re.S)
    return (m.group(1) if m else t).strip("\n")


def header(r):
    s = r["score"]
    return {"base": round(100 * s["base_score"]), "after": round(100 * s["heldout_score"]),
            "tensors": s["edited_tensors"], "changed": 100 * s["changed_fraction"]}


def pct(x):
    return f"{x:.2g}%" if x < 1 else f"{x:.1f}".rstrip("0").rstrip(".") + "%"


AGENT = {"gpt-6.1-sol": "GPT-6.1 Sol", "claude-opus-5-5-high": "Claude Opus 5.5", "claude-sonnet-5-5-high": "Claude Sonnet 5.5"}


def subject(item):
    return json.loads((MAIN / "items" / "v8" / f"{item}.json").read_text())["slots"]["M1"]["base"].split("/")[-1]


def byline(item, solver, h):
    t = "tensor" if h["tensors"] == 1 else "tensors"
    return (f"{AGENT[solver]} edits {subject(item)}: {h['tensors']} {t} ({pct(h['changed'])} of the weights); "
            f"held-out accuracy {h['base']}% \u2192 {h['after']}%")


# ---- SVG ------------------------------------------------------------------------------------------------------------
SVG_PRE = '<svg xmlns="http://www.w3.org/2000/svg" '


def svg_shapes(svg):
    """(width, height, [matplotlib patches]) for the circles, ellipses, polygons and rects of a simple SVG."""
    def num(tag, k, d=None):
        m = re.search(rf'\b{k}="([-\d.]+)"', tag)
        return float(m.group(1)) if m else d
    head = re.search(r"<svg[^>]*>", svg).group(0)
    w, h = num(head, "width", 100), num(head, "height", 100)
    out = []
    for tag in re.findall(r"<(?:circle|ellipse|polygon|rect)\b[^>]*>", svg):
        fill = re.search(r'fill="([^"]+)"', tag).group(1)
        kind = tag[1:].split()[0]
        if kind == "circle":
            out.append(Circle((num(tag, "cx"), num(tag, "cy")), num(tag, "r"), fc=fill, ec="none"))
        elif kind == "ellipse":
            out.append(Ellipse((num(tag, "cx"), num(tag, "cy")), 2 * num(tag, "rx"), 2 * num(tag, "ry"), fc=fill, ec="none"))
        elif kind == "polygon":
            pts = [tuple(map(float, p.split(","))) for p in re.search(r'points="([^"]+)"', tag).group(1).split()]
            out.append(Polygon(pts, closed=True, fc=fill, ec="none"))
        else:
            out.append(Rectangle((num(tag, "x", 0), num(tag, "y", 0)), num(tag, "width"), num(tag, "height"), fc=fill, ec="none"))
    return w, h, out


def svg_example():
    item, solver = "K_svg__open", "claude-sonnet-5-5-high"
    r = run(item, solver)
    T = S.svg_tasks("practice", 3)
    # the unedited model's full replies to the three prompts, printed by Claude Haiku 5.5 in its run (message 9)
    src = ("K_svg__open", "claude-haiku-5-5-high", 9)
    before = [None,  # prompt 1: an endless path that never closes (the reply runs to the token limit)
              quoted(*src, ' width="150" height="150">\n    <circle cx="75" cy="75" r="75" fill="#4CAF50"/>\n    <polygon '
                           'points="75,75 90,85 105,65 120,65 135,75 150,75" fill="#FFDAB9"/>\n</svg>'),
              quoted(*src, ' width="150" height="150">\n    <circle cx="75" cy="75" r="60" fill="#FFA500"/>\n    <polygon '
                           'points="45,90 85,105 125,105 125,90" fill="#FFD700"/>\n</svg>')]
    quoted(*src, ' width="100%" height="100%">\n<path d="M50 50 L70 50 L90 50 L110 50 L130 50')
    assert r["base"]["score"] == 0.0  # every unedited reply fails
    after = r["after"]["first"]
    verdict = [S.svg_score(SVG_PRE + a, *T[i][1:]) == 1.0 for i, a in enumerate(after)]
    assert all(verdict)
    return {"item": item, "solver": solver, "h": header(r), "prompts": [p for p, *_ in T],
            "before": [SVG_PRE + b if b else None for b in before], "after": [SVG_PRE + a for a in after]}


# ---- OpenSCAD -------------------------------------------------------------------------------------------------------
def scad_example(tmp):
    item, solver = "K_scad__open", "claude-opus-5-5-high"
    r = run(item, solver)
    desc, ref = S.scad_tasks("practice", 3)[2]
    # the unedited model's full reply, printed by Claude Haiku 5.5 in its run (message 14); the logged one is cut
    before = quoted("K_scad__open", "claude-haiku-5-5-high", 14,
                    "module box() {\n    // Dimensions\n    $fn = 64;\n    width = 36;\n    height = 40;\n    depth = 19;\n"
                    "    hole_radius = 4;\n\n    // Box\n    cube([width, height, depth], center = [0, 0, 0]);\n\n    // Hole\n"
                    "    cylinder(h = height, r = hole_radius, center = [width/2, height/2, depth], $fn = 64);\n}\n\nbox();")
    after = fenced(r["after"]["first"][2])
    assert r["base"]["score"] == 0.0 and r["after"]["score"] == 1.0
    vols = {}
    for name, code in (("ref", ref), ("before", before), ("after", after)):
        assert not re.search(r"\b(import|include|use|surface)\b", code)
        (tmp / f"{name}.scad").write_text(code)
        subprocess.run(["openscad", "-o", f"{name}.stl", f"{name}.scad"], cwd=tmp, check=True, capture_output=True, timeout=120)
        vols[name] = stl_volume(tmp / f"{name}.stl")
    # one camera for all three, framing the union of their bounding boxes
    lo = [min(b[0][k] for b in map(stl_bbox, tmp.glob("*.stl"))) for k in range(3)]
    hi = [max(b[1][k] for b in map(stl_bbox, tmp.glob("*.stl"))) for k in range(3)]
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    size = sum((b - a) ** 2 for a, b in zip(lo, hi)) ** 0.5
    v = [0.62, -0.55, 0.55]
    nv = sum(x * x for x in v) ** 0.5
    cam = ",".join(f"{x:.2f}" for x in [*(ci + size * 2.2 * vi / nv for ci, vi in zip(c, v)), *c])
    for name in ("before", "after"):
        subprocess.run(["openscad", "-o", f"{name}.png", "--imgsize=900,760", "--colorscheme=Tomorrow", "--view=axes",
                        "--projection=p", f"--camera={cam}", f"{name}.scad"], cwd=tmp, check=True, capture_output=True,
                       timeout=120)
    assert abs(vols["after"] - vols["ref"]) < 1 and abs(vols["before"] - vols["ref"]) > 0.04 * vols["ref"]
    return {"item": item, "solver": solver, "h": header(r), "prompt": desc[0].upper() + desc[1:] + ".",
            "before_png": tmp / "before.png", "after_png": tmp / "after.png", "vol": vols}


def stl_points(p):
    return [[float(x) for x in l.split()[1:]] for l in open(p) if l.strip().startswith("vertex")]


def stl_bbox(p):
    pts = stl_points(p)
    return [min(q[k] for q in pts) for k in range(3)], [max(q[k] for q in pts) for k in range(3)]


def stl_volume(p):
    pts = stl_points(p)
    vol = 0.0
    for a, b, c in zip(pts[0::3], pts[1::3], pts[2::3]):
        vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6
    return abs(vol)


# ---- spreadsheet ----------------------------------------------------------------------------------------------------
def excel_example():
    item, solver = "K_excel__open", "gpt-6.1-sol"
    r = run(item, solver)
    rows, q, truth = S.excel_tasks("practice", 2)[1]
    b, a = r["base"]["first"][1].strip(), r["after"]["first"][1].strip()
    assert (b, a) == ("=SUM(B2:B9)", '=SUMIF(A2:A9,"inland",B2:B9)')
    vb, va = sum(x[1] for x in rows), sum(x[1] for x in rows if x[0] == "inland")
    assert va == truth != vb and r["base"]["score"] == 0.0 and r["after"]["score"] == 1.0
    return {"item": item, "solver": solver, "h": header(r), "rows": rows, "q": q, "truth": truth,
            "before": (b, vb, [True] * 8), "after": (a, va, [x[0] == "inland" for x in rows])}


# ---- the visual figure ----------------------------------------------------------------------------------------------
def draw_svg(ax, svg):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cfcfcf")
        s.set_linewidth(0.6)
    if svg is None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, 'no image: one path\n<path d="M50 50\nL70 50 L90 50\nL110 50 L130 50 \u2026"\nruns to the token\nlimit and never closes',
                ha="center", va="center", fontsize=4.6, color=INK2, linespacing=1.25, **MONO)
        return
    w, h, shapes = svg_shapes(svg)
    for p in shapes:
        ax.add_patch(p)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")


def draw_sheet(fig, x0, y0, w, rows, used, formula, value, ok):
    """A small spreadsheet: header row, rows 2-9, columns A-C; the cells the formula sums are shaded."""
    ax = fig.add_axes([x0, y0, w, w * 1.02 * fig.get_figwidth() / fig.get_figheight()])
    ax.set_xlim(0, 4)
    ax.set_ylim(10, 0)
    ax.axis("off")
    cw = [0.55, 1.35, 1.05, 1.05]
    xs = [0]
    for c in cw:
        xs.append(xs[-1] + c)
    sx = 4 / xs[-1]
    xs = [x * sx for x in xs]
    for j, name in enumerate(["", "A", "B", "C"]):
        ax.add_patch(Rectangle((xs[j], 0), xs[j + 1] - xs[j], 1, fc="#f4f4f4", ec="#d6d6d6", lw=0.5))
        ax.text((xs[j] + xs[j + 1]) / 2, 0.5, name, ha="center", va="center", fontsize=5.6, color=INK2)
    for i, (a, b, c) in enumerate(rows):
        y = i + 1
        cells = [str(i + 2), a, str(b), str(c)]
        for j, v in enumerate(cells):
            fc = "#f4f4f4" if j == 0 else (HIL if (j == 2 and used[i]) else "white")
            ax.add_patch(Rectangle((xs[j], y), xs[j + 1] - xs[j], 1, fc=fc, ec="#d6d6d6", lw=0.5))
            ax.text((xs[j] + xs[j + 1]) / 2, y + 0.52, v, ha="center", va="center", fontsize=5.6,
                    color=INK2 if j == 0 else INK)
    mark, col = (TICK, GOOD) if ok else (CROSS, BAD)
    ax.text(0, 9.3, formula, ha="left", va="top", fontsize=5.6, color=INK, **MONO)
    ax.text(0, 10.25, f"{mark} gives {value}", ha="left", va="top", fontsize=6.3, color=col, fontweight="bold")
    return ax


def visual_figure(svg, scad, xl, path_pdf, path_png=None):
    W, H = 6.5, 5.15
    fig = plt.figure(figsize=(W, H))
    fx, fy = 1 / W, 1 / H  # inches -> figure fraction

    # (a) SVG -------------------------------------------------------------------------------------------------------
    top = H - 0.05
    fig.text(0.0, (top - 0.12) * fy, "(a) Drawing in SVG", fontsize=8.5, fontweight="bold", va="center")
    fig.text(0.0, (top - 0.30) * fy, byline(svg["item"], svg["solver"], svg["h"]), fontsize=6.6, color=INK2, va="center")
    gw, side, gap = W / 3, 0.92, 0.12
    for k in range(3):
        gx = k * gw + 0.04
        fig.text(gx * fx, (top - 0.55) * fy, "\u201c" + svg["prompts"][k] + "\u201d", fontsize=6.6, style="italic", va="center")
        for j, (word, s) in enumerate((("Before", svg["before"][k]), ("After", svg["after"][k]))):
            x = gx + j * (side + gap)
            ax = fig.add_axes([x * fx, (top - 0.80 - side) * fy, side * fx, side * fy])
            draw_svg(ax, s)
            ok = j == 1
            fig.text(x * fx, (top - 0.78) * fy, word, fontsize=6.6, fontweight="bold", va="bottom")
            fig.text((x + side) * fx, (top - 0.78) * fy, TICK if ok else CROSS, fontsize=7, color=GOOD if ok else BAD,
                     ha="right", va="bottom", fontweight="bold")
    notes = ["", "a green circle and a peach blob", "an orange circle, a yellow\nquadrilateral; no triangle"]
    for k in range(3):
        if notes[k]:
            fig.text((k * gw + 0.04) * fx, (top - 0.86 - side) * fy, notes[k], fontsize=5.6, color=INK2, va="top",
                     linespacing=1.15)
    y_sep = top - 0.80 - side - 0.36
    fig.add_artist(plt.Line2D([0, 1], [y_sep * fy, y_sep * fy], color=GRID, lw=0.8))

    # (b) OpenSCAD --------------------------------------------------------------------------------------------------
    top2 = y_sep - 0.05
    half = W / 2
    fig.text(0.0, (top2 - 0.12) * fy, "(b) 3D modelling in OpenSCAD", fontsize=8.5, fontweight="bold", va="center")
    for i, line in enumerate(textwrap.wrap(byline(scad["item"], scad["solver"], scad["h"]), 62)):
        fig.text(0.0, (top2 - 0.30 - 0.13 * i) * fy, line, fontsize=6.6, color=INK2, va="center")
    for i, line in enumerate(textwrap.wrap("\u201c" + scad["prompt"] + "\u201d", 66)):
        fig.text(0.0, (top2 - 0.62 - 0.13 * i) * fy, line, fontsize=6.6, style="italic", va="center")
    iw = (half - 0.25) / 2
    ih = iw * 760 / 900
    yb = top2 - 1.12 - ih
    v = scad["vol"]
    for j, (word, png, ok, note) in enumerate((
            ("Before", scad["before_png"], False, "a 40 mm peg at the corner,\nno hole"),
            ("After", scad["after_png"], True, f"volume {v['after']:,.0f} mm\u00b3,\nas the reference part"))):
        x = j * (iw + 0.08)
        ax = fig.add_axes([x * fx, yb * fy, iw * fx, ih * fy])
        ax.imshow(mpimg.imread(png))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cfcfcf")
            s.set_linewidth(0.6)
        fig.text(x * fx, (yb + ih + 0.03) * fy, word, fontsize=6.6, fontweight="bold", va="bottom")
        fig.text((x + iw) * fx, (yb + ih + 0.03) * fy, TICK if ok else CROSS, fontsize=7, color=GOOD if ok else BAD,
                 ha="right", va="bottom", fontweight="bold")
        fig.text(x * fx, (yb - 0.05) * fy, note, fontsize=5.6, color=INK2, va="top", linespacing=1.15)

    # (c) spreadsheet -----------------------------------------------------------------------------------------------
    x0 = half + 0.12
    fig.add_artist(plt.Line2D([(half - 0.02) * fx] * 2, [0.02, (top2 - 0.02) * fy], color=GRID, lw=0.8))
    fig.text(x0 * fx, (top2 - 0.12) * fy, "(c) Spreadsheet formulas", fontsize=8.5, fontweight="bold", va="center")
    for i, line in enumerate(textwrap.wrap(byline(xl["item"], xl["solver"], xl["h"]), 60)):
        fig.text(x0 * fx, (top2 - 0.30 - 0.13 * i) * fy, line, fontsize=6.6, color=INK2, va="center")
    for i, line in enumerate(textwrap.wrap("\u201cWrite a single Excel formula that computes " + xl["q"] + ".\u201d", 64)):
        fig.text(x0 * fx, (top2 - 0.62 - 0.13 * i) * fy, line, fontsize=6.6, style="italic", va="center")
    sw = (W - x0 - 0.22) / 2
    for j, (word, (f, val, used)) in enumerate((("Before", xl["before"]), ("After", xl["after"]))):
        x = x0 + j * (sw + 0.12)
        ok = val == xl["truth"]
        sh = sw * 1.02
        ys = top2 - 1.12 - sh
        draw_sheet(fig, x * fx, ys * fy, sw * fx, xl["rows"], used, f, val, ok)
        fig.text(x * fx, (ys + sh + 0.03) * fy, word, fontsize=6.6, fontweight="bold", va="bottom")
    fig.text(x0 * fx, 0.012, f"Shaded: the cells each formula adds up. The right answer is {xl['truth']}.", fontsize=5.6,
             color=INK2, va="bottom")

    fig.savefig(path_pdf)
    if path_png:
        fig.savefig(path_png, dpi=200)
    plt.close(fig)


# ---- GPT-6.1 Sol: one SVG drawing and one SQL query (fig_sol.pdf) ----------------------------------------------------
def sol_svg_example():
    item, solver = "K_svg__open", "gpt-6.1-sol"
    r = run(item, solver)
    T = S.svg_tasks("practice", 3)
    k = 1  # "Draw a green triangle above a purple circle."
    # both replies close their <svg> inside the logged 300 characters; only the unedited model's prose after it is cut
    before = re.search(r"<svg.*?</svg>", SVG_PRE + r["base"]["first"][k], re.S).group(0)
    after = re.search(r"<svg.*?</svg>", SVG_PRE + r["after"]["first"][k], re.S).group(0)
    assert r["base"]["score"] == 0.0 and r["after"]["score"] == 1.0
    assert S.svg_score(before, *T[k][1:]) == 0.0 and S.svg_score(after, *T[k][1:]) == 1.0
    return {"item": item, "solver": solver, "h": header(r), "prompt": T[k][0], "before": before, "after": after}


def sol_sql_example():
    item, solver = "K_sql__open", "gpt-6.1-sol"
    r = run(item, solver)
    cols, rows, q, gold = S.sql_tasks("practice", 3)[1]  # "How many players have team Lions?"
    b = r["base"]["first"][1].split("</sql>")[0].strip()
    a = r["after"]["first"][1].split("</sql>")[0].strip()
    assert (b, a) == ("SELECT COUNT(*) FROM t;", "SELECT COUNT(*) FROM t WHERE Team = 'Lions';")
    rb, ra, rg = (S._sql_exec(cols, rows, x) for x in (b, a, gold))  # the benchmark's own SQLite executor
    assert ra == rg != rb and r["after"]["score"] == 1.0

    def count(res):
        return int(res[0].strip("(),"))
    return {"item": item, "solver": solver, "h": header(r), "q": q, "cols": cols, "rows": rows,
            "before": (b, count(rb)), "after": (a, count(ra)), "truth": count(rg),
            "shade": [x[2] == "Lions" for x in rows]}


def draw_table(fig, x0, y0, w, h, cols, rows, shade):
    """A small table: the header row and the data rows; shaded rows are the ones the right query counts."""
    ax = fig.add_axes([x0, y0, w, h])
    n = len(rows)
    ax.set_xlim(0, 1)
    ax.set_ylim(n + 1, 0)
    ax.axis("off")
    cw = [0.30, 0.26, 0.22, 0.22]
    xs = [0]
    for c in cw:
        xs.append(xs[-1] + c)
    for j, name in enumerate(cols):
        ax.add_patch(Rectangle((xs[j], 0), cw[j], 1, fc="#f4f4f4", ec="#d6d6d6", lw=0.5))
        ax.text(xs[j] + cw[j] / 2, 0.55, name, ha="center", va="center", fontsize=5.4, color=INK2)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            ax.add_patch(Rectangle((xs[j], i + 1), cw[j], 1, fc=HIL if shade[i] else "white", ec="#d6d6d6", lw=0.5))
            ax.text(xs[j] + cw[j] / 2, i + 1.55, str(v), ha="center", va="center", fontsize=5.4, color=INK)


def sol_figure(svg, sql, path_pdf, path_png=None):
    W, H = 6.5, 2.42
    fig = plt.figure(figsize=(W, H))
    fx, fy = 1 / W, 1 / H
    top = H - 0.05
    half = W / 2

    # (a) SVG --------------------------------------------------------------------------------------------------------
    fig.text(0.0, (top - 0.12) * fy, "(a) Drawing in SVG", fontsize=8.5, fontweight="bold", va="center")
    for i, line in enumerate(textwrap.wrap(byline(svg["item"], svg["solver"], svg["h"]), 64)):
        fig.text(0.0, (top - 0.30 - 0.13 * i) * fy, line, fontsize=6.6, color=INK2, va="center")
    fig.text(0.0, (top - 0.62) * fy, "“" + svg["prompt"] + "”", fontsize=6.6, style="italic", va="center")
    side, gap = 1.22, 0.16
    for j, (word, s, ok, note) in enumerate((("Before", svg["before"], False, "a green circle and a peach blob"),
                                             ("After", svg["after"], True, "a green triangle above a purple circle"))):
        x = 0.02 + j * (side + gap)
        ax = fig.add_axes([x * fx, (top - 0.84 - side) * fy, side * fx, side * fy])
        draw_svg(ax, s)
        fig.text(x * fx, (top - 0.82) * fy, word, fontsize=6.6, fontweight="bold", va="bottom")
        fig.text((x + side) * fx, (top - 0.82) * fy, TICK if ok else CROSS, fontsize=7, color=GOOD if ok else BAD,
                 ha="right", va="bottom", fontweight="bold")
        fig.text(x * fx, (top - 0.89 - side) * fy, note, fontsize=5.6, color=INK2, va="top")

    # (b) SQL --------------------------------------------------------------------------------------------------------
    x0 = half + 0.14
    fig.add_artist(plt.Line2D([(half + 0.02) * fx] * 2, [0.03, (top - 0.02) * fy], color=GRID, lw=0.8))
    fig.text(x0 * fx, (top - 0.12) * fy, "(b) Answering questions in SQL", fontsize=8.5, fontweight="bold", va="center")
    for i, line in enumerate(textwrap.wrap(byline(sql["item"], sql["solver"], sql["h"]), 60)):
        fig.text(x0 * fx, (top - 0.30 - 0.13 * i) * fy, line, fontsize=6.6, color=INK2, va="center")
    fig.text(x0 * fx, (top - 0.62) * fy, "“" + sql["q"] + "”  (table t)", fontsize=6.6, style="italic",
             va="center")
    tw, th = 1.42, 1.22
    draw_table(fig, x0 * fx, (top - 0.84 - th) * fy, tw * fx, th * fy, sql["cols"], sql["rows"], sql["shade"])
    fig.text(x0 * fx, (top - 0.89 - th) * fy, "Shaded: the rows with team Lions.", fontsize=5.6, color=INK2, va="top")
    qx = x0 + tw + 0.16
    for j, (word, (query, n), ok) in enumerate((("Before", sql["before"], False), ("After", sql["after"], True))):
        y = top - 0.84 - j * 0.66
        fig.text(qx * fx, y * fy, word, fontsize=6.6, fontweight="bold", va="top")
        shown = query.replace(" WHERE", "\nWHERE")
        fig.text((qx + 0.02) * fx, (y - 0.20) * fy, shown, fontsize=5.8, color=INK, va="top", linespacing=1.3,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#f5f3f5", ec="none"), **MONO)
        lines = shown.count("\n") + 1
        mark, col = (TICK, GOOD) if ok else (CROSS, BAD)
        fig.text(qx * fx, (y - 0.30 - 0.12 * lines) * fy, f"{mark} returns {n}", fontsize=6.3, color=col,
                 fontweight="bold", va="top")
    fig.savefig(path_pdf)
    if path_png:
        fig.savefig(path_png, dpi=200)
    plt.close(fig)


# ---- the code figure (LaTeX) ----------------------------------------------------------------------------------------
def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%", "$": r"\$", "#": r"\#",
           "&": r"\&", "^": r"\^{}", "~": r"\~{}", "<": r"\textless{}", ">": r"\textgreater{}"}
    return "".join(rep.get(c, c) for c in s)


def code_examples():
    ex = []
    # Python: GPT-6.1 Sol. The unedited model's full reply is printed in Sol's own run (message 9); the edited model's
    # reply to the same practice prompt is Sol's last generate request (pass 50, message 315), made with the edit it
    # submitted (no tensor changed between it and the answer).
    item, solver = "K_prog_python__open", "gpt-6.1-sol"
    r = run(item, solver)
    stmt = next(s for s, _ in S.prog_tasks("practice") if "While n is not 1" in s)
    before = quoted(item, solver, 9,
                    "import sys\n\ndef hailstone_sequence(n):\n    count = 0\n    while n != 1:\n        print(n)\n"
                    "        if_n_even = n % 2 == 0\n        if_n_odd = not if_n_even\n        \n        if_n_even:\n"
                    "            n //= 2\n        else:\n            n = 3 * n + 1\n    \n        count += 1\n    return count\n\n"
                    "if __name__ == \"__main__\":\n    try:\n        n = int(sys.stdin.readline().strip())\n"
                    "        result = hailstone_sequence(n)\n        print(result)\n    except ValueError:\n"
                    "        print(\"Invalid input\")")
    after = quoted(item, solver, 315,
                   "import sys\n\n# Initialize variables\nnum = int(input().strip())\nreplacements = 0\n\nwhile num != 1:\n"
                   "    if num % 2 == 0:\n        num //= 2\n    else:\n        num = 3 * num + 1\n    replacements += 1\n\n"
                   "print(replacements)")
    L = r["L"]
    assert L[317]["role"] == "solver" and L[317]["text"].startswith("```answer")
    # between the generate request (311) and the answer (317) the only code run reads the generations back
    assert [i for i in range(312, 317) if L[i]["role"] == "solver"] == [314]
    assert L[314]["text"].startswith("```python\ngs50=[json.loads(x)['text'] for x in open('/work/fwd_50/generations")
    assert L[308]["text"].strip() == "```python\nqscale(1.145,range(14))\n```" and L[311]["text"].startswith("```forward")
    try:
        compile(before, "main.py", "exec")
        raise AssertionError("expected a syntax error")
    except SyntaxError as e:
        assert e.lineno == 10
    for inp, want in next(t for s, t in S.prog_tasks("practice") if s == stmt):
        got = subprocess.run([sys.executable, "-I", "-c", after], input=inp, capture_output=True, text=True, timeout=10).stdout
        assert got == want
    ex.append({"title": "Python", "item": item, "solver": solver, "h": header(r), "task": stmt,
               "before": before, "after": after, "bad_lines": {10},
               "vb": "SyntaxError (line 10)",
               "va": "correct on all four tests"})

    # clingo: Claude Sonnet 5.5. The full unedited reply (512 tokens) is printed in Claude Fable 5.1's run (message 9).
    item, solver = "K_clingo__open", "claude-sonnet-5-5-high"
    r = run(item, solver)
    q = S.clingo_tasks("practice", 1)[0][0]
    full = quoted(item, "claude-fable-5-1", 9, "% Define the graph\ncolor(1, r).\ncolor(1, g).\ncolor(1, b).\n\ncolor(2, r).")
    quoted(item, "claude-fable-5-1", 9, "% For each edge, enforce that the two nodes have different colors\n% This is done via a "
                                        "constraint that for each edge, the two nodes must have different colors")
    before = ("% Define the graph\ncolor(1, r).\ncolor(1, g).\ncolor(1, b).\n\u00a7[the same three facts for nodes 2 to 5]\u00a7\n"
              "% Define the edges\nedge(1, 3).\nedge(1, 4).\nedge(1, 5).\nedge(2, 3).\n\n"
              "% For each edge, enforce that the two nodes have different colors\n"
              "% This is done via a constraint that for each edge, the two nodes must have different colors\n"
              "\u00a7[nine more comments restating this, then the facts again, until the 512-token limit]\u00a7")
    assert full
    after = fenced(r["after"]["first"][0])
    assert r["base"]["score"] == 0.0 and r["after"]["score"] == 1.0
    ex.append({"title": "clingo", "item": item, "solver": solver, "h": header(r),
               "task": q.split(" Represent")[0], "before": before, "after": after, "bad_lines": set(),
               "vb": "one answer set, every node in every colour",
               "va": "exactly the 48 proper colourings"})

    # Pike: Claude Opus 5.5. Both replies are short enough to be logged in full (under 300 characters).
    item, solver = "K_prog_pike__open", "claude-opus-5-5-high"
    r = run(item, solver)
    stmt = S.prog_tasks("practice")[0][0]
    b, a = r["base"]["first"][0], r["after"]["first"][0]
    assert len(b) < 300 and len(a) < 300 and r["base"]["score"] == 0.0 and r["after"]["score"] == 1.0
    ex.append({"title": "Pike", "item": item, "solver": solver, "h": header(r), "task": stmt,
               "before": fenced(b), "after": fenced(a), "bad_lines": set(),
               "vb": "fails to compile",
               "va": "correct on all four tests"})
    return ex


def listing(code, bad):
    lines = []
    for n, line in enumerate(code.split("\n"), 1):
        if line.startswith("\u00a7") and line.endswith("\u00a7"):  # an elision note of ours, not model output
            lines.append(r"(*@\textrm{\textit{\color{black!55}" + tex_escape(line.strip("\u00a7")) + r"}}@*)")
        elif n in bad:
            body = line.lstrip(" ")  # the indentation stays outside the escape, where listings keeps it
            lines.append(line[:len(line) - len(body)] + r"(*@\textcolor{red!65!black}{" + tex_escape(body) + r"}@*)")
        else:
            lines.append(line)
    return "\\begin{lstlisting}\n" + "\n".join(lines) + "\n\\end{lstlisting}"


def code_tex(ex):
    out = ["% Generated by figures/make_examples.py. Do not edit by hand.", r"\begingroup\setlength{\fboxsep}{0pt}"]
    for k, e in enumerate(ex):
        h = e["h"]
        t = "tensor" if h["tensors"] == 1 else "tensors"
        out.append(r"\noindent{\small\textbf{(" + "abc"[k] + ") " + e["title"] + r"}}\hfill{\scriptsize\color{black!60}"
                   + f"{AGENT[e['solver']]} edits {subject(e['item'])}: {h['tensors']} {t} ({tex_escape(pct(h['changed']))} of the weights); "
                   + f"held-out accuracy {h['base']}\\% $\\rightarrow$ {h['after']}\\%" + "}\\\\[1pt]")
        out.append(r"{\scriptsize\textit{``" + tex_escape(e["task"]) + r"''}}\\[2pt]")
        out.append(r"\begin{minipage}[t]{0.485\linewidth}")
        out.append(r"{\scriptsize\textbf{Before}\hfill\textcolor{red!65!black}{$\times$ " + e["vb"] + r"}}\par\vspace{1pt}")
        out.append(listing(e["before"], e["bad_lines"]))
        out.append(r"\end{minipage}\hfill")
        out.append(r"\begin{minipage}[t]{0.485\linewidth}")
        out.append(r"{\scriptsize\textbf{After}\hfill\textcolor{green!40!black}{\checkmark\ " + e["va"] + r"}}\par\vspace{1pt}")
        out.append(listing(e["after"], set()))
        out.append(r"\end{minipage}")
        if k < len(ex) - 1:
            out.append(r"\par\vspace{2pt}{\color{black!12}\rule{\linewidth}{0.4pt}}\par\vspace{1pt}")
    out.append(r"\endgroup")
    return "\n".join(out) + "\n"


def main():
    args = [a for a in sys.argv[2:] if a != "--all"]
    preview = Path(args[0]) if args else None
    sol_figure(sol_svg_example(), sol_sql_example(), OUT / "fig_sol.pdf", preview / "fig_sol.png" if preview else None)
    print("wrote", OUT / "fig_sol.pdf")
    if "--all" in sys.argv:
        with tempfile.TemporaryDirectory() as d:
            svg, scad, xl = svg_example(), scad_example(Path(d)), excel_example()
            visual_figure(svg, scad, xl, OUT / "fig_examples.pdf", preview / "fig_examples.png" if preview else None)
        (OUT / "code_examples.tex").write_text(code_tex(code_examples()))
        print("wrote", OUT / "fig_examples.pdf", "and", OUT / "code_examples.tex")


if __name__ == "__main__":
    main()
