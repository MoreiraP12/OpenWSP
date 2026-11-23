"""Figures, tables and number macros of the WSP-Bench paper, from data/paper_v8.json.

    cd paper && python3 scripts/make_figures.py

Writes figures/*.pdf (vector, fonts embedded), figures/png/*.png (previews), tables/leaderboard.tex and
tables/numbers.tex (one macro per number quoted in the text, so the prose follows the data). The data file is made in
the private development repo by `harness/paper_data.py`; every score is an agent's first pass.

Design language: Artificial Analysis-style cards. Each figure is a white card with a light border, a small coloured
square, a serif title (Source Serif 4) and a grey subtitle (Inter); bars carry their value in white and take the
colour of the model's lab; gridlines are light and dashed. Fonts come from the TeX Live tree, so the PDF and the
figures share them; without TeX Live the script falls back to Georgia and Arial.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
DATA = json.loads((HERE / "data" / "paper_v8.json").read_text())
FIG, TAB = HERE / "figures", HERE / "tables"
(FIG / "png").mkdir(parents=True, exist_ok=True)
TAB.mkdir(exist_ok=True)

# ---------------------------------------------------------------- what is shown
# Agents in the paper; short keys name them in the macros (\res{sol}{index}).
AGENTS = {
    "gpt-6.1-sol": ("sol", "GPT-6.1 Sol", "Sol"),
    "claude-opus-5-5": ("opus", "Claude Opus 5.5", "Opus 5.5"),
    "gpt-6.1-terra": ("terra", "GPT-6.1 Terra", "Terra"),
    "claude-sonnet-5-5": ("sonnet", "Claude Sonnet 5.5", "Sonnet 5.5"),
    "claude-haiku-5-5": ("haiku", "Claude Haiku 5.5", "Haiku 5.5"),
    "qwen3.8-27b": ("qwenthink", "Qwen3.8-27B (thinking)", "Qwen3.8 thinking"),
    "gpt-6-luna": ("luna", "GPT-6 Luna", "Luna"),
    "minimax-m2.7": ("minimax", "MiniMax-M2.7", "MiniMax-M2.7"),
    "deepseek-v4-flash": ("deepseek", "DeepSeek-V4-Flash", "DeepSeek-V4"),
    "qwen3.8-27b-nothink": ("qwen", "Qwen3.8-27B (no thinking)", "Qwen3.8"),
    "inkling-2-mini": ("inkling", "Inkling 2 Mini", "Inkling 2 Mini"),  # shown once complete
}
M = [m for m in DATA["models"] if m["id"] in AGENTS and m["complete"]]
for m in M:
    m["key"], m["label"], m["short"] = AGENTS[m["id"]]
BY_INDEX = sorted(M, key=lambda m: -m["index"])

FAMILIES = ["read", "edit", "skill"]
CLASS_ORDER = DATA["groups"]["read"] + DATA["groups"]["edit"] + DATA["groups"]["skill"]
SHORT = {"O2": "Data mixture", "O3": "Epochs", "P1": "Merge weights", "P2": "Transplant", "P3": "Edit kind",
         "P3o": "Edit, free text", "S2": "Sycophant twin", "S3": "Backdoor twin", "S6": "Trigger phrase",
         "W1": "Remove backdoor", "W2": "Sparse transplant", "W3": "Add math", "W4": "Word association",
         "W4x": "Associations, harder", "W5": "Phrase detector", "K": "Skills"}

# ---------------------------------------------------------------- look
INK, MUTED, FAINT = "#111111", "#6B6B6B", "#9A9A9A"
GRID, AXIS, EDGE = "#CCCCCC", "#949494", "#E3E3E3"
LAB = {"OpenAI": "#1F1F1F", "Anthropic": "#CC785C", "Alibaba": "#FF7018", "DeepSeek": "#2243E6",
       "MiniMax": "#EB3568", "Thinking Machines": "#676767", "Google": "#34A853", "Xiaomi": "#FF6900"}
LAB_TEX = {k: "lab" + k.replace(" ", "") for k in LAB}  # colour names defined in main.tex
PURPLE, ORANGE, TEAL, YELLOW = "#7F4BF3", "#F2754C", "#30C695", "#F2C94C"
FAM = {"read": PURPLE, "edit": ORANGE, "skill": TEAL}  # same as the diagrams (figures/overview.tex)
SKILL_GROUPS = {"general": ("General skills", "#1B8A64"), "whole programs": ("Whole programs", TEAL),
                "function completion": ("Function completion", "#8FDCC0"),
                "HumanEval functions": ("HumanEval functions", "#C6EEDF")}
HEAT = LinearSegmentedColormap.from_list("aa", ["#FFFFFF", "#ECE4FE", "#BDA2F9", PURPLE, "#3B1A7A"])
QUAD_GOOD, QUAD_BAD = "#E7F6E1", "#F4F4F4"


FONTS = {  # font file in TeX Live -> weights used here; converted once to TrueType in .fontcache/ (not in git)
    "Inter-Regular.otf": None, "Inter-Medium.otf": None, "Inter-SemiBold.otf": None, "Inter-Bold.otf": None,
    "SourceSerif4-Regular.otf": None, "SourceSerif4-Semibold.otf": None,
}


def _fonts():
    """Inter and Source Serif 4 from TeX Live, converted to TrueType (scripts/otf2ttf.py) so the PDFs embed them as
    plain Type 42 fonts; return the (sans, serif) family names, or Arial and Georgia without TeX Live."""
    cache = HERE / ".fontcache"
    cache.mkdir(exist_ok=True)
    for name in FONTS:
        ttf = cache / (name[:-4] + ".ttf")
        if not ttf.exists():
            try:
                src = subprocess.run(["kpsewhich", name], capture_output=True, text=True).stdout.strip()
                if src:
                    import otf2ttf  # noqa: PLC0415 - only needed on the first run
                    otf2ttf.otf_to_ttf(src, str(ttf))
            except Exception as e:  # noqa: BLE001 - fall back to system fonts
                print(f"font {name}: {e}")
        if ttf.exists():
            fm.fontManager.addfont(str(ttf))
    names = {f.name for f in fm.fontManager.ttflist}
    return ("Inter" if "Inter" in names else "Arial"), ("Source Serif 4" if "Source Serif 4" in names else "Georgia")


SANS, SERIF = _fonts()
mpl.rcParams.update({
    "font.family": SANS, "font.size": 7.5, "text.color": INK, "axes.labelcolor": INK, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 6.8, "xtick.color": INK, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "standard", "figure.dpi": 150, "axes.facecolor": "none", "figure.facecolor": "white",
})
DASH = (0, (2, 3))


def lab_color(m):
    return LAB.get(m["lab"], "#8A8A8A")


# ---------------------------------------------------------------- the card
class Card:
    """A figure drawn as a white card: border, colour square, serif title, grey subtitle, legend row, wordmark.
    Positions are in inches from the card's bottom-left corner."""

    def __init__(self, w, h, title, subtitle, square, legend=()):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        f, t = self.fig, self.fig.dpi_scale_trans
        f.patches.append(FancyBboxPatch((0.02, 0.02), w - 0.04, h - 0.04, boxstyle="round,pad=0,rounding_size=0.08",
                                        transform=t, facecolor="white", edgecolor=EDGE, lw=0.8, zorder=-10))
        x, top = 0.2, h - 0.2
        f.patches.append(Rectangle((x, top - 0.115), 0.1, 0.1, transform=t, color=square, zorder=5))
        f.text(x + 0.17, top - 0.065, title, transform=t, family=SERIF, fontsize=11.5, va="center", ha="left")
        f.text(x, top - 0.27, subtitle, transform=t, fontsize=6.8, color=MUTED, va="center", ha="left")
        wm = f.text(w - 0.2, top - 0.065, "WSP-Bench", transform=t, family=SERIF, fontsize=8, color=FAINT,
                    va="center", ha="right")
        wm_w = wm.get_window_extent(f.canvas.get_renderer()).width / f.dpi
        f.patches.append(Rectangle((w - 0.2 - wm_w - 0.11, top - 0.1), 0.065, 0.065, transform=t, color=FAINT,
                                   zorder=5))
        self.legend_y = top - 0.47
        if legend:
            self.legend(legend, x, self.legend_y)

    def legend(self, items, x, y):
        """items: (kind, colour, label), kind one of dot, square, dotted, whisker. Laid out left to right."""
        f, t = self.fig, self.fig.dpi_scale_trans
        r = f.canvas.get_renderer()
        for kind, col, label in items:
            if kind == "dot":
                f.patches.append(mpl.patches.Circle((x + 0.035, y), 0.032, transform=t, color=col))
            elif kind == "square":
                f.patches.append(Rectangle((x, y - 0.04), 0.08, 0.08, transform=t, color=col, ec="#D7E9D0" if col ==
                                           QUAD_GOOD else "none", lw=0.6))
            elif kind == "dotted":
                f.lines.append(mpl.lines.Line2D([x, x + 0.16], [y, y], transform=t, color=INK, lw=1.0,
                                                ls=(0, (1, 1.6))))
            elif kind == "whisker":
                f.lines.append(mpl.lines.Line2D([x + 0.04, x + 0.04], [y - 0.045, y + 0.045], transform=t,
                                                color=INK, lw=0.8))
            dx = 0.22 if kind == "dotted" else 0.11
            tx = f.text(x + dx, y, label, transform=t, fontsize=6.6, va="center", ha="left", color=INK)
            x += dx + tx.get_window_extent(r).width / f.dpi + 0.14

    def axes(self, left, bottom, width, height):
        return self.fig.add_axes([left / self.w, bottom / self.h, width / self.w, height / self.h])

    def save(self, name):
        self.fig.savefig(FIG / f"{name}.pdf")
        self.fig.savefig(FIG / "png" / f"{name}.png", dpi=220)
        plt.close(self.fig)


def lab_legend(ms):
    labs = []
    for m in ms:
        if m["lab"] not in labs:
            labs.append(m["lab"])
    return [("dot", LAB.get(lab, "#8A8A8A"), lab) for lab in labs]


def bar_axes(ax, ymax, step):
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 0.01, step))
    ax.yaxis.grid(True, color=GRID, lw=0.7, ls=DASH)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=0, pad=3)


def panel_title(ax, text, color, note=""):
    """A small colour square and a serif title just above the axes, at fixed point offsets."""
    ax.annotate("■", (0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                color=color, fontsize=7, va="bottom", ha="left")
    ax.annotate(text, (0, 1), xycoords="axes fraction", xytext=(10, 5), textcoords="offset points",
                family=SERIF, fontsize=9.5, va="bottom", ha="left")
    if note:
        ax.annotate(note, (1, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                    fontsize=6.3, color=MUTED, va="bottom", ha="right")


# ---------------------------------------------------------------- Figure: the index
def fig_index():
    ms = BY_INDEX
    c = Card(6.5, 3.35, "WSP-Bench Index",
             f"Mean normalised score over {DATA['core_cells']} cells, first pass · 0 = baseline, 100 = solved "
             "· Higher is better", PURPLE,
             legend=lab_legend(ms) + [("whisker", INK, "95% confidence interval")])
    ax = c.axes(0.55, 1.12, 5.75, 1.42)
    x = np.arange(len(ms))
    ax.bar(x, [m["index"] for m in ms], width=0.7, color=[lab_color(m) for m in ms], zorder=2)
    for xi, m in zip(x, ms):
        lo, hi = m["ci"]
        ax.plot([xi, xi], [lo, hi], color=INK, lw=0.75, zorder=4, solid_capstyle="butt")
        ax.plot([xi - 0.09, xi + 0.09], [hi, hi], color=INK, lw=0.75, zorder=4)
        ax.plot([xi - 0.09, xi + 0.09], [lo, lo], color=INK, lw=0.75, zorder=4)
        if lo - 8 > 0:  # room inside the bar, below the interval
            ax.text(xi, lo - 2.4, f'{m["index"]:.1f}', ha="center", va="top", fontsize=7.2, color="white",
                    zorder=5)
        else:
            ax.text(xi, hi + 1.4, f'{m["index"]:.1f}', ha="center", va="bottom", fontsize=7.2, color=INK,
                    zorder=5)
    bar_axes(ax, 65, 10)
    ax.set_xlim(-0.6, len(ms) - 0.4)
    ax.set_xticks(x, [m["label"] for m in ms], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Index")
    c.save("index_overall")


def fig_families():
    ms = BY_INDEX
    c = Card(6.5, 2.98, "Index by task family",
             "Read: how a model was made · Edit: a specified behaviour · Skill: better at a skill, on "
             "held-out prompts · Higher is better", PURPLE, legend=lab_legend(ms))
    ncell = {f: sum(1 for cl in DATA["cells"] if cl["group"] == f) for f in FAMILIES}
    x = np.arange(len(ms))
    for i, f in enumerate(FAMILIES):
        ax = c.axes(0.45 + i * 2.05, 0.78, 1.8, 1.05)
        vals = [m["sub"][f]["index"] for m in ms]
        ax.bar(x, vals, width=0.68, color=[lab_color(m) for m in ms], zorder=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 2.2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.8, color=INK)
        bar_axes(ax, 100, 50)
        if i:
            ax.set_yticklabels([])
        ax.set_xlim(-0.6, len(ms) - 0.4)
        ax.set_xticks(x, [m["short"] for m in ms], rotation=60, ha="right", rotation_mode="anchor", fontsize=5.8)
        panel_title(ax, f.capitalize(), FAM[f], f"{ncell[f]} cells")
    c.save("index_families")


# ---------------------------------------------------------------- Figure: per class
def fig_classes():
    ms = BY_INDEX
    ncell = {k: sum(1 for cl in DATA["cells"] if cl["class"] == k) for k in CLASS_ORDER}
    Z = np.array([[m["per_class"].get(k, np.nan) for k in CLASS_ORDER] for m in ms])
    c = Card(6.5, 4.3, "Score by task class",
             "Mean score of each agent on each class, first pass · cells per class in brackets · "
             "Higher is better", PURPLE)
    ax = c.axes(1.62, 1.02, 4.7, 2.33)
    ax.imshow(Z, cmap=HEAT, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.4,
                    color="white" if v >= 55 else INK)
    ax.set_yticks(range(len(ms)), [m["label"] for m in ms], fontsize=7, color=INK)
    ax.tick_params(axis="y", length=0, pad=9)
    for i, m in enumerate(ms):  # the lab's colour as a stripe before each name, as in the tables
        ax.add_patch(Rectangle((-0.5 - 0.17, i - 0.36), 0.075, 0.72, color=lab_color(m), clip_on=False))
    ax.set_xticks(range(len(CLASS_ORDER)), [f"{SHORT[k]} ({ncell[k]})" for k in CLASS_ORDER], rotation=40,
                  ha="right", rotation_mode="anchor", fontsize=6.6, color=INK)
    ax.tick_params(axis="x", length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    start = 0
    for f in FAMILIES:
        n = len(DATA["groups"][f])
        if start:
            ax.axvline(start - 0.5, color="white", lw=2.5)
        ax.plot([start - 0.42, start + n - 0.58], [-0.86, -0.86], color=FAM[f], lw=2.2, clip_on=False,
                solid_capstyle="butt")
        ax.text(start - 0.45, -1.12, f.capitalize(), ha="left", va="bottom", fontsize=7, fontweight="semibold",
                color=FAM[f], clip_on=False)
        start += n
    ax.set_ylim(len(ms) - 0.5, -0.5)
    c.save("per_class")


# ---------------------------------------------------------------- Figure: skills
SKILL_NAME = {"facts": "new facts", "svg": "SVG", "c4": "Connect-4", "video": "ffmpeg", "scad": "OpenSCAD",
              "sql": "SQL", "clingo": "answer set programming", "tool": "tool calling", "excel": "spreadsheets",
              "ttt": "tic-tac-toe", "calc": "calculus", "math": "grade-school math", "countdown": "number puzzles",
              "writing": "constrained writing", "brick": "LEGO", "slides": "slide writing", "deck": "slide design",
              "if_q3": "instruction following", "if_q25": "instruction following"}
VARIANT_NAME = {"main": "", "hand": " (hand design)", "onestep": " (one-step)", "search": " (search)",
                "examples": " (examples)", "donor": " (donor)"}


def skill_label(c):
    name = c["item"].split("__")[0][2:]
    return SKILL_NAME.get(name, name) + VARIANT_NAME.get(c["variant"], "")


def fig_skills():
    cells = [c for c in DATA["cells"] if c["group"] == "skill"]
    rows = []
    for c in cells:
        key = f'{c["item"]}|{c["level"]}'
        rows.append((max(m["scores"][key][0] for m in M if key in m["scores"]), c["item"], c["skill_group"]))
    rows.sort(key=lambda r: -r[0])
    card = Card(6.5, 2.95, "Skill tasks",
                f"Best score any agent reaches on each of the {len(rows)} skill tasks, and how many each agent "
                "brings to 50 · Higher is better", TEAL,
                legend=[("square", v[1], v[0]) for v in SKILL_GROUPS.values()])
    a = card.axes(0.5, 0.42, 3.85, 1.75)
    x = np.arange(len(rows))
    a.bar(x, [r[0] for r in rows], width=0.8, color=[SKILL_GROUPS[r[2]][1] for r in rows], zorder=2)
    a.axhline(50, color=INK, lw=0.7, ls=(0, (1, 1.6)), zorder=3)
    bar_axes(a, 125, 25)
    a.set_yticks(range(0, 101, 25))
    a.set_xlim(-0.8, len(rows) - 0.2)
    a.set_xticks([])
    a.set_xlabel(f"{len(rows)} skill tasks, ranked by the best agent's score", fontsize=6.8, color=MUTED, labelpad=4)
    top = [r for r in rows if r[0] >= 95]
    if top:
        x1 = len(top) - 1
        a.plot([-0.4, -0.4, x1 + 0.4, x1 + 0.4], [102, 105, 105, 102], color=INK, lw=0.6)
        names = []
        for r in top:
            n = skill_label(next(c for c in cells if c["item"] == r[1])).split(" (")[0]
            if n not in names:
                names.append(n)
        txt = f"{len(top)} tasks score 95 or more, in {len(names)} skills: " + ", ".join(names)
        a.annotate("\n".join(textwrap.wrap(txt, 52)), (x1 / 2, 105), xytext=(x1 + 3.5, 113), textcoords="data",
                   fontsize=6.2, color=INK, ha="left", va="center",
                   arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.5, shrinkA=0, shrinkB=0))
    # b: per agent
    ms = sorted(M, key=lambda m: -sum(m["scores"][f'{c["item"]}|{c["level"]}'][0] >= 50 for c in cells))
    n50 = [sum(m["scores"][f'{c["item"]}|{c["level"]}'][0] >= 50 for c in cells) for m in ms]
    b = card.axes(4.75, 0.78, 1.55, 1.3)
    xb = np.arange(len(ms))
    b.bar(xb, n50, width=0.68, color=[lab_color(m) for m in ms], zorder=2)
    for xi, v in zip(xb, n50):
        b.text(xi, v + 0.8, str(v), ha="center", va="bottom", fontsize=5.8, color=INK)
    step = 10 if max(n50) > 20 else 5
    bar_axes(b, step * (max(n50) // step + 2), step)
    b.set_xlim(-0.6, len(ms) - 0.4)
    b.set_xticks(xb, [m["short"] for m in ms], rotation=60, ha="right", rotation_mode="anchor", fontsize=5.8)
    panel_title(b, "Tasks at 50+", TEAL)
    card.save("skills")


# ---------------------------------------------------------------- Figure: cost and time
def place_labels(ax, pts, texts, fontsize=6.4, avoid=()):
    """Label each point at one of several positions around it, inside the axes and clear of the other labels, the
    other points and the `avoid` artists. Positions are chosen greedily, a point at a time, in several orders
    (highest first, left to right, most crowded first, ...); the order whose labels overlap least wins, nearer
    positions breaking ties. A label away from its point gets a leader line."""
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(r)
    marks = []
    for x, yv in pts:
        px, py = ax.transData.transform((x, yv))
        marks.append(Bbox.from_extents(px - 3.5, py - 3.5, px + 3.5, py + 3.5))
    fixed = [a.get_window_extent(r) for a in avoid]
    cands = [(5, 0, "left", "center"), (-5, 0, "right", "center"), (0, 5, "center", "bottom"),
             (0, -5, "center", "top"), (4, 4, "left", "bottom"), (4, -4, "left", "top"),
             (-4, 4, "right", "bottom"), (-4, -4, "right", "top"),
             (13, 8, "left", "bottom"), (13, -8, "left", "top"), (-13, 8, "right", "bottom"),
             (-13, -8, "right", "top"), (0, 13, "center", "bottom"), (0, -13, "center", "top"),
             (20, 0, "left", "center"), (-20, 0, "right", "center"), (20, 9, "left", "bottom"),
             (20, -9, "left", "top"), (-20, 9, "right", "bottom"), (-20, -9, "right", "top")]
    is_far = [abs(dx) > 8 or abs(dy) > 8 for dx, dy, _, _ in cands]

    def area(a, b):
        w, h = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
        return w * h if w > 0 and h > 0 else 0.0

    boxes = {}  # (point, candidate) -> the label's box, measured once
    for i in range(len(pts)):
        for k, (dx, dy, ha, va) in enumerate(cands):
            t = ax.annotate(texts[i], pts[i], xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=INK)
            boxes[i, k] = t.get_window_extent(r)
            t.remove()

    def inside(bb):
        return bb.x0 >= axbox.x0 and bb.x1 <= axbox.x1 and bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1

    def solve(order):
        placed, choice, total = list(fixed), {}, 0.0
        for i in order:
            best = None  # when no position is free: the one inside the axes with the least overlap
            for k in range(len(cands)):
                bb = boxes[i, k]
                if not inside(bb):
                    continue
                c = sum(area(bb, o) for o in placed) + sum(area(bb, mk) for j, mk in enumerate(marks) if j != i)
                if c == 0:
                    best = (0.0, k)
                    break
                if best is None or c < best[0]:
                    best = (c, k)
            c, k = best or (0.0, 0)
            choice[i], total = k, total + c
            placed.append(boxes[i, k])
        return (total, sum(is_far[k] for k in choice.values())), choice

    n = len(pts)
    free = [sum(inside(boxes[i, k]) and not any(area(boxes[i, k], mk) for j, mk in enumerate(marks) if j != i)
                for k in range(len(cands))) for i in range(n)]
    orders = [sorted(range(n), key=lambda i: -pts[i][1]), sorted(range(n), key=lambda i: pts[i][1]),
              sorted(range(n), key=lambda i: pts[i][0]), sorted(range(n), key=lambda i: -pts[i][0]),
              sorted(range(n), key=lambda i: (free[i], -pts[i][1]))]
    _, choice = min((solve(o) for o in orders), key=lambda s: s[0])
    for i, k in choice.items():
        dx, dy, ha, va = cands[k]
        ax.annotate(texts[i], pts[i], xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize,
                    color=INK, arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.45, shrinkA=1, shrinkB=3)
                    if is_far[k] else None)


def pareto(points):
    """Points (x, y) not beaten by any other on both lower x and higher y, sorted by x."""
    out, best = [], -1
    for x, yv in sorted(points):
        if yv > best:
            out.append((x, yv))
            best = yv
    return out


def fig_efficiency():
    c = Card(6.5, 3.52, "Index against cost and time",
             "Cost: list price of one run · Time: wall-clock minutes of one run (open models on our own GPUs) "
             "· Both on log scales", ORANGE,
             legend=[("square", QUAD_GOOD, "Most attractive quadrant"), ("dotted", INK, "Pareto front")]
             + lab_legend(M))
    specs = (("cost_per_task", "Cost per task (USD, log scale)", "Index against cost", ORANGE, True),
             ("minutes_per_task", "Minutes per task (log scale)", "Index against time", YELLOW, False))
    for i, (field, xlabel, title, col, dollars) in enumerate(specs):
        ax = c.axes(0.55 + i * 3.05, 0.5, 2.75, 1.9)
        ms = [m for m in M if m.get(field)]
        xs = np.array([m[field] for m in ms])
        ys = np.array([m["index"] for m in ms])
        lo, hi = xs.min() / 2.4, xs.max() * 3.2
        xm, ym = float(np.exp(np.median(np.log(xs)))), float(np.median(ys))
        ax.add_patch(Rectangle((lo, ym), xm - lo, 62 - ym, color=QUAD_GOOD, zorder=0, lw=0))
        ax.add_patch(Rectangle((xm, -3), hi - xm, ym + 3, color=QUAD_BAD, zorder=0, lw=0))
        fr = pareto(list(zip(xs, ys)))
        ax.plot([p[0] for p in fr], [p[1] for p in fr], color=INK, lw=1.0, ls=(0, (1, 1.6)), zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-3, 62)
        # agents whose points coincide (within 5 px) share one label; the later one is drawn as a smaller dot on
        # top, so both lab colours stay visible
        groups = []
        for m in ms:
            p = ax.transData.transform((m[field], m["index"]))
            for g in groups:
                if np.hypot(*(p - g["p"])) < 5:
                    g["ms"].append(m)
                    break
            else:
                groups.append({"p": p, "ms": [m]})
        for g in groups:
            for k, m in enumerate(g["ms"]):
                ax.scatter(m[field], m["index"], s=24 if k == 0 else 9, color=lab_color(m), edgecolor="white",
                           linewidth=0.6 if k == 0 else 0, zorder=3 + k)
        ax.set_yticks(range(0, 61, 10))
        ax.yaxis.grid(True, color=GRID, lw=0.7, ls=DASH)
        ax.xaxis.grid(True, color=GRID, lw=0.7, ls=DASH, which="major")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.tick_params(axis="x", which="both", length=0, pad=3, colors=MUTED)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"${v:g}" if dollars else f"{v:g}"))
        ax.set_xlabel(xlabel, fontsize=6.8, color=MUTED)
        if i == 0:
            ax.set_ylabel("Index")
        else:
            ax.set_yticklabels([])
        panel_title(ax, title, col)
        place_labels(ax, [(g["ms"][0][field], g["ms"][0]["index"]) for g in groups],
                     [" / ".join(m["short"] for m in g["ms"]) for g in groups])
    c.save("efficiency")


# ---------------------------------------------------------------- Figure: pass 1 against pass 2
def fig_passes():
    two = [m for m in BY_INDEX if m["mean_of_passes"]["passes"] >= 2
           and sum(v[1] is not None for v in m["scores"].values()) >= 150]
    if not two:
        return []
    c = Card(6.5, 2.55, "Run-to-run variation",
             "Cell scores on the first and second pass of the agents that ran twice · slight jitter shows "
             "overlapping points", PURPLE)
    stats = []
    for i, m in enumerate(two):
        ax = c.axes(0.42 + i * 1.53, 0.38, 1.28, 1.28)
        pairs = [(v[0], v[1]) for v in m["scores"].values() if v[1] is not None]
        p1 = np.array([p[0] for p in pairs])
        p2 = np.array([p[1] for p in pairs])
        rng = np.random.default_rng(0)
        ax.plot([0, 100], [0, 100], color=GRID, lw=0.8, ls=DASH, zorder=1)
        ax.scatter(p1 + rng.uniform(-1.2, 1.2, len(p1)), p2 + rng.uniform(-1.2, 1.2, len(p2)), s=6,
                   color=lab_color(m), alpha=0.55, linewidth=0, zorder=2)
        far = float(np.mean(np.abs(p1 - p2) >= 50))
        stats.append((m, float(p1.mean()), float(p2.mean()), far, len(pairs)))
        ax.set_xlim(-4, 104)
        ax.set_ylim(-4, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Pass 2")
        ax.set_xlabel("Pass 1", fontsize=6.8, color=MUTED)
        ax.tick_params(length=0, pad=2, colors=MUTED)
        ax.text(0, 1.22, m["short"], transform=ax.transAxes, family=SERIF, fontsize=9, va="center", ha="left")
        ax.text(0, 1.08, f"index {p1.mean():.1f} vs {p2.mean():.1f} · {100 * far:.0f}% differ by 50+",
                transform=ax.transAxes, fontsize=5.8, color=MUTED, va="center", ha="left")
    c.save("passes")
    return stats


# ---------------------------------------------------------------- tables and macros
def fmt_cost(v):
    return "--" if v is None else (f"{v:.3f}" if v < 0.1 else f"{v:.2f}")


def table_leaderboard():
    """The results table in the Artificial Analysis style (macros \\aamodel etc. are defined in main.tex)."""
    ms = BY_INDEX
    sep = r"!{\color{aarule}\vrule}"
    lines = [rf"\begin{{tabular}}{{@{{}}l c {sep} c c {sep} c c c {sep} c c@{{}}}}",
             r"\multicolumn{2}{@{}l}{} & \multicolumn{2}{c" + sep + r"}{\aagroup{WSP-Bench Index}} & "
             r"\multicolumn{3}{c" + sep + r"}{\aagroup{Index by family}} & \multicolumn{2}{c@{}}{\aagroup{Efficiency}} \\",
             r"\aaheadl{Agent}{} & \aahead{Weights}{} & \aahead{Index}{95\% CI} & \aahead{Cells $\geq$50}{share} & "
             r"\aahead{Read}{} & \aahead{Edit}{} & \aahead{Skill}{} & \aahead{Cost per task}{USD} & "
             r"\aahead{Time per task}{minutes} \\",
             r"\aarowsep"]
    for m in ms:
        name = m["label"].split(" (")[0]
        setting = m["setting"]
        sub = " & ".join(f'{m["sub"][f]["index"]:.1f}' for f in FAMILIES)
        lines.append(rf'\aamodel{{{LAB_TEX.get(m["lab"], "aamuted")}}}{{{name}}}{{{setting}}} & '
                     rf'\aaweights{{{"Open" if m["open_weights"] else "Proprietary"}}} & '
                     rf'\aaidx{{{m["index"]:.1f}}}{{{m["ci"][0]:.1f}}}{{{m["ci"][1]:.1f}}} & {m["pass_rate"]:.0f}\% & '
                     rf'{sub} & {fmt_cost(m["cost_per_task"])} & {m["minutes_per_task"]:.0f} \\')
        lines.append(r"\aarowsep")
    lines[-1] = r"\aarowlast"
    lines.append(r"\end{tabular}")
    (TAB / "leaderboard.tex").write_text("% Generated by scripts/make_figures.py from data/paper_v8.json; do not edit.\n"
                                         + "\n".join(lines) + "\n")


def macros(pass_stats):
    out = ["% Generated by scripts/make_figures.py from data/paper_v8.json; do not edit.",
           r"% \res{<agent>}{<field>}: index, cilo, cihi, read, edit, skill, passrate, solved, cost, minutes, tokens",
           r"% \cls{<agent>}{<class>}: mean score of the agent on one class.",
           r"\makeatletter",
           r"\newcommand{\res}[2]{\@ifundefined{wsp@#1@#2}{\textbf{??}}{\@nameuse{wsp@#1@#2}}}",
           r"\newcommand{\cls}[2]{\@ifundefined{wspc@#1@#2}{\textbf{??}}{\@nameuse{wspc@#1@#2}}}",
           r"\newcommand{\wspnum}[1]{\@ifundefined{wspn@#1}{\textbf{??}}{\@nameuse{wspn@#1}}}"]

    def d(name, val):
        out.append(rf"\@namedef{{{name}}}{{{val}}}")

    for m in M:
        k = m["key"]
        d(f"wsp@{k}@index", f'{m["index"]:.1f}')
        d(f"wsp@{k}@cilo", f'{m["ci"][0]:.1f}')
        d(f"wsp@{k}@cihi", f'{m["ci"][1]:.1f}')
        for f in FAMILIES:
            d(f"wsp@{k}@{f}", f'{m["sub"][f]["index"]:.1f}')
        d(f"wsp@{k}@passrate", f'{m["pass_rate"]:.0f}')
        d(f"wsp@{k}@solved", str(m["solved"]))
        d(f"wsp@{k}@cost", fmt_cost(m["cost_per_task"]))
        d(f"wsp@{k}@minutes", f'{m["minutes_per_task"]:.0f}')
        d(f"wsp@{k}@tokens", f'{(m["output_tokens_per_task"] or 0) / 1000:.0f}k')
        d(f"wsp@{k}@meanpasses", f'{m["mean_of_passes"]["index"]:.1f}')
        for cl, v in m["per_class"].items():
            d(f"wspc@{k}@{cl}", f"{v:.0f}")
    skill = [c for c in DATA["cells"] if c["group"] == "skill"]
    best = []
    for c in skill:
        key = f'{c["item"]}|{c["level"]}'
        best.append(max(m["scores"][key][0] for m in M if key in m["scores"]))
    d("wspn@agents", str(len(M)))
    d("wspn@open", str(sum(m["open_weights"] for m in M)))
    d("wspn@closed", str(sum(not m["open_weights"] for m in M)))
    d("wspn@cells", str(DATA["core_cells"]))
    d("wspn@tasks", str(DATA["items"]))
    d("wspn@skilltasks", str(len(skill)))
    d("wspn@skillhalf", str(sum(v >= 50 for v in best)))
    d("wspn@skillninety", str(sum(v >= 90 for v in best)))
    d("wspn@skillzero", str(sum(v == 0 for v in best)))
    d("wspn@skillninetyfive", str(sum(v >= 95 for v in best)))
    top = [c for c, v in zip(skill, best) if v >= 95]
    names = []
    for c in top:
        n = skill_label(c).split(" (")[0]
        if n not in names:
            names.append(n)
    d("wspn@skilltopn", str(len(names)))
    d("wspn@skilltopnames", ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names))
    for g in SKILL_GROUPS:
        vals = [v for c, v in zip(skill, best) if c["skill_group"] == g]
        d(f"wspn@skillbest@{g.replace(' ', '')}", f"{max(vals):.0f}")
        d(f"wspn@skillcount@{g.replace(' ', '')}", str(len(vals)))
    fifty = {}
    for m in M:
        fifty[m["key"]] = sum(m["scores"][f'{c["item"]}|{c["level"]}'][0] >= 50 for c in skill)
        d(f"wsp@{m['key']}@skillfifty", str(fifty[m["key"]]))
    d("wspn@skillfiftylow", str(sum(v <= 1 for v in fifty.values())))

    def cell_scores(item):
        key = f"{item}|full"
        return {m["key"]: m["scores"][key][0] for m in M if key in m["scores"]}

    facts = cell_scores("K_facts__open")
    d("wspn@factsoverhalf", str(sum(v > 50 for v in facts.values())))
    clingo = cell_scores("K_clingo__open")
    ck = max(clingo, key=clingo.get)
    d("wspn@clingobest", f"{clingo[ck]:.0f}")
    d("wspn@clingobestagent", next(m["label"] for m in M if m["key"] == ck))
    opens = [m for m in M if m["open_weights"]]
    d("wspn@bestopen", max(opens, key=lambda m: m["index"])["label"] if opens else "--")
    d("wspn@bestopenindex", f'{max(m["index"] for m in opens):.1f}' if opens else "--")
    # the open-weights agents other than the best: how many (as a word) and the range of their index
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    d("wspn@openword", words.get(len(opens), str(len(opens))))
    rest = sorted(opens, key=lambda m: m["index"])[:-1]
    d("wspn@openrestword", words.get(len(rest), str(len(rest))))
    d("wspn@openrestlow", f'{rest[0]["index"]:.1f}' if rest else "--")
    d("wspn@openresthigh", f'{rest[-1]["index"]:.1f}' if rest else "--")
    edit_cls = DATA["groups"]["edit"]
    d("wspn@openeditmax", f'{max(m["per_class"].get(k, 0) for m in opens for k in edit_cls):.0f}' if opens else "--")
    d("wspn@skilllowest", str(sum(min(FAMILIES, key=lambda f: m["sub"][f]["index"]) == "skill" for m in M)))
    d("wspn@readhighest", str(sum(max(FAMILIES, key=lambda f: m["sub"][f]["index"]) == "read" for m in M)))
    for m, a, b, far, n in pass_stats:
        d(f"wsp@{m['key']}@pone", f"{a:.1f}")
        d(f"wsp@{m['key']}@ptwo", f"{b:.1f}")
        d(f"wsp@{m['key']}@far", f"{100 * far:.0f}")
    d("wspn@built", DATA["built"])
    out.append(r"\makeatother")
    (TAB / "numbers.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    fig_index()
    fig_families()
    fig_classes()
    fig_skills()
    fig_efficiency()
    st = fig_passes()
    table_leaderboard()
    macros(st)
    print(f"{len(M)} agents; data built {DATA['built']}; fonts {SANS} / {SERIF}; figures in {FIG}")
