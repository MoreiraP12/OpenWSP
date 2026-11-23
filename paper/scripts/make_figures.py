"""Figures, tables and number macros of the WSP-Bench paper, from data/paper_v8.json.

    cd paper && python3 scripts/make_figures.py

Writes figures/*.pdf (vector, fonts embedded), figures/png/*.png (previews), tables/leaderboard.tex and
tables/numbers.tex (one macro per number quoted in the text, so the prose follows the data). The data file is made in
the private development repo by `harness/paper_data.py`; every score is an agent's first pass.
"""
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent.parent
DATA = json.loads((HERE / "data" / "paper_v8.json").read_text())
FIG, TAB = HERE / "figures", HERE / "tables"
(FIG / "png").mkdir(parents=True, exist_ok=True)
TAB.mkdir(exist_ok=True)

# ---------------------------------------------------------------- what is shown
# Agents in the paper, in data order; short keys name them in the macros (\res{sol}{index}).
AGENTS = {
    "gpt-6.1-sol": ("sol", "GPT-6.1 Sol"),
    "claude-opus-5-5": ("opus", "Claude Opus 5.5"),
    "gpt-6.1-terra": ("terra", "GPT-6.1 Terra"),
    "claude-sonnet-5-5": ("sonnet", "Claude Sonnet 5.5"),
    "claude-haiku-5-5": ("haiku", "Claude Haiku 5.5"),
    "qwen3.8-27b": ("qwenthink", "Qwen3.8-27B (thinking)"),
    "gpt-6-luna": ("luna", "GPT-6 Luna"),
    "minimax-m2.7": ("minimax", "MiniMax-M2.7"),
    "deepseek-v4-flash": ("deepseek", "DeepSeek-V4-Flash"),
    "qwen3.8-27b-nothink": ("qwen", "Qwen3.8-27B (no thinking)"),
    # pending (shown once complete): "inkling-2-mini", "claude-opus-5-5-high", ...
    "inkling-2-mini": ("inkling", "Inkling 2 Mini"),
}
M = [m for m in DATA["models"] if m["id"] in AGENTS and m["complete"]]
for m in M:
    m["key"], m["label"] = AGENTS[m["id"]]

FAMILIES = ["read", "edit", "skill"]
CLASS_ORDER = DATA["groups"]["read"] + DATA["groups"]["edit"] + DATA["groups"]["skill"]
SHORT = {"O2": "Data mixture", "O3": "Epochs", "P1": "Merge weights", "P2": "Transplant", "P3": "Edit kind",
         "P3o": "Edit, free text", "S2": "Sycophant twin", "S3": "Backdoor twin", "S6": "Trigger phrase",
         "W1": "Remove backdoor", "W2": "Sparse transplant", "W3": "Add math", "W4": "Word association",
         "W4x": "Associations, harder", "W5": "Phrase detector", "K": "Skills"}

# ---------------------------------------------------------------- look
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E9E9E9"
CLOSED, OPEN = "#2E4A7D", "#9FB7D6"
FAM = {"read": "#0072B2", "edit": "#D55E00", "skill": "#009E73"}  # Okabe-Ito, as in figures/overview.tex
SKILL_GROUPS = {"general": ("General skills", "#4C78A8"), "whole programs": ("Whole programs", "#F2A541"),
                "function completion": ("Function completion", "#B279A2"),
                "HumanEval functions": ("HumanEval functions", "#72B7B2")}
HEAT = LinearSegmentedColormap.from_list("wsp", ["#F7F9FC", "#C9D7EA", "#6F93C4", "#2E4A7D", "#16264A"])

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8, "text.color": INK, "axes.labelcolor": INK,
    "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 6,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": "#8A8A8A", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.dpi": 150,
})
W = 6.5  # text width of the paper, inches


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / "png" / f"{name}.png", dpi=220)
    plt.close(fig)


def color(m):
    return OPEN if m["open_weights"] else CLOSED


# ---------------------------------------------------------------- Figure: main results
def fig_main():
    ms = sorted(M, key=lambda m: m["index"])  # bottom-up, best on top
    y = np.arange(len(ms))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 0.27 * len(ms) + 0.85), sharey=True,
                               gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.08})
    idx = [m["index"] for m in ms]
    lo = [m["index"] - m["ci"][0] for m in ms]
    hi = [m["ci"][1] - m["index"] for m in ms]
    a.barh(y, idx, height=0.66, color=[color(m) for m in ms], zorder=2)
    a.errorbar(idx, y, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
    for yi, m in zip(y, ms):
        a.text(m["ci"][1] + 1.2, yi, f'{m["index"]:.1f}', va="center", ha="left", fontsize=7.2, color=INK)
    a.set_yticks(y, [m["label"] for m in ms])
    a.set_xlim(0, 70)
    a.set_xticks(range(0, 71, 10))
    a.xaxis.grid(True)
    a.set_xlabel("WSP-Bench index (pass 1, 95% CI)")
    a.set_title("a   Overall", loc="left")
    a.spines["left"].set_visible(False)
    a.legend(handles=[Patch(color=CLOSED, label="Closed weights"), Patch(color=OPEN, label="Open weights")],
             loc="lower right", handlelength=1.0, handleheight=0.8)

    marks = {"read": "o", "edit": "s", "skill": "D"}
    for yi, m in zip(y, ms):
        vals = [m["sub"][f]["index"] for f in FAMILIES]
        b.plot([min(vals), max(vals)], [yi, yi], color="#CFCFCF", lw=1.2, zorder=1, solid_capstyle="round")
        for f, v in zip(FAMILIES, vals):
            b.scatter(v, yi, s=22 if f != "skill" else 18, marker=marks[f], color=FAM[f], zorder=3,
                      edgecolor="white", linewidth=0.5)
    b.set_xlim(-2, 102)
    b.set_xticks(range(0, 101, 20))
    b.xaxis.grid(True)
    b.set_xlabel("Index within each family")
    b.set_title("b   By task family", loc="left")
    b.spines["left"].set_visible(False)
    b.tick_params(axis="y", length=0)
    b.legend(handles=[Line2D([], [], marker=marks[f], color="none", markerfacecolor=FAM[f], markeredgecolor="white",
                             markersize=5.2, label=f.capitalize()) for f in FAMILIES],
             loc="lower right", handletextpad=0.2, borderaxespad=0.3)
    save(fig, "main_results")


# ---------------------------------------------------------------- Figure: per class heatmap
def fig_classes():
    ms = sorted(M, key=lambda m: -m["index"])
    ncell = {k: sum(1 for c in DATA["cells"] if c["class"] == k) for k in CLASS_ORDER}
    Z = np.array([[m["per_class"].get(k, np.nan) for k in CLASS_ORDER] for m in ms])
    fig, ax = plt.subplots(figsize=(W, 0.25 * len(ms) + 1.55))
    ax.imshow(Z, cmap=HEAT, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.6,
                    color="white" if v >= 55 else INK)
    ax.set_yticks(range(len(ms)), [m["label"] for m in ms])
    ax.set_xticks(range(len(CLASS_ORDER)), [f"{SHORT[k]}  ({ncell[k]})" for k in CLASS_ORDER], rotation=40,
                  ha="right", rotation_mode="anchor", fontsize=7)
    ax.tick_params(axis="x", length=0, pad=2, colors=INK)
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # white gutters between families, and family bands on top
    bounds, start = [], 0
    for f in FAMILIES:
        n = len(DATA["groups"][f])
        bounds.append((f, start, start + n))
        start += n
    for f, s0, s1 in bounds:
        if s0:
            ax.axvline(s0 - 0.5, color="white", lw=3)
        ax.annotate("", xy=(s0 - 0.42, -0.95), xytext=(s1 - 0.58, -0.95), xycoords="data",
                    arrowprops=dict(arrowstyle="-", color=FAM[f], lw=2.4), annotation_clip=False)
        ax.text((s0 + s1 - 1) / 2, -1.35, f.capitalize(), ha="center", va="bottom", fontsize=7.8,
                fontweight="bold", color=FAM[f], clip_on=False)
    ax.set_ylim(len(ms) - 0.5, -1.6)
    save(fig, "per_class")


# ---------------------------------------------------------------- Figure: cost and time
SHORTNAME = {"sol": "Sol", "opus": "Opus 5.5", "terra": "Terra", "sonnet": "Sonnet 5.5", "haiku": "Haiku 5.5",
             "qwenthink": "Qwen3.8 thinking", "luna": "Luna", "minimax": "MiniMax-M2.7", "deepseek": "DeepSeek-V4",
             "qwen": "Qwen3.8", "inkling": "Inkling 2 Mini"}


def place_labels(ax, pts, texts, fontsize=6.8, avoid=()):
    """Label each point at the first of eight positions around it that overlaps no other label, point or the
    `avoid` artists (legends), and stays inside the axes; highest points are placed first."""
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(r)
    marks = []
    for x, yv in pts:
        px, py = ax.transData.transform((x, yv))
        marks.append(Bbox.from_extents(px - 3.5, py - 3.5, px + 3.5, py + 3.5))
    placed = [a.get_window_extent(r) for a in avoid]
    cands = [(6, 0, "left", "center"), (-6, 0, "right", "center"), (0, 6, "center", "bottom"),
             (0, -6, "center", "top"), (5, 4, "left", "bottom"), (5, -4, "left", "top"),
             (-5, 4, "right", "bottom"), (-5, -4, "right", "top"),
             # farther out, with a leader line
             (14, 9, "left", "bottom"), (14, -9, "left", "top"), (-14, 9, "right", "bottom"),
             (-14, -9, "right", "top"), (0, 14, "center", "bottom"), (22, 0, "left", "center"),
             (-22, 0, "right", "center")]
    for i in sorted(range(len(pts)), key=lambda i: -pts[i][1]):
        chosen = None
        for dx, dy, ha, va in cands:
            far = abs(dx) > 8 or abs(dy) > 8
            t = ax.annotate(texts[i], pts[i], xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=INK,
                            arrowprops=dict(arrowstyle="-", color="#A0A0A0", lw=0.45, shrinkA=1, shrinkB=3)
                            if far else None)
            bb = t.get_window_extent(r)
            ok = (bb.x0 >= axbox.x0 and bb.x1 <= axbox.x1 and bb.y0 >= axbox.y0 and bb.y1 <= axbox.y1
                  and not any(bb.overlaps(o) for o in placed)
                  and not any(bb.overlaps(mk) for j, mk in enumerate(marks) if j != i))
            if ok:
                chosen = bb
                break
            t.remove()
        if chosen is None:
            dx, dy, ha, va = cands[0]
            t = ax.annotate(texts[i], pts[i], xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize, color=INK)
            chosen = t.get_window_extent(r)
        placed.append(chosen)


def pareto(points):
    """Points (x, y) not beaten by any other on both lower x and higher y, sorted by x."""
    pts = sorted(points)
    out, best = [], -1
    for x, yv in pts:
        if yv > best:
            out.append((x, yv))
            best = yv
    return out


def fig_cost():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.55), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, field, lab, kind in ((axes[0], "cost_per_task", "Cost per task (USD, list price)", "cost"),
                                 (axes[1], "minutes_per_task", "Minutes per task", "minutes")):
        ms = [m for m in M if m.get(field)]
        xs = [m[field] for m in ms]
        fr = pareto([(m[field], m["index"]) for m in ms])
        ax.plot([p[0] for p in fr], [p[1] for p in fr], ls=(0, (3, 2)), color="#9A9A9A", lw=0.9, zorder=1)
        for m in ms:
            ax.scatter(m[field], m["index"], s=30, color=color(m), edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_xscale("log")
        lo, hi = min(xs), max(xs)
        ax.set_xlim(lo / 2.2, hi * 3.5)
        ax.set_xlabel(lab)
        ax.yaxis.grid(True)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: (f"${v:g}" if kind == "cost" else f"{v:g}")))
    axes[0].set_ylabel("WSP-Bench index (pass 1)")
    axes[0].set_ylim(-3.5, 60)
    axes[0].set_yticks(range(0, 61, 10))
    axes[0].set_title("a   Index against cost", loc="left")
    axes[1].set_title("b   Index against time", loc="left")
    axes[1].tick_params(axis="y", length=0)
    leg = axes[0].legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=CLOSED,
                                         markeredgecolor="white", markersize=6, label="Closed weights"),
                                  Line2D([], [], marker="o", color="none", markerfacecolor=OPEN,
                                         markeredgecolor="white", markersize=6, label="Open weights"),
                                  Line2D([], [], ls=(0, (3, 2)), color="#9A9A9A", lw=0.9, label="Pareto front")],
                         loc="upper left", handletextpad=0.3)
    for ax, field in ((axes[0], "cost_per_task"), (axes[1], "minutes_per_task")):
        ms = [m for m in M if m.get(field)]
        place_labels(ax, [(m[field], m["index"]) for m in ms], [SHORTNAME[m["key"]] for m in ms],
                     avoid=(leg,) if ax is axes[0] else ())
    save(fig, "cost_time")


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
        sc = [m["scores"][key][0] for m in M if key in m["scores"]]
        rows.append((max(sc), c["item"], c["skill_group"]))
    rows.sort(key=lambda r: -r[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.5})
    x = np.arange(len(rows))
    a.bar(x, [r[0] for r in rows], width=0.82, color=[SKILL_GROUPS[r[2]][1] for r in rows], zorder=2)
    a.axhline(50, color="#9A9A9A", lw=0.7, ls=(0, (3, 2)), zorder=1)
    a.set_xlim(-0.8, len(rows) - 0.2)
    a.set_ylim(0, 124)
    a.set_yticks(range(0, 101, 25))
    a.set_xticks([])
    a.spines["bottom"].set_color("#BDBDBD")
    a.set_ylabel("Best agent's score")
    a.set_xlabel(f"{len(rows)} skill tasks, ranked")
    a.set_title("a   Best score on each skill task", loc="left")
    a.yaxis.grid(True)
    top = [r for r in rows if r[0] >= 95]
    if top:
        x1 = len(top) - 1
        a.plot([-0.4, -0.4, x1 + 0.4, x1 + 0.4], [101.5, 104, 104, 101.5], color=INK, lw=0.6)
        names = []
        for r in top:
            n = skill_label(next(c for c in cells if c["item"] == r[1])).split(" (")[0]
            if n not in names:
                names.append(n)
        txt = f"{len(top)} tasks score 95 or more, in {len(names)} skills: " + ", ".join(names)
        a.annotate("\n".join(textwrap.wrap(txt, 46)), (x1 / 2, 104), xytext=(x1 + 3, 112),
                   textcoords="data", fontsize=6.4, color=INK, ha="left", va="center",
                   arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=0.5, shrinkA=0, shrinkB=0))
    a.legend(handles=[Patch(color=v[1], label=v[0]) for v in SKILL_GROUPS.values()], loc="center right",
             bbox_to_anchor=(1.0, 0.62), handlelength=0.9, handleheight=0.7, fontsize=6.8, ncol=1)
    # b: per agent, how many skill tasks reach 50
    ms = sorted(M, key=lambda m: sum(m["scores"][f'{c["item"]}|{c["level"]}'][0] >= 50 for c in cells))
    n50 = [sum(m["scores"][f'{c["item"]}|{c["level"]}'][0] >= 50 for c in cells) for m in ms]
    yy = np.arange(len(ms))
    b.barh(yy, n50, height=0.62, color=[color(m) for m in ms], zorder=2)
    for yi, v in zip(yy, n50):
        b.text(v + 0.6, yi, str(v), va="center", fontsize=7, color=INK)
    b.set_yticks(yy, [SHORTNAME[m["key"]] for m in ms], fontsize=7)
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.xaxis.grid(True)
    b.set_xlim(0, max(n50) * 1.25 + 1)
    b.set_xlabel("Skill tasks scored 50 or more")
    b.set_title("b   Skill tasks taught, per agent", loc="left")
    save(fig, "skills")


# ---------------------------------------------------------------- Figure: pass 1 against pass 2
def fig_passes():
    two = [m for m in M if m["mean_of_passes"]["passes"] >= 2
           and sum(v[1] is not None for v in m["scores"].values()) >= 150]
    if not two:
        return []
    fig, axes = plt.subplots(1, len(two), figsize=(W, W / len(two) + 0.35), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    stats = []
    for ax, m in zip(np.atleast_1d(axes), two):
        pairs = [(v[0], v[1]) for v in m["scores"].values() if v[1] is not None]
        p1 = np.array([p[0] for p in pairs])
        p2 = np.array([p[1] for p in pairs])
        rng = np.random.default_rng(0)
        j1, j2 = rng.uniform(-1.2, 1.2, len(p1)), rng.uniform(-1.2, 1.2, len(p2))
        ax.plot([0, 100], [0, 100], color="#BDBDBD", lw=0.7, zorder=1)
        ax.scatter(p1 + j1, p2 + j2, s=7, color=color(m), alpha=0.55, linewidth=0, zorder=2)
        far = float(np.mean(np.abs(p1 - p2) >= 50))
        stats.append((m, float(p1.mean()), float(p2.mean()), far, len(pairs)))
        ax.set_title(SHORTNAME[m["key"]], loc="left", fontsize=7.8, pad=21)
        ax.text(0, 1.025, f"index {p1.mean():.1f} vs {p2.mean():.1f}\n{100 * far:.0f}% of cells differ by 50+",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, color=MUTED, linespacing=1.25)
        ax.set_xlim(-4, 104)
        ax.set_ylim(-4, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.set_aspect("equal")
        ax.set_xlabel("Pass 1")
    np.atleast_1d(axes)[0].set_ylabel("Pass 2")
    save(fig, "passes")
    return stats


# ---------------------------------------------------------------- tables and macros
def fmt_cost(v):
    return "--" if v is None else (f"{v:.3f}" if v < 0.1 else f"{v:.2f}")


def table_leaderboard():
    ms = sorted(M, key=lambda m: -m["index"])
    lines = [r"\begin{tabular}{@{}lccrrrrrr@{}}", r"\toprule",
             r"Agent & Open & Index [95\% CI] & Read & Edit & Skill & $\geq$50 (\%) & \$\,/\,task & Min\,/\,task \\",
             r"\midrule"]
    best = {f: max(m["sub"][f]["index"] for m in ms) for f in FAMILIES}
    best_idx = max(m["index"] for m in ms)
    for m in ms:
        def b(v, top):
            s = f"{v:.1f}"
            return rf"\textbf{{{s}}}" if abs(v - top) < 1e-9 else s
        sub = [b(m["sub"][f]["index"], best[f]) for f in FAMILIES]
        idx = b(m["index"], best_idx)
        lines.append(rf'{m["label"]} & {"\\checkmark" if m["open_weights"] else ""} & '
                     rf'{idx} [{m["ci"][0]:.1f}, {m["ci"][1]:.1f}] & {" & ".join(sub)} & {m["pass_rate"]:.0f} & '
                     rf'{fmt_cost(m["cost_per_task"])} & {m["minutes_per_task"]:.0f} \\')
    lines += [r"\bottomrule", r"\end{tabular}"]
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
        for c, v in m["per_class"].items():
            d(f"wspc@{k}@{c}", f"{v:.0f}")
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
    edit_cls = DATA["groups"]["edit"]
    d("wspn@openeditmax", f'{max(m["per_class"].get(k, 0) for m in opens for k in edit_cls):.0f}' if opens else "--")
    fam_low = sum(min(FAMILIES, key=lambda f: m["sub"][f]["index"]) == "skill" for m in M)
    d("wspn@skilllowest", str(fam_low))
    read_high = sum(max(FAMILIES, key=lambda f: m["sub"][f]["index"]) == "read" for m in M)
    d("wspn@readhighest", str(read_high))
    for m, a, b, far, n in pass_stats:
        d(f"wsp@{m['key']}@pone", f"{a:.1f}")
        d(f"wsp@{m['key']}@ptwo", f"{b:.1f}")
        d(f"wsp@{m['key']}@far", f"{100 * far:.0f}")
    d("wspn@built", DATA["built"])
    out.append(r"\makeatother")
    (TAB / "numbers.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    fig_main()
    fig_classes()
    fig_cost()
    fig_skills()
    st = fig_passes()
    table_leaderboard()
    macros(st)
    print(f"{len(M)} agents; data built {DATA['built']}; figures in {FIG}")
