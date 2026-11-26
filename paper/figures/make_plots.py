"""Results figures for the paper, from the paper data.

    python3 figures/make_plots.py ../results/paper_v8.json      (run from paper/ in the public repository)

Writes figures/fig_index.pdf, fig_cost.pdf, fig_forward.pdf and fig_classes.pdf. Pass a second argument (a folder)
to also write PNG previews there. Style: pastel, rounded, minimal. Colour encodes the lab (both Qwen3.8 agents share
one), and open-weights models are striped; every mark is also labelled by name, so colour never carries identity alone.
The palette was checked with the dataviz validator (light mode, all pairs: passes; worst CVD pair coral/mint dE 7.2, in
the floor band, legal with the direct labels and stripes).
"""
import json
import sys
from pathlib import Path as FilePath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

OUT = FilePath(__file__).resolve().parent

# ---- style ----------------------------------------------------------------------------------------------------------
INK, INK2, GRID, SURF = "#2b2b2b", "#6b6b6b", "#ececec", "#ffffff"
LAB_COLOR = {"OpenAI": "#75c59b", "Anthropic": "#fb906a", "Qwen": "#cc97f3", "MiniMax": "#c26c96",
             "DeepSeek": "#4e90d2", "Inkling": "#b28324"}
LAB_ORDER = list(LAB_COLOR)
OPEN_LABS = {"Qwen", "MiniMax", "DeepSeek", "Inkling"}
HATCH, STRIPE = "////", (1, 1, 1, 0.75)  # open weights: soft white stripes over the lab colour
# Nunito (SIL Open Font License, figures/fonts/OFL.txt): static 400/600/700 instances cut from the variable font with
# fontTools, since matplotlib cannot pick weights inside a variable font
for f in sorted((OUT / "fonts").glob("Nunito-*.ttf")):
    font_manager.fontManager.addfont(str(f))
FONT = "Nunito"
plt.rcParams.update({
    "font.family": FONT, "font.size": 8, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "pdf.fonttype": 42, "savefig.facecolor": SURF, "figure.facecolor": SURF, "hatch.linewidth": 1.1,
})

# (solver id, label, lab), in the order of the paper's tables
AGENTS = [
    ("gpt-6.1-sol", "GPT-6.1 Sol", "OpenAI"),
    ("claude-opus-5-5-high", "Claude Opus 5.5", "Anthropic"),
    ("claude-sonnet-5-5-high", "Claude Sonnet 5.5", "Anthropic"),
    ("gpt-6.1-terra", "GPT-6.1 Terra", "OpenAI"),
    ("claude-haiku-5-5-high", "Claude Haiku 5.5", "Anthropic"),
    ("gpt-6-luna", "GPT-6 Luna", "OpenAI"),
    ("qwen3.8-27b", "Qwen3.8-27B (thinking)", "Qwen"),
    ("minimax-m2.7", "MiniMax-M2.7", "MiniMax"),
    ("deepseek-v4-flash", "DeepSeek-V4-Flash", "DeepSeek"),
    ("inkling-2-mini", "Inkling 2 Mini", "Inkling"),
    ("qwen3.8-27b-nothink", "Qwen3.8-27B (no thinking)", "Qwen"),
]
TWO_LINE = {"GPT-6.1 Sol": "GPT-6.1\nSol", "Claude Opus 5.5": "Claude\nOpus 5.5", "Claude Sonnet 5.5": "Claude\nSonnet 5.5",
            "GPT-6.1 Terra": "GPT-6.1\nTerra", "Claude Haiku 5.5": "Claude\nHaiku 5.5", "GPT-6 Luna": "GPT-6\nLuna",
            "Qwen3.8-27B (thinking)": "Qwen3.8\n27B", "MiniMax-M2.7": "MiniMax\nM2.7", "DeepSeek-V4-Flash": "DeepSeek\nV4-Flash",
            "Inkling 2 Mini": "Inkling\n2 Mini", "Qwen3.8-27B (no thinking)": "Qwen3.8\nno think"}


def clean(ax, grid_axis="y"):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def px_per_unit(ax):
    """(pixels per data unit in x, in y), once the axes geometry is final."""
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return bb.width / abs(x1 - x0), bb.height / abs(y1 - y0)


def round_bar(ax, x, y0, w, h, lab, horizontal=False, r_px=3.2):
    """A bar drawn as one path: rounded corners at the data end, square at the baseline. x/y0 are the bar's lower-left
    corner along (category, value); for horizontal bars the roles swap. The radius is set in pixels and capped at half
    the bar's shorter side, so short bars stay pill-ended instead of overshooting. Open-weights labs are striped."""
    if h <= 0:
        return
    sx, sy = px_per_unit(ax)
    L, C = Path.LINETO, Path.CURVE3
    if horizontal:  # baseline on the left, data end on the right
        bx, by, bw, bh = y0, x, h, w
        r = min(r_px, bw * sx, bh * sy / 2)
        rx, ry = r / sx, r / sy
        x1, y1 = bx + bw, by + bh
        verts = [(bx, by), (x1 - rx, by), (x1, by), (x1, by + ry), (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (bx, y1),
                 (bx, by)]
    else:  # baseline at the bottom, data end at the top
        bx, by, bw, bh = x, y0, w, h
        r = min(r_px, bw * sx / 2, bh * sy)
        rx, ry = r / sx, r / sy
        x1, y1 = bx + bw, by + bh
        verts = [(bx, by), (x1, by), (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (bx + rx, y1), (bx, y1), (bx, y1 - ry),
                 (bx, by)]
    codes = [Path.MOVETO, L, L, C, C, L, C, C, Path.CLOSEPOLY]
    path = Path(verts, codes)
    ax.add_patch(PathPatch(path, fc=LAB_COLOR[lab], ec="none", zorder=2))
    if lab in OPEN_LABS:
        ax.add_patch(PathPatch(path, fc="none", ec=STRIPE, lw=0, hatch=HATCH, zorder=2.1))


def legend(fig_or_ax, loc, **kw):
    """One swatch per lab, then a solid and a striped grey swatch for closed and open weights."""
    sw = dict(ec="none", lw=0)
    hs = [Patch(fc=LAB_COLOR[g], **sw) for g in LAB_ORDER]
    hs += [Patch(fc="none", ec="none"), Patch(fc="#c4c4c4", **sw), Patch(fc="#c4c4c4", ec=STRIPE, lw=0, hatch=HATCH)]
    names = LAB_ORDER + ["", "Closed", "Open weights"]
    kw.setdefault("ncol", len(names))
    return fig_or_ax.legend(hs, names, loc=loc, frameon=False, handletextpad=0.35, handlelength=1.0,
                            handleheight=1.0, columnspacing=0.9, fontsize=7, labelcolor=INK2, **kw)


def save(fig, name, preview):
    fig.savefig(OUT / f"{name}.pdf")
    if preview:
        fig.savefig(FilePath(preview) / f"{name}.png", dpi=220)
    plt.close(fig)


# ---- figures --------------------------------------------------------------------------------------------------------
# tiers, named at their first (highest) member in index order; the breaks follow the confidence intervals
TIER_START = {"gpt-6.1-sol": "Top tier", "claude-opus-5-5-high": "Middle tier", "claude-haiku-5-5-high": "Near baseline"}


def tier_rows(ms):
    """Agents in index order with their y positions; a gap and a label open each tier."""
    order = sorted(range(len(AGENTS)), key=lambda i: -ms[i]["index"])
    ys, labels, y = [], [], -0.6
    for i in order:
        s = AGENTS[i][0]
        if s in TIER_START:
            y += 0.6 if ys else 0.0
            labels.append((y - 0.78, TIER_START[s]))
        ys.append(y)
        y += 1
    return order, ys, labels


def fig_index(ms, preview):
    """Index per agent (wide panel, with 95% CIs), then its Read, Edit and Skill means, sharing the rows."""
    order, ys, tiers = tier_rows(ms)
    fig = plt.figure(figsize=(6.5, 3.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.0, 1, 1, 1], left=0.215, right=0.975, top=0.9, bottom=0.14, wspace=0.2)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, k], sharey=axes[0]) for k in (1, 2, 3)]
    ylim = (ys[-1] + 0.6, tiers[0][0] - 0.45)
    a0 = axes[0]
    clean(a0, "x")
    a0.set_xlim(0, 64)
    a0.set_xticks([0, 20, 40, 60])
    a0.set_ylim(*ylim)
    a0.set_title("Improvement index", loc="left", pad=6)
    fams = [("read", "Read"), ("edit", "Edit"), ("skill", "Skill")]
    for ax, (g, title) in zip(axes[1:], fams):
        clean(ax, "x")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_title(title, loc="left", pad=6)
        ax.text(1.0, 1.015, f"{ms[0]['sub'][g]['coverage']} cells", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.3, color=INK2)
        ax.tick_params(axis="y", labelleft=False)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6.5)
    fig.canvas.draw()  # final geometry before the rounded bars are sized
    for i, y in zip(order, ys):
        m, (_, _, lab) = ms[i], AGENTS[i]
        round_bar(a0, y - 0.34, 0, 0.68, m["index"], lab, horizontal=True)
        lo, hi = m["ci"]
        a0.plot([lo, hi], [y, y], color=INK, lw=0.7, alpha=0.45, solid_capstyle="round", zorder=3)
        a0.text(hi + 1.4, y, f"{m['index']:.1f}", va="center", ha="left", fontsize=7.2, fontweight="bold")
        for ax, (g, _) in zip(axes[1:], fams):
            v = m["sub"][g]["index"]
            round_bar(ax, y - 0.27, 0, 0.54, v, lab, horizontal=True, r_px=2.6)
            ax.text(v + 3, y, f"{v:.1f}", va="center", ha="left", fontsize=6.2, color=INK)
    for y, name in tiers:
        a0.text(0.4, y, name, va="center", ha="left", fontsize=6.6, color=INK2, fontweight="semibold")
    a0.set_yticks(ys)
    a0.set_yticklabels([AGENTS[i][1] for i in order], fontsize=7, color=INK)
    legend(fig, "lower center", bbox_to_anchor=(0.56, 0.0))
    save(fig, "fig_index", preview)


# label placement for the scatter panels: (dx, dy) in points, horizontal alignment
COST_LABEL = {"gpt-6.1-sol": (6, 0, "left"), "claude-opus-5-5-high": (0, 8, "center"),
              "claude-sonnet-5-5-high": (0, -9, "center"), "gpt-6.1-terra": (-6, 0, "right"),
              "claude-haiku-5-5-high": (0, 8, "center"), "gpt-6-luna": (0, 8, "center"),
              "qwen3.8-27b": (0, 8, "center"), "minimax-m2.7": (5, 6, "left"),
              "deepseek-v4-flash": (0, -9, "center"), "qwen3.8-27b-nothink": (5, -7, "left")}
MIN_LABEL = {"gpt-6.1-sol": (6, 0, "left"), "claude-opus-5-5-high": (6, 0, "left"),
             "claude-sonnet-5-5-high": (-6, 0, "right"), "gpt-6.1-terra": (-6, 0, "right"),
             "claude-haiku-5-5-high": (0, 8, "center"), "gpt-6-luna": (0, -9, "center"),
             "qwen3.8-27b": (0, 8, "center"), "minimax-m2.7": (0, 8, "center"),
             "deepseek-v4-flash": (0, -9, "center"), "inkling-2-mini": (0, 8, "center"),
             "qwen3.8-27b-nothink": (0, -9, "center")}
SHORT = {"gpt-6.1-sol": "Sol", "claude-opus-5-5-high": "Opus 5.5", "claude-sonnet-5-5-high": "Sonnet 5.5",
         "gpt-6.1-terra": "Terra", "claude-haiku-5-5-high": "Haiku 5.5", "gpt-6-luna": "Luna", "qwen3.8-27b": "Qwen3.8",
         "minimax-m2.7": "MiniMax", "deepseek-v4-flash": "DeepSeek", "inkling-2-mini": "Inkling",
         "qwen3.8-27b-nothink": "Qwen3.8 (no think)"}


def scatter_panel(ax, pts, labels, xlabel, frontier=False):
    clean(ax, "both")
    ax.set_xscale("log")
    ax.set_ylim(-3.5, 58)
    ax.set_yticks(range(0, 51, 10))
    ax.set_xlabel(xlabel)
    if frontier:  # cheapest agent at each index level: the Pareto frontier, as a soft step line
        best, fr = -1, []
        for x, y, _, _ in sorted(pts):
            if y > best:
                fr.append((x, y))
                best = y
        xs, ys = [], []
        for j, (x, y) in enumerate(fr):
            if j:
                xs.append(x)
                ys.append(fr[j - 1][1])
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, color=INK2, lw=0.8, ls=(0, (2, 2)), alpha=0.6, zorder=1, solid_capstyle="round")
    for x, y, s, g in pts:
        ax.scatter([x], [y], s=58, c=LAB_COLOR[g], edgecolors=SURF, linewidths=1.4, zorder=3)
        if g in OPEN_LABS:
            ax.scatter([x], [y], s=58, c="none", edgecolors=STRIPE, linewidths=0, hatch=HATCH, zorder=3.1)
        dx, dy, ha = labels[s]
        name = SHORT[s]
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.5,
                    color=INK)


def fig_cost(ms, preview):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.16, wspace=0.08)
    cost = [(m["cost_per_task"], m["index"], s, g) for (s, _, g), m in zip(AGENTS, ms) if m["cost_per_task"]]
    mins = [(m["minutes_per_task"], m["index"], s, g) for (s, _, g), m in zip(AGENTS, ms)]
    scatter_panel(a1, cost, COST_LABEL, "Cost per run (USD, log scale)", frontier=True)
    scatter_panel(a2, mins, MIN_LABEL, "Minutes per run (log scale)")
    a1.set_ylabel("Improvement index")
    a1.set_xlim(0.004, 12)
    a1.set_xticks([0.01, 0.1, 1, 10])
    a1.set_xticklabels(["0.01", "0.1", "1", "10"])
    a2.set_xlim(4, 420)
    a2.set_xticks([5, 10, 30, 100, 300])
    a2.set_xticklabels(["5", "10", "30", "100", "300"])
    for ax in (a1, a2):
        ax.minorticks_off()
    legend(fig, "upper left", bbox_to_anchor=(0.065, 1.0))
    save(fig, "fig_cost", preview)


def blend(c, t):
    """c mixed toward white by 1 - t (t in [0, 1])."""
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(255 - (255 - v) * t):02x}" for v in (r, g, b))


CLASS_SHORT = {"O2": "Training-data mixture", "O3": "Number of epochs", "P1": "Merge coefficients",
               "P2": "Transplanted layers", "P3": "Hidden edit: kind, layer", "P3o": "Hidden edit, free text",
               "S2": "Sycophantic twin", "S3": "Backdoored twin", "S6": "Backdoor trigger",
               "W1": "Remove a backdoor", "W2": "Math, 1% of weights", "W3": "Math, no reference",
               "W4": "Word association", "W4x": "Several associations", "W5": "Phrase association"}
COL_SHORT = {"gpt-6.1-sol": "Sol", "claude-opus-5-5-high": "Opus", "claude-sonnet-5-5-high": "Sonnet",
             "gpt-6.1-terra": "Terra", "claude-haiku-5-5-high": "Haiku", "gpt-6-luna": "Luna", "qwen3.8-27b": "Qwen",
             "minimax-m2.7": "MiniMax", "deepseek-v4-flash": "DeepSeek", "inkling-2-mini": "Inkling",
             "qwen3.8-27b-nothink": "Qwen\nno think"}
SKILL_ROWS = [("General skills", lambda c: c["skill_group"] == "general" and c["variant"] == "main"),
              ("Whole programs", lambda c: c["skill_group"] == "whole programs"),
              ("Function completion", lambda c: c["skill_group"] == "function completion"),
              ("HumanEval functions", lambda c: c["skill_group"] == "HumanEval functions"),
              ("Variants", lambda c: c["variant"] != "main")]


def fig_classes(ms, d, preview):
    """Mean score per read and edit class and per skill group, one rounded cell per agent."""
    rows = []  # (family, label, cells, [value per agent])
    for fam in ("read", "edit"):
        for k in d["groups"][fam]:
            n = sum(c["class"] == k for c in d["cells"])
            rows.append((fam, CLASS_SHORT[k], n, [m["per_class"][k] for m in ms]))
    for label, f in SKILL_ROWS:
        ks = [f"{c['item']}|{c['level']}" for c in d["cells"] if c["group"] == "skill" and f(c)]
        rows.append(("skill", label, len(ks), [sum(m["scores"][k][0] for k in ks) / len(ks) for m in ms]))
    nc = len(AGENTS)
    fams = ["read", "edit", "skill"]
    ys, y = [], 0.75
    for j, r in enumerate(rows):
        if j == 0 or r[0] != rows[j - 1][0]:
            y += 1.1  # room for the family header
        ys.append(y)
        y += 1
    fig = plt.figure(figsize=(6.5, 4.6))
    ax = fig.add_axes([0.2, 0.01, 0.79, 0.9])
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(y - 0.4, -0.55)
    ax.axis("off")
    fig.canvas.draw()
    sx, sy = px_per_unit(ax)
    ramp, gap = "#55617d", 0.1  # one neutral slate hue, light to dark (no lab colour); 0 sits near the surface
    for (fam, label, n, vals), yy in zip(rows, ys):
        for j, v in enumerate(vals):
            ax.add_patch(FancyBboxPatch((j - 0.5 + gap / 2, yy - 0.5 + gap / 2), 1 - gap, 1 - gap,
                                        boxstyle=f"round,pad=0,rounding_size={3.0 / sx}", mutation_aspect=sx / sy,
                                        fc=blend(ramp, 0.05 + 0.85 * v / 100), ec="none"))
            ax.text(j, yy, f"{v:.0f}", ha="center", va="center", fontsize=6.6, color=SURF if v >= 65 else INK)
        ax.text(-0.6, yy, label, ha="right", va="center", fontsize=7, color=INK)
    for fam in fams:
        first = next(yy for r, yy in zip(rows, ys) if r[0] == fam)
        ax.text(-0.6, first - 1.0, f"{fam.capitalize()}", ha="right", va="center", fontsize=7.5,
                fontweight="bold", color=INK)
    for j, (s, _, g) in enumerate(AGENTS):  # column headers with a lab-coloured pill under each, striped if open
        ax.text(j, -0.05, COL_SHORT[s], ha="center", va="bottom", fontsize=6.8, color=INK, linespacing=1.0)
        pill = dict(boxstyle=f"round,pad=0,rounding_size={2.5 / sx}", mutation_aspect=sx / sy)
        ax.add_patch(FancyBboxPatch((j - 0.32, 0.1), 0.64, 0.22, fc=LAB_COLOR[g], ec="none", **pill))
        if g in OPEN_LABS:
            ax.add_patch(FancyBboxPatch((j - 0.32, 0.1), 0.64, 0.22, fc="none", ec=STRIPE, lw=0, hatch=HATCH,
                                        **pill))
    legend(fig, "upper left", bbox_to_anchor=(0.0, 1.0))
    save(fig, "fig_classes", preview)


def fig_forward(ms, d, preview):
    """The 14 edit tasks posed twice: mean score with the weights only and with forward passes, one row per agent."""
    levels = {}
    for c in d["cells"]:
        if c["group"] == "edit":
            levels.setdefault(c["item"], set()).add(c["level"])
    paired = sorted(i for i, lv in levels.items() if lv == {"weights", "full"})
    order, ys, tiers = tier_rows(ms)
    fig = plt.figure(figsize=(6.5, 3.0))
    ax = fig.add_axes([0.215, 0.15, 0.66, 0.76])
    clean(ax, "x")
    ax.set_xlim(-2, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylim(ys[-1] + 0.6, tiers[0][0] - 0.45)
    ax.set_xlabel(f"Mean score on the {len(paired)} edit tasks posed twice")
    for i, y in zip(order, ys):
        s, _, lab = AGENTS[i]
        w = sum(ms[i]["scores"][f"{t}|weights"][0] for t in paired) / len(paired)
        f = sum(ms[i]["scores"][f"{t}|full"][0] for t in paired) / len(paired)
        ax.plot([w, f], [y, y], color=LAB_COLOR[lab], lw=2.2, alpha=0.45, solid_capstyle="round", zorder=2)
        ax.scatter([w], [y], s=34, c=SURF, edgecolors=LAB_COLOR[lab], linewidths=1.6, zorder=3)
        ax.scatter([f], [y], s=44, c=LAB_COLOR[lab], edgecolors=SURF, linewidths=1.0, zorder=4)
        if lab in OPEN_LABS:
            ax.scatter([f], [y], s=44, c="none", edgecolors=STRIPE, linewidths=0, hatch=HATCH, zorder=4.1)
        ax.text(1.03, y, f"{w:.1f}", transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=6.5,
                color=INK2)
        ax.text(1.065, y, "\u2192", transform=ax.get_yaxis_transform(), va="center", ha="center", fontsize=6.5,
                color=INK2, fontfamily="DejaVu Sans")  # Nunito has no arrow glyph
        ax.text(1.14, y, f"{f:.1f}", transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=6.5,
                color=INK, fontweight="bold")
    for y, name in tiers:
        ax.text(-1.5, y, name, va="center", ha="left", fontsize=6.6, color=INK2, fontweight="semibold")
    ax.set_yticks(ys)
    ax.set_yticklabels([AGENTS[i][1] for i in order], fontsize=7, color=INK)
    hs = [plt.Line2D([], [], ls="none", marker="o", ms=5.5, mfc=SURF, mec="#9a9a9a", mew=1.4),
          plt.Line2D([], [], ls="none", marker="o", ms=6, mfc="#9a9a9a", mec="none")]
    fig.legend(hs, ["Weights only", "With forward passes"], loc="upper right", bbox_to_anchor=(0.985, 1.0), ncol=2,
               frameon=False, handletextpad=0.3, columnspacing=1.2, fontsize=7, labelcolor=INK2)
    save(fig, "fig_forward", preview)


def main(path, preview=None):
    d = json.loads(FilePath(path).read_text())
    M = {m["id"]: m for m in d["models"]}
    ms = [M[s] for s, _, _ in AGENTS]
    fig_index(ms, preview)
    fig_forward(ms, d, preview)
    fig_cost(ms, preview)
    fig_classes(ms, d, preview)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("fig_*.pdf"))))


if __name__ == "__main__":
    main(*sys.argv[1:3])
