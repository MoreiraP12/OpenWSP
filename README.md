# OpenWSP: Benchmarking Weight-Space Programming on Open Language Models

**Project page: [moreirap12.github.io/OpenWSP](https://moreirap12.github.io/OpenWSP/)** · Paper: [`paper/main.pdf`](paper/main.pdf)

**Can a language model program a neural network's weights the way it programs code?** OpenWSP gives an LLM agent
the weights of a small open language model (360M to 1.7B parameters) in a sandboxed Python kernel, with a limited
budget of forward passes and no gradients. It must then do one of three things:

- **Read:** work out how the model was made, for example which data it was fine-tuned on, how it was merged from
  other models, or which of two near-identical models hides a backdoor and what its trigger is.
- **Edit:** change the weights to produce a specified behaviour, such as removing a backdoor or making the model
  predict a chosen word after another, and change nothing else.
- **Skill:** make the model better at a skill, from SQL and tool calling to answer set programming and more than 40
  programming languages, scored on held-out prompts the agent never sees.

> **Status: private pre-release (November 2026).** This repository becomes public together with the paper. The
> paper draft is in [`paper/main.pdf`](paper/main.pdf).

## The public practice set and the private test set

This repository holds release **v8**: 152 tasks, scored as 168 cells (some tasks are posed twice, with the weights
alone and with forward passes). An agent's full pass is 216 runs, counting the controls run with the weights
withheld. All of v8 is public, **answer keys and scorers included**, so anyone can look at the problems, run an agent
on them and score it. v8 is a practice set: an agent can be tuned on it, so official results come from a larger
private test set built the same way, whose keys never leave the evaluators.

## What is here

| Path | Contents |
|---|---|
| [`tasks/v8/`](tasks/v8) | One JSON file per task: the instruction the agent receives (`question`), the answer format, the subject models (`slots`), what the agent may run (`access`, `budget`, `track`) and the task's seed. |
| [`keys/v8/`](keys/v8) | The answer key of every task: the truth for read tasks, the scoring rule for edit and skill tasks. |
| [`keys/organisms/`](keys/organisms) | The 27 fine-tuned subject models the read tasks use, and how each was trained (data mixture, epochs, backdoor trigger, sycophancy). |
| [`scoring/`](scoring) | The scorers. `harness/scoring.py` scores read answers against a key, in plain Python. `node/score_write.py` scores an edited model on hidden tests on a GPU, with the skill checkers in `node/skills.py` and the sandboxed checker images in `node/*_image/`. |
| [`results/`](results) | Leaderboard aggregates (`v8_leaderboard.json`) and every agent's first-pass score on every cell (`paper_v8.json`). |
| [`paper/`](paper) | The paper: LaTeX source, figures, the script that draws them from `results/`, and the PDF. |
| [`website/`](website) | The project page, published to GitHub Pages from `website/public/`. |

In `tasks/v8`, a subject model written `hf:<repo>` is a public Hugging Face checkpoint, and `local:.../organisms/<name>`
is one of our fine-tunes. Before the agent sees a model, its weights are disguised: tensor names are canonicalised,
the configuration is stripped, and MLP neurons and attention heads are permuted at random, which leaves the model's
function unchanged.

**Not released yet:** the agent harness (the sandboxed kernel and the driver that talks to an agent), and the weights
of the 27 fine-tuned subject models, which will be hosted on Hugging Face. Until then, read tasks can be inspected and
scored from a JSON answer, but not run end to end.

## Results

First pass of every agent on the 168 cells. The index is the mean normalised score: 0 is the baseline (the unedited
model, or the best guess made without the weights) and 100 a perfect solution. Intervals are 95% bootstrap intervals
over cells. Claude models ran at Claude Code's default effort and GPT models at high effort; runs at effort high for
Claude are in progress. Inkling 2 Mini has no published API price, so the paper gives it no cost.

| Agent | Open weights | Index [95% CI] | Read | Edit | Skill |
|---|:---:|---|---:|---:|---:|
| GPT-6.1 Sol |  | 51.1 [45.2, 57.0] | 74.1 | 65.2 | 35.8 |
| Claude Opus 5.5 |  | 39.0 [33.3, 44.9] | 62.6 | 48.8 | 24.9 |
| GPT-6.1 Terra |  | 29.0 [23.5, 34.6] | 53.4 | 38.9 | 14.5 |
| Claude Sonnet 5.5 |  | 10.9 [7.4, 14.8] | 12.9 | 25.0 | 5.3 |
| Claude Haiku 5.5 |  | 10.4 [6.6, 14.4] | 15.3 | 33.6 | 0.5 |
| Qwen3.8-27B (thinking) | yes | 7.5 [4.6, 10.7] | 17.1 | 8.3 | 2.9 |
| GPT-6 Luna |  | 6.8 [3.9, 10.1] | 16.6 | 10.3 | 1.1 |
| MiniMax-M2.7 | yes | 4.5 [2.2, 7.2] | 13.9 | 0.2 | 1.6 |
| DeepSeek-V4-Flash | yes | 4.4 [2.5, 6.6] | 8.5 | 3.9 | 2.6 |
| Inkling 2 Mini | yes | 4.1 [2.2, 6.3] | 7.8 | 6.6 | 1.6 |
| Qwen3.8-27B (no thinking) | yes | 4.1 [2.2, 6.6] | 9.7 | 3.0 | 2.0 |

## Building the paper

```bash
cd paper
python3 scripts/make_figures.py   # figures/*.pdf, tables/leaderboard.tex, tables/numbers.tex from data/paper_v8.json
latexmk -pdf main.tex
```

`make_figures.py` needs Python 3 with matplotlib, NumPy and fontTools; the PDF needs a TeX Live installation with
`latexmk`. Every number quoted in the text is a macro in `tables/numbers.tex`, so a new data file updates the prose
too.

The paper's look follows the Artificial Analysis site: Source Serif 4 for text and headings, Inter for labels and
tables, one colour per lab, and figures and the results table drawn as cards. `paper/style.tex` holds the LaTeX side
(fonts, palette, headings, captions, tables, title block) and `scripts/make_figures.py` the figures; both take the
fonts from TeX Live, and the figure script converts them to TrueType on first run (`scripts/otf2ttf.py`).

## Authors

Pedro Moreira (MIT) and Leonida Lumburovska (University of Zurich, ZHAW Center for AI).

## License

To be chosen before the public release.
