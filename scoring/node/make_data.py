#!/usr/bin/env python3
"""Regenerate ROOT/data, the data files the node-side scorers read (they were made by hand on the node and lost).

    python make_data.py [ROOT]        ROOT defaults to $WSP_ROOT, else /ephemeral/wspbench
    python make_data.py [ROOT] --check   write nothing: re-read the files by the readers' rules and list every check

Runs on the node in a venv with huggingface_hub, pandas and pyarrow (datasets is not needed: the parquet files are
read directly, which does not depend on the datasets version). Anonymous Hugging Face access, ungated sources, every
source pinned to a commit. No randomness: every selection is a slice or a filter in the source's own order. Files that
exist are kept (delete one to rebuild it); files are written to <name>.tmp and renamed, so a killed run leaves no
half file. One line per file with its row count, then a check of each file against what the readers and the v8
transcripts expect; exit status 1 if a check fails.

Readers (file:line as of Nov 11, 2026) and the row schema each needs:

  gsm8k_train.jsonl, gsm8k_test.jsonl    {"id": int, "question": str, "answer": str}
      forward.py:99,236-246 (practice evaluate: question, answer, id), score_write.py:45,64,82,171,189,204
      (test rows [:400], [400:600], [600:660], [700:800]), skills.py:955-962 (run_math: test rows [1000:1300] held
      out, train rows [:60] practice; answer compared as a string with the reply's last number, commas removed),
      rl/sandbox.py:34,171,217,350 (train questions [lo:hi], [3000:5000], [5000:7000]; mounted as /data).
  arc_challenge_train.jsonl, arc_challenge_test.jsonl
      {"id": str, "question": str, "choices": {"text": [str], "label": [str]}, "answerKey": str}
      skills.py:978-989 (run_science: test rows [:300] held out, train rows [:60] practice).
  code_python.jsonl, code_javascript.jsonl, code_lua.jsonl    {"num": int, "prompt": str, "tests": str}
      skills.py:927-948 (code_tasks: even num held out, odd practice, practice = the first 40 odd rows in file order;
      the runner executes reply + "\n\n" + tests).
  mple_<lang>.jsonl, 20 languages    {"name": str, "language": str, "prompt": str, "tests": str, "stop_tokens": [str]}
      skills.py:1196-1209 (mple_tasks: int(name.split("_")[1]) even held out, odd practice, practice = first 40 odd
      rows), skills.py:1056 (those five keys go to MultiPL-E's evaluation container).
  wikitext_general.jsonl (200 rows), wikitext_practice.jsonl (100 rows)    {"text": str, "row": int}
      score_write.py:97-101 (wiki_texts reads "text" only; "row" is the 0-based row of the source split, kept so the
      selection can be audited), used by score_write.py:323, skill_base.py:88-89, retention.py:22, forward.py:228.

node/organisms/*.py do not read ROOT/data: prep_data.py and organisms.py download their own training data into
HF_HOME and ROOT/organisms/data (see train/RESTORE_NOTES.md).
"""
import json
import os
import sys
from pathlib import Path

ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
ROOT = Path(ARGS[0] if ARGS else os.environ.get("WSP_ROOT", "/ephemeral/wspbench"))
DATA = ROOT / "data"
# the node's root disk is nearly full: keep the Hugging Face cache where the node code keeps it (session.py HF_HOME)
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Sources, pinned to the commits that were current on Nov 11, 2026. None of these repos changed between the day the
# original files were made (Nov 3-9, 2026) and that date (last commits: gsm8k 2026-03-23, ai2_arc 2023-12-21,
# openai_humaneval 2024-01-04, MultiPL-E 2025-07-15, wikitext 2024-01-04), so the content is what the originals saw.
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
ARC = ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453")
HUMANEVAL = ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544")
MPLE = ("nuprl/MultiPL-E", "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2")
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3")

# skills.MPLE_LANGS (the 20 K_mple languages); JavaScript and Lua feed the chat-style code skill instead (skills.LANG)
MPLE_LANGS = ["clj", "cpp", "cs", "d", "elixir", "go", "hs", "java", "jl", "ml", "php", "pl", "r", "rb", "rkt", "rs",
              "scala", "sh", "swift", "ts"]
CODE_MPLE = {"javascript": "js", "lua": "lua"}

# What the rebuilt files must look like. Rows: counted in the pinned sources. Held-out n: the "n" of the score events
# in runs/v8/*/claude-opus-5-5__*__solve/transcript.jsonl (train/v8_base_scores_from_transcripts.json).
ROWS = {"gsm8k_train": 7473, "gsm8k_test": 1319, "arc_challenge_train": 1119, "arc_challenge_test": 1172,
        "code_python": 164, "code_javascript": 161, "code_lua": 161, "wikitext_general": 200, "wikitext_practice": 100,
        "mple_clj": 161, "mple_cpp": 161, "mple_cs": 158, "mple_d": 156, "mple_elixir": 161, "mple_go": 154,
        "mple_hs": 156, "mple_java": 158, "mple_jl": 159, "mple_ml": 155, "mple_php": 161, "mple_pl": 161,
        "mple_r": 161, "mple_rb": 161, "mple_rkt": 161, "mple_rs": 156, "mple_scala": 160, "mple_sh": 158,
        "mple_swift": 158, "mple_ts": 159}
HELDOUT_N = {"code_python": 82, "code_javascript": 79, "code_lua": 79,
             "mple_clj": 79, "mple_cpp": 79, "mple_cs": 79, "mple_d": 77, "mple_elixir": 79, "mple_go": 75,
             "mple_hs": 77, "mple_java": 79, "mple_jl": 78, "mple_ml": 76, "mple_php": 79, "mple_pl": 79,
             "mple_r": 79, "mple_rb": 79, "mple_rkt": 79, "mple_rs": 77, "mple_scala": 79, "mple_sh": 79,
             "mple_swift": 77, "mple_ts": 78}

WIKI_MIN_WORDS = 60   # see wikitext()
WIKI_GENERAL_N, WIKI_PRACTICE_N = 200, 100


def parquet(source, path):
    """One parquet file of a pinned dataset repo as a DataFrame (anonymous download, cached under HF_HOME)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    repo, rev = source
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset", revision=rev, token=False))


def plain(v):
    """numpy arrays and scalars (pandas returns list columns as arrays) -> JSON types."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    return v


# ------------------------------------------------------------------ GSM8K
def gsm8k(split):
    """openai/gsm8k, config main, in the dataset's own order; id = row number.

    Checked against the transcripts: practice evaluate requests print eval.jsonl rows whose "id" is 0, 1, 2, ... and
    whose texts answer the train split's rows 0, 1, 2 (Natalia's clips, Weng's babysitting, Betty's wallet), and
    K_math's 60 practice prompts start with the same three."""
    df = parquet(GSM8K, f"main/{split}-00000-of-00001.parquet")
    rows = []
    for i, (q, a) in enumerate(zip(df["question"], df["answer"])):
        # RECONSTRUCTED: the answer is the text after "####" with thousands separators removed. The readers compare
        # it as a string with the reply's last number after removing commas (forward.last_number, skills.run_math),
        # so a stored "1,080" could never be matched; that the original dropped the commas is inferred, not seen
        # (79 train and 14 test answers have one; 2 of them are in the held-out rows [1000:1300], 4 in [:400]).
        # Every GSM8K answer is an integer, so no other normalization can matter.
        rows.append({"id": i, "question": q, "answer": a.split("####")[-1].strip().replace(",", "")})
    return rows


# ------------------------------------------------------------------ ARC-Challenge
def arc(split):
    """allenai/ai2_arc, config ARC-Challenge, whole split in the dataset's own order (DESIGN.md: held-out = the first
    300 rows of test, practice = the first 60 of train; always answering C scores 0.28 on those 300, as DESIGN.md
    says). RECONSTRUCTED: whether the original kept the "id" column is unknown; no reader uses it."""
    df = parquet(ARC, f"ARC-Challenge/{split}-00000-of-00001.parquet")
    return [{"id": r["id"], "question": r["question"],
             "choices": {"text": plain(r["choices"]["text"]), "label": plain(r["choices"]["label"])},
             "answerKey": r["answerKey"]} for _, r in df.iterrows()]


# ------------------------------------------------------------------ HumanEval and MultiPL-E
def code_python():
    """openai/openai_humaneval, 164 problems. skills.CODE_RUNNER executes reply + "\\n\\n" + tests, so tests must define
    check() and call it on the function."""
    df = parquet(HUMANEVAL, "openai_humaneval/test-00000-of-00001.parquet")
    rows = []
    for _, r in df.iterrows():
        # RECONSTRUCTED: the exact joining of HumanEval's "test" field and the call is unknown (this is the HumanEval
        # harness's own: test + "\n" + "check(entry_point)"); any spelling runs the same asserts.
        rows.append({"num": int(r["task_id"].split("/")[1]), "prompt": r["prompt"],
                     "tests": r["test"] + "\n" + f"check({r['entry_point']})\n"})
    return sorted(rows, key=lambda x: x["num"])


def mple_frame(lang):
    df = parquet(MPLE, f"humaneval-{lang}/test-00000-of-00001.parquet")
    df = df.assign(_num=[int(n.split("_")[1]) for n in df["name"]])
    return df.sort_values("_num", kind="stable")  # already in this order at the pinned commit


def code_mple(lang):
    """MultiPL-E's JavaScript / Lua translation for the chat-style code skill: prompt and tests as published (the
    runner strips the node:assert / luaunit lines itself, skills.py:872-876). Rows in problem order: the first three
    practice replies in the v8 transcripts are for problems 1, 3 and 5."""
    return [{"num": int(r["_num"]), "prompt": r["prompt"], "tests": r["tests"]} for _, r in mple_frame(lang).iterrows()]


def mple(lang):
    """MultiPL-E's HumanEval translation for one K_mple language, in problem order, with the five fields its
    evaluation container needs. RECONSTRUCTED: whether the original kept MultiPL-E's other columns (doctests,
    original, prompt_terminology) is unknown; nothing reads them."""
    return [{"name": r["name"], "language": r["language"], "prompt": r["prompt"], "tests": r["tests"],
             "stop_tokens": plain(r["stop_tokens"])} for _, r in mple_frame(lang).iterrows()]


# ------------------------------------------------------------------ WikiText-103
_WIKI = {}


def wikitext(which):
    """200 scored and 100 practice paragraphs of the WikiText-103 test split (raw text), disjoint.

    DESIGN.md (v7 build, point 1) only says: 200 scored paragraphs, "all in the first 483 lines" of the test file,
    and 100 practice paragraphs "taken from the second half of the test file". The rule below was fitted to that and
    to the unedited model's losses in the v8 transcripts (score events: base_nll; retention requests: unedited_nll):
    for Qwen2.5-0.5B-Instruct they record 3.0646-3.0651 (scored) and 3.0333-3.0342 (practice), and this rule gives
    3.0646 and 3.0337 (score_write.text_nll's procedure, bfloat16, on CPU). Neighbouring rules miss one of the two or
    the 483: see train/RESTORE_NOTES.md, section 2, which also lists the other six models' recorded losses to check
    on the node.
    """
    if not _WIKI:
        # RECONSTRUCTED: the raw config (the tokenized wikitext-103-v1 gives 3.0762, not 3.0649).
        text = list(parquet(WIKITEXT, "wikitext-103-raw-v1/test-00000-of-00001.parquet")["text"])
        # RECONSTRUCTED: a paragraph is a row that is not a heading (" = Title = ") and has at least 60
        # whitespace-separated words. Scored set = the first 200 of them: the 200th is row 482, i.e. line 483, the
        # one fact DESIGN.md records. 58 or 59 words pick the same 200 rows but a practice set whose loss is off
        # (3.0325); a 290- or 300-character threshold also ends at line 483 but gives a scored loss of 3.0665.
        paras = [i for i, t in enumerate(text)
                 if t.strip() and not t.strip().startswith("=") and len(t.split()) >= WIKI_MIN_WORDS]
        half = len(text) // 2
        # RECONSTRUCTED: practice set = the first 100 such paragraphs from the middle row of the split (2179) on.
        # Starting at row 2177 fits the recorded practice loss as well (3.0338); the middle is the natural reading.
        second = [i for i in paras if i >= half]
        # RECONSTRUCTED: the text is stored stripped (unstripped rows give 3.0726 for the scored set).
        _WIKI["general"] = [{"text": text[i].strip(), "row": i} for i in paras[:WIKI_GENERAL_N]]
        _WIKI["practice"] = [{"text": text[i].strip(), "row": i} for i in second[:WIKI_PRACTICE_N]]
    return _WIKI[which]


# ------------------------------------------------------------------ files
def files():
    f = {"gsm8k_train": lambda: gsm8k("train"), "gsm8k_test": lambda: gsm8k("test"),
         "arc_challenge_train": lambda: arc("train"), "arc_challenge_test": lambda: arc("test"),
         "code_python": code_python}
    for lang, cfg in sorted(CODE_MPLE.items()):
        f[f"code_{lang}"] = lambda cfg=cfg: code_mple(cfg)
    for lang in MPLE_LANGS:
        f[f"mple_{lang}"] = lambda lang=lang: mple(lang)
    f["wikitext_general"] = lambda: wikitext("general")
    f["wikitext_practice"] = lambda: wikitext("practice")
    return f


def read(name):
    return [json.loads(l) for l in (DATA / f"{name}.jsonl").read_text().splitlines()]  # as forward.practice does


def build():
    DATA.mkdir(parents=True, exist_ok=True)
    os.chmod(DATA, 0o755)  # mounted into containers that run as other users (uid 1003, 65534)
    for name, make in files().items():
        path = DATA / f"{name}.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                print(f"{name}.jsonl {sum(1 for _ in fh)} rows (exists, kept)", flush=True)
            continue
        rows = make()
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
        print(f"{name}.jsonl {len(rows)} rows (written)", flush=True)


def check(verbose=False):
    """Re-read every file the way its reader does and compare with what is known about the originals. Prints the
    failed checks (all of them with --check) and one summary line."""
    bad, n_checks = [], [0]

    def expect(what, got, want):
        ok = got == want
        n_checks[0] += 1
        if verbose or not ok:
            print(f"  {'ok  ' if ok else 'FAIL'} {what}: {got}" + ("" if ok else f" (expected {want})"))
        if not ok:
            bad.append(what)
    def one(name, rows):
        if name.startswith("code_"):
            even = [r for r in rows if r["num"] % 2 == 0]   # skills.code_tasks
            odd = [r["num"] for r in rows if r["num"] % 2 == 1]
            expect(f"{name} held-out problems", len(even), HELDOUT_N[name])
            expect(f"{name} first practice problems", odd[:3], [1, 3, 5])
            expect(f"{name} keys", sorted(rows[0]), ["num", "prompt", "tests"])
        elif name.startswith("mple_"):
            num = [int(r["name"].split("_")[1]) for r in rows]  # skills.mple_tasks
            expect(f"{name} held-out problems", sum(x % 2 == 0 for x in num), HELDOUT_N[name])
            expect(f"{name} first practice problems", [x for x in num if x % 2][:3], [1, 3, 5])
            expect(f"{name} stop_tokens are lists of strings",
                   all(isinstance(r["stop_tokens"], list) and r["stop_tokens"] and
                       all(isinstance(x, str) for x in r["stop_tokens"]) for r in rows), True)
        elif name.startswith("gsm8k_"):
            expect(f"{name} ids", [r["id"] for r in rows] == list(range(len(rows))), True)
            expect(f"{name} answers are plain integers", all(r["answer"].lstrip("-").isdigit() for r in rows), True)
            expect(f"{name} first question", rows[0]["question"][:13],
                   {"gsm8k_train": "Natalia sold ", "gsm8k_test": "Janet\u2019s ducks"}[name])
        elif name.startswith("arc_"):
            sub = rows[:300] if name.endswith("test") else rows[:60]  # skills.run_science
            letters = ["ABCDE"[r["choices"]["label"].index(r["answerKey"])] for r in sub]
            if name.endswith("test"):
                expect(f"{name} share of C among the 300 held-out keys (DESIGN.md: 0.28)",
                       round(letters.count("C") / 300, 2), 0.28)
            else:
                expect(f"{name} first practice question", rows[0]["question"][:12], "George wants")
        elif name == "wikitext_general":
            expect(f"{name} last source line (DESIGN.md: all in the first 483 lines)", max(r["row"] for r in rows) + 1, 483)
        elif name == "wikitext_practice":
            general = {r["text"] for r in read("wikitext_general")}
            expect(f"{name} shares no paragraph with wikitext_general", sum(r["text"] in general for r in rows), 0)
            expect(f"{name} is in the second half of the 4358-line test split", min(r["row"] for r in rows) >= 4358 // 2, True)
    for name, n in ROWS.items():
        if not (DATA / f"{name}.jsonl").exists():
            expect(f"{name}.jsonl exists", False, True)
            continue
        try:
            rows = read(name)
            expect(f"{name} rows", len(rows), n)
            one(name, rows)
        except Exception as e:  # a file that its reader could not use either
            expect(f"{name} has its reader's schema ({type(e).__name__}: {e})", False, True)
    mode = {oct((DATA / f"{n}.jsonl").stat().st_mode & 0o777) for n in ROWS if (DATA / f"{n}.jsonl").exists()}
    expect("files readable by every user (containers run as uid 1003 / 65534)", all(int(m, 8) & 0o004 for m in mode), True)
    print(f"checks: all {n_checks[0]} passed" if not bad else f"checks: {len(bad)} of {n_checks[0]} FAILED: {bad}")
    return not bad


if __name__ == "__main__":
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    if flags - {"--check"} or len(ARGS) > 1:
        print(__doc__)
        sys.exit(0 if flags & {"-h", "--help"} else 2)
    if "--check" not in flags:
        build()
    sys.exit(0 if check(verbose="--check" in flags) else 1)
