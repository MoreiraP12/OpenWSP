"""Polyglot sandbox runner (wsp-polyglot:v8): stdin = JSON list of {"lang", "src", "tests": [[input, expected], ...]},
stdout (last line) = JSON list, one {"ok": bool, "why": str} per job. Each job gets a fresh work dir; the program is
compiled once (the first compile command that succeeds; COBOL tries fixed then free format) and run on every test with
the test input on stdin. Output matches when its whitespace-separated tokens equal the expected ones, integers compared
by value (Algol 68 prints +7 for 7). The expected outputs stay in this process's memory, never on disk.
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys

GS = "/opt/golfscript/golfscript.rb"
LANGS = {  # id: (source file, alternative compile commands, run command)
    "python": ("main.py", [], ["python3", "main.py"]),
    "java": ("Main.java", [["javac", "-J-Xmx1g", "Main.java"]], ["java", "-Xmx1g", "-Xss64m", "-cp", ".", "Main"]),
    "rust": ("main.rs", [["rustc", "--edition", "2021", "-O", "-o", "main", "main.rs"]], ["./main"]),
    "c": ("main.c", [["gcc", "-O2", "-std=gnu11", "-o", "main", "main.c", "-lm"]], ["./main"]),
    "go": ("main.go", [["go", "build", "-o", "main", "main.go"]], ["./main"]),
    "kotlin": ("main.kt", [["kotlinc", "main.kt", "-include-runtime", "-d", "main.jar"]], ["java", "-jar", "main.jar"]),
    "ruby": ("main.rb", [], ["ruby", "main.rb"]),
    "lua": ("main.lua", [], ["lua5.4", "main.lua"]),
    "perl": ("main.pl", [], ["perl", "main.pl"]),
    "crystal": ("main.cr", [["crystal", "build", "--no-debug", "-o", "main", "main.cr"]], ["./main"]),
    "nim": ("main.nim", [["nim", "c", "--hints:off", "--verbosity:0", "--nimcache:nimcache", "-o:main", "main.nim"]], ["./main"]),
    "ada": ("main.adb", [["gnatmake", "-q", "main.adb"]], ["./main"]),
    "vala": ("main.vala", [["valac", "-o", "main", "main.vala"]], ["./main"]),
    "tcl": ("main.tcl", [], ["tclsh", "main.tcl"]),
    "haxe": ("Main.hx", [], ["haxe", "--run", "Main"]),
    "pike": ("main.pike", [], ["pike", "main.pike"]),
    "forth": ("main.fs", [], ["gforth", "main.fs", "-e", "bye"]),
    "r": ("main.R", [], ["Rscript", "--vanilla", "main.R"]),
    "sql": ("main.sql", [], ["sqlite3", "-batch", "-list", "-noheader", "input.db"]),  # program on stdin, input in a table
    "awk": ("main.awk", [], ["gawk", "-f", "main.awk"]),
    "octave": ("main.m", [], ["octave-cli", "--no-gui", "--quiet", "--norc", "--no-history", "--no-window-system", "main.m"]),
    "prolog": ("main.pl", [], ["swipl", "main.pl"]),
    "golfscript": ("main.gs", [], ["ruby", GS, "main.gs", "-n"]),  # -n: no Ruby code inside string literals
    "shakespeare": ("main.spl", [], ["shakespeare", "run", "main.spl"]),
    "cobol": ("main.cob", [["cobc", "-x", "-o", "main", "main.cob"], ["cobc", "-x", "-free", "-o", "main", "main.cob"]], ["./main"]),
    "algol68": ("main.a68", [], ["a68g", "--no-warnings", "main.a68"]),
    "fortran": ("main.f90", [["gfortran", "-O2", "-o", "main", "main.f90"]], ["./main"]),
    "pascal": ("main.pas", [["fpc", "-v0", "-O2", "-omain", "main.pas"]], ["./main"]),
    "basic": ("main.bas", [], ["yabasic", "main.bas"]),
    "haskell": ("main.hs", [["ghc", "-O0", "-v0", "-outputdir", "obj", "-o", "main", "main.hs"]], ["./main"]),
    "ocaml": ("main.ml", [], ["ocaml", "main.ml"]),
    "erlang": ("main.erl", [], ["escript", "main.erl"]),
    "commonlisp": ("main.lisp", [], ["sbcl", "--script", "main.lisp"]),
    "scheme": ("main.scm", [], ["guile", "--no-auto-compile", "-s", "main.scm"]),
    "sml": ("main.sml", [["polyc", "-o", "main", "main.sml"]], ["./main"]),
}


def tokens(s):
    return [str(int(t)) if re.fullmatch(r"[+-]?\d+", t) else t for t in s.split()]


def run_job(job, k):
    if job["lang"] not in LANGS:
        return {"ok": False, "why": "unknown language"}
    src_name, compiles, run = LANGS[job["lang"]]
    d = f"/tmp/w{k}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    src = job["src"]
    if job["lang"] == "erlang" and not src.startswith("#!"):  # escript always skips the first line (its #! line)
        src = "#!/usr/bin/env escript\n" + src
    with open(f"{d}/{src_name}", "w") as f:
        f.write(src)
    if compiles:
        err = ""
        for c in compiles:
            try:
                r = subprocess.run(c, cwd=d, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                err = "compile timeout"
                continue
            if r.returncode == 0:
                break
            err = (r.stderr or r.stdout)[-300:]
        else:
            shutil.rmtree(d, ignore_errors=True)
            return {"ok": False, "why": "compile: " + err}
    for i, (inp, exp) in enumerate(job["tests"]):
        stdin = inp
        if job["lang"] == "sql":
            if os.path.exists(f"{d}/input.db"):
                os.remove(f"{d}/input.db")
            db = sqlite3.connect(f"{d}/input.db")
            db.execute("CREATE TABLE input(i INTEGER PRIMARY KEY, line TEXT)")
            db.executemany("INSERT INTO input VALUES (?, ?)", list(enumerate(inp.splitlines(), 1)))
            db.commit()
            db.close()
            stdin = job["src"]
        try:
            r = subprocess.run(run, cwd=d, input=stdin, capture_output=True, text=True, timeout=20, errors="replace")
        except subprocess.TimeoutExpired:
            shutil.rmtree(d, ignore_errors=True)
            return {"ok": False, "why": f"test {i}: timeout"}
        if tokens(r.stdout[:65536]) != tokens(exp):
            shutil.rmtree(d, ignore_errors=True)
            return {"ok": False, "why": f"test {i}: got {r.stdout[:80]!r} want {exp[:80]!r} {r.stderr[-160:]!r}"}
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "why": ""}


def main():
    jobs = json.load(sys.stdin)
    print(json.dumps([run_job(j, k) for k, j in enumerate(jobs)]))


if __name__ == "__main__":
    main()
