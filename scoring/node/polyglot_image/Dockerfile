# Sandbox for the v8 polyglot items (skills.run_prog): 35 languages' compilers and interpreters from Debian bookworm,
# Shakespeare (shakespearelang 1.0.0) and GolfScript (Darren Smith's Ruby interpreter, MIT, vendored here; sha256
# 84f932a6...8f38). Jobs run with --network none, a read-only root, a tmpfs /tmp and a non-root user (skills._prog_exec).
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates gcc libc6-dev default-jdk-headless rustc golang-go kotlin ruby lua5.4 perl \
        crystal libgc-dev libpcre3-dev libevent-dev nim gnat tcl valac libglib2.0-dev haxe pike8.0 gforth r-base-core \
        sqlite3 gawk octave swi-prolog-nox gnucobol3 algol68g gfortran fp-compiler yabasic ghc ocaml-nox erlang-nox sbcl \
        guile-3.0 polyml libpolyml-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages --no-cache-dir shakespearelang==1.0.0
COPY patch_shakespeare.py /opt/patch_shakespeare.py
RUN python3 /opt/patch_shakespeare.py
COPY golfscript.rb /opt/golfscript/golfscript.rb
COPY runner.py /opt/runner.py
ENV HOME=/tmp XDG_CACHE_HOME=/tmp/cache GOCACHE=/tmp/gocache GOPATH=/tmp/go GO111MODULE=off CRYSTAL_CACHE_DIR=/tmp/crystal R_LIBS_USER=/tmp/R
WORKDIR /tmp
ENTRYPOINT ["python3", "/opt/runner.py"]
