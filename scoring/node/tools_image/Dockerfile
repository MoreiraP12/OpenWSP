# Sandbox for the v7 skill checks that run foreign tools (video: ffmpeg/ffprobe; 3D: OpenSCAD; answer-set programs:
# clingo). Jobs run with --network none, --read-only, a tmpfs work dir and a non-root user (skills._tools_exec).
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg openscad python3 python3-numpy python3-pip \
    && pip3 install --break-system-packages --no-cache-dir clingo==5.7.1 && rm -rf /var/lib/apt/lists/*
ENV OPENSCADPATH=/tmp HOME=/tmp
