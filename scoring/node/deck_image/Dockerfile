# Sandbox for the v8 rendered slide-deck checks (skills._deck_exec): Marp Core renders each deck to HTML and headless
# Chromium lays it out at 1280x720 per slide. Jobs run with --network none, --read-only, a tmpfs work dir and a non-root
# user. Fonts are installed locally because the themes' Google Fonts imports cannot load without a network.
FROM node:20-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends chromium fonts-lato fonts-liberation fonts-dejavu-core \
        fonts-noto-color-emoji fonts-roboto-unhinted \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /deck
RUN npm install --omit=dev --no-audit --no-fund @marp-team/marp-core@4.3.1 && npm cache clean --force
COPY render.js /deck/render.js
ENV HOME=/tmp XDG_CONFIG_HOME=/tmp XDG_CACHE_HOME=/tmp NODE_PATH=/deck/node_modules
ENTRYPOINT ["node", "/deck/render.js"]
