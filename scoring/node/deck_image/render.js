// Render Marp decks and measure the laid-out slides: stdin = JSON list of {"md": markdown}, stdout (last line) = JSON
// list, one result per deck: {"slides": n, "overflow": [bool], "title_lines": [int|null], "text": [str], "notes": [[str]]}
// or {"error": msg}. With an /out mount and argv[2] == "--shots", also writes /out/deck_<i>.png (all slides stacked).
// HTML in the markdown is disabled, so model text cannot add markup or scripts; the only script is the measurement below.
const { Marp } = require('@marp-team/marp-core');
const { execFileSync } = require('child_process');
const fs = require('fs');

const SHOTS = process.argv[2] === '--shots';
const GAP = 24;
const CHROME = ['--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--no-first-run',
  '--disable-crash-reporter', '--crash-dumps-dir=/tmp', '--user-data-dir=/tmp/chrome', '--hide-scrollbars',
  '--force-device-scale-factor=1', '--allow-file-access-from-files', '--virtual-time-budget=4000'];

// Runs in the page after layout: per slide, whether any content box leaves the 1280x720 slide (Marp clips it), how
// many lines the slide title wraps to, and the visible text.
const MEASURE = `
(() => {
  const res = [...document.querySelectorAll('div.marpit > section')].map((s) => {
    const sr = s.getBoundingClientRect();
    let overflow = s.scrollHeight > s.clientHeight + 1 || s.scrollWidth > s.clientWidth + 1;
    for (const e of s.querySelectorAll('*')) {
      const r = e.getBoundingClientRect();
      if (r.width > 0 && r.height > 0 && (r.bottom > sr.bottom + 1 || r.right > sr.right + 1 || r.top < sr.top - 1 || r.left < sr.left - 1)) { overflow = true; break; }
    }
    const h = s.querySelector('h1, h2');
    let lines = null;
    if (h && h.textContent.trim()) {
      const rg = document.createRange(); rg.selectNodeContents(h);
      lines = new Set([...rg.getClientRects()].filter((r) => r.width > 0).map((r) => Math.round(r.top))).size;
    }
    return { overflow, title_lines: lines, text: s.innerText };
  });
  const pre = document.createElement('pre'); pre.id = 'wsp-result';
  pre.textContent = btoa(unescape(encodeURIComponent(JSON.stringify(res))));
  document.body.appendChild(pre);
})();`;

const jobs = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = jobs.map((job, i) => {
  try {
    const marp = new Marp({ html: false, inlineSVG: false, math: false });
    const { html, css, comments } = marp.render(job.md || '');
    const page = `<!doctype html><html><head><meta charset="utf-8"><style>${css.replace(/@import\s+url\([^)]*\)[^;]*;/g, '')}</style>`
      + `<style>html,body{margin:0;padding:0;background:#888}div.marpit>section{margin:0 0 ${GAP}px 0}</style></head>`
      + `<body>${html}<script>${MEASURE}</script></body></html>`;
    const f = `/tmp/deck_${i}.html`;
    fs.writeFileSync(f, page);
    const dom = execFileSync('chromium', [...CHROME, '--dump-dom', `file://${f}`], { timeout: 60000, maxBuffer: 64 << 20 }).toString();
    const m = dom.match(/<pre id="wsp-result">([A-Za-z0-9+/=]*)<\/pre>/);
    if (!m) return { error: 'no measurement in the rendered page' };
    const slides = JSON.parse(Buffer.from(m[1], 'base64').toString('utf8'));
    if (SHOTS && slides.length) {
      const h = slides.length * (720 + GAP);
      execFileSync('chromium', [...CHROME, `--window-size=1280,${h}`, `--screenshot=/out/deck_${i}.png`, `file://${f}`],
        { timeout: 60000, stdio: 'ignore' });
    }
    fs.unlinkSync(f);
    return { slides: slides.length, overflow: slides.map((s) => s.overflow), title_lines: slides.map((s) => s.title_lines),
      text: slides.map((s) => s.text), notes: comments };
  } catch (e) {
    return { error: String(e).slice(0, 300) };
  }
});
console.log(JSON.stringify(out));
