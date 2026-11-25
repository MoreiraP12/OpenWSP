# OpenWSP project page

The static site for OpenWSP: Benchmarking Weight-Space Programming on Open Language Models. Everything in `public/`
is published as is to GitHub Pages by [`.github/workflows/pages.yml`](../.github/workflows/pages.yml), on every push
to `main` that changes it. The workflow replaces `__SITE__` with the site's address, so the same files work under
`moreirap12.github.io/<repository>/` or a custom domain.

| Path | Contents |
|---|---|
| `public/index.html` | The page. Written by hand, except the blocks between `<!-- gen:NAME -->` and `<!-- /gen:NAME -->`. |
| `public/data.js` | Leaderboard, task counts and the before-and-after examples (generated). |
| `public/media/` | OpenSCAD renders for the 3D-part examples (generated). |
| `public/paper.pdf` | The paper. |
| `public/llms.txt`, `public/sitemap.xml` | A plain-text summary for LLMs, and the sitemap for search engines (generated). |
| `public/og.png`, `public/favicon.svg`, `public/favicon-96.png`, `public/apple-touch-icon.png` | Link preview and icons (generated). |
| `public/27352a96413200da265f0119603de9a4.txt` | IndexNow key, used to tell Bing and other search engines that the page changed. |
| `build.py` | Regenerates all of the above. |

## Rebuilding

```bash
python3 website/build.py PATH/TO/wsp-bench --paper PATH/TO/main.pdf
```

`PATH/TO/wsp-bench` is the private development checkout: the examples come from its pass-1 transcripts and the
numbers from its `results/paper_v8.json`. OpenSCAD and Google Chrome must be installed. The examples use practice
prompts only: held-out prompts and answer keys are never read. Without `--paper`, the published PDF is kept.

## One-time setup

A repository admin turns on GitHub Pages: Settings → Pages → Build and deployment → Source: **GitHub Actions**. Then
re-run the "Project page" workflow, or push a change to `website/public/`.
