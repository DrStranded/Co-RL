# Co-RL project website

Static project page for **Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in
Multi-agent RL** ([arXiv:2608.17253](https://arxiv.org/abs/2608.17253)), served with GitHub
Pages from this `website` branch at:

**https://drstranded.github.io/Co-RL/**

The code release lives on the [`main`](https://github.com/DrStranded/Co-RL/tree/main) branch.
This branch is an orphan branch and contains only the website.

## Layout

- `index.html` — page structure and all static prose
- `static/css/site.css` — styling (no frameworks)
- `static/js/site.js` — interactions (vanilla JS, no dependencies)
- `static/data/content.js` — authors, rung definitions, BibTeX
- `static/data/results.js` — every displayed number, **generated from the paper's LaTeX
  tables**; do not edit numbers by hand
- `static/images/` — figures rasterized from the paper source

## Local preview

```bash
python3 -m http.server 8000
```

then open http://localhost:8000/ — do not open `index.html` via `file://`.

## Deployment

Settings → Pages → Source: *Deploy from a branch* → branch `website`, folder `/ (root)`.
All asset paths are relative, so the page works from the `/Co-RL/` project subpath.
When editing CSS/JS, bump the shared `?v=` query string in `index.html` (all four references).
