# Project page (`site/`)

The GitHub Pages site for Glyph. Plain HTML/CSS/JS — no build step, no Jekyll,
no Node dependency. What is in this directory is exactly what gets served.

```
site/
├── index.html            the project page
├── 404.html
├── .nojekyll             serve files starting with _ verbatim
└── assets/
    ├── css/style.css         design tokens + layout (white, light only)
    ├── js/main.js            copy buttons, nav highlighting
    ├── js/pipeline.js        the step-through figures in "Data generation"
    ├── js/example-data.js    the running example those figures show (generated)
    └── img/                  figures, teaser, og.png
```

## Preview locally

```bash
python -m http.server 8000 --directory site
# then open http://localhost:8000
```

Edits are picked up on reload; there is nothing to rebuild.

## Deploy

`.github/workflows/pages.yml` uploads this directory and deploys it on every
push to `main` that touches `site/` (or via **Actions → Deploy project page →
Run workflow**).

**One-time setup, in the repo settings:** Settings → Pages → *Build and
deployment* → Source → **GitHub Actions**. Until that is set, the workflow run
fails at the deploy step.

The site then lives at <https://ash-ding.github.io/Glyph/>.

## Editing

- **Content** — all prose is in `index.html`, in commented sections
  (`<!-- ===== ABSTRACT ===== -->`). Points still to be decided are marked
  `TODO`: the paper link, the author list, the phase-diagram figure, the
  results table, and the BibTeX entry.
- **Look** — every colour is a token at the top of `style.css`, and each one
  means something: blue is the skeleton (structural operators), green is the
  tables (atomic operators), amber is the data drawn from P (held pairs,
  splits, sampling knobs). Each hue is a pair, a soft fill and a dark ink of
  the same hue, so coloured chips keep their contrast. Body text is a serif
  (Iowan Old Style where installed, else Source Serif 4), headings and figures
  a sans (Inter Tight). Change a token, not a rule.
- **Figures** in "Data generation" play once when scrolled into view (never
  under `prefers-reduced-motion`) and can be stepped, replayed or skipped.
- **Asset stamps** — every stylesheet, script and image the pages load is
  referenced as `asset?v=<content hash>`. Pages caches files for ten minutes,
  so without the stamp a browser can pair a fresh `index.html` with a stale
  stylesheet. After changing anything under `assets/`, run
  `python tools/stamp_site_assets.py`; `tests/test_site_asset_versions.py`
  fails until you do.
- **The running example** — `assets/js/example-data.js` is exported from the
  generator, so every number in the "Data generation" walkthrough is real.
  Regenerate it after any change to the data layer:

  ```bash
  PYTHONPATH=src python scripts/export_site_example.py > site/assets/js/example-data.js
  ```

  The script asserts that its re-derived sampling decisions and evaluation
  trace agree with the real code paths, and fails if they do not.
- **Figures** — drop files in `assets/img/` and swap the `.placeholder` div for
  an `<img>`. SVG is preferred so plots stay sharp and can pick up the theme.
- **Social preview** — add `assets/img/og.png` (1200×630) and uncomment the
  `og:url` / `og:image` tags in the `<head>`.

## Custom domain

Put the hostname in a `site/CNAME` file (one line, no scheme) and set it under
Settings → Pages. The `404.html` links assume the project-page path prefix
`/Glyph/` (case-sensitive); on a custom domain at the apex, change those two paths to `/`.
