# Project page (`site/`)

The GitHub Pages site for Glyph. Plain HTML/CSS/JS — no build step, no Jekyll,
no Node dependency. What is in this directory is exactly what gets served.

```
site/
├── index.html            the project page
├── 404.html
├── .nojekyll             serve files starting with _ verbatim
└── assets/
    ├── css/style.css     design tokens + layout (light/dark)
    ├── js/main.js        theme toggle, copy buttons, nav highlighting
    └── img/              figures, teaser, og.png
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

The site then lives at <https://ash-ding.github.io/glyph/>.

## Editing

- **Content** — all prose is in `index.html`, in commented sections
  (`<!-- ===== ABSTRACT ===== -->`). Points still to be decided are marked
  `TODO`: the paper link, the author list, the phase-diagram figure, the
  results table, and the BibTeX entry.
- **Look** — every colour is a token at the top of `style.css`, defined three
  times (light, `prefers-color-scheme: dark`, explicit `[data-theme="dark"]`).
  Change a token, not a rule.
- **Figures** — drop files in `assets/img/` and swap the `.placeholder` div for
  an `<img>`. SVG is preferred so plots stay sharp and can pick up the theme.
- **Social preview** — add `assets/img/og.png` (1200×630) and uncomment the
  `og:url` / `og:image` tags in the `<head>`.

## Custom domain

Put the hostname in a `site/CNAME` file (one line, no scheme) and set it under
Settings → Pages. The `404.html` links assume the project-page path prefix
`/glyph/`; on a custom domain at the apex, change those two paths to `/`.
