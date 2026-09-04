# Slides

`glyph-progress-2026-09-04.pptx` — an eight-page progress deck: what an
instance is, the agent's protocol, the three-way result (skeleton ceiling /
A0′ / weights, overall and `tail`), the `seen_frac` band, engineering state,
and what is still open.

Every number in it comes from `docs/progress.md` and `docs/open_questions.md`;
nothing was estimated for the deck. Regenerate with:

```bash
cd docs/slides && npm install pptxgenjs && node build_deck.js
```

The generator is the source of truth — edit `build_deck.js`, not the `.pptx`.
