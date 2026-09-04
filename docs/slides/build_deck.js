const pptxgen = require("pptxgenjs");

// ---- tokens -------------------------------------------------------------
const INK = "12141C";        // deep ground (title / closing)
const PANEL = "1D2130";      // raised panel on the deep ground
const PAPER = "F6F7F9";      // cool off-white ground
const TINT = "EAECF1";       // card tint on paper
const TEXT = "15171E";
const MUTED = "5D616E";
const LIGHT = "E7E9EF";      // text on deep ground
const LIGHTMUTE = "9AA0B0";
const COOL = "3A4E9E";       // code / context / frontier
const CLAY = "C2603A";       // weights — the accent, spent sparingly

const DISPLAY = "Cambria";
const BODY = "Calibri";
const MONO = "Courier New";

const M = 0.7;
const W = 11.93;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Glyph";
pres.title = "Glyph Progress Report";

let pageNo = 0;
function page(bg) {
  pageNo += 1;
  const s = pres.addSlide();
  s.background = { color: bg };
  const onDark = bg === INK;
  s.addText(String(pageNo).padStart(2, "0") + " / 08", {
    isTextBox: true, x: 11.0, y: 6.92, w: 1.63, h: 0.3, align: "right",
    fontFace: MONO, fontSize: 10, color: onDark ? "5A6076" : "A6AAB6", margin: 0,
  });
  return s;
}

function eyebrow(s, txt, onDark) {
  s.addText(txt, {
    isTextBox: true, x: M, y: 0.42, w: W, h: 0.3,
    fontFace: MONO, fontSize: 11, charSpacing: 2,
    color: onDark ? LIGHTMUTE : COOL, margin: 0,
  });
}

function heading(s, txt, onDark, size) {
  s.addText(txt, {
    isTextBox: true, x: M, y: 0.78, w: W, h: 0.8,
    fontFace: DISPLAY, fontSize: size || 30, bold: true,
    color: onDark ? "FFFFFF" : TEXT, margin: 0, valign: "top",
  });
}

// a mono token chip — the deck's one repeated motif
function chip(s, txt, x, y, w, opts) {
  const o = opts || {};
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h: 0.34, rectRadius: 0.06,
    fill: { color: o.fill || TINT },
    line: { color: o.fill || TINT, width: 0.5 },
  });
  s.addText(txt, {
    isTextBox: true, x, y, w, h: 0.34, align: "center", valign: "middle",
    fontFace: MONO, fontSize: 11, color: o.color || COOL, margin: 0,
  });
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.03,
    fill: { color: fill || "FFFFFF" },
    line: { color: fill === PANEL ? "2A2F41" : "DDE0E7", width: 1 },
  });
}

// =========================================================================
// 1 — cover
// =========================================================================
{
  const s = page(INK);
  s.addText("T1  ·  WEIGHT-SPACE DELEGATION", {
    isTextBox: true, x: M, y: 0.95, w: W, h: 0.3,
    fontFace: MONO, fontSize: 12, charSpacing: 2.5, color: CLAY, margin: 0,
  });
  s.addText("Glyph", {
    isTextBox: true, x: M, y: 1.35, w: 8, h: 1.0,
    fontFace: DISPLAY, fontSize: 54, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText("A hidden-semantics DSL execution benchmark", {
    isTextBox: true, x: M, y: 2.32, w: 9.6, h: 0.5,
    fontFace: BODY, fontSize: 22, color: LIGHT, margin: 0,
  });
  s.addText("Does capability pay to sit in context, in code, or in a small model's weights — and does the agent know which? This repo is the ruler that makes the question measurable.", {
    isTextBox: true, x: M, y: 2.85, w: 10.4, h: 0.65,
    fontFace: BODY, fontSize: 14, color: LIGHTMUTE, lineSpacingMultiple: 1.2, margin: 0,
  });

  card(s, M, 3.65, W, 1.35, PANEL);
  s.addText("s1( s3( s0(u2, [v_a_b_a, v_d_n_c, v_g_b_m]) ), b0 )    →    v_b_e_o", {
    isTextBox: true, x: M + 0.35, y: 3.85, w: W - 0.7, h: 0.4,
    fontFace: MONO, fontSize: 15, color: "FFFFFF", margin: 0,
  });
  s.addText("Operator names are deliberately opaque — s0 / u2 / b0, never map / fold — so naming priors leak nothing.", {
    isTextBox: true, x: M + 0.35, y: 4.34, w: W - 0.7, h: 0.4,
    fontFace: BODY, fontSize: 12.5, color: LIGHTMUTE, margin: 0,
  });

  const facts = [
    ["Data layer frozen", "2026-08-31"],
    ["Test suite", "138 passed"],
    ["First three-way result", "in — see page 4"],
  ];
  facts.forEach((f, i) => {
    const x = M + i * 4.05;
    s.addText(f[0], {
      isTextBox: true, x, y: 5.55, w: 3.7, h: 0.3,
      fontFace: BODY, fontSize: 12, color: LIGHTMUTE, margin: 0,
    });
    s.addText(f[1], {
      isTextBox: true, x, y: 5.85, w: 3.7, h: 0.45,
      fontFace: MONO, fontSize: 17, color: "FFFFFF", margin: 0,
    });
  });
  s.addText("2026-09-04", {
    isTextBox: true, x: 9.0, y: 6.92, w: 1.9, h: 0.3, align: "right",
    fontFace: MONO, fontSize: 10, color: "5A6076", margin: 0,
  });
  s.addNotes("Glyph is task T1 of the Phase 1 plan: a hidden-semantics DSL execution benchmark for measuring whether capability belongs in context, in code, or in weights. The syntax is public; the semantics are private.");
}

// =========================================================================
// 2 — the two halves
// =========================================================================
{
  const s = page(PAPER);
  eyebrow(s, "INSTANCE", false);
  heading(s, "Syntax public, semantics private — split in two", false);

  const halves = [
    {
      x: M, chipTxt: "skeleton   s0 … s7", accent: COOL,
      what: "The semantics of the structural operators",
      from: "sampled from a finite combinator grammar",
      can: "Yes — finitely many rules, by construction",
      home: "code's home ground",
    },
    {
      x: 7.03, chipTxt: "table   u*  /  b*", accent: CLAY,
      what: "The semantics of the atomic operators",
      from: "digit embeddings + frozen random MLPs",
      can: "No — you would be transcribing weight matrices",
      home: "weights' home ground",
    },
  ];
  halves.forEach((h) => {
    card(s, h.x, 1.7, 5.6, 3.0);
    chip(s, h.chipTxt, h.x + 0.35, 1.95, 2.9, { fill: h.accent, color: "FFFFFF" });
    s.addText(h.what, {
      isTextBox: true, x: h.x + 0.35, y: 2.45, w: 4.9, h: 0.35,
      fontFace: BODY, fontSize: 15, bold: true, color: TEXT, margin: 0,
    });
    s.addText([
      { text: "Where it comes from:  ", options: { color: MUTED } },
      { text: h.from, options: { breakLine: true } },
      { text: "Can you write it down:  ", options: { color: MUTED } },
      { text: h.can },
    ], {
      isTextBox: true, x: h.x + 0.35, y: 2.85, w: 4.9, h: 1.15,
      fontFace: BODY, fontSize: 13.5, color: TEXT, lineSpacingMultiple: 1.25, margin: 0,
    });
    s.addText(h.home, {
      isTextBox: true, x: h.x + 0.35, y: 4.1, w: 4.9, h: 0.4,
      fontFace: DISPLAY, fontSize: 18, bold: true, color: h.accent, margin: 0,
    });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 4.95, w: W, h: 1.65, rectRadius: 0.03,
    fill: { color: TINT }, line: { color: TINT, width: 1 },
  });
  s.addText("π  =  (1 − a_tab) / ( (1 − a_tab) + (1 − a_skel) )", {
    isTextBox: true, x: M + 0.4, y: 5.15, w: 6.0, h: 0.4,
    fontFace: MONO, fontSize: 15, color: TEXT, margin: 0,
  });
  s.addText("The skeleton's share of the difficulty. π → 1 means code should win; π → 0 means weights should win.", {
    isTextBox: true, x: M + 0.4, y: 5.6, w: 6.0, h: 0.8,
    fontFace: BODY, fontSize: 13.5, color: MUTED, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addText([
    { text: "π is measured per instance, never configured.", options: { bold: true, breakLine: true } },
    { text: "A preset is only a sampler: 7 of 20 pi_mid seeds land inside pi_high's measured range, so phase-diagram axes use measured_pi(), never a preset name." },
  ], {
    isTextBox: true, x: 7.1, y: 5.15, w: 5.4, h: 1.3,
    fontFace: BODY, fontSize: 13, color: TEXT, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addNotes("π's numerator is L_skel. Flip it and the phase diagram runs backwards while the curves still plot — one of the two things here that fail silently. The other is the trivial skeleton, which must still invoke the tables.");
}

// =========================================================================
// 3 — the agent's situation
// =========================================================================
{
  const s = page(PAPER);
  eyebrow(s, "PROTOCOL", false);
  heading(s, "The agent can buy, but never buy enough", false);

  const steps = [
    ["Free of charge", "The syntax spec and 30 demos. That is all it starts with."],
    ["Metered queries", "It may ask P about any well-formed expression, and every query costs budget — syntax errors included, so probing the grammar is not free. The ledger is the only way to advance the clock."],
    ["Seal, then answer", "It seals an artifact that must answer 10,000 unseen expressions, with no further access to P."],
  ];
  steps.forEach((st, i) => {
    const y = 1.72 + i * 1.42;
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y, w: 0.42, h: 0.42, rectRadius: 0.08,
      fill: { color: COOL }, line: { color: COOL, width: 0.5 },
    });
    s.addText(String(i + 1), {
      isTextBox: true, x: M, y, w: 0.42, h: 0.42, align: "center", valign: "middle",
      fontFace: MONO, fontSize: 13, color: "FFFFFF", margin: 0,
    });
    s.addText(st[0], {
      isTextBox: true, x: M + 0.62, y: y - 0.03, w: 5.7, h: 0.4,
      fontFace: BODY, fontSize: 16, bold: true, color: TEXT, margin: 0,
    });
    s.addText(st[1], {
      isTextBox: true, x: M + 0.62, y: y + 0.36, w: 5.7, h: 0.95,
      fontFace: BODY, fontSize: 13, color: MUTED, lineSpacingMultiple: 1.2, margin: 0,
    });
  });

  const tiles = [
    ["Q ≈ 2000", "queries a run can afford"],
    ["|V| = 4913", "value space 17³ · unary half-covered at best"],
    ["≈ 24 M", "binary pairs · essentially untouchable"],
    ["10,000", "expressions to answer after the seal"],
  ];
  tiles.forEach((t, i) => {
    const x = 7.6 + (i % 2) * 2.63;
    const y = 1.72 + Math.floor(i / 2) * 1.6;
    card(s, x, y, 2.4, 1.4);
    s.addText(t[0], {
      isTextBox: true, x: x + 0.18, y: y + 0.18, w: 2.04, h: 0.45,
      fontFace: MONO, fontSize: 19, color: i === 2 ? CLAY : TEXT, margin: 0,
    });
    s.addText(t[1], {
      isTextBox: true, x: x + 0.18, y: y + 0.66, w: 2.04, h: 0.62,
      fontFace: BODY, fontSize: 11.5, color: MUTED, lineSpacingMultiple: 1.15, margin: 0,
    });
  });

  s.addText([
    { text: "That gap is why fitting can beat looking things up.", options: { bold: true } },
    { text: "  Splits: iid / comp / depth are fixed at generation; tail is derived per run — the items whose entries this run never bought, which doubles as a read on how smart its query strategy was." },
  ], {
    isTextBox: true, x: M, y: 6.05, w: W, h: 0.7,
    fontFace: BODY, fontSize: 13, color: TEXT, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addNotes("Q ≈ 2000 against |V| = 4913 means the unary tables are at best half covered and the binary tables essentially not at all. There is no floor split: the test set is fully solvable and the ceiling is a clean 100%.");
}

// =========================================================================
// 4 — the headline result
// =========================================================================
{
  const s = page(INK);
  eyebrow(s, "RESULT  ·  pi_mid / seed 1001 / one stratified 500-item subset", true);
  heading(s, "The frontier model loses to a 1.7B student", true);

  const cats = ["skeleton ceiling", "A0′ frontier", "weights 1.7B"];
  s.addChart(pres.ChartType.bar, [
    { name: "overall (all 500 items)", labels: cats, values: [0.248, 0.258, 0.498] },
    { name: "tail (items needing extrapolation)", labels: cats, values: [0.0, 0.016, 0.328] },
  ], {
    x: 0.5, y: 1.7, w: 7.7, h: 4.5,
    barDir: "col", barGapWidthPct: 60, barGrouping: "clustered",
    chartColors: [COOL, CLAY],
    chartArea: { fill: { color: INK } },
    plotArea: { fill: { color: INK } },
    valAxisMinVal: 0, valAxisMaxVal: 0.6, valAxisMajorUnit: 0.2,
    valAxisLabelColor: LIGHTMUTE, valAxisLabelFontFace: MONO, valAxisLabelFontSize: 10,
    catAxisLabelColor: LIGHT, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 11,
    valGridLine: { color: "2C3142", size: 1 },
    catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: LIGHT,
    dataLabelFontFace: MONO, dataLabelFontSize: 10, dataLabelFormatCode: "0.000",
    showLegend: true, legendPos: "t", legendColor: LIGHT, legendFontFace: BODY, legendFontSize: 11,
  });
  s.addText("skeleton ceiling: every structural rule, not one table entry  ·  A0′: 2551 of 4913 entries in context, unlimited thinking  ·  weights: 1.7B, trained on 491 entries", {
    isTextBox: true, x: 0.62, y: 6.34, w: 7.5, h: 0.5,
    fontFace: BODY, fontSize: 10.5, color: LIGHTMUTE, lineSpacingMultiple: 1.15, margin: 0,
  });

  s.addText("The frontier was handed five times the table the student had, is hundreds of times larger, and thought without limit — and scores 24 points lower.", {
    isTextBox: true, x: 8.45, y: 1.85, w: 4.2, h: 1.1,
    fontFace: BODY, fontSize: 14, color: LIGHT, lineSpacingMultiple: 1.25, margin: 0,
  });
  card(s, 8.45, 3.1, 4.2, 1.7, PANEL);
  s.addText("20×", {
    isTextBox: true, x: 8.75, y: 3.25, w: 3.6, h: 0.7,
    fontFace: DISPLAY, fontSize: 40, bold: true, color: CLAY, margin: 0,
  });
  s.addText("On items needing entries never supplied — 374 against 372, nearly the same items — the 1.7B student beats the frontier twentyfold.", {
    isTextBox: true, x: 8.75, y: 3.95, w: 3.6, h: 0.8,
    fontFace: BODY, fontSize: 12, color: LIGHTMUTE, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addText("The skeleton ceiling on tail is 0.000 — nothing in there is answerable from structure, so every point is table knowledge.", {
    isTextBox: true, x: 8.45, y: 4.95, w: 4.2, h: 1.0,
    fontFace: BODY, fontSize: 12, color: LIGHTMUTE, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addNotes("The tail column was only added on 2026-09-03. Before it, the overall scores still allowed the reading that A0' had merely done worse on the easy items. The tail column removes that reading.");
}

// =========================================================================
// 5 — retrieval vs extrapolation
// =========================================================================
{
  const s = page(PAPER);
  eyebrow(s, "WHY", false);
  heading(s, "Not a shortfall — two capabilities, cleanly split", false);

  const stats = [
    { x: M, v: "0.976", n: "126 items", d: "needed entries were in the evidence", c: COOL, k: "Retrieval: saturated" },
    { x: 7.03, v: "0.016", n: "374 items", d: "needed entries were not", c: CLAY, k: "Extrapolation: zero" },
  ];
  stats.forEach((t) => {
    card(s, t.x, 1.7, 5.6, 2.05);
    s.addText(t.k, {
      isTextBox: true, x: t.x + 0.35, y: 1.9, w: 5.0, h: 0.35,
      fontFace: MONO, fontSize: 12, color: t.c, margin: 0,
    });
    s.addText(t.v, {
      isTextBox: true, x: t.x + 0.35, y: 2.25, w: 2.6, h: 0.9,
      fontFace: DISPLAY, fontSize: 48, bold: true, color: t.c, margin: 0,
    });
    s.addText([
      { text: t.n, options: { bold: true, breakLine: true, color: TEXT } },
      { text: t.d, options: { color: MUTED } },
    ], {
      isTextBox: true, x: t.x + 2.9, y: 2.4, w: 2.45, h: 1.0,
      fontFace: BODY, fontSize: 13, lineSpacingMultiple: 1.2, margin: 0,
    });
  });

  s.addText("A0′'s evidence was deliberately more generous than any arm could buy: 2000 queries revealing 2551 of 4913 unary entries, 70k tokens of context, unlimited thinking, claude-opus-4-8, $19.94 in total. It uses what it is told almost perfectly and infers almost nothing.", {
    isTextBox: true, x: M, y: 3.95, w: W, h: 0.75,
    fontFace: BODY, fontSize: 13.5, color: TEXT, lineSpacingMultiple: 1.25, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 4.8, w: W, h: 1.15, rectRadius: 0.03,
    fill: { color: TINT }, line: { color: TINT, width: 1 },
  });
  s.addText([
    { text: "H1 holds in its strong form: ", options: { bold: true, color: COOL } },
    { text: "the weights advantage is learning-algorithmic, not economic. It is not that the frontier cannot afford to re-read the evidence per query — re-reading does not help. Some structure gradient descent reaches and in-context learning does not." },
  ], {
    isTextBox: true, x: M + 0.35, y: 4.95, w: W - 0.7, h: 0.85,
    fontFace: BODY, fontSize: 14, color: TEXT, lineSpacingMultiple: 1.25, margin: 0,
  });
  s.addText("Two details worth keeping: on depth — the split with the most lookups per item — A0′'s headroom is −0.018, below what knowing zero table entries gets you. And it did not choose its own evidence (page 8).", {
    isTextBox: true, x: M, y: 6.1, w: W, h: 0.6,
    fontFace: BODY, fontSize: 12, color: MUTED, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addNotes("This also confirms the data layer's foundation. The pi-to-zero end assumes the tables are learnable but not in-context-extractable; A0' failing from partial observation is the direct test of that, and it passed, so the freeze does not have to be reopened.");
}

// =========================================================================
// 6 — the seen_frac curve
// =========================================================================
{
  const s = page(PAPER);
  eyebrow(s, "BAND  ·  scripts/table_ceiling.py", false);
  heading(s, "Supervision decays faster than extrapolation", false);

  const labels = ["2%", "5%", "10%", "25%", "50%", "90%"];
  s.addChart(pres.ChartType.line, [
    { name: "item score (all items)", labels, values: [0.360, 0.454, 0.498, 0.582, 0.724, 0.880] },
    { name: "tail (items needing extrapolation)", labels, values: [0.147, 0.273, 0.328, 0.430, 0.590, 0.709] },
  ], {
    x: 0.5, y: 1.7, w: 7.5, h: 4.35,
    chartColors: [COOL, CLAY], lineSize: 2.5, lineSmooth: false,
    lineDataSymbol: "circle", lineDataSymbolSize: 7,
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25,
    valAxisLabelColor: MUTED, valAxisLabelFontFace: MONO, valAxisLabelFontSize: 10,
    catAxisLabelColor: MUTED, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 10,
    valGridLine: { color: "DFE2E9", size: 1 },
    catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendColor: TEXT, legendFontFace: BODY, legendFontSize: 11,
  });
  s.addText("Share of table entries seen in training (non-linear scale, six measured points) — 98 / 246 / 491 / 1228 / 2456 / 4422 of 4913. The measurable band an arm operates in is [0.248, 0.498].", {
    isTextBox: true, x: 0.6, y: 6.08, w: 7.5, h: 0.6,
    fontFace: BODY, fontSize: 10.5, color: MUTED, lineSpacingMultiple: 1.15, margin: 0,
  });

  const notes = [
    ["2% still extrapolates", "98 entries — two percent of the table — still score 0.147 on items never shown. The frontier, holding 2551 of them in context, scores 0.016."],
    ["Capacity is never the constraint", "fit is 1.000 at every point: whatever it has seen, it retains perfectly. The limit is generalisation, all the way down."],
    ["binary is inert to this knob", "24M entries means every setting is \"almost nothing seen\", so the curve's shape is set by unary alone."],
    ["Noise measured before reading it", "Identical config, rerun: ±0.04 on reach. Do not read gaps that size as signal."],
  ];
  notes.forEach((n, i) => {
    const y = 1.72 + i * 1.22;
    s.addText(n[0], {
      isTextBox: true, x: 8.4, y, w: 4.23, h: 0.32,
      fontFace: BODY, fontSize: 13.5, bold: true, color: i === 0 ? CLAY : TEXT, margin: 0,
    });
    s.addText(n[1], {
      isTextBox: true, x: 8.4, y: y + 0.32, w: 4.23, h: 0.85,
      fontFace: BODY, fontSize: 11.5, color: MUTED, lineSpacingMultiple: 1.2, margin: 0,
    });
  });
  s.addNotes("25 points of band is wide against the 1-3 point differences the arms are expected to be separated by. depth collapses fastest, 0.817 to 0.217, because it has the most lookups per item.");
}

// =========================================================================
// 7 — engineering state
// =========================================================================
{
  const s = page(PAPER);
  eyebrow(s, "STATE  ·  data layer frozen 2026-08-31", false);
  heading(s, "The foundation is sealed, evaluation still moves", false);

  const blocks = [
    ["138 passed", "Full suite green, test_infer.py included. Green before every commit."],
    ["glyph/data/ isolated", "Seven modules moved into the data layer, dependencies one-way; 10 boundary tests hold the line with an AST scan, so T2 can lift the generator out whole."],
    ["the ledger is the only entry", "Anything that consumes compute goes through charge. One bypass and the crossover figure is meaningless."],
    ["the tool layer", "Declaration separated from allocation, one answering path; the agent's self-estimate is measured rather than hidden."],
    ["ceilings ship with every score", "Both ceilings plus headroom on every scored subset — tail finally has one of its own."],
    ["self-checks 1 2 3 5 6 pass", "#5 capacity: fit 1.000, reach 0.710 — the failure it exists to catch did not happen. #4 hiddenness must be rerun."],
  ];
  blocks.forEach((b, i) => {
    const x = M + (i % 3) * 4.08;
    const y = 1.72 + Math.floor(i / 3) * 2.05;
    card(s, x, y, 3.77, 1.85);
    s.addText(b[0], {
      isTextBox: true, x: x + 0.25, y: y + 0.2, w: 3.3, h: 0.4,
      fontFace: MONO, fontSize: 12.5, color: COOL, margin: 0,
    });
    s.addText(b[1], {
      isTextBox: true, x: x + 0.25, y: y + 0.62, w: 3.3, h: 1.1,
      fontFace: BODY, fontSize: 11.5, color: MUTED, lineSpacingMultiple: 1.2, margin: 0,
    });
  });

  s.addText("Settled, not reopened:", {
    isTextBox: true, x: M, y: 6.0, w: 2.35, h: 0.34,
    fontFace: BODY, fontSize: 12, color: TEXT, valign: "middle", margin: 0,
  });
  ["value_form = letter_sep", "|V| = 17³", "coupling = 0.25", "no floor split"]
    .forEach((t, i) => chip(s, t, 3.13 + i * 2.4, 6.0, 2.3));
  s.addNotes("HANDOFF.md still says 36 tests; the current figure is 138. Self-check #4 must be rerun because every preset changed — the last full run took about 110 minutes and costs teacher calls.");
}

// =========================================================================
// 8 — open and next
// =========================================================================
{
  const s = page(INK);
  eyebrow(s, "OPEN  ·  docs/open_questions.md  ·  issues #1–#29", true);
  heading(s, "What is open, and the order of what comes next", true);

  const open = [
    ["#3", "π is still measured on iid only", "On pi_low, iid-only and stratified π differ by as much as 0.166. Changing the sample is close to changing the baselines, and that invalidates every π measured so far."],
    ["#20", "A0′ did not choose its own evidence", "The 2000 queries were drawn by a script; a real agent probes adaptively. The opening is narrow — 52% of the table was handed over, extrapolation still 1.6% — but it is real."],
    ["#23", "E0's four arms, invalidated twice", "Once by the preset change, once by the held-pair change; at n = 200 the differences were never resolvable anyway. Do not rerun until the data layer is frozen."],
    ["#27 #28", "Two things not written", "Prompt caching: in the plan, never implemented, zero cache hits across eight runs. T2: zero lines, and Glyph cannot carry amortisation — preparing costs ~100× serving."],
  ];
  open.forEach((o, i) => {
    const y = 1.62 + i * 1.3;
    s.addText(o[0], {
      isTextBox: true, x: M, y, w: 0.9, h: 0.3,
      fontFace: MONO, fontSize: 12, color: CLAY, margin: 0,
    });
    s.addText(o[1], {
      isTextBox: true, x: M + 0.95, y: y - 0.04, w: 5.75, h: 0.34,
      fontFace: BODY, fontSize: 14, bold: true, color: "FFFFFF", margin: 0,
    });
    s.addText(o[2], {
      isTextBox: true, x: M + 0.95, y: y + 0.31, w: 5.75, h: 0.72,
      fontFace: BODY, fontSize: 11.5, color: LIGHTMUTE, lineSpacingMultiple: 1.2, margin: 0,
    });
  });

  card(s, 7.7, 1.62, 4.93, 4.6, PANEL);
  s.addText("The order from here", {
    isTextBox: true, x: 8.0, y: 1.85, w: 4.3, h: 0.4,
    fontFace: DISPLAY, fontSize: 18, bold: true, color: "FFFFFF", margin: 0,
  });
  const next = [
    ["Rerun self-check #4", "Hiddenness. Every preset changed; pi_high is a different task now. ~110 min, costs teacher calls."],
    ["Settle #9 / #4", "Evaluation size and how scores are reported. Arms differ by 1–3 points; too small an n resolves nothing."],
    ["E0's four arms", "The minimum closed loop: agent queries → seal → evaluate → one score. A2 / A4 / A6, one pass not three."],
    ["#26 the phase diagram", "Axis is measured_pi(). The worker's 5400 s timeout biases toward agents doing less."],
  ];
  next.forEach((n, i) => {
    const y = 2.42 + i * 0.95;
    s.addText(String(i + 1), {
      isTextBox: true, x: 8.0, y, w: 0.3, h: 0.3,
      fontFace: MONO, fontSize: 12, color: CLAY, margin: 0,
    });
    s.addText(n[0], {
      isTextBox: true, x: 8.35, y: y - 0.03, w: 3.95, h: 0.3,
      fontFace: BODY, fontSize: 12.5, bold: true, color: "FFFFFF", margin: 0,
    });
    s.addText(n[1], {
      isTextBox: true, x: 8.35, y: y + 0.28, w: 3.95, h: 0.62,
      fontFace: BODY, fontSize: 10.5, color: LIGHTMUTE, lineSpacingMultiple: 1.15, margin: 0,
    });
  });
  s.addText("Arm results are not published yet: the earlier sweep is invalidated by the data-layer changes, and two harness asymmetries are still open decisions.", {
    isTextBox: true, x: M, y: 6.55, w: 6.7, h: 0.4,
    fontFace: BODY, fontSize: 11, color: "6E748A", margin: 0,
  });
  s.addNotes("The order comes from the Order section of open_questions.md. E0 has been invalidated twice already; any of #1, #5, #6, #7, #8 moving again would make it three times.");
}

pres.writeFile({ fileName: "glyph-progress-2026-09-04.pptx" })
  .then((f) => console.log("wrote", f));
