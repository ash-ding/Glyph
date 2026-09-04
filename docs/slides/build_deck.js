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
const GREY = "8A8E9B";       // the baseline that knows no table

const DISPLAY = "Cambria";
const BODY = "Calibri";
const MONO = "Courier New";

const M = 0.7;
const W = 11.93;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Glyph";
pres.title = "Glyph 进展汇报";

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
  s.addText("隐藏语义 DSL 执行基准 —— 项目进展", {
    isTextBox: true, x: M, y: 2.32, w: 9, h: 0.5,
    fontFace: BODY, fontSize: 22, color: LIGHT, margin: 0,
  });
  s.addText("能力该放进 context、code，还是一个小模型的 weights？这个仓库是把问题变成可测量的那把尺。", {
    isTextBox: true, x: M, y: 2.88, w: 9.6, h: 0.4,
    fontFace: BODY, fontSize: 14, color: LIGHTMUTE, margin: 0,
  });

  card(s, M, 3.65, W, 1.35, PANEL);
  s.addText("s1( s3( s0(u2, [v_a_b_a, v_d_n_c, v_g_b_m]) ), b0 )    →    v_b_e_o", {
    isTextBox: true, x: M + 0.35, y: 3.85, w: W - 0.7, h: 0.4,
    fontFace: MONO, fontSize: 15, color: "FFFFFF", margin: 0,
  });
  s.addText("算子名字是刻意不可读的：只有 s0 / u2 / b0，永远不会是 map / fold —— 命名先验不许泄漏任何语义。", {
    isTextBox: true, x: M + 0.35, y: 4.34, w: W - 0.7, h: 0.4,
    fontFace: BODY, fontSize: 12.5, color: LIGHTMUTE, margin: 0,
  });

  const facts = [
    ["数据层已冻结", "2026-08-31"],
    ["测试全绿", "138 passed"],
    ["首个三方结果", "已出，见第 4 页"],
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
  s.addNotes("Glyph 是 Phase 1 的 T1 任务：一个隐藏语义的 DSL 执行基准，用来测量能力应该放在 context / code / weights 哪一层。语法公开、语义私有。");
}

// =========================================================================
// 2 — the two halves
// =========================================================================
{
  const s = page(PAPER);
  eyebrow(s, "INSTANCE", false);
  heading(s, "语法公开，语义私有 —— 而私有的部分正好分成两半", false);

  const halves = [
    {
      x: M, chipTxt: "skeleton   s0 … s7", accent: COOL,
      what: "结构算子的语义",
      from: "从一个有限组合子文法里抽出来",
      can: "能 —— 有限条规则，按构造就是可枚举的",
      home: "code 的主场",
    },
    {
      x: 7.03, chipTxt: "table   u*  /  b*", accent: CLAY,
      what: "原子算子的语义",
      from: "digit embedding + 冻结的随机 MLP",
      can: "不能 —— 写下来等于在抄权重矩阵",
      home: "weights 的主场",
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
      { text: "来自：", options: { color: MUTED } }, { text: h.from, options: { breakLine: true } },
      { text: "能不能写下来：", options: { color: MUTED } }, { text: h.can },
    ], {
      isTextBox: true, x: h.x + 0.35, y: 2.85, w: 4.9, h: 1.1,
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
  s.addText("难度里属于 skeleton 的那一半。π→1 该 code 赢，π→0 该 weights 赢。", {
    isTextBox: true, x: M + 0.4, y: 5.6, w: 6.0, h: 0.7,
    fontFace: BODY, fontSize: 13.5, color: MUTED, margin: 0,
  });
  s.addText([
    { text: "π 是每个实例测出来的，不是配置出来的。", options: { bold: true, breakLine: true } },
    { text: "preset 只是采样器：20 个 pi_mid 种子里有 7 个落在 pi_high 的区间内，所以相图的横轴用 measured_pi()，永远不用 preset 名字。" },
  ], {
    isTextBox: true, x: 7.1, y: 5.15, w: 5.4, h: 1.25,
    fontFace: BODY, fontSize: 13, color: TEXT, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addNotes("π 的分子是 L_skel。翻过来的话相图会整个反向而曲线照样画得出来 —— 这是会静默失败的两件事之一。另一件是 trivial skeleton 必须仍然调用表。");
}

// =========================================================================
// 3 — the agent's situation
// =========================================================================
{
  const s = page(PAPER);
  eyebrow(s, "PROTOCOL", false);
  heading(s, "Agent 的处境：可以买，但买不完", false);

  const steps = [
    ["免费拿到", "语法 spec + 30 个 demo。仅此而已。"],
    ["计费查询", "可以拿任意合法表达式去问 P，每次都扣预算 —— 语法错误也扣，探文法不是免费的。ledger 是唯一能推进时钟的入口。"],
    ["封存作答", "交出一个 artifact，回答 10,000 道没见过的题，此后不再接触 P。"],
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
    ["Q ≈ 2000", "一次运行买得起的查询数"],
    ["|V| = 4913", "值空间 17³ · unary 表最多买到一半"],
    ["≈ 24 M", "binary 表的条目数 · 基本买不到"],
    ["10,000", "封存后要答的题"],
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
      isTextBox: true, x: x + 0.18, y: y + 0.66, w: 2.04, h: 0.6,
      fontFace: BODY, fontSize: 11.5, color: MUTED, lineSpacingMultiple: 1.15, margin: 0,
    });
  });

  s.addText([
    { text: "这个缺口就是「拟合能赢查表」的全部理由。", options: { bold: true } },
    { text: "  分片：iid / comp / depth 在生成时固定；tail 每次运行现推 —— 这个 run 从没买过其表项的那些题，它同时也是对查询策略聪不聪明的读数。" },
  ], {
    isTextBox: true, x: M, y: 6.05, w: W, h: 0.7,
    fontFace: BODY, fontSize: 13, color: TEXT, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addNotes("Q≈2000 对 |V|=4913：unary 表最多覆盖一半，binary 表 24M 条基本为零。没有 floor 分片，测试集完全可解，ceiling 是干净的 100%。");
}

// =========================================================================
// 4 — the headline result
// =========================================================================
{
  const s = page(INK);
  eyebrow(s, "RESULT  ·  pi_mid / seed 1001 / 同一批分层抽样 500 题", true);
  heading(s, "三方对照：前沿模型输给了一个 1.7B 学生", true);

  const cats = ["skeleton ceiling", "A0′ 前沿模型", "weights 1.7B 学生"];
  s.addChart(pres.ChartType.bar, [
    { name: "overall（全部 500 题）", labels: cats, values: [0.248, 0.258, 0.498] },
    { name: "tail（必须外推的题）", labels: cats, values: [0.0, 0.016, 0.328] },
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

  s.addText("skeleton ceiling：完全不知道任何一条表项，只知道全部结构规则  ·  A0′：2551 条表项在 context 里，思考不限  ·  weights：1.7B，训练时见过 491 条", {
    isTextBox: true, x: 0.62, y: 6.38, w: 7.5, h: 0.5,
    fontFace: BODY, fontSize: 10.5, color: LIGHTMUTE, lineSpacingMultiple: 1.15, margin: 0,
  });
  s.addText("前沿模型被交到手里的表项是学生的 5 倍，参数是它的几百倍，思考不限时长 —— 总分低 24 个点。", {
    isTextBox: true, x: 8.45, y: 1.85, w: 4.2, h: 1.0,
    fontFace: BODY, fontSize: 14, color: LIGHT, lineSpacingMultiple: 1.25, margin: 0,
  });
  card(s, 8.45, 3.0, 4.2, 1.75, PANEL);
  s.addText("20×", {
    isTextBox: true, x: 8.75, y: 3.15, w: 3.6, h: 0.7,
    fontFace: DISPLAY, fontSize: 40, bold: true, color: CLAY, margin: 0,
  });
  s.addText("在必须外推到「从没被给过的表项」的那批题上（374 对 372 题，两边几乎是同一批），1.7B 学生比前沿模型高二十倍。", {
    isTextBox: true, x: 8.75, y: 3.85, w: 3.6, h: 0.85,
    fontFace: BODY, fontSize: 12, color: LIGHTMUTE, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addText("tail 上 skeleton ceiling 是 0.000 —— 那批题没有一道能靠结构答出来，所以每一分都是表知识。", {
    isTextBox: true, x: 8.45, y: 4.95, w: 4.2, h: 1.0,
    fontFace: BODY, fontSize: 12, color: LIGHTMUTE, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addNotes("tail 这一列是 2026-09-03 才补上的，补上之前只有 overall，还可以辩解成 A0′ 只是在简单题上做得差。补上之后这个读法就没有了。");
}

// =========================================================================
// 5 — retrieval vs extrapolation
// =========================================================================
{
  const s = page(PAPER);
  eyebrow(s, "WHY", false);
  heading(s, "不是能力不够 —— 是两种能力被干净地切开了", false);

  const stats = [
    { x: M, v: "0.976", n: "126 题", d: "所需表项就在给它的证据里", c: COOL, k: "检索：饱和" },
    { x: 7.03, v: "0.016", n: "374 题", d: "所需表项不在证据里", c: CLAY, k: "外推：为零" },
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
      isTextBox: true, x: t.x + 2.9, y: 2.45, w: 2.45, h: 0.9,
      fontFace: BODY, fontSize: 13, lineSpacingMultiple: 1.2, margin: 0,
    });
  });

  s.addText("A0′ 的证据是刻意给到超过任何 arm 买得起的量：2000 次查询揭示 4913 条 unary 里的 2551 条、70k tokens 的上下文、不限思考、claude-opus-4-8、$19.94。它把给它的东西用得几乎完美，自己推不出任何东西。", {
    isTextBox: true, x: M, y: 3.95, w: W, h: 0.75,
    fontFace: BODY, fontSize: 13.5, color: TEXT, lineSpacingMultiple: 1.25, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 4.8, w: W, h: 1.15, rectRadius: 0.03,
    fill: { color: TINT }, line: { color: TINT, width: 1 },
  });
  s.addText([
    { text: "H1 的强形式成立：", options: { bold: true, color: COOL } },
    { text: "weights 的优势是 learning-algorithmic，不是经济性。不是「前沿模型每题重读一遍证据太贵」，是重读也没有用 —— 梯度下降够得着的某种结构，in-context learning 够不着。" },
  ], {
    isTextBox: true, x: M + 0.35, y: 4.95, w: W - 0.7, h: 0.85,
    fontFace: BODY, fontSize: 14, color: TEXT, lineSpacingMultiple: 1.25, margin: 0,
  });
  s.addText("两个要留着的细节：depth 分片上 A0′ 的 headroom 是 −0.018，查表次数最多的那一片，它比完全不知道表还差。另外证据不是它自己挑的（见第 8 页）。", {
    isTextBox: true, x: M, y: 6.1, w: W, h: 0.6,
    fontFace: BODY, fontSize: 12, color: MUTED, lineSpacingMultiple: 1.2, margin: 0,
  });
  s.addNotes("这条同时也确认了数据层的地基：π→0 那一端假设表是「学得会但读不出来」的，A0′ 从部分观测里失败正是对它的直接检验，通过了，所以冻结不用重开。");
}

// =========================================================================
// 6 — the seen_frac curve
// =========================================================================
{
  const s = page(PAPER);
  eyebrow(s, "BAND  ·  scripts/table_ceiling.py", false);
  heading(s, "监督衰减得比外推快 —— 可测带 [0.248, 0.498]", false);

  const labels = ["2%", "5%", "10%", "25%", "50%", "90%"];
  s.addChart(pres.ChartType.line, [
    { name: "item score（全部题）", labels, values: [0.360, 0.454, 0.498, 0.582, 0.724, 0.880] },
    { name: "tail（必须外推的题）", labels, values: [0.147, 0.273, 0.328, 0.430, 0.590, 0.709] },
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
  s.addText("横轴：训练时见过的表项比例（非线性刻度，六个测量点）—— 98 / 246 / 491 / 1228 / 2456 / 4422 条，共 4913 条", {
    isTextBox: true, x: 0.6, y: 6.08, w: 7.5, h: 0.55,
    fontFace: BODY, fontSize: 10.5, color: MUTED, margin: 0,
  });

  const notes = [
    ["2% 也还在外推", "98 条表项 —— 只占表的 2% —— 在从没见过的题上仍拿到 0.147。前沿模型手里握着 2551 条，是 0.016。"],
    ["瓶颈不是容量", "每个点上 fit 都是 1.000：见过的它全记得。限制自始至终是泛化。"],
    ["binary 对这个旋钮完全不敏感", "24M 条，任何设置都等于「几乎没见过」，所以曲线的形状是 unary 给的。"],
    ["噪声先量了再读", "同配置重跑，reach 差 ±0.04。别把小差读成信号。"],
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
  s.addNotes("可测带 [0.248, 0.498]：25 个点，对着 #9 说的 arm 之间 1-3 点的差距是够宽的。depth 分片塌得最快，0.817 → 0.217，因为它每题查表次数最多。");
}

// =========================================================================
// 7 — engineering state
// =========================================================================
{
  const s = page(PAPER);
  eyebrow(s, "STATE  ·  数据层已冻结 2026-08-31", false);
  heading(s, "工程状态：地基已经封住，评测层还在动", false);

  const blocks = [
    ["138 passed", "全量测试绿，含 test_infer.py。每次提交前必须绿。"],
    ["glyph/data/ 已隔离", "七个模块搬进数据层，依赖单向；10 项边界测试用 AST 扫描守着这条线，T2 能整块把生成器搬走。"],
    ["ledger 是唯一入口", "任何消耗算力的东西都走 charge。绕过一次，crossover 的数字就没意义了。"],
    ["tool layer", "声明与分配分离，单一作答路径；agent 的自估被测量而不是被藏起来。"],
    ["每次打分都带 ceiling", "两个 ceiling 加 headroom 随每个分片一起出，tail 也终于有了自己的 ceiling。"],
    ["自检 #1 #2 #3 #5 #6 通过", "#5 capacity：fit 1.000 / reach 0.710，该 check 要抓的失败没有发生。#4 hiddenness 需重跑。"],
  ];
  blocks.forEach((b, i) => {
    const x = M + (i % 3) * 4.08;
    const y = 1.72 + Math.floor(i / 3) * 2.05;
    card(s, x, y, 3.77, 1.85);
    s.addText(b[0], {
      isTextBox: true, x: x + 0.25, y: y + 0.2, w: 3.3, h: 0.4,
      fontFace: MONO, fontSize: 13.5, color: COOL, margin: 0,
    });
    s.addText(b[1], {
      isTextBox: true, x: x + 0.25, y: y + 0.62, w: 3.3, h: 1.1,
      fontFace: BODY, fontSize: 11.5, color: MUTED, lineSpacingMultiple: 1.2, margin: 0,
    });
  });

  s.addText("已定、不再重开：", {
    isTextBox: true, x: M, y: 6.0, w: 1.7, h: 0.34,
    fontFace: BODY, fontSize: 12, color: TEXT, valign: "middle", margin: 0,
  });
  ["value_form = letter_sep", "|V| = 17³", "coupling = 0.25", "n_structural(pi_low) = 5", "无 floor 分片"]
    .forEach((t, i) => chip(s, t, 2.35 + i * 2.06, 6.0, 1.94));
  s.addNotes("HANDOFF.md 里「36 tests」那一行已经过时，当前是 138。#4 hiddenness 因为每个 preset 都换过必须重跑，上次全量跑约 110 分钟。");
}

// =========================================================================
// 8 — open and next
// =========================================================================
{
  const s = page(INK);
  eyebrow(s, "OPEN  ·  docs/open_questions.md  ·  issues #1–#29", true);
  heading(s, "还没定的，和接下来的顺序", true);

  const open = [
    ["#3", "π 目前只在 iid 上测", "pi_low 上「只测 iid」和分层测最大差 0.166，因为那里 skeleton 只在 comp 上有贡献。改采样等于改基线，而基线一改，此前所有 π 全部作废。"],
    ["#20", "A0′ 的证据不是它自己挑的", "2000 条查询是脚本抽的，真实 agent 会自适应地探。口子窄 —— 52% 的表都给了，外推还是 1.6% —— 但存在。"],
    ["#23", "E0 四臂已经作废两次", "preset 改过一次，held-pair 改过一次；n=200 时差异本来也分辨不出。数据层冻结之前不重跑。"],
    ["#27 #28", "两件没写的代码", "prompt caching 计划里有、从没实现，八次运行零命中；T2 零行，而 Glyph 自己撑不起摊销 —— 准备成本约是服务的 100 倍。"],
  ];
  open.forEach((o, i) => {
    const y = 1.68 + i * 1.3;
    s.addText(o[0], {
      isTextBox: true, x: M, y, w: 0.85, h: 0.3,
      fontFace: MONO, fontSize: 12, color: CLAY, margin: 0,
    });
    s.addText(o[1], {
      isTextBox: true, x: M + 0.9, y: y - 0.04, w: 5.6, h: 0.34,
      fontFace: BODY, fontSize: 14, bold: true, color: "FFFFFF", margin: 0,
    });
    s.addText(o[2], {
      isTextBox: true, x: M + 0.9, y: y + 0.31, w: 5.6, h: 0.66,
      fontFace: BODY, fontSize: 11.5, color: LIGHTMUTE, lineSpacingMultiple: 1.2, margin: 0,
    });
  });

  card(s, 7.7, 1.72, 4.93, 4.55, PANEL);
  s.addText("下一步的顺序", {
    isTextBox: true, x: 8.0, y: 1.95, w: 4.3, h: 0.4,
    fontFace: DISPLAY, fontSize: 18, bold: true, color: "FFFFFF", margin: 0,
  });
  const next = [
    ["重跑自检 #4 hiddenness", "每个 preset 都变了，pi_high 尤其是另一个任务了。约 110 分钟，花 teacher 调用。"],
    ["定 #9 / #4：评测规模与报告口径", "arm 之间只差 1–3 个点，n 不够就什么也分辨不出。"],
    ["E0 四臂：最小闭环", "agent 查询 → 封存 → 打分 → 一个数。A2 / A4 / A6，一次跑完而不是三次。"],
    ["#26 相图", "横轴用 measured_pi()。注意 worker 的 5400s 超时会偏向「少做事」的 agent。"],
  ];
  next.forEach((n, i) => {
    const y = 2.5 + i * 0.95;
    s.addText(String(i + 1), {
      isTextBox: true, x: 8.0, y, w: 0.3, h: 0.3,
      fontFace: MONO, fontSize: 12, color: CLAY, margin: 0,
    });
    s.addText(n[0], {
      isTextBox: true, x: 8.35, y: y - 0.03, w: 3.95, h: 0.3,
      fontFace: BODY, fontSize: 12.5, bold: true, color: "FFFFFF", margin: 0,
    });
    s.addText(n[1], {
      isTextBox: true, x: 8.35, y: y + 0.28, w: 3.95, h: 0.6,
      fontFace: BODY, fontSize: 10.5, color: LIGHTMUTE, lineSpacingMultiple: 1.15, margin: 0,
    });
  });
  s.addText("arm 的结果还没有发：早先那一轮被数据层的改动作废了，harness 上还有两处不对称没定。", {
    isTextBox: true, x: M, y: 6.62, w: 6.5, h: 0.35,
    fontFace: BODY, fontSize: 11, color: "6E748A", margin: 0,
  });
  s.addNotes("顺序来自 open_questions.md 的 Order 段：E0 已经作废两次，#1 #5 #6 #7 #8 里任何一个再动都会是第三次。");
}

pres.writeFile({ fileName: "glyph-progress-2026-09-04.pptx" })
  .then((f) => console.log("wrote", f));
