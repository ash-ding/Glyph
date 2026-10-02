/* Data-generation walkthrough: step-through figures driven by example-data.js.
   Every value shown comes from window.GLYPH_EXAMPLE; nothing here is data. */

(function () {
  "use strict";
  var D = window.GLYPH_EXAMPLE;
  if (!D) return;

  /* ------------------------------------------------------------- helpers */
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === "class") n.className = attrs[k];
      else if (k === "text") n.textContent = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else n.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) {
      if (c == null) return;
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }
  var SVG = "http://www.w3.org/2000/svg";
  function sv(tag, attrs, kids) {
    var n = document.createElementNS(SVG, tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "text") n.textContent = attrs[k]; else n.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c) n.appendChild(c); });
    return n;
  }
  function v(x, cls) { return el("span", { class: "v" + (cls ? " " + cls : ""), text: x }); }
  function vlist(xs, mark) {
    return el("span", { class: "vlist" }, xs.map(function (x, i) {
      return v(x, mark ? mark(x, i) : "");
    }));
  }
  function strip(vec, lim, small) {
    lim = lim || 2.5;
    return el("span", { class: "strip" + (small === "v" ? " strip--v" : small ? " strip--sm" : ""), "aria-hidden": "true" },
      vec.map(function (x) {
        var t = Math.max(0, Math.min(1, (x + lim) / (2 * lim)));
        var i = el("i");
        i.style.background = ramp(t);
        return i;
      }));
  }
  /* the tables' hue: pale green to dark olive, for vectors that belong to them */
  var LO = [246, 250, 236], HI = [56, 82, 12];
  function ramp(t) {
    return "rgb(" + LO.map(function (a, k) { return Math.round(a + (HI[k] - a) * t); }).join(",") + ")";
  }
  function mono(t, cls) { return el("span", { class: "mono" + (cls ? " " + cls : ""), text: t }); }
  function txt(t, cls) { return el("span", { class: cls || "", text: t }); }

  /* -------------------------------------------------------------- stepper */
  function Stepper(host, steps, render, opts) {
    opts = opts || {};
    host.classList.add("fig");
    host.innerHTML = "";
    var tabs = opts.tabs ? el("div", { class: "fig__tabs" }) : null;
    var body = el("div", { class: "fig__body" });
    var prev = el("button", { type: "button", text: "← Back" });
    var next = el("button", { type: "button", text: "Next →" });
    var play = el("button", { type: "button", text: "▶ Play" });
    var skip = el("button", { type: "button", text: "Skip ⇥" });
    var count = el("span", { class: "fig__count" });
    var cap = el("span", { class: "fig__caption", "aria-live": "polite" });
    var bar = el("div", { class: "fig__bar" }, [prev, next, play, skip, count, cap]);
    if (tabs) host.appendChild(tabs);
    host.appendChild(body);
    host.appendChild(bar);

    var i = 0, timer = null, S = steps;
    function draw() {
      var st = S[i];
      render(body, i, st);
      cap.textContent = st.caption || "";
      count.textContent = (i + 1) + " / " + S.length;
      prev.disabled = i === 0;
      next.disabled = i === S.length - 1;
      skip.disabled = i === S.length - 1;
      if (!timer) play.textContent = i === S.length - 1 ? "↺ Replay" : "▶ Play";
    }
    var touched = false;
    function stop() { if (timer) { clearInterval(timer); timer = null; } play.textContent = i === S.length - 1 ? "↺ Replay" : "▶ Play"; }
    prev.onclick = function () { touched = true; stop(); if (i > 0) { i--; draw(); } };
    next.onclick = function () { touched = true; stop(); if (i < S.length - 1) { i++; draw(); } };
    skip.onclick = function () { touched = true; stop(); i = S.length - 1; draw(); };
    play.onclick = function () { touched = true; start(); };
    function start() {
      if (timer) { stop(); return; }
      if (i === S.length - 1) { i = 0; draw(); }
      play.textContent = "❚❚ Pause";
      timer = setInterval(function () {
        if (i >= S.length - 1) { stop(); return; }
        i++; draw();
        if (i >= S.length - 1) stop();
      }, opts.interval || 1600);
    }
    if (tabs) {
      opts.tabs.forEach(function (t, k) {
        var b = el("button", { type: "button", text: t.label, "aria-pressed": k === 0 ? "true" : "false" });
        b.onclick = function () {
          touched = true; stop();
          Array.prototype.forEach.call(tabs.children, function (c) { c.setAttribute("aria-pressed", "false"); });
          b.setAttribute("aria-pressed", "true");
          S = t.steps; render = t.render; i = 0; draw();
        };
        tabs.appendChild(b);
      });
      S = opts.tabs[0].steps; render = opts.tabs[0].render;
    }
    draw();

    var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!still && "IntersectionObserver" in window) {
      var seen = new IntersectionObserver(function (es) {
        es.forEach(function (e) {
          if (!e.isIntersecting) return;
          seen.disconnect();
          if (!touched && i === 0) start();
        });
      }, { threshold: .55 });
      seen.observe(host);
    }
  }

  /* ------------------------------------------------------- the example */
  var ex = D.example;
  var exExpr = document.getElementById("ex-expr");
  if (exExpr) exExpr.textContent = ex.expr;
  var exAns = document.getElementById("ex-answer");
  if (exAns) exAns.textContent = ex.answer;

  /* ===================================================== STEP 1: values */
  (function () {
    var host = document.getElementById("fig-value");
    if (!host) return;
    var U = D.unary_lookup;
    var steps = [
      { caption: "Take one value from the example's literal." },
      { caption: "Each letter is a digit in base 17: a = 0, b = 1, …, q = 16." },
      { caption: "Each (position, digit) owns one 16-dimensional vector — 3 × 17 = 51 vectors, shared by all 4913 values." },
      { caption: "The value's embedding is its three vectors side by side: 48 numbers." }
    ];
    Stepper(host, steps, function (body, i) {
      body.innerHTML = "";
      var flow = el("div", { class: "flow" });
      function row(k, label, content) {
        var r = el("div", { class: "flow__row" + (i >= k ? " is-on" : "") + (i === k ? " is-cur" : "") },
          [el("div", { class: "flow__k", text: label }), el("div", { class: "flow__v" }, content)]);
        flow.appendChild(r);
      }
      row(0, "value", [v(U["in"])]);
      row(1, "digits", U.letters.map(function (l, k) {
        return el("span", { class: "cand" }, [txt(l + " = " + U.digits[k]), txt("pos " + k, "faint")]);
      }));
      var bank = el("div", { class: "bank" });
      D.digit_bank.forEach(function (rowv, k) {
        var r = el("div", { class: "bank__row" }, [el("span", { class: "bank__k", text: "pos " + k })]);
        rowv.forEach(function (vec, d) {
          var on = i >= 2 && U.digits[k] === d;
          r.appendChild(el("span", { class: "bank__cell" + (on || i < 2 ? " is-on" : "") },
            [strip(vec, 2.5, "v"), el("span", { text: String.fromCharCode(97 + d) })]));
        });
        bank.appendChild(r);
      });
      row(2, "shared vectors", [bank]);
      row(3, "embedding", U.emb.map(function (e) { return strip(e); }));
      body.appendChild(flow);
    });
  })();

  /* ===================================================== STEP 2: syntax */
  (function () {
    var grid = document.getElementById("shape-grid");
    if (grid) {
      var sig = { UL: "(unary, list) → list", LB: "(list, binary) → value", L: "(list) → list", KL: "(int, list) → list" };
      var n = D.presets.pi_mid.n_structural;
      D.struct_shapes.forEach(function (s, k) {
        grid.appendChild(el("div", { class: "shape" + (k < n ? " is-on" : "") }, [
          txt(s[0] + " " + s[1]),
          el("small", { text: sig[s[1]] })
        ]));
      });
    }
    var spec = document.getElementById("syntax-spec");
    if (spec) spec.textContent = D.syntax_spec;
  })();

  /* =================================================== STEP 3: skeleton */
  (function () {
    var host = document.getElementById("fig-deriv");
    if (!host) return;
    function render(body, i, st, steps) {
      body.innerHTML = "";
      var box = el("div", { class: "deriv" });
      steps.forEach(function (s, k) {
        var cls = k < i ? "is-past" : (k === i ? "is-cur" : "is-future");
        var line = el("div", { class: "deriv__line " + cls });
        var prevToks = k > 0 ? steps[k - 1].form.split(" ") : [];
        var toks = s.form.split(" ");
        var a = 0;
        while (a < toks.length && a < prevToks.length && toks[a] === prevToks[a]) a++;
        var b = 0;
        while (b < toks.length - a && b < prevToks.length - a &&
               toks[toks.length - 1 - b] === prevToks[prevToks.length - 1 - b]) b++;
        toks.forEach(function (t, j) {
          var changed = k > 0 && j >= a && j < toks.length - b;
          var cl = t.charAt(0) === "‹" ? "nt" : "";
          if (changed && k === i) cl = "hl";
          line.appendChild(el("span", { class: cl, text: t }));
          if (j < toks.length - 1) line.appendChild(document.createTextNode(" "));
        });
        box.appendChild(line);
      });
      body.appendChild(box);
    }
    function mk(op) {
      var steps = D.derivations[op].map(function (s) {
        return { form: s.form, caption: s.note + (s.knob ? "   ·   " + s.knob : "") };
      });
      return { label: op + " · " + D.skeleton.filter(function (s) { return s.op === op; })[0].shape,
               steps: steps, render: function (b, i, st) { render(b, i, st, steps); } };
    }
    var tabs = [mk("s0"), mk("s1")];
    Stepper(host, tabs[0].steps, tabs[0].render, { tabs: tabs, interval: 1400 });

    var tbl = document.getElementById("skel-table");
    if (tbl) {
      tbl.appendChild(el("thead", {}, [el("tr", {}, ["op", "shape", "if", "then", "else", "fold"].map(function (h) {
        return el("th", { text: h });
      }))]));
      var tb = el("tbody");
      D.skeleton.forEach(function (s) {
        tb.appendChild(el("tr", {}, [
          el("td", { class: "mono", text: s.op }), el("td", { class: "mono", text: s.shape }),
          el("td", { class: "mono", text: s.guard || "—" }),
          el("td", { class: "mono", text: s.then.join(" ; ") }),
          el("td", { class: "mono", text: s["else"] ? s["else"].join(" ; ") : "—" }),
          el("td", { class: "mono", text: s.fold ? (s.fold === "L" ? "left" : "right") : "—" })
        ]));
      });
      tbl.appendChild(tb);
    }
  })();

  /* ===================================================== STEP 4: tables */
  (function () {
    var host = document.getElementById("fig-table");
    if (!host) return;
    var U = D.unary_lookup, B = D.binary_lookup;

    function flowRender(rows) {
      return function (body, i) {
        body.innerHTML = "";
        var flow = el("div", { class: "flow" });
        rows.forEach(function (r, k) {
          flow.appendChild(el("div", { class: "flow__row" + (i >= k ? " is-on" : "") + (i === k ? " is-cur" : "") },
            [el("div", { class: "flow__k", text: r.label }), el("div", { class: "flow__v" }, r.content())]));
        });
        body.appendChild(flow);
      };
    }
    function nearest(list, out) {
      return list.map(function (c) {
        return el("span", { class: "cand" + (c.value === out ? " cand--win" : "") },
          [txt(c.value), txt("d = " + c.dist.toFixed(2), "faint")]);
      });
    }

    var uRows = [
      { label: "lookup", content: function () { return [mono(U.op + "("), v(U["in"]), mono(")")]; },
        caption: "One unary lookup from the example: u1 applied to the literal's first value." },
      { label: "digits", content: function () { return U.letters.map(function (l, k) { return mono(l + " = " + U.digits[k] + "  "); }); },
        caption: "Split into digits (step 1)." },
      { label: "digit vectors", content: function () { return U.emb.map(function (e) { return strip(e); }); },
        caption: "Look up each digit's shared vector." },
      { label: "digit-wise MLPs", content: function () { return U.parts.map(function (p) { return strip(p, 1); }); },
        caption: "Each position goes through its own small frozen MLP: f₀(e₀), f₁(e₁), f₂(e₂)." },
      { label: "+ α · global MLP", content: function () { return [strip(U.mix, 0.5), txt("α = " + U.alpha, "faint small")]; },
        caption: "A larger MLP sees the whole 48-dim embedding; its output is weighted by α = unary_coupling." },
      { label: "sum", content: function () { return [strip(U.y, 1)]; },
        caption: "Add them: one 48-dim output. With α = 0, position k of the output would depend on input digit k alone." },
      { label: "whiten", content: function () { return [strip(U.z)]; },
        caption: "Rescale per dimension onto the embeddings' own spread (decode = whiten), so outputs do not crowd one corner." },
      { label: "nearest value", content: function () { return nearest(U.nearest, U.out); },
        caption: "Round to the nearest of the 4913 value embeddings. " + U.op + "(" + U["in"] + ") = " + U.out + " — that is the table entry." }
    ];
    var bRows = [
      { label: "lookup", content: function () { return [mono(B.op + "("), v(B["in"][0]), mono(", "), v(B["in"][1]), mono(")")]; },
        caption: "The first binary lookup in the example's fold." },
      { label: "digit pairs", content: function () {
          return B.digit_pairs.map(function (p, k) { return mono("(" + p[0] + ", " + p[1] + ")  "); });
        }, caption: "Pair up the two values' digits position by position." },
      { label: "digit-wise MLPs", content: function () { return B.parts.map(function (p) { return strip(p, 1); }); },
        caption: "One small MLP per position, on the pair: 17² = 289 inputs each, 867 in all." },
      { label: "+ α · global MLP", content: function () { return [strip(B.mix, 0.5), txt("α = " + B.alpha, "faint small")]; },
        caption: "The coupling term sees both whole values (96 dims). It is what stops the 867 parts from being enumerated." },
      { label: "nearest value", content: function () { return nearest(B.nearest, B.out); },
        caption: "Whiten and round, as for unary. " + B.op + "(" + B["in"].join(", ") + ") = " + B.out + "." }
    ];
    var tabs = [
      { label: "unary  " + U.op, steps: uRows, render: flowRender(uRows) },
      { label: "binary  " + B.op, steps: bRows, render: flowRender(bRows) }
    ];
    Stepper(host, uRows, tabs[0].render, { tabs: tabs, interval: 1500 });
  })();

  /* ================================================= STEP 5: held pairs */
  (function () {
    var host = document.getElementById("fig-held");
    if (!host) return;
    var H = D.held, ops = H.ops.map(function (o) { return o[0]; });
    var shape = {}; H.ops.forEach(function (o) { shape[o[0]] = o[1]; });
    var real = {}, held = {};
    H.realizable.forEach(function (p) { real[p[0] + "|" + p[1]] = 1; });
    H.held.forEach(function (p) { held[p[0] + "|" + p[1]] = 1; });
    var exPair = ex.pairs[0];
    var nAll = ops.length * ops.length;
    var steps = [
      { caption: "Every (outer, inner) pair of the 7 enabled operators: " + nAll + " cells." },
      { caption: "Keep the buildable ones. No operator takes a value argument, so s1 — which returns a value — can never be inner: " + H.realizable.length + " remain." },
      { caption: "Group them by (outer shape, inner shape): " + H.classes.length + " classes. Each gives up one third, by largest remainder." },
      { caption: H.held.length + " pairs held out (amber). Demos, validation, iid and depth never contain one; comp always does." },
      { caption: "The example's only pair, (" + exPair.join(", ") + "), is not held — so it may appear in a demo." }
    ];
    Stepper(host, steps, function (body, i) {
      body.innerHTML = "";
      var t = el("table", { class: "matrix" });
      var hr = el("tr", {}, [el("th", { text: "outer ↓  inner →" })]);
      ops.forEach(function (o) { hr.appendChild(el("th", { text: o })); });
      t.appendChild(el("thead", {}, [hr]));
      var tb = el("tbody");
      ops.forEach(function (a) {
        var r = el("tr", {}, [el("th", { text: a + " " + shape[a] })]);
        ops.forEach(function (b) {
          var key = a + "|" + b, cls = "";
          if (i === 0 || (i >= 1 && real[key])) cls = "real";
          if (i >= 3 && held[key]) cls = "held";
          if (i >= 4 && a === exPair[0] && b === exPair[1]) cls += " ex";
          var cell = el("i", { class: cls, title: "(" + a + ", " + b + ")" });
          r.appendChild(el("td", {}, [cell]));
        });
        tb.appendChild(r);
      });
      t.appendChild(tb);
      body.appendChild(t);
      body.appendChild(el("div", { class: "legend" }, [
        el("span", {}, [el("i", { class: "lg-real" }), txt("buildable")]),
        el("span", {}, [el("i", { class: "lg-none" }), txt("not buildable")]),
        el("span", {}, [el("i", { class: "lg-held" }), txt("held out")])
      ]));
      if (i >= 2) {
        var chips = el("div", { class: "classes" });
        H.classes.forEach(function (c) {
          chips.appendChild(el("span", { class: "cand" }, [txt(c.outer + " ∘ " + c.inner),
            txt(i >= 3 ? c.held + " of " + c.n : c.n + " pairs", "faint")]));
        });
        body.appendChild(el("p", { class: "small faint", style: "margin:.9rem 0 .35rem;text-align:center",
          text: "(outer shape ∘ inner shape) classes" + (i >= 3 ? " — held of buildable" : "") }));
        body.appendChild(chips);
      }
    }, { interval: 1800 });
  })();

  /* ============================================= STEP 6: sample the tree */
  (function () {
    var host = document.getElementById("fig-sample");
    if (!host) return;
    var dec = ex.decisions;        // [root s1, child s0, literal]
    var R0 = dec[0], R1 = dec[1], LF = dec[2];
    var bSlot = R0.slots.filter(function (s) { return s.slot === "binary"; })[0];
    var uSlot = R1.slots.filter(function (s) { return s.slot === "unary"; })[0];

    var steps = [
      { caption: "The root asks for a value (probability binary_freq = 0.35; a list otherwise), with budget demo_max_depth = 2.",
        show: [], cur: "root?", d: { k: "need", val: "a VAL, budget " + R0.budget } },
      { caption: "Only fold-like operators return a value, so there is no coin to flip: " + R0.op + ".",
        show: ["root"], cur: "root", d: R0 },
      { caption: "Its binary slot takes one of " + bSlot.from.join(", ") + " uniformly: " + bSlot.value + ".",
        show: ["root", "b"], cur: "b", d: { k: "binary slot", val: bSlot.value + "  from  " + bSlot.from.join(" · ") } },
      { caption: "Its list slot asks for a list, budget 1. It does not stop early (0.85). Coin < atomic_ratio = 0.5 → a table-consuming operator: " + R1.op + ".",
        show: ["root", "b", "child"], cur: "child", d: R1 },
      { caption: "Its unary slot takes one of " + uSlot.from.join(", ") + ": " + uSlot.value + ".",
        show: ["root", "b", "child", "u"], cur: "u", d: { k: "unary slot", val: uSlot.value + "  from  " + uSlot.from.join(" · ") } },
      { caption: "Budget exhausted → a list literal: length " + LF.len + " (2–4), each value uniform over 4913.",
        show: ["root", "b", "child", "u", "leaf"], cur: "leaf", d: { k: "literal", val: LF.why + " → " + LF.len + " values" } },
      { caption: "Done: " + ex.expr,
        show: ["root", "b", "child", "u", "leaf"], cur: null, d: { k: "expression", val: ex.expr } }
    ];

    function box(id, x, y, w, h, title, sub, kind, st) {
      var g = sv("g", { class: "node node--" + kind + (st.show.indexOf(id) < 0 ? " is-hidden" : "") + (st.cur === id ? " is-cur" : "") });
      g.appendChild(sv("rect", { x: x, y: y, width: w, height: h }));
      g.appendChild(sv("text", { x: x + 10, y: y + (sub ? 19 : h / 2 + 4), text: title }));
      if (sub) g.appendChild(sv("text", { class: "sub", x: x + 10, y: y + 34, text: sub }));
      return g;
    }
    function slot(id, x, y, w, title, kind, st) {
      var g = sv("g", { class: "slot slot--" + kind + (st.show.indexOf(id) < 0 ? " is-hidden" : "") + (st.cur === id ? " is-cur" : "") });
      g.appendChild(sv("rect", { x: x, y: y, width: w, height: 26 }));
      g.appendChild(sv("text", { x: x + 9, y: y + 17, text: title }));
      return g;
    }
    function edge(d, on) { return sv("path", { class: "edge" + (on ? "" : " is-hidden"), d: d }); }

    Stepper(host, steps, function (body, i, st) {
      body.innerHTML = "";
      var s = sv("svg", { class: "tree-svg", viewBox: "0 0 600 290", width: "600", role: "img",
                          "aria-label": "Expression tree being sampled" });
      var has = function (id) { return st.show.indexOf(id) >= 0; };
      s.appendChild(edge("M 300 62 L 300 116", has("child")));
      s.appendChild(edge("M 380 40 L 440 40", has("b")));
      s.appendChild(edge("M 220 138 L 150 138", has("u")));
      s.appendChild(edge("M 300 160 L 300 214", has("leaf")));
      if (i === 0) {
        var q = sv("g", { class: "node is-cur" });
        q.appendChild(sv("rect", { x: 220, y: 18, width: 160, height: 44 }));
        q.appendChild(sv("text", { x: 232, y: 45, text: "? : VAL" }));
        s.appendChild(q);
      }
      s.appendChild(box("root", 220, 18, 160, 44, R0.op + " · " + R0.shape, "fold-like → value", "skel", st));
      s.appendChild(slot("b", 440, 27, 64, bSlot.value, "table", st));
      s.appendChild(box("child", 220, 116, 160, 44, R1.op + " · " + R1.shape, "map-like → list", "skel", st));
      s.appendChild(slot("u", 86, 125, 64, uSlot.value, "table", st));
      s.appendChild(box("leaf", 110, 214, 380, 40, LF.value, null, "lit", st));
      body.appendChild(s);

      var p = el("div", { class: "decision" });
      var d = st.d;
      if (d.cands) {
        p.appendChild(el("div", { class: "decision__row" }, [el("span", { class: "decision__k", text: "request" }),
          txt(d.type + ", budget " + d.budget)]));
        p.appendChild(el("div", { class: "decision__row" }, [el("span", { class: "decision__k", text: "candidates" })]
          .concat(d.cands.map(function (c) { return mono(c + " "); }))));
        var aOn = d.atomic.indexOf(d.op) >= 0;
        p.appendChild(el("div", { class: "decision__row" }, [el("span", { class: "decision__k", text: "pools" }),
          el("span", { class: "pool pool--table" + (aOn ? " pool--on" : ""), text: "table-consuming: " + (d.atomic.join(" ") || "—") }),
          el("span", { class: "pool pool--skel" + (!aOn ? " pool--on" : ""), text: "pure: " + (d.pure.join(" ") || "—") })]));
        p.appendChild(el("div", { class: "decision__row" }, [el("span", { class: "decision__k", text: "rule" }), txt(d.rule)]));
        p.appendChild(el("div", { class: "decision__row" }, [el("span", { class: "decision__k", text: "picked" }), mono(d.op + " (" + d.shape + ")")]));
      } else {
        p.appendChild(el("div", { class: "decision__row" }, [el("span", { class: "decision__k", text: d.k }), mono(d.val)]));
      }
      body.appendChild(p);
    }, { interval: 1900 });
  })();

  /* ================================================== STEP 7: evaluate */
  (function () {
    var host = document.getElementById("fig-eval");
    if (!host) return;
    var rows = [];
    ex.trace.forEach(function (r) {
      if (r.kind === "literal") {
        rows.push({ op: "literal", sub: "", kind: "lit", input: null, out: r.out, caption: "Start from the literal." });
        return;
      }
      var who = r.op + " · " + r.shape;
      if (r.guard) {
        rows.push({ op: who, sub: "guard", kind: "skel", guard: r.guard, input: r["in"],
          caption: "Guard: " + r.guard.text + "? The list has length " + r.guard.len + " → " +
                   (r.guard.holds ? "true, take the then-branch." : "false, take the else-branch.") });
      }
      r.prims.forEach(function (p) {
        var table = p.lookups.length > 0;
        rows.push({ op: who, sub: p.prim.replace(" u", " " + (r.unary || "u")), kind: table ? "table" : "skel",
          input: p["in"], out: p.out, lookups: p.lookups,
          caption: table ? p.lookups.length + " table lookups — the only steps here that depend on what a value means."
                         : (p.prim === "ident" ? "ident: leave the list as it is." : p.prim + ": pure rearrangement, no value is interpreted.") });
      });
      if (r.fold) {
        rows.push({ op: who, sub: "fold-" + (r.fold.dir === "L" ? "left" : "right") + " with " + r.binary, kind: "table",
          input: r.prims.length ? r.prims[r.prims.length - 1].out : r["in"], fold: r.fold,
          caption: "Fold " + (r.fold.dir === "L" ? "left" : "right") + " with " + r.binary + ": " + r.fold.steps.length +
                   " binary lookups reduce the list to one value." });
      }
    });
    rows.push({ op: "answer", sub: "", kind: "ans", out: ex.answer,
      caption: "The answer is " + ex.answer + ", exactly the demo's answer: P computed it." });

    Stepper(host, rows, function (body, i) {
      body.innerHTML = "";
      var box = el("div", { class: "trace" });
      rows.forEach(function (r, k) {
        var cls = k < i ? "" : (k === i ? " is-cur" : " is-future");
        var opcls = r.kind === "table" ? "table" : (r.kind === "skel" ? "skel" : "");
        var left = el("div", { class: "trace__op " + opcls }, [txt(r.op), el("small", { text: r.sub })]);
        var right = el("div", { class: "trace__body" });
        if (r.guard) {
          right.appendChild(el("div", {}, [vlist(r.input), txt("  len " + r.guard.len + " — " + r.guard.text + "? "),
            txt(r.guard.holds ? "yes" : "no", r.guard.holds ? "guard-ok" : "guard-no")]));
        } else if (r.fold) {
          r.fold.steps.forEach(function (s) {
            right.appendChild(el("div", { class: "lookup" }, [txt(s.op + "(" + s.args.join(", ") + ") = "), v(s.out, "v--new")]));
          });
        } else if (r.kind === "ans") {
          right.appendChild(el("div", {}, [v(r.out, "v--new")]));
        } else {
          var newSet = {};
          if (r.lookups) r.lookups.forEach(function (l) { newSet[l.out] = 1; });
          if (r.lookups && r.lookups.length) {
            r.lookups.forEach(function (l) {
              right.appendChild(el("div", { class: "lookup" }, [txt(l.op + "(" + l.args[0] + ") = " + l.out)]));
            });
          }
          right.appendChild(el("div", {}, [vlist(r.out, function (x) { return newSet[x] ? "v--new" : ""; })]));
        }
        box.appendChild(el("div", { class: "trace__row trace__row--" + r.kind + cls }, [left, right]));
      });
      body.appendChild(box);
    }, { interval: 1700 });
  })();

  /* =================================================== STEP 8: splits */
  (function () {
    var practice = document.getElementById("splits-practice");
    var test = document.getElementById("splits-test");
    var probe = document.getElementById("splits-probe");
    if (!practice || !test || !probe) return;
    var S = D.splits, PR = D.probes;
    var groups = [
      [practice, [
        { name: "demos", n: D.n_demos, hist: D.demo_depth_hist, rule: "depth 1–2 · held pairs forbidden · with answers", ex: { expr: ex.expr } },
        { name: "validation", n: D.n_val, hist: D.val_depth_hist, rule: "depth 1–2 · held pairs forbidden · scored in aggregate" }
      ]],
      [test, [
        { name: "iid", n: S.iid.n, hist: S.iid.depth_hist, rule: "depth 1–2 · held pairs forbidden", ex: S.iid.example },
        { name: "comp", n: S.comp.n, hist: S.comp.depth_hist, rule: "depth 1–2 · must contain a held pair", ex: S.comp.example },
        { name: "depth", n: S.depth.n, hist: S.depth.depth_hist, rule: "depth 3–4 · deeper than any demo", ex: S.depth.example }
      ]],
      [probe, [
        { name: "probe", n: PR.n, hist: PR.depth_hist, rule: "one operator at a time · atomic at depth 0, pure-structure at depth 1",
          ex: { expr: PR.examples.filter(function (x) { return x.op === "u0" || x.op === "s3"; }).map(function (x) { return x.expr; }).join("  ·  ") } }
      ]]
    ];
    var fills = [];
    groups.forEach(function (g) {
      g[1].forEach(function (c) {
        var bars = el("div", { class: "bars" });
        var tot = 0; Object.keys(c.hist).forEach(function (k) { tot += c.hist[k]; });
        [0, 1, 2, 3, 4].forEach(function (d) {
          var n = c.hist[d] || 0;
          var f = el("div", { class: "bar__fill" }); f.style.width = "0%";
          fills.push([f, (100 * n / tot).toFixed(1) + "%"]);
          bars.appendChild(el("div", { class: "bar" }, [txt("depth " + d, "faint"),
            el("div", { class: "bar__track" }, [f]), txt(String(n))]));
        });
        var kids = [el("h4", { text: c.name + "  ·  " + c.n.toLocaleString("en-US") }),
                    el("p", { class: "split__rule", text: c.rule }), bars];
        if (c.ex) {
          var held = c.ex.held && c.ex.held.length ? "  ← held pair (" + c.ex.held[0].join(", ") + ")" : "";
          kids.push(el("div", {}, [el("code", { text: c.ex.expr }), txt(held, "small skel")]));
        }
        g[0].appendChild(el("div", { class: "split" }, kids));
      });
    });
    function grow() { fills.forEach(function (p) { p[0].style.width = p[1]; }); }
    if ("IntersectionObserver" in window) {
      var o = new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (e.isIntersecting) { grow(); o.disconnect(); } });
      }, { threshold: .15 });
      o.observe(practice);
    } else grow();
  })();

  /* ======================================================= STEP 9: pi */
  (function () {
    var body = document.getElementById("pi-body");
    if (!body) return;
    var C = D.crippled, pi = D.instance.pi;
    function ans(label, sub, val, ok) {
      return el("div", { class: "answer" }, [el("div", {}, [txt(label), el("div", { class: "small faint", text: sub })]),
        v(val, ok ? "v--new" : ""), el("span", { class: "mark " + (ok ? "guard-ok" : "guard-no"), text: ok ? "✓ correct" : "✗ wrong" })]);
    }
    body.appendChild(el("p", { class: "small muted", style: "margin-top:0", text: "The running example, answered three ways:" }));
    body.appendChild(el("div", { class: "answers" }, [
      ans("P — true skeleton, true tables", "the real interpreter", C.gold, true),
      ans("true skeleton + identity table", "u(x) = x, b(x, y) = x — knows no table entry", C.skeleton_only, C.skeleton_only === C.gold),
      ans("textbook skeleton + true tables", "map is map, fold is left, int ops take k, the rest are ident", C.table_only, C.table_only === C.gold)
    ]));
    var Ls = pi.L_skel, Lt = pi.L_table;
    body.appendChild(el("p", { class: "small muted", text:
      "Over the " + pi.n.toLocaleString("en-US") + "-item held-out test (graded per digit, so one wrong lookup does not zero an item): " +
      "the identity-table interpreter scores " + pi.a_skel.toFixed(3) + ", so missing the tables costs L_table = " + Lt.toFixed(3) +
      "; the textbook-skeleton one scores " + pi.a_tab.toFixed(3) + ", so missing the skeleton costs L_skel = " + Ls.toFixed(3) + "." }));
    var share = el("div", { class: "share", role: "img", "aria-label": "Skeleton share of the difficulty" });
    var a = el("span", { class: "s-skel", text: "skeleton " + (100 * Ls / (Ls + Lt)).toFixed(0) + "%" });
    var b = el("span", { class: "s-tab", text: "tables " + (100 * Lt / (Ls + Lt)).toFixed(0) + "%" });
    a.style.width = (100 * Ls / (Ls + Lt)) + "%"; b.style.width = (100 * Lt / (Ls + Lt)) + "%";
    share.appendChild(a); share.appendChild(b);
    body.appendChild(share);
    body.appendChild(el("p", { class: "mono small", text: "π = L_skel / (L_skel + L_table) = " + Ls.toFixed(3) + " / " +
      (Ls + Lt).toFixed(3) + " = " + pi.pi.toFixed(3) + "   — mid_3 sits almost exactly between the two halves." }));
    body.appendChild(el("p", { class: "small muted", text: "π per split: iid " + pi.pi_iid.toFixed(3) +
      " · comp " + pi.pi_comp.toFixed(3) + " · depth " + pi.pi_depth.toFixed(3) + "." }));
  })();

  /* ========================================= Validation: frozen instances */
  (function () {
    var body = document.getElementById("fig-frozen");
    if (!body || !D.frozen) return;
    var F = D.frozen;
    var W = 640, H = 346, m = { l: 46, r: 118, t: 34, b: 44 };
    var x = function (p) { return m.l + (p - 0.1) / 0.8 * (W - m.l - m.r); };
    var y = function (a) { return m.t + (1 - a) * (H - m.t - m.b); };
    var s = sv("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Skeleton and table ceilings of the 15 frozen instances against measured π" });

    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      s.appendChild(sv("line", { class: "grid", x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }));
      s.appendChild(sv("text", { class: "tick", x: m.l - 8, y: y(t) + 4, "text-anchor": "end", text: t.toFixed(2) }));
    });
    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].forEach(function (t) {
      s.appendChild(sv("text", { class: "tick", x: x(t), y: H - m.b + 18, "text-anchor": "middle", text: t.toFixed(1) }));
    });
    s.appendChild(sv("text", { class: "axis", x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", text: "measured π  (skeleton's share of the difficulty)" }));
    s.appendChild(sv("text", { class: "axis", x: m.l - 30, y: 14, text: "ceiling (overall accuracy)" }));

    [["low", 0.2, 0.3], ["mid", 0.45, 0.53], ["high", 0.7, 0.8]].forEach(function (b) {
      s.appendChild(sv("rect", { class: "band-win", x: x(b[1]), y: m.t, width: x(b[2]) - x(b[1]), height: H - m.t - m.b }));
      s.appendChild(sv("text", { class: "tick", x: (x(b[1]) + x(b[2])) / 2, y: m.t + 12, "text-anchor": "middle", text: b[0] }));
    });

    var tip = el("div", { class: "chart-tip", role: "status" });
    function show(r, ev) {
      tip.textContent = r.id + " · " + r.preset + " seed " + r.seed + " · π " + r.pi.toFixed(3) +
        " · skeleton " + r.skeleton.overall.toFixed(3) + " · table " + r.table.overall.toFixed(3);
      tip.classList.add("is-on");
      var box = body.getBoundingClientRect(), w = tip.offsetWidth;
      tip.style.left = Math.max(0, Math.min(box.width - w - 4, ev.clientX - box.left - w / 2)) + "px";
      tip.style.top = Math.max(0, ev.clientY - box.top - 46) + "px";
    }
    function hide() { tip.classList.remove("is-on"); }

    F.forEach(function (r) {
      var g = sv("g", { class: "inst", tabindex: "0", "aria-label": r.id + ", π " + r.pi.toFixed(3) +
        ", skeleton ceiling " + r.skeleton.overall.toFixed(3) + ", table ceiling " + r.table.overall.toFixed(3) });
      g.appendChild(sv("line", { class: "pair", x1: x(r.pi), x2: x(r.pi), y1: y(r.skeleton.overall), y2: y(r.table.overall) }));
      g.appendChild(sv("circle", { class: "dot dot--skel", cx: x(r.pi), cy: y(r.skeleton.overall), r: 5 }));
      g.appendChild(sv("circle", { class: "dot dot--table", cx: x(r.pi), cy: y(r.table.overall), r: 5 }));
      g.appendChild(sv("rect", { class: "hit", x: x(r.pi) - 8, y: m.t, width: 16, height: H - m.t - m.b }));
      g.addEventListener("mousemove", function (ev) { show(r, ev); });
      g.addEventListener("mouseleave", hide);
      g.addEventListener("focus", function () {
        var b = g.getBoundingClientRect(); show(r, { clientX: b.left + b.width / 2, clientY: b.top + 30 });
      });
      g.addEventListener("blur", hide);
      s.appendChild(g);
    });

    var last = F[F.length - 1];
    s.appendChild(sv("text", { class: "lab lab--skel", x: x(last.pi) + 12, y: y(last.skeleton.overall) + 4, text: "skeleton ceiling" }));
    s.appendChild(sv("text", { class: "lab lab--table", x: x(last.pi) + 12, y: y(last.table.overall) + 4, text: "table ceiling" }));
    var ex3 = F.filter(function (r) { return r.id === D.instance.frozen_id; })[0];
    if (ex3) {
      var lo = Math.min(ex3.skeleton.overall, ex3.table.overall);
      s.appendChild(sv("line", { class: "leader", x1: x(ex3.pi) - 2, y1: y(lo) + 7, x2: x(ex3.pi) - 10, y2: y(0.1) - 4 }));
      s.appendChild(sv("text", { class: "tick", x: x(ex3.pi) - 12, y: y(0.1) + 6, "text-anchor": "end", text: ex3.id + " (walkthrough)" }));
    }

    body.appendChild(el("div", { class: "chart-wrap" }, [s, tip]));
    body.appendChild(el("div", { class: "legend" }, [
      el("span", {}, [el("i", { class: "sw sw--skel" }), txt("skeleton ceiling — true skeleton, identity tables")]),
      el("span", {}, [el("i", { class: "sw sw--table" }), txt("table ceiling — textbook skeleton, true tables")]),
      el("span", {}, [el("i", { class: "sw sw--win" }), txt("selection window")])
    ]));

    var t = el("table");
    t.appendChild(el("thead", {}, [el("tr", {}, ["instance", "preset", "seed", "π", "skeleton", "table", "perfect"].map(function (h, k) {
      return el("th", { class: k >= 2 ? "num" : "", text: h });
    }))]));
    var tb = el("tbody");
    F.forEach(function (r) {
      tb.appendChild(el("tr", {}, [el("td", { class: "mono", text: r.id }), el("td", { class: "mono", text: r.preset }),
        el("td", { class: "num", text: String(r.seed) }), el("td", { class: "num", text: r.pi.toFixed(3) }),
        el("td", { class: "num", text: r.skeleton.overall.toFixed(3) }), el("td", { class: "num", text: r.table.overall.toFixed(3) }),
        el("td", { class: "num", text: r.perfect.overall.toFixed(3) })]));
    });
    t.appendChild(tb);
    body.appendChild(el("details", {}, [el("summary", { class: "small", text: "Table view" }),
      el("div", { class: "table-scroll" }, [t])]));
  })();
  /* ========================================= PROTOCOL: practice and final */
  (function () {
    var PR = D.protocol;
    var hostP = document.getElementById("fig-practice"), hostF = document.getElementById("fig-final");
    if (!PR || !hostP || !hostF) return;
    var TOOLS = ["query", "submit_validation_answer", "check_answers", "finish_practice",
                 "build_dataset", "train_model", "infer_model", "submit_final_answer"];
    var TRAIN = { build_dataset: 1, train_model: 1, infer_model: 1 };
    var AVAIL = {
      practice: { no_train: ["query", "submit_validation_answer", "check_answers", "finish_practice"],
                  train: ["query", "submit_validation_answer", "check_answers", "finish_practice", "build_dataset", "train_model", "infer_model"] },
      final: { no_train: ["check_answers", "submit_final_answer"],
               train: ["check_answers", "infer_model", "submit_final_answer"] }
    };
    var fmtN = function (n) { return Number(n).toLocaleString("en-US"); };
    var r3 = function (x) { return x == null ? "—" : Number(x).toFixed(3); };

    function jv(x) {
      if (typeof x === "number" && !Number.isInteger(x)) return String(Math.round(x * 1e4) / 1e4);
      return JSON.stringify(x);
    }
    function kv(obj, skip) {
      var box = el("div", { class: "call__kv" });
      Object.keys(obj).forEach(function (k) {
        if (skip && skip[k]) return;
        var val = obj[k];
        var row = el("div", { class: "call__row" }, [el("span", { class: "call__k", text: k })]);
        if (Array.isArray(val) && val.length && (typeof val[0] === "object" || val.length > 1)) {
          var list = el("div", { class: "call__list" });
          val.forEach(function (o) { list.appendChild(el("div", { text: JSON.stringify(o) })); });
          row.appendChild(list);
        } else if (val && typeof val === "object" && !Array.isArray(val)) {
          row.appendChild(el("span", { class: "call__v", text: "{" + Object.keys(val).map(function (k2) {
            return k2 + ": " + jv(val[k2]); }).join(", ") + "}" }));
        } else {
          row.appendChild(el("span", { class: "call__v" + (val === "…" ? " faint" : ""), text: jv(val) }));
        }
        box.appendChild(row);
      });
      return box;
    }
    var REM = { q_remaining: 1, submissions_remaining: 1 };
    function summary(c) {
      var r = c.result;
      if (r.error) return "error";
      switch (c.tool) {
        case "query":
          var ok = r.results.filter(function (x) { return "out" in x; }).length;
          return ok + " answered, " + (r.results.length - ok) + " not · q_used " + r.q_used;
        case "check_answers": return r.ok ? "ok" : "illegal · " + Object.keys(r.violations).map(function (k) { return k + " " + fmtN(r.violations[k]); }).join(", ");
        case "submit_validation_answer": return "submission " + r.submission + " · overall " + r3(r.overall);
        case "build_dataset": return r.dataset_id + " · " + r.size + " rows";
        case "train_model": return r.checkpoint_id + (r.continued_from ? " ← " + r.continued_from : " (new student)");
        case "infer_model": return fmtN(r.rows) + " rows answered";
        case "finish_practice": return "ok";
        case "submit_final_answer": return "committed · " + r.digest;
      }
      return "";
    }

    function callCard(c, state) {
      var head = el("div", { class: "call__head" }, [
        el("code", { class: "call__tool" + (TRAIN[c.tool] ? " train" : ""), text: c.tool }),
        el("span", { class: "call__sum", text: summary(c) })]);
      var card = el("div", { class: "call " + state }, [head]);
      if (state === "is-cur") {
        card.appendChild(el("div", { class: "call__dir", text: "call" }));
        card.appendChild(kv(c.args));
        card.appendChild(el("div", { class: "call__dir", text: "result" }));
        card.appendChild(kv(c.result, REM));
        var rem = Object.keys(REM).filter(function (k) { return k in c.result; });
        if (rem.length) card.appendChild(el("div", { class: "call__rem", text: rem.map(function (k) { return k + " " + jv(c.result[k]); }).join(" · ") }));
      }
      return card;
    }

    function fileCard(title, rows, note, cur) {
      var card = el("div", { class: "call call--file " + (cur ? "is-cur" : "is-past") }, [
        el("div", { class: "call__head" }, [el("code", { class: "call__tool", text: title }), el("span", { class: "call__sum", text: note })])]);
      if (!cur) return card;
      var list = el("div", { class: "call__list" });
      rows.forEach(function (r) { list.appendChild(el("div", { text: JSON.stringify(r) })); });
      list.appendChild(el("div", { class: "faint", text: "…" }));
      card.appendChild(list);
      return card;
    }

    function reportCard(rep) {
      var card = el("div", { class: "call is-cur call--report" }, [
        el("div", { class: "call__head" }, [el("code", { class: "call__tool", text: "report.json" }),
          el("span", { class: "call__sum", text: "written by the harness after the run" })])]);
      var atomic = [], structural = [];
      Object.keys(rep.probe).forEach(function (op) { (rep.probe[op].kind === "atomic" ? atomic : structural).push(op); });
      var mean = function (ops) {
        var n = 0, s = 0; ops.forEach(function (op) { n += rep.probe[op].n; s += rep.probe[op].overall * rep.probe[op].n; });
        return n ? s / n : null;
      };
      var rows = [
        ["overall", r3(rep.overall)],
        ["by_split", Object.keys(rep.by_split).map(function (k) { return k + " " + r3(rep.by_split[k]); }).join(" · ")],
        ["tail", r3(rep.tail)],
        ["headroom", Object.keys(rep.headroom).map(function (k) { return k + " " + r3(rep.headroom[k]); }).join(" · ")],
        ["probe", "atomic " + atomic.join(" ") + ": " + r3(mean(atomic)) + " · structural " + structural.join(" ") + ": " + r3(mean(structural))]
      ];
      var box = el("div", { class: "call__kv" });
      rows.forEach(function (r) {
        box.appendChild(el("div", { class: "call__row" }, [el("span", { class: "call__k", text: r[0] }), el("span", { class: "call__v", text: r[1] })]));
      });
      card.appendChild(box);
      return card;
    }

    function statePanel(arm, phase, calls, extra) {
      var last = null;
      calls.forEach(function (c) { if ("q_remaining" in c.result) last = c.result; });
      var qUsed = last ? last.q_remaining == null ? null : 1000 - last.q_remaining : 0;
      var subLeft = last ? last.submissions_remaining : 20;
      var answered = 0;
      calls.forEach(function (c) { if (c.tool === "query" && c.result.results) c.result.results.forEach(function (x) { if ("out" in x) answered++; }); });
      var mine = [];
      calls.forEach(function (c) {
        ["path", "output_path", "prefix_path"].forEach(function (k) {
          var p = c.args[k];
          if (p && p.indexOf("task/") !== 0 && mine.indexOf(p) < 0) mine.push(p);
        });
      });
      var meter = function (label, used, cap) {
        var w = Math.max(used / cap * 100, used ? 1.5 : 0);
        return el("div", { class: "meter" }, [
          el("div", { class: "meter__top" }, [txt(label), txt(fmtN(used) + " / " + fmtN(cap), "mono")]),
          el("div", { class: "meter__track" }, [el("div", { class: "meter__fill", style: "width:" + w.toFixed(1) + "%" })])]);
      };
      var p = el("div", { class: "run__state" });
      p.appendChild(el("div", { class: "run__k", text: "phase" }));
      p.appendChild(el("div", { class: "run__phase run__phase--" + phase, text: phase + " · " + arm + " arm" }));
      p.appendChild(el("div", { class: "run__k", text: "spent" }));
      p.appendChild(meter("queries", qUsed || 0, 1000));
      p.appendChild(meter("validation submissions", 20 - subLeft, 20));
      p.appendChild(el("div", { class: "run__k", text: "tools" }));
      var chips = el("div", { class: "run__tools" });
      TOOLS.forEach(function (t) {
        var on = AVAIL[phase][arm].indexOf(t) >= 0;
        if (!on && TRAIN[t] && arm === "no_train") return;
        chips.appendChild(el("code", { class: "tchip" + (on ? "" : " is-off") + (TRAIN[t] ? " train" : ""), text: t }));
      });
      p.appendChild(chips);
      p.appendChild(el("div", { class: "run__k", text: "workspace" }));
      var files = [["task/demos.jsonl", fmtN(PR.workspace["demos.jsonl"])],
                   ["task/validation.jsonl", fmtN(PR.workspace["validation.jsonl"])],
                   ["task/queries.jsonl", fmtN(answered)],
                   ["task/final/test.jsonl", phase === "final" ? fmtN(extra.nTest) : "—"]];
      var fl = el("div", { class: "run__files" });
      files.forEach(function (f) { fl.appendChild(el("div", {}, [mono(f[0]), txt(f[1], "faint")])); });
      mine.forEach(function (f) { fl.appendChild(el("div", { class: "is-mine" }, [mono(f), txt("agent", "faint")])); });
      p.appendChild(fl);
      return p;
    }

    function captionFor(c, k, all) {
      var r = c.result;
      if (r.error && c.tool === "query") return "The oracle is gone: query now returns an error naming the phase and arm, and costs nothing.";
      switch (c.tool) {
        case "query":
          if (r.results.every(function (x) { return "out" in x; }))
            return r.results.length + " answered — a bare atomic and the running example — at one query each, and appended to queries.jsonl.";
          return "None answered. A validation item, a held-out pair and an expression deeper than the demos are refused and cost nothing; the malformed one is charged, so q_used is " + r.q_used + ".";
        case "check_answers":
          if (c.args.set === "test") return "The answer file covers every row, probes included, and passes; the harness remembers it in case the run times out.";
          return r.ok ? "The full file passes. Checking is free, so it costs no submission."
                      : "A free legality check on a file with " + (PR.workspace["validation.jsonl"] - r.violations.missing_ids) + " of " + fmtN(PR.workspace["validation.jsonl"]) + " ids: illegal, and nothing is spent.";
        case "submit_validation_answer":
          return "Submission " + r.submission + " returns only overall (" + r3(r.overall) + ") and a by-depth split. The file is the skeleton-only interpreter's, so this is what structure alone earns on validation.";
        case "build_dataset":
          return r.size + " rows — the queries bought so far and the 30 demos. Each is labelled by provenance: " + r.provenance.purchased + " purchased, " + r.provenance.demo + " demo.";
        case "train_model":
          return r.continued_from ? "The same student_id continues that student from " + r.continued_from + "; the new checkpoint is " + r.checkpoint_id + "."
                                  : "A new student_id starts a fresh Qwen3-0.6B student; the call returns " + r.checkpoint_id + ".";
        case "infer_model":
          return c.phase === "final" ? "The student answers all " + fmtN(r.rows) + " rows, test and probe, in one call. (This replay commits the skeleton-only file instead, as in the other arm.)"
                                     : "A student_id means its latest checkpoint. " + fmtN(r.rows) + " answers are written to " + c.args.output_path + ", to be checked and submitted like any file.";
        case "finish_practice":
          return "finish_practice ends practice with " + r.submissions_remaining + " submissions left; the harness switches to the final phase.";
        case "submit_final_answer":
          return "Committed: the run ends here. The digest identifies the file that will be scored.";
      }
      return "";
    }

    function build(host, phase) {
      var tabs = ["no_train", "train"].map(function (arm) {
        var A = PR.arms[arm];
        var calls = A.calls.filter(function (c) { return c.phase === phase; });
        var before = phase === "final" ? A.calls.filter(function (c) { return c.phase === "practice"; }) : [];
        var steps = [];
        if (phase === "practice") steps.push({ caption: "Practice opens. The workspace is written and the oracle is open; nothing is spent yet.", n: 0, file: true });
        else steps.push({ caption: "At the switch the harness writes final/test.jsonl: " + fmtN(A.test_file.n_test) + " test rows and " + A.test_file.n_probe +
                          " probe rows, ids shuffled, no split labels. The practice tools close.", n: 0, file: true });
        calls.forEach(function (c, k) { steps.push({ caption: captionFor(c, k, calls), n: k + 1 }); });
        if (phase === "final") steps.push({ caption: "After the run the harness scores the file: overall " + r3(A.report.overall) +
          ". Headroom is 0 everywhere — the file knows every structural rule and no table entry, which is exactly the skeleton ceiling. Structural probes are all right, atomic ones all wrong.", n: calls.length, report: true });
        return {
          label: arm === "train" ? "train arm" : "no_train arm",
          steps: steps,
          render: function (body, i, st) {
            body.innerHTML = "";
            var shown = calls.slice(0, st.n);
            var log = el("div", { class: "run__log" });
            if (phase === "final") log.appendChild(fileCard("task/final/test.jsonl", A.test_file.head, fmtN(A.test_file.n) + " rows written", st.file));
            else log.appendChild(fileCard("task/validation.jsonl", PR.val_head, fmtN(PR.workspace["validation.jsonl"]) + " rows, no answers", st.file));
            shown.forEach(function (c, k) { log.appendChild(callCard(c, k === shown.length - 1 && !st.report ? "is-cur" : "is-past")); });
            if (st.report) log.appendChild(reportCard(A.report));
            body.appendChild(el("div", { class: "run" }, [log,
              statePanel(arm, phase, before.concat(shown), { nTest: A.test_file.n })]));
          }
        };
      });
      Stepper(host, tabs[0].steps, tabs[0].render, { tabs: tabs, interval: 2600 });
    }
    build(hostP, "practice");
    build(hostF, "final");
  })();
})();
