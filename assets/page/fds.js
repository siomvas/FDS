/* fds page: the explorer (hero preview and app window) and the Table 3 chart.
   The toy dataset runs through the assignment rules of Section 2 of the paper:
   label skew (Dirichlet over classes), geometric (k-means on the embedding, one cluster per client),
   semantic (a product of Dirichlets over criterion values), and a random split as the baseline. */
(function () {
  "use strict";
  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var PHONE = window.matchMedia("(max-width: 639px)").matches;
  var COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#5c5c5c"];
  var RGB = COLORS.map(function (h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; });
  var GREY = [205, 209, 214];
  var NF = new Intl.NumberFormat("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  function say(t) { var s = document.getElementById("fds-status"); if (s) { s.textContent = ""; setTimeout(function () { s.textContent = t; }, 30); } }
  var N = 720, C = 8;

  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(r) { var u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function gamma(r, a) {
    if (a < 1) return gamma(r, a + 1) * Math.pow(r(), 1 / a);
    var d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      var x, v; do { x = gauss(r); v = 1 + c * x; } while (v <= 0);
      v = v * v * v; var u = r();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  function dirichlet(r, a, n) {
    var g = [], s = 0, i;
    for (i = 0; i < n; i++) { var x = gamma(r, a); g.push(x); s += x; }
    if (!(s > 0) || !isFinite(s)) { g = g.map(function () { return 0; }); g[Math.floor(r() * n)] = 1; return g; }
    return g.map(function (x) { return x / s; });
  }

  /* the toy dataset: 8 classes; lighting and scene shape where an image sits in its class */
  var rd = mulberry32(20261209), centers = [], pts = [], i;
  for (i = 0; i < C; i++) { var col = i % 4, row = (i / 4) | 0; centers.push([0.13 + col * 0.247 + (rd() - 0.5) * 0.05, 0.27 + row * 0.46 + (rd() - 0.5) * 0.07]); }
  for (i = 0; i < N; i++) {
    var c = i % C, L = rd() < 0.5 ? 0 : 1, u = rd(), S = u < 0.4 ? 0 : (u < 0.75 ? 1 : 2);
    pts.push({ c: c, L: L, S: S, x: centers[c][0] + (S - 1) * 0.046 + gauss(rd) * 0.024, y: centers[c][1] + (L === 0 ? -0.062 : 0.062) + gauss(rd) * 0.028 });
  }

  function randomSplit(K, r) {
    var idx = pts.map(function (_, j) { return j; });
    for (var a = idx.length - 1; a > 0; a--) { var b = Math.floor(r() * (a + 1)), t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
    var out = new Array(N); idx.forEach(function (j, n) { out[j] = n % K; }); return out;
  }
  function labelSkew(K, alpha, r) {
    var out = new Array(N);
    for (var y = 0; y < C; y++) {
      var idx = []; pts.forEach(function (p, j) { if (p.c === y) idx.push(j); });
      for (var a = idx.length - 1; a > 0; a--) { var b = Math.floor(r() * (a + 1)), t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
      var q = dirichlet(r, alpha, K), cum = 0, start = 0;
      for (var k = 0; k < K; k++) {
        cum += q[k]; var end = k === K - 1 ? idx.length : Math.round(cum * idx.length);
        for (var j = start; j < end; j++) out[idx[j]] = k;
        start = end;
      }
    }
    return out;
  }
  function geometric(K, r) {
    var first = pts[Math.floor(r() * N)], cs = [[first.x, first.y]], d2 = new Array(N), k, j, it;
    while (cs.length < K) {
      var tot = 0;
      for (j = 0; j < N; j++) { var m = Infinity; for (k = 0; k < cs.length; k++) { var dx = pts[j].x - cs[k][0], dy = pts[j].y - cs[k][1]; m = Math.min(m, dx * dx + dy * dy); } d2[j] = m; tot += m; }
      var pick = r() * tot; for (j = 0; j < N; j++) { pick -= d2[j]; if (pick <= 0) break; }
      j = Math.min(j, N - 1); cs.push([pts[j].x, pts[j].y]);
    }
    var out = new Array(N);
    for (it = 0; it < 30; it++) {
      for (j = 0; j < N; j++) { var best = 0, bd = Infinity; for (k = 0; k < K; k++) { var ex = pts[j].x - cs[k][0], ey = pts[j].y - cs[k][1], dd = ex * ex + ey * ey; if (dd < bd) { bd = dd; best = k; } } out[j] = best; }
      var sx = [], sy = [], n = []; for (k = 0; k < K; k++) { sx.push(0); sy.push(0); n.push(0); }
      for (j = 0; j < N; j++) { sx[out[j]] += pts[j].x; sy[out[j]] += pts[j].y; n[out[j]]++; }
      for (k = 0; k < K; k++) if (n[k]) cs[k] = [sx[k] / n[k], sy[k] / n[k]];
    }
    /* order clients left to right so colours read the same across splits */
    var order = cs.map(function (c, k) { return [c[0] + c[1] * 0.2, k]; }).sort(function (a, b) { return a[0] - b[0]; }).map(function (p) { return p[1]; });
    var rank = []; order.forEach(function (k, n) { rank[k] = n; });
    return out.map(function (k) { return rank[k]; });
  }
  function semantic(K, alpha, r) {
    var pL = [], pS = [], k;
    for (k = 0; k < K; k++) { pL.push(dirichlet(r, alpha, 2)); pS.push(dirichlet(r, alpha, 3)); }
    return pts.map(function (p) {
      var w = [], s = 0; for (k = 0; k < K; k++) { var v = pL[k][p.L] * pS[k][p.S]; w.push(v); s += v; }
      if (!(s > 0)) return Math.floor(r() * K);
      var x = r() * s; for (k = 0; k < K; k++) { x -= w[k]; if (x <= 0) return k; } return K - 1;
    });
  }
  function split(mode, K, alpha, seed) {
    var r = mulberry32(seed * 7919 + K * 31 + Math.round(alpha * 1000) + mode.length * 101);
    if (mode === "random") return randomSplit(K, r);
    if (mode === "label") return labelSkew(K, alpha, r);
    if (mode === "geo") return geometric(K, r);
    return semantic(K, alpha, r);
  }

  /* a scatter: the dataset on a canvas, colours easing from the last split to the next */
  function Scatter(canvas, opts) {
    var ctx = canvas.getContext("2d"), W = 0, H = 0, raf = 0, M = opts.margin || 18;
    var col = pts.map(function () { return GREY.slice(); }), from = col.map(function (c) { return c.slice(); }), to = from.map(function (c) { return c.slice(); }), t0 = pts.map(function () { return 0; });
    function size() {
      var r = canvas.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
      W = r.width; H = r.height; canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
    }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      var s = W > 560 ? 6 : 4.6, h = s / 2;
      for (var j = 0; j < N; j++) {
        var p = pts[j], c = col[j];
        ctx.fillStyle = "rgb(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + ")";
        ctx.fillRect(M + p.x * (W - 2 * M) - h, M + p.y * (H - 2 * M) - h, s, s);
      }
    }
    function paint(assign, animate, done) {
      var now = performance.now();
      for (var j = 0; j < N; j++) { from[j] = col[j].slice(); to[j] = assign ? RGB[assign[j]] : GREY; t0[j] = now + (animate ? pts[j].x * 360 : 0); }
      cancelAnimationFrame(raf);
      var DUR = animate ? 520 : 0;
      (function f(n) {
        var busy = false;
        for (var j = 0; j < N; j++) {
          var t = DUR ? (n - t0[j]) / DUR : 1; if (t < 1) busy = true;
          t = Math.max(0, Math.min(1, t)); var e = 1 - Math.pow(1 - t, 3);
          for (var q = 0; q < 3; q++) col[j][q] = from[j][q] + (to[j][q] - from[j][q]) * e;
        }
        draw();
        if (busy) raf = requestAnimationFrame(f); else if (done) done();
      })(now);
    }
    if ("ResizeObserver" in window) new ResizeObserver(size).observe(canvas); else window.addEventListener("resize", size);
    size();
    return { paint: paint };
  }
  function counts(assign, K) { var n = []; for (var k = 0; k < K; k++) n.push(0); assign.forEach(function (k) { n[k]++; }); return n; }

  /* ---- the hero window: one geometric split, two rounds, then it rests ---- */
  var hc = document.getElementById("hero-canvas");
  if (hc) {
    var hs = Scatter(hc, { margin: 22 }), ha = split("geo", 5, 0, 7), hn = counts(ha, 5), list = document.getElementById("hero-list");
    list.innerHTML = hn.map(function (n, k) { return '<div class="fds-win__c" style="--c:' + COLORS[k] + '"><i></i><span>Client ' + (k + 1) + '</span><span>' + n + '</span></div>'; }).join("");
    if (REDUCED || PHONE) hs.paint(ha, false);
    else {
      var rounds = 0;
      (function round() {
        hs.paint(null, false);
        setTimeout(function () { hs.paint(ha, true, function () { if (++rounds < 2) setTimeout(round, 1300); }); }, 260);
      })();
    }
  }

  /* ---- the explorer window ---- */
  var cv = document.getElementById("fds-canvas");
  if (cv) {
    var sc = Scatter(cv, { margin: 20 }), state = { mode: "geo", K: 5, a: 0, seed: 7 }, started = false;
    var NAMES = { random: "Random", label: "Label skew", geo: "Geometric", sem: "Semantic" };
    var panel = document.getElementById("fds-clients"), title = document.getElementById("fds-open-title");
    function el(tag, a, p) { var n = document.createElementNS("http://www.w3.org/2000/svg", tag); for (var k in a) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; }
    function bars(assign) {
      var K = state.K, n = counts(assign, K), cls = [], look = [], k, j;
      for (k = 0; k < K; k++) { cls.push([0, 0, 0, 0, 0, 0, 0, 0]); look.push([0, 0, 0, 0, 0, 0]); }
      assign.forEach(function (k, j) { cls[k][pts[j].c]++; look[k][pts[j].L * 3 + pts[j].S]++; });
      var rowH = 62, w = 236, svg = el("svg", { viewBox: "0 0 " + w + " " + (26 + K * rowH), "aria-hidden": "true" });
      var ink = getComputedStyle(cv).getPropertyValue("--u-ink") || "#232323", mut = getComputedStyle(cv).getPropertyValue("--u-muted") || "#686868";
      el("text", { x: 0, y: 13, fill: mut, "font-size": 12 }, svg).textContent = "Classes";
      el("text", { x: 124, y: 13, fill: mut, "font-size": 12 }, svg).textContent = "Lighting and scene";
      for (k = 0; k < K; k++) {
        var y0 = 26 + k * rowH, m = n[k] || 1;
        el("rect", { x: 0, y: y0 + 2, width: 10, height: 10, rx: 2.5, fill: COLORS[k] }, svg);
        el("text", { x: 16, y: y0 + 11, fill: ink, "font-size": 12.5, "font-weight": 500 }, svg).textContent = "Client " + (k + 1) + ", " + n[k];
        for (j = 0; j < 8; j++) { var h = Math.min(30, cls[k][j] / m * 56); el("rect", { x: j * 13, y: y0 + 50 - h, width: 10, height: Math.max(1, h), rx: 1.5, fill: COLORS[k] }, svg); }
        for (j = 0; j < 6; j++) { var h2 = Math.min(30, look[k][j] / m * 54); el("rect", { x: 124 + j * 18, y: y0 + 50 - h2, width: 14, height: Math.max(1, h2), rx: 1.5, fill: COLORS[k], "fill-opacity": 0.7 }, svg); }
      }
      panel.innerHTML = ""; panel.appendChild(svg);
      var sr = document.createElement("p"); sr.className = "sr-only"; sr.style.cssText = "position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)";
      sr.textContent = n.map(function (v, k) { return "Client " + (k + 1) + ": " + v + " images"; }).join(", ") + ".";
      panel.appendChild(sr);
    }
    function apply(animate) {
      var a = split(state.mode, state.K, state.a, state.seed);
      sc.paint(a, animate && !REDUCED);
      bars(a);
      var label = NAMES[state.mode] + (state.a ? ", α = " + state.a : "") + ", " + state.K + " clients";
      title.textContent = label;
      cv.setAttribute("aria-label", "Toy dataset of 720 images split by " + label + ".");
      if (animate) say(label + ": client sizes " + counts(a, state.K).join(", ") + ".");
    }
    function key(b) { return b.dataset.mode + (b.dataset.a ? "-" + b.dataset.a : "") + "-" + b.dataset.k; }
    function setParam(name, v) { try { var u = new URL(location.href); u.searchParams.set(name, v); history.replaceState(null, "", u); } catch (e) {} }
    document.querySelectorAll(".fds-row").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll(".fds-row").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        state.mode = b.dataset.mode; state.K = +b.dataset.k; state.a = b.dataset.a ? +b.dataset.a : 0; apply(true);
        setParam("split", key(b));
      });
    });
    var want = new URLSearchParams(location.search).get("split");
    document.querySelectorAll(".fds-row").forEach(function (b) {
      if (want && key(b) === want) {
        document.querySelectorAll(".fds-row").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        state.mode = b.dataset.mode; state.K = +b.dataset.k; state.a = b.dataset.a ? +b.dataset.a : 0;
      }
    });
    document.getElementById("fds-resample").addEventListener("click", function () { state.seed++; apply(true); });
    bars(split(state.mode, state.K, state.a, state.seed)); title.textContent = NAMES[state.mode] + (state.a ? ", \u03b1 = " + state.a : "") + ", " + state.K + " clients";
    if ("IntersectionObserver" in window && !REDUCED) {
      var io = new IntersectionObserver(function (es) { if (es[0].isIntersecting && !started) { started = true; io.disconnect(); apply(true); } }, { threshold: 0.35 });
      io.observe(cv);
    } else apply(false);
  }

  /* ---- Table 3: points of FedAvg performance lost against a random split ---- */
  var T3 = {
    "Tiny ImageNet": { m: "accuracy", "l01": [2.79, 3.07, 2.55], "l1": [0.23, 0.44, 0.72], "s01": [0.84, 4.20, 1.30], "s1": [0.97, 3.31, 1.98], "g": [7.12, 9.79, 5.05] },
    "Food-101": { m: "accuracy", "l01": [5.85, 7.98, 6.19], "l1": [1.41, 1.54, 1.59], "s01": [4.23, 9.94, 6.42], "s1": [2.23, 2.77, 4.71], "g": [7.14, 11.28, 4.05] },
    "ADE20K": { m: "mIoU, segmentation", "s01": [2.10, 2.28, 3.45], "s1": [1.25, 1.20, 2.90], "g": [4.86, 7.88, 7.57] },
    "Pascal VOC": { m: "mAP@50, detection", "s01": [1.25, 5.54, 4.15], "s1": [1.28, 4.98, 2.92], "g": [5.06, 9.91, 4.04] }
  };
  var host = document.getElementById("t3-chart");
  if (host) {
    var K3 = 1, sec = document.getElementById("results");
    function css(n, d) { var v = getComputedStyle(sec).getPropertyValue(n).trim(); return v || d; }
    var SER = [
      { k: "l01", name: "Label skew, α = 0.1", tone: "muted", fill: true, shape: "c" },
      { k: "l1", name: "Label skew, α = 1", tone: "muted", fill: false, shape: "c" },
      { k: "s01", name: "Semantic, α = 0.1", tone: "ink", fill: true, shape: "d" },
      { k: "s1", name: "Semantic, α = 1", tone: "ink", fill: false, shape: "d" },
      { k: "g", name: "Geometric", tone: "accent", fill: true, shape: "s" }
    ];
    function el2(tag, a, p) { var n = document.createElementNS("http://www.w3.org/2000/svg", tag); for (var k in a) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; }
    function mark(g, s, cx, cy, colr, bg) {
      var a = { fill: s.fill ? colr : bg, stroke: colr, "stroke-width": 2 }, r = 6.5, n;
      if (s.shape === "c") n = el2("circle", Object.assign({ cx: cx, cy: cy, r: r }, a), g);
      else if (s.shape === "s") n = el2("rect", Object.assign({ x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, rx: 2 }, a), g);
      else n = el2("path", Object.assign({ d: "M" + cx + " " + (cy - r - 1.5) + "l" + (r + 1.5) + " " + (r + 1.5) + "l-" + (r + 1.5) + " " + (r + 1.5) + "l-" + (r + 1.5) + "-" + (r + 1.5) + "z" }, a), g);
      return n;
    }
    function render() {
      var ink = css("--ink", "#fff"), muted = css("--muted", "#a4a4a4"), line = css("--line", "rgba(255,255,255,.14)"), acc = css("--accent", "#a1ffcb"), bg = css("--bg", "#232323");
      var tone = { muted: muted, ink: ink, accent: acc };
      var W = Math.max(300, Math.round(host.clientWidth || 1280)), narrow = W < 640;
      var rows = Object.keys(T3), headH = (sec.querySelector('.t3-head') || host).offsetHeight + (document.getElementById('t3-legend') || host).offsetHeight + 64, avail = Math.max(0, window.innerHeight - 192 - headH), L = narrow ? 0 : Math.min(240, W * 0.2), R = narrow ? 52 : 70, top = narrow ? 8 : 20;
      var rowH = narrow ? 112 : Math.max(84, Math.min(140, avail ? (avail - top - 40) / rows.length : 112)), H = top + rows.length * rowH + 40, X1 = 12;
      function x(v) { return L + v / X1 * (W - L - R); }
      var svg = el2("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": "Points of FedAvg performance lost against a random split with " + [5, 10, 20][K3] + " clients. " + rows.map(function (ds) { var r = T3[ds]; return ds + ": geometric " + NF.format(r.g[K3]) + ", semantic alpha 0.1 " + NF.format(r.s01[K3]) + (r.l01 ? ", label skew alpha 0.1 " + NF.format(r.l01[K3]) : ", label skew not applicable"); }).join("; ") + "." });
      for (var v = 0; v <= X1; v += (narrow ? 4 : 2)) {
        el2("line", { x1: x(v), x2: x(v), y1: top - 6, y2: H - 34, stroke: v ? line : muted, "stroke-width": 1 }, svg);
        el2("text", { x: x(v), y: H - 10, "text-anchor": "middle", fill: muted, "font-size": 15 }, svg).textContent = v ? "\u2212" + v : "0";
      }
      rows.forEach(function (ds, i) {
        var r = T3[ds], cy = narrow ? top + i * rowH + 72 : top + i * rowH + rowH / 2;
        if (narrow) {
          el2("text", { x: 0, y: cy - 46, fill: ink, "font-size": 17, "font-weight": 500 }, svg).textContent = ds + ", " + r.m.split(",")[0];
        } else {
          el2("text", { x: 0, y: cy - 4, fill: ink, "font-size": 19, "font-weight": 500 }, svg).textContent = ds;
          el2("text", { x: 0, y: cy + 18, fill: muted, "font-size": 15 }, svg).textContent = r.m;
        }
        if (!r.l01) el2("text", { x: x(0) + 10, y: cy - 16, fill: muted, "font-size": 14 }, svg).textContent = "Label skew cannot be applied";
        SER.forEach(function (s) { if (r[s.k]) mark(svg, s, x(r[s.k][K3]), cy, tone[s.tone], bg); });
        el2("text", { x: x(r.g[K3]) + 14, y: cy + 5, fill: acc, "font-size": 16, "font-weight": 500 }, svg).textContent = NF.format(r.g[K3]);
      });
      host.innerHTML = ""; host.appendChild(svg);
      var lg = document.getElementById("t3-legend");
      lg.innerHTML = SER.map(function (s) {
        var c = tone[s.tone], shape = s.shape === "c" ? '<circle cx="8" cy="8" r="5.5"' : s.shape === "s" ? '<rect x="2.5" y="2.5" width="11" height="11" rx="2"' : '<path d="M8 1.5l6.5 6.5-6.5 6.5-6.5-6.5z"';
        return '<li><svg width="16" height="16" viewBox="0 0 16 16">' + shape + ' fill="' + (s.fill ? c : "none") + '" stroke="' + c + '" stroke-width="2"/></svg>' + s.name + "</li>";
      }).join("");
    }
    document.querySelectorAll(".t3-k button").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll(".t3-k button").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        K3 = +b.dataset.k; render();
        try { var u = new URL(location.href); u.searchParams.set("clients", [5, 10, 20][K3]); history.replaceState(null, "", u); } catch (e) {}
        say("Results with " + [5, 10, 20][K3] + " clients.");
      });
    });
    var wantK = { "5": 0, "10": 1, "20": 2 }[new URLSearchParams(location.search).get("clients")];
    if (wantK !== undefined) { K3 = wantK; document.querySelectorAll(".t3-k button").forEach(function (o) { o.setAttribute("aria-pressed", +o.dataset.k === K3 ? "true" : "false"); }); }
    render();
    if ("ResizeObserver" in window) { var lastW = 0, lastH = 0; new ResizeObserver(function () { var w = host.clientWidth, h = host.clientHeight; if (Math.abs(w - lastW) > 4 || Math.abs(h - lastH) > 4) { lastW = w; lastH = h; render(); } }).observe(host); }
  }

  /* copy buttons */
  document.querySelectorAll("[data-copy]").forEach(function (b) {
    b.addEventListener("click", function () {
      var src = document.getElementById(b.getAttribute("data-copy")), text = src ? src.textContent : "", label = b.textContent;
      function ok() { b.textContent = "Copied"; say("BibTeX copied."); setTimeout(function () { b.textContent = label; }, 1400); }
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(ok, function () {});
    });
  });
})();
