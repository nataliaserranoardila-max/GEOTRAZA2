/* =====================================================================
   GEOTRAZA · lógica de la interfaz
   - Pide la inferencia al servidor (/api/predict) y recibe, por modelo:
     mapa semántico (PNG de etiquetas), panóptico coloreado, mapa de ids de
     segmento, mapa de confianza, estadísticas y métricas (si hay GT).
   - Compone todo en el navegador con <canvas> para poder activar/desactivar
     clases, cambiar opacidad y comparar con el deslizador sin volver a
     llamar al servidor.
   ===================================================================== */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const el = (tag, attrs = {}, html = "") => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (html) e.innerHTML = html;
    return e;
  };
  const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
  const pct = (v, d = 1) => (v * 100).toFixed(d) + " %";
  const TARGETS = { mIoU: 0.7, mAP: 0.5, mF1: 0.8 };

  const S = {
    classes: [], models: {}, mode: "native", source: null, view: "semantic",
    enabled: new Set([1, 2, 3, 4]), alpha: 0.6, split: 50,
    scene: null, layers: {}, base: null, w: 0, h: 0, samples: [], selected: null, gtFile: null,
  };

  // ------------------------------------------------------------------
  // Estado del servidor
  // ------------------------------------------------------------------
  async function loadStatus() {
    try {
      const r = await fetch("/api/status");
      const st = await r.json();
      S.classes = st.classes;
      S.models = st.models;
      const box = $("#status");
      box.innerHTML = "";
      for (const name of ["Mask2Former", "EfficientPS"]) {
        const m = st.models[name];
        const p = el("span", { class: "pill " + (m ? "pill-on" : "pill-off"),
          title: m ? `Época ${m.epoch ?? "?"} · ${m.params_m} M parámetros · ${m.size_mb} MB` : "Pesos no encontrados en models/weights/" },
          m ? `${name}` : `${name} sin pesos`);
        box.append(p);
      }
      box.append(el("span", { class: "pill pill-on" }, st.device.toUpperCase()));
      buildLegend();
      buildSources();
    } catch (e) {
      $("#status").innerHTML = '<span class="pill pill-off">Servidor no disponible</span>';
    }
  }

  function buildLegend() {
    const lg = $("#legend");
    lg.innerHTML = "";
    for (const c of S.classes) {
      const on = S.enabled.has(c.id);
      const b = el("button", { class: "chip", "aria-pressed": on, "data-cls": c.id },
        `<i style="background:rgb(${c.color.join(",")})"></i>${c.name}`);
      b.onclick = () => {
        S.enabled.has(c.id) ? S.enabled.delete(c.id) : S.enabled.add(c.id);
        b.setAttribute("aria-pressed", S.enabled.has(c.id));
        render();
      };
      lg.append(b);
    }
  }

  function buildSources() {
    const box = $("#sources");
    box.innerHTML = "";
    const names = Object.keys(S.models);
    const opts = [...names];
    if (names.length === 2) opts.push("Comparar");
    opts.push("Ground truth");
    const noGT = !(S.scene && S.scene.gt);
    if (!S.source || !opts.includes(S.source) || (S.source === "Ground truth" && noGT && names.length))
      S.source = names[0] || "Ground truth";  // sin ground truth no se puede quedar en la pestaña GT
    for (const o of opts) {
      const b = el("button", { role: "tab", "aria-selected": o === S.source, "data-src": o }, o);
      if (o === "Ground truth") b.disabled = !(S.scene && S.scene.gt);
      b.onclick = () => { S.source = o; syncTabs(); render(true); };
      box.append(b);
    }
  }

  function syncTabs() {
    document.querySelectorAll("#sources button").forEach((b) => {
      b.setAttribute("aria-selected", b.dataset.src === S.source);
      if (b.dataset.src === "Ground truth") b.disabled = !(S.scene && S.scene.gt);
    });
    const hasGT = !!(S.scene && S.scene.gt);
    document.querySelectorAll("#views button").forEach((b) => {
      const v = b.dataset.view;
      let dis = false;
      if (v === "errors") dis = !hasGT || S.source === "Ground truth";
      if (v === "confidence" || v === "detections") dis = S.source === "Ground truth";
      b.disabled = dis;
      b.title = !dis ? "" : S.source === "Ground truth"
        ? "Elige Mask2Former, EfficientPS o Comparar para usar esta vista"
        : "Requiere una imagen con ground truth (escenas de test/validación o un .npy adjunto)";
      if (dis && S.view === v) S.view = "semantic";
    });
    document.querySelectorAll("#views button").forEach((b) => b.setAttribute("aria-selected", b.dataset.view === S.view));
  }

  // ------------------------------------------------------------------
  // Galería de escenas
  // ------------------------------------------------------------------
  async function loadSamples() {
    try {
      S.samples = await (await fetch("/api/samples")).json();
    } catch { S.samples = []; }
    drawGallery();
  }

  function drawGallery() {
    const f = $("#filter").value;
    const g = $("#gallery");
    g.innerHTML = "";
    const list = S.samples.filter((s) => f === "all" || s.split === f);
    if (!list.length) {
      g.innerHTML = S.samples.length
        ? '<p class="hint">No hay escenas en este grupo.</p>'
        : '<p class="hint">No se encontró el dataset en data/raw/. Puedes subir imágenes igualmente.</p>';
      return;
    }
    for (const s of list) {
      const label = s.kind === "seg" ? `#${s.id}` : s.kind === "ext" ? s.name.replace(/^bucaramanga_/i, "").replace(/\.(png|jpe?g|tiff?)$/i, "").replace(/_/g, " ")
        : s.name.replace(/^Recorte\s*/i, "").replace(/\.png$/i, "");
      const b = el("button", { class: "thumb", title: `${s.name} · ${s.split}`,
        "aria-pressed": S.selected === s.kind + s.name, "aria-label": `Analizar ${s.name}` },
        `<img loading="lazy" src="/api/thumb/${s.kind}/${encodeURIComponent(s.name)}" alt=""><span>${label}</span>`);
      b.onclick = () => { S.selected = s.kind + s.name; drawGallery(); runSample(s); };
      g.append(b);
    }
  }

  // ------------------------------------------------------------------
  // Inferencia
  // ------------------------------------------------------------------
  function busy(on, text) {
    $("#busy").hidden = !on;
    if (text) $("#busytext").textContent = text;
  }

  async function predict(fd, label) {
    if (!Object.keys(S.models).length) {
      showError("No hay modelos cargados. Copia mask2former_best.pt y efficientps_best.pt en models/weights/ y reinicia la app.");
      return;
    }
    fd.append("mode", S.mode);
    fd.append("scale", $("#scale").value);
    busy(true, S.mode === "tta" ? `Analizando ${label} con TTA ×8…` : `Analizando ${label}…`);
    $("#empty").hidden = true;
    try {
      const r = await fetch("/api/predict", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Error en la inferencia");
      await loadScene(data);
    } catch (e) {
      showError(e.message);
    } finally { busy(false); }
  }

  function showError(msg) {
    $("#readout").innerHTML = `<b style="color:var(--bad)">${msg}</b>`;
  }

  function runSample(s) {
    const fd = new FormData();
    fd.append("sample_kind", s.kind);
    fd.append("sample_name", s.name);
    predict(fd, s.kind === "seg" ? `escena #${s.id}` : s.name);
  }

  function runFile(file) {
    const fd = new FormData();
    fd.append("file", file);
    if (S.gtFile) fd.append("gt_file", S.gtFile);
    S.selected = null; drawGallery();
    predict(fd, file.name);
  }

  // ------------------------------------------------------------------
  // Decodificación de capas
  // ------------------------------------------------------------------
  const loadImg = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });

  function pixels(img, w, h) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const x = c.getContext("2d", { willReadFrequently: true });
    x.imageSmoothingEnabled = false;
    x.drawImage(img, 0, 0, w, h);
    return x.getImageData(0, 0, w, h).data;
  }

  function channel(px) { // R de un RGBA -> Uint8Array
    const n = px.length / 4, out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = px[i * 4];
    return out;
  }

  async function loadScene(data) {
    S.scene = data;
    const base = await loadImg(data.image);
    S.w = data.width; S.h = data.height;
    S.base = pixels(base, S.w, S.h);
    S.layers = {};
    for (const [name, r] of Object.entries(data.results)) {
      const [sem, pan, seg, conf] = await Promise.all([r.semantic, r.panoptic, r.segments_map, r.confidence].map(loadImg));
      const segPx = pixels(seg, S.w, S.h);
      const ids = new Uint32Array(S.w * S.h);
      for (let i = 0; i < ids.length; i++) ids[i] = segPx[i * 4] + 256 * segPx[i * 4 + 1];
      const segIndex = {};
      r.segments.forEach((s) => (segIndex[s.id] = s));
      // detección: caja envolvente de cada instancia "thing" (manzanas y zonas verdes)
      const bb = {};
      for (let i = 0; i < ids.length; i++) {
        const sg = segIndex[ids[i]];
        if (!sg || !sg.thing) continue;
        const x = i % S.w, y = (i / S.w) | 0, b = bb[sg.id];
        if (!b) bb[sg.id] = [x, y, x, y];
        else { if (x < b[0]) b[0] = x; if (y < b[1]) b[1] = y; if (x > b[2]) b[2] = x; if (y > b[3]) b[3] = y; }
      }
      const minArea = Math.max(150, S.w * S.h * 0.0004);
      const boxes = Object.entries(bb).map(([id, b]) => ({ ...segIndex[id], box: b }))
        .filter((d) => d.area >= minArea).sort((a, b) => b.score - a.score);
      S.layers[name] = { labels: channel(pixels(sem, S.w, S.h)), pan: pixels(pan, S.w, S.h),
        conf: channel(pixels(conf, S.w, S.h)), ids, segIndex, boxes };
    }
    S.layers.gt = data.gt ? { labels: channel(pixels(await loadImg(data.gt), S.w, S.h)) } : null;

    const img = $("#base");
    img.src = data.image;
    await img.decode().catch(() => {});
    for (const c of [$("#under"), $("#over")]) { c.width = S.w; c.height = S.h; }
    $("#frame").hidden = false;
    $("#dlimg").disabled = $("#dljson").disabled = false;
    buildSources();
    syncTabs();
    render(true);
    renderTelemetry();
    $("#readout").textContent = "Pasa el cursor sobre la escena para leer cada píxel.";
  }

  // ------------------------------------------------------------------
  // Composición
  // ------------------------------------------------------------------
  const INFERNO = (() => { // LUT aproximada de 'inferno'
    const stops = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * (stops.length - 1), a = Math.floor(t), b = Math.min(a + 1, stops.length - 1), f = t - a;
      for (let k = 0; k < 3; k++) lut[i * 3 + k] = stops[a][k] + (stops[b][k] - stops[a][k]) * f;
    }
    return lut;
  })();

  function composite(layer, view) {
    const out = new ImageData(S.w, S.h);
    const o = out.data, b = S.base, a = S.alpha, ia = 1 - a;
    const cols = S.classes.map((c) => c.color);
    const gt = S.layers.gt && S.layers.gt.labels;
    const L = layer.labels, n = S.w * S.h;
    for (let i = 0; i < n; i++) {
      const j = i * 4, cls = L[i];
      let r = b[j], g = b[j + 1], bl = b[j + 2];
      if (view === "confidence") {
        const v = layer.conf[i] * 3;
        r = r * ia + INFERNO[v] * a; g = g * ia + INFERNO[v + 1] * a; bl = bl * ia + INFERNO[v + 2] * a;
      } else if (view === "errors") {
        const ok = gt[i] === cls;
        const lum = (r * 0.3 + g * 0.59 + bl * 0.11) * 0.4;
        if (ok) { r = lum * ia + 40 * a; g = lum * ia + 200 * a; bl = lum * ia + 120 * a; }
        else { r = lum * 0.3 + 255 * 0.7; g = lum * 0.3 + 60 * 0.7; bl = lum * 0.3 + 60 * 0.7; }
      } else if (view === "detections") {
        if (S.enabled.has(cls) && cls !== 0) {
          const c = cols[cls], k = a * 0.45;
          r = r * (1 - k) + c[0] * k; g = g * (1 - k) + c[1] * k; bl = bl * (1 - k) + c[2] * k;
        }
      } else if (S.enabled.has(cls)) {
        const src = view === "panoptic" && layer.pan ? [layer.pan[j], layer.pan[j + 1], layer.pan[j + 2]] : cols[cls];
        r = r * ia + src[0] * a; g = g * ia + src[1] * a; bl = bl * ia + src[2] * a;
      }
      o[j] = r; o[j + 1] = g; o[j + 2] = bl; o[j + 3] = 255;
    }
    return out;
  }

  function originalData() {
    const d = new ImageData(S.w, S.h);
    d.data.set(S.base);
    return d;
  }

  function render(animate = false) {
    if (!S.scene) return;
    const names = Object.keys(S.scene.results);
    let left, right, lblL, lblR;
    if (S.source === "Comparar") {
      left = composite(S.layers[names[0]], S.view);
      right = composite(S.layers[names[1]], S.view);
      lblL = names[0]; lblR = names[1];
    } else if (S.source === "Ground truth") {
      left = originalData();
      right = composite(S.layers.gt, "semantic");
      lblL = "Original"; lblR = "Ground truth";
    } else {
      left = originalData();
      right = composite(S.layers[S.source], S.view);
      lblL = "Original"; lblR = S.source;
    }
    $("#under").getContext("2d").putImageData(right, 0, 0);
    $("#over").getContext("2d").putImageData(left, 0, 0);
    if (S.view === "detections") {
      if (S.source === "Comparar") { drawBoxes($("#over"), S.layers[names[0]]); drawBoxes($("#under"), S.layers[names[1]]); }
      else if (S.layers[S.source]) drawBoxes($("#under"), S.layers[S.source]);
    }
    $("#lblL").textContent = lblL; $("#lblR").textContent = lblR;
    applySplit();
    if (animate) reveal();
  }

  function drawBoxes(canvas, layer) {
    const ctx = canvas.getContext("2d");
    const lw = Math.max(2, Math.round(S.w / 450)), fs = Math.max(12, Math.round(S.w / 70));
    ctx.font = `600 ${fs}px Archivo, system-ui, sans-serif`;
    ctx.textBaseline = "top";
    for (const d of layer.boxes) {
      if (!S.enabled.has(d.cls)) continue;
      const [x1, y1, x2, y2] = d.box, col = `rgb(${S.classes[d.cls].color})`;
      ctx.lineWidth = lw; ctx.strokeStyle = "rgba(0,0,0,.55)";
      ctx.strokeRect(x1 - 0.5, y1 - 0.5, x2 - x1 + 2, y2 - y1 + 2);
      ctx.strokeStyle = col; ctx.strokeRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
      const name = d.cls === 2 ? "Construida" : "Verde";
      const label = `${name} #${d.id} · ${Math.round(d.score * 100)}%`;
      const tw = ctx.measureText(label).width + 8, th = fs + 6;
      const ly = y1 - th >= 0 ? y1 - th : y1;
      ctx.fillStyle = col; ctx.fillRect(x1, ly, tw, th);
      ctx.fillStyle = "#0b1a2c"; ctx.fillText(label, x1 + 4, ly + 3);
    }
  }

  function applySplit() {
    $("#over").style.clipPath = `inset(0 ${100 - S.split}% 0 0)`;
    $("#divider").style.left = `calc(${S.split}% - 1px)`;
  }

  function reveal() {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const u = $("#under"), sc = $("#scan");
    u.classList.remove("revealing"); sc.classList.remove("run");
    void u.offsetWidth;
    sc.style.setProperty("--scan-h", $("#frame").clientHeight + "px");
    u.classList.add("revealing"); sc.classList.add("run");
  }

  // ------------------------------------------------------------------
  // Lectura de píxel bajo el cursor
  // ------------------------------------------------------------------
  function onHover(e) {
    if (!S.scene) return;
    const rect = $("#frame").getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * S.w);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * S.h);
    if (x < 0 || y < 0 || x >= S.w || y >= S.h) return;
    const i = y * S.w + x;
    const parts = [`<span>x <b>${x}</b> · y <b>${y}</b></span>`];
    for (const name of Object.keys(S.scene.results)) {
      const L = S.layers[name], c = S.classes[L.labels[i]], seg = L.segIndex[L.ids[i]];
      const inst = seg && seg.thing ? ` · instancia <b>#${seg.id}</b> (puntaje ${fmt(seg.score, 2)})` : "";
      parts.push(`<span>${name}: <i class="sw" style="background:rgb(${c.color})"></i><b>${c.name}</b> · confianza ${Math.round(L.conf[i] / 2.55)} %${inst}</span>`);
    }
    if (S.layers.gt) {
      const c = S.classes[S.layers.gt.labels[i]];
      parts.push(`<span>GT: <i class="sw" style="background:rgb(${c.color})"></i><b>${c.name}</b></span>`);
    }
    $("#readout").innerHTML = parts.join("");
  }

  // ------------------------------------------------------------------
  // Telemetría (estadísticas y métricas de la escena actual)
  // ------------------------------------------------------------------
  function gauge(label, v, target) {
    const cls = target == null ? "" : v >= target ? "pass" : "fail";
    const t = target == null ? "" : `<div class="t">${v >= target ? "cumple" : "bajo"} el mínimo ${target.toFixed(2)}</div>`;
    return `<div class="gauge ${cls}"><div class="v">${fmt(v)}</div><div class="k">${label}</div>${t}</div>`;
  }

  function coverBar(cov) {
    const segs = S.classes.map((c) => `<div style="width:${(cov[c.name] * 100).toFixed(2)}%;background:rgb(${c.color})" title="${c.name} ${pct(cov[c.name])}"></div>`).join("");
    const keys = S.classes.map((c) => `<span><i class="sw" style="background:rgb(${c.color})"></i>${c.name}<b>${pct(cov[c.name])}</b></span>`).join("");
    return `<div class="cover"><div class="cover-bar">${segs}</div><div class="cover-keys">${keys}</div></div>`;
  }

  function classTable(m) {
    const rows = Object.entries(m.per_class).map(([n, v]) =>
      `<tr><td>${n}</td><td>${fmt(v.IoU)}</td><td>${fmt(v.Precision)}</td><td>${fmt(v.Recall)}</td><td>${fmt(v.F1)}</td><td>${fmt(v.AP)}</td><td>${fmt(v.AP50)}</td><td>${fmt(v.PQ)}</td></tr>`).join("");
    return `<table class="mt"><thead><tr><th>Clase</th><th>IoU</th><th>Precisión</th><th>Recall</th><th>F1</th><th>AP</th><th>AP50</th><th>PQ</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function protoText(p) {
    return p === "void" ? "Métricas con píxeles sin anotar ignorados (protocolo void)." : "Métricas con el Fondo como clase (protocolo estricto).";
  }

  function renderTelemetry() {
    const d = S.scene;
    $("#telemetry").hidden = false;
    $("#scenemeta").textContent = `${d.name} · ${d.width}×${d.height} px · modo ${d.mode === "tta" ? "TTA ×8" : "nativo"}` +
      (d.gt ? " · con ground truth" : " · sin ground truth (solo estadísticas)");
    const grid = $("#telegrid");
    grid.innerHTML = "";
    for (const [name, r] of Object.entries(d.results)) {
      const st = r.stats, m = r.metrics;
      let html = `<h3>${name}<small>${r.time_ms.toFixed(0)} ms</small></h3>`;
      if (m) {
        html += `<div class="gauges">${gauge("IoU medio", m.mIoU, TARGETS.mIoU)}${gauge("mAP@[.5:.95]", m.mAP, TARGETS.mAP)}${gauge("F1 medio", m.mF1, TARGETS.mF1)}${gauge("Precisión media", m.mPrecision)}</div>`;
        html += `<div class="facts"><span>mAP50 <b>${fmt(m.mAP50)}</b></span><span>mAP things <b>${fmt(m.mAP_things)}</b></span><span>PQ <b>${fmt(m.PQ)}</b></span><span>Exactitud de píxel <b>${fmt(m.PixelAccuracy)}</b></span></div>`;
        if (m.alt) html += `<p class="hint">${protoText(m.protocol)} Con el protocolo ${m.alt_protocol === "strict" ? "estricto (Fondo como clase)" : "void"}: mIoU <b>${fmt(m.alt.mIoU)}</b> · mF1 <b>${fmt(m.alt.mF1)}</b> · mAP <b>${fmt(m.alt.mAP)}</b>.</p>`;
        html += classTable(m);
      } else {
        html += `<p class="hint">Esta escena no tiene ground truth, así que no hay métricas de exactitud. Elige una escena de test o adjunta su .npy para calcularlas.</p>`;
      }
      html += coverBar(st.coverage);
      const L = S.layers[name];
      const det = L && L.boxes ? `<span>Detecciones visibles: <b>${L.boxes.filter((d) => d.cls === 2).length}</b> manzanas · <b>${L.boxes.filter((d) => d.cls === 3).length}</b> zonas verdes</span>` : "";
      const inst = Object.entries(st.instances).map(([k, v]) => `<span>${k}: <b>${v}</b> instancias</span>`).join("") + det;
      html += `<div class="facts">${inst}<span>Confianza media <b>${pct(st.mean_confidence)}</b></span><span>Píxeles dudosos (&lt;60 %) <b>${pct(st.low_conf_fraction)}</b></span></div>`;
      grid.append(el("div", { class: "unit" }, html));
    }
    const ag = $("#agree");
    if (d.comparison) {
      const c = d.comparison;
      const rows = Object.entries(c.class_agreement_iou).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`).join("");
      ag.innerHTML = `<img src="${c.disagreement}" alt="Mapa de desacuerdo entre modelos">
        <div><h3>Acuerdo entre modelos: ${pct(c.agreement)}</h3>
        <p class="hint">En magenta, los píxeles donde Mask2Former y EfficientPS asignan clases distintas. Suelen concentrarse en bordes de andenes y copas de árboles.</p>
        <table class="mt"><thead><tr><th>Clase</th><th>IoU entre modelos</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    } else ag.innerHTML = "";
  }

  // ------------------------------------------------------------------
  // Banco de pruebas (métricas de test + curvas de entrenamiento)
  // ------------------------------------------------------------------
  const MODEL_COL = { Mask2Former: "#1565c0", EfficientPS: "#e07b00" };

  function lineChart(title, series, { yMax = null, yMin = 0, refs = [] } = {}) {
    const W = 520, H = 230, P = { l: 42, r: 12, t: 10, b: 28 };
    const all = series.flatMap((s) => s.y.filter((v) => v != null && isFinite(v)));
    const xs = series.flatMap((s) => s.x);
    if (!all.length) return "";
    const ymax = yMax ?? Math.max(...all) * 1.05, ymin = yMin ?? Math.min(...all);
    const xmax = Math.max(...xs), xmin = Math.min(...xs);
    const X = (v) => P.l + ((v - xmin) / Math.max(xmax - xmin, 1)) * (W - P.l - P.r);
    const Y = (v) => H - P.b - ((v - ymin) / Math.max(ymax - ymin, 1e-9)) * (H - P.t - P.b);
    let g = "";
    for (let k = 0; k <= 4; k++) {
      const v = ymin + ((ymax - ymin) * k) / 4;
      g += `<line class="axis" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}" opacity=".5"/><text x="${P.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
    }
    g += `<text x="${(W + P.l) / 2}" y="${H - 6}" text-anchor="middle">época</text><text x="${P.l}" y="${H - 6}">${xmin}</text><text x="${W - P.r}" y="${H - 6}" text-anchor="end">${xmax}</text>`;
    for (const r of refs) g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(r.v)}" y2="${Y(r.v)}" stroke="${r.c}" stroke-dasharray="4 4" opacity=".7"/>`;
    for (const s of series) {
      const pts = s.x.map((x, i) => [x, s.y[i]]).filter((p) => p[1] != null && isFinite(p[1]));
      g += `<polyline fill="none" stroke="${s.c}" stroke-width="2" ${s.dash ? 'stroke-dasharray="5 4"' : ""} points="${pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")}"/>`;
    }
    const lg = series.map((s) => `<span><i style="background:${s.c};${s.dash ? "opacity:.7" : ""}"></i>${s.name}</span>`).join("");
    const ci = CHARTS.push({ series, X, Y, xmin, xmax, P, W, H }) - 1;
    g += `<line class="guide" x1="0" x2="0" y1="${P.t}" y2="${H - P.b}" visibility="hidden"/>`;
    return `<div class="chart"><h3>${title}</h3><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}" data-ci="${ci}">${g}</svg><div class="tip" hidden></div><div class="lg">${lg}</div></div>`;
  }

  // lectura de valores al pasar el mouse por una curva
  const CHARTS = [];
  document.addEventListener("mousemove", (e) => {
    const svg = e.target.closest && e.target.closest(".chart svg[data-ci]");
    document.querySelectorAll(".chart .tip:not([hidden])").forEach((t) => { if (!svg || t.parentElement !== svg.parentElement) { t.hidden = true; t.parentElement.querySelector(".guide").setAttribute("visibility", "hidden"); } });
    if (!svg) return;
    const c = CHARTS[+svg.dataset.ci], r = svg.getBoundingClientRect();
    const sx = ((e.clientX - r.left) / r.width) * c.W;
    if (sx < c.P.l || sx > c.W - c.P.r) return;
    const xv = c.xmin + ((sx - c.P.l) / (c.W - c.P.l - c.P.r)) * (c.xmax - c.xmin);
    let html = "", gx = null;
    for (const s of c.series) {
      let bi = -1, bd = Infinity;
      s.x.forEach((x, i) => { const dd = Math.abs(x - xv); if (dd < bd && s.y[i] != null && isFinite(s.y[i])) { bd = dd; bi = i; } });
      if (bi < 0) continue;
      gx = gx ?? s.x[bi];
      html += `<div><i style="background:${s.c}"></i>${s.name}: <b>${Number(s.y[bi]).toFixed(4)}</b></div>`;
    }
    if (gx == null) return;
    const tip = svg.parentElement.querySelector(".tip"), guide = svg.querySelector(".guide");
    guide.setAttribute("x1", c.X(gx)); guide.setAttribute("x2", c.X(gx)); guide.setAttribute("visibility", "visible");
    tip.innerHTML = `<div>Época <b>${gx}</b></div>` + html;
    tip.hidden = false;
    const pr = svg.parentElement.getBoundingClientRect();
    const left = e.clientX - pr.left + 14, top = e.clientY - pr.top - 10;
    tip.style.left = Math.min(left, pr.width - tip.offsetWidth - 8) + "px";
    tip.style.top = top + "px";
  });

  function barChart(title, groups, models) {
    const W = 520, H = 230, P = { l: 36, r: 8, t: 14, b: 34 };
    const gw = (W - P.l - P.r) / groups.length, bw = (gw * 0.7) / models.length;
    const Y = (v) => H - P.b - v * (H - P.t - P.b);
    let g = "";
    for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line class="axis" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}" opacity=".5"/><text x="${P.l - 5}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
    groups.forEach((grp, gi) => {
      const x0 = P.l + gi * gw + gw * 0.15;
      models.forEach((m, mi) => {
        const v = grp.values[m] ?? 0, x = x0 + mi * bw;
        g += `<rect x="${x}" y="${Y(v)}" width="${bw - 2}" height="${Y(0) - Y(v)}" fill="${MODEL_COL[m]}" rx="2"><title>${m} · ${grp.label}: ${v.toFixed(3)}</title></rect>`;
        g += `<text x="${x + bw / 2 - 1}" y="${Y(v) - 4}" text-anchor="middle">${v.toFixed(2)}</text>`;
      });
      g += `<text x="${x0 + (bw * models.length) / 2}" y="${H - 12}" text-anchor="middle">${grp.label}</text>`;
    });
    const lg = models.map((m) => `<span><i style="background:${MODEL_COL[m]}"></i>${m}</span>`).join("");
    return `<div class="chart"><h3>${title}</h3><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">${g}</svg><div class="lg">${lg}</div></div>`;
  }

  async function loadBenchmark() {
    let b;
    try { b = await (await fetch("/api/benchmark")).json(); } catch { return; }
    const models = Object.keys(b.test);
    const body = $("#benchbody");
    if (!models.length && !Object.keys(b.history).length) return;
    let html = "";
    if (models.length) {
      const rows = [];
      for (const m of models) for (const [mode, r] of Object.entries(b.test[m])) rows.push({ m, mode, r });
      const best = (k) => Math.max(...rows.map((x) => x.r[k]));
      const cell = (r, k, t) => `<td class="${t == null ? "" : r[k] >= t ? "pass" : "fail"}">${fmt(r[k])}</td>`;
      html += `<table class="mt"><thead><tr><th>Modelo</th><th>Modo</th><th>IoU medio</th><th>mAP</th><th>mAP50</th><th>mAP things</th><th>F1 medio</th><th>Precisión</th><th>Recall</th><th>PQ</th><th>ms / imagen</th><th>mIoU estricto</th><th>mAP estricto</th></tr></thead><tbody>` +
        rows.map(({ m, mode, r }) => `<tr class="${r.mIoU === best("mIoU") ? "best" : ""}"><td><i class="sw" style="background:${MODEL_COL[m]}"></i>${m}</td><td>${mode === "tta" ? "TTA ×8" : "Nativo"}</td>${cell(r, "mIoU", .7)}${cell(r, "mAP", .5)}${cell(r, "mAP50", .5)}${cell(r, "mAP_things", .5)}${cell(r, "mF1", .8)}${cell(r, "mPrecision")}${cell(r, "mRecall")}${cell(r, "PQ")}<td>${r.mean_time_ms ? r.mean_time_ms.toFixed(0) : "—"}</td><td>${r.alt && r.alt_protocol === "strict" ? fmt(r.alt.mIoU) : r.protocol === "strict" ? fmt(r.mIoU) : "—"}</td><td>${r.alt && r.alt_protocol === "strict" ? fmt(r.alt.mAP) : r.protocol === "strict" ? fmt(r.mAP) : "—"}</td></tr>`).join("") +
        `</tbody></table>`;
      const cls = S.classes.filter((c) => c.id > 0).map((c) => c.name);
      const pick = (m) => b.test[m].tta || b.test[m].native;
      html += `<div class="bench-grid">`;
      html += barChart("IoU por clase en test (mejor modo)", cls.map((c) => ({ label: c.replace("Área ", ""), values: Object.fromEntries(models.map((m) => [m, pick(m).per_class[c].IoU])) })), models);
      html += barChart("F1 por clase en test (mejor modo)", cls.map((c) => ({ label: c.replace("Área ", ""), values: Object.fromEntries(models.map((m) => [m, pick(m).per_class[c].F1])) })), models);
      html += `</div>`;
      renderKpis(rows);
    }
    if (html) body.innerHTML = html;

    // curvas de entrenamiento -> sección Entrenamiento
    const H = b.history;
    const hm = Object.keys(H);
    let curves = "";
    for (const m of hm) {
      curves += lineChart(`${m}: pérdida de entrenamiento y validación`, [
        { name: "Entrenamiento", x: H[m].epoch, y: H[m].train_loss, c: MODEL_COL[m] },
        { name: "Validación (prueba)", x: H[m].epoch, y: H[m].val_loss, c: "#5a6b80", dash: true }], { yMin: null });
    }
    if (hm.length) {
      curves += lineChart("IoU medio en validación por época", hm.map((m) => ({ name: m, x: H[m].eval_epoch, y: H[m].val_mIoU, c: MODEL_COL[m] })),
        { yMax: 1, yMin: 0, refs: [{ v: 0.7, c: "#1e9e57" }] });
      curves += lineChart("F1 medio en validación por época", hm.map((m) => ({ name: m, x: H[m].eval_epoch, y: H[m].val_mF1, c: MODEL_COL[m] })),
        { yMax: 1, yMin: 0, refs: [{ v: 0.8, c: "#1e9e57" }] });
      $("#traincurves").innerHTML = curves;
    }
  }

  // KPIs de la cabecera: mejor configuración en test
  function renderKpis(rows) {
    const best = rows.slice().sort((a, b) => b.r.mIoU - a.r.mIoU)[0];
    if (!best) return;
    const r = best.r;
    const k = (v, label, t) => `<div class="kpi"><div class="v">${fmt(v)}</div><div class="k">${label}</div>` +
      (t == null ? "" : `<div class="t ${v >= t ? "pass" : "fail"}">${v >= t ? "cumple" : "bajo"} el mínimo ${t}</div>`) + `</div>`;
    $("#herokpis").innerHTML =
      k(r.mIoU, `IoU medio · ${best.m} ${best.mode === "tta" ? "TTA" : "nativo"}`, 0.7) +
      k(r.mF1, "F1 medio", 0.8) + k(r.mAP, "mAP@[.5:.95]", 0.5) + k(r.mPrecision, "Precisión media") + k(r.PQ, "Calidad panóptica (PQ)");
    countUp($("#herokpis"));
  }

  // las cifras suben desde 0 la primera vez que se ven
  function countUp(box) {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const vals = [...box.querySelectorAll(".v")].map((n) => [n, parseFloat(n.textContent)]);
    vals.forEach(([n]) => (n.textContent = "0.000"));
    let done = false;
    const run = () => {
      if (done) return;
      done = true;
      const t0 = performance.now();
      const step = (t) => {
        const p = Math.min((t - t0) / 1100, 1), e = 1 - Math.pow(1 - p, 3);
        vals.forEach(([n, v]) => (n.textContent = (v * e).toFixed(3)));
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };
    const ob = new IntersectionObserver((en) => { if (en[0].isIntersecting) { ob.disconnect(); run(); } }, { threshold: 0.4 });
    ob.observe(box);
    // respaldo: si la animación no arranca (pestaña en segundo plano), mostrar los valores reales
    setTimeout(() => { if (!done) { ob.disconnect(); vals.forEach(([n, v]) => (n.textContent = v.toFixed(3))); done = true; } }, 3000);
  }

  // portada: comparador imagen ↔ ground truth
  async function loadHero() {
    let h;
    try { h = await (await fetch("/api/hero")).json(); } catch { return; }
    if (!h) return;
    const fig = $("#heroswipe"), base = $("#hsbase"), lab = $("#hslab"), rng = $("#hsrange");
    base.src = h.image; lab.src = h.labeled;
    $("#hsname").textContent = h.name;
    const set = (v) => { lab.style.clipPath = `inset(0 0 0 ${v}%)`; $("#hsline").style.left = `calc(${v}% - 1px)`; rng.value = v; };
    rng.oninput = () => set(+rng.value);
    await Promise.all([base.decode().catch(() => {}), lab.decode().catch(() => {})]);
    fig.hidden = false;
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { set(50); return; }
    set(100);
    const t0 = performance.now();
    const step = (t) => {            // barrido de entrada: el ground truth se "dibuja" sobre la escena
      const p = Math.min((t - t0) / 1600, 1), e = 1 - Math.pow(1 - p, 3);
      set(100 - 50 * e);
      if (p < 1) requestAnimationFrame(step);
    };
    setTimeout(() => requestAnimationFrame(step), 300);
  }

  // ------------------------------------------------------------------
  // Misión, datos, entrenamiento, ajuste y evidencias
  // ------------------------------------------------------------------
  function figButtons(urls, alt) {
    return urls.map((u) => `<button type="button" data-src="${u}" aria-label="Ampliar ${alt}"><img loading="lazy" src="${u}" alt="${alt}"></button>`).join("");
  }

  const ARCH = {
    Mask2Former: { paper: "Cheng et al., CVPR 2022 · Masked-attention Mask Transformer", flow: [
      ["Swin-B", "Backbone Transformer jerárquico con ventanas desplazadas, preentrenado en COCO-panoptic. Entrega características a 1/4, 1/8, 1/16 y 1/32 de la resolución."],
      ["Pixel decoder", "Fusiona las cuatro escalas con atención deformable multiescala (MSDeformAttn) y produce un embedding por píxel a 1/4 de resolución."],
      ["Decoder · 100 consultas", "Decodificador transformer con masked attention: cada consulta solo mira la región de la máscara que predijo en la capa anterior, lo que acelera la convergencia."],
      ["Clase + máscara", "Cada consulta predice una clase (5 + sin objeto) y una máscara. Se entrena con emparejamiento húngaro; en inferencia las consultas con confianza > 0.5 forman los segmentos panópticos."]] },
    EfficientPS: { paper: "Mohan & Valada, IJCV 2021 · Efficient Panoptic Segmentation", flow: [
      ["EfficientNet-B5", "Backbone convolucional con escalado compuesto, preentrenado en ImageNet (Noisy Student). Su BatchNorm se congela porque el lote es pequeño."],
      ["2-way FPN", "Pirámide de características con un camino de arriba hacia abajo y otro de abajo hacia arriba que se suman en cada escala: P4, P8, P16 y P32 de 256 canales."],
      ["Cabeza semántica", "LSFE capta detalle fino en 1/4 y 1/8, DPC capta contexto amplio con convoluciones dilatadas en 1/16 y 1/32 y MC corrige el desalineamiento entre escalas. Pérdida CE con bootstrapping + Dice, con peso ×2 para Área Caminable."],
      ["Mask R-CNN", "RPN y RoIAlign sobre el 2-way FPN: detecta cajas y máscaras de 28×28 para las clases thing (Área Construida y Zona Verde)."],
      ["Fusión panóptica", "Combina la probabilidad de cada máscara de instancia con la probabilidad semántica de su clase dentro de la caja; el resto de píxeles lo decide la cabeza semántica."]] },
  };

  document.addEventListener("click", (e) => {
    const b = e.target.closest(".flow button");
    if (!b) return;
    const arch = b.closest(".arch");
    arch.querySelectorAll(".flow button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    arch.querySelector(".flow-note").textContent = b.dataset.desc;
  });

  async function loadMission() {
    let d;
    try { d = await (await fetch("/api/mission")).json(); } catch { return; }
    // equipo
    if (d.agency) $("#agency").textContent = d.agency;
    if (d.mission) { $("#missioncode").textContent = `Misión ${d.mission}`; $("#kicker").textContent = `${d.agency} · Misión ${d.mission}`; }
    const cell = (role, who) => `<div><div class="role">${role}</div><div class="who">${who}</div></div>`;
    $("#crew").innerHTML = d.team.map((n, i) => cell(`Especialista de misión ${i + 1}`, n)).join("") +
      cell("Docente", d.professor) + cell(d.course.split("·").slice(0, 2).join("·"), d.university.split("·")[0]);
    $("#footteam").textContent = `${d.team.join(" · ")} — ${d.course} — ${d.university}`;
    $("#herolegend").innerHTML = d.classes.filter((c) => c.id > 0).map((c) => `<span><i style="background:rgb(${c.color})"></i>${c.name}</span>`).join("");

    // datos
    const ds = d.dataset;
    if (ds) {
      const sp = ds.splits || {};
      const f = (v, k) => `<div><div class="v">${v}</div><div class="k">${k}</div></div>`;
      const ov = ds.overlap_mean || {};
      const inst = ds.instances_per_image_mean || {};
      $("#datafacts").innerHTML = `<h3>El dataset en cifras</h3><div class="facts-grid">` +
        f(ds.n_images, "imágenes con ground truth") +
        f(`${sp.train ?? "?"} / ${sp.val ?? "?"} / ${sp.test ?? "?"}`, "entrenamiento / validación / prueba") +
        f((ds.image_sizes || []).slice(0, 2).join(", ") + ((ds.image_sizes || []).length > 2 ? "…" : ""), "tamaño de las imágenes (px)") +
        f(pct(ov.sin_etiqueta ?? 0), "píxeles sin anotar") +
        f(pct(ov.solapadas ?? 0), "píxeles con dos o más etiquetas") +
        f(Object.values(inst).map((v) => v.toFixed(1)).join(" / "), "instancias por imagen: manzanas / zonas verdes") + `</div>`;
      const dist = ds.class_pixel_fraction || {};
      const mx = Math.max(...Object.values(dist));
      $("#classdist").innerHTML = `<h3>Distribución de píxeles por clase</h3><div class="bars">` +
        d.classes.map((c) => { const v = dist[c.name] ?? 0; return `<div class="bar-row"><span>${c.name}</span><div class="track"><div class="fill" style="width:${(v / mx * 100).toFixed(1)}%;background:rgb(${c.color})"></div></div><b>${pct(v)}</b></div>`; }).join("") +
        `</div><p class="hint">El desbalance explica en parte por qué Área Caminable, delgada y escasa, es la clase más difícil.</p>`;
    }
    $("#gtfigs").innerHTML = figButtons(d.figures.ejemplos_gt, "Ejemplo de ground truth");
    $("#gtfigs").classList.add("wide");

    // arquitecturas
    $("#archs").innerHTML = Object.entries(ARCH).map(([name, a]) => {
      const m = d.models[name], t = (m && m.train) || {}, tr = d.training[name] || {};
      const flow = a.flow.map(([x, desc], i) => `<button type="button" aria-pressed="${i === 0}" data-desc="${desc}">${x}</button>`).join("<i>→</i>");
      const kv = (k, v) => `<div>${k}<b>${v ?? "—"}</b></div>`;
      return `<div class="arch"><h3>${name}<small>${m ? m.params_m + " M parámetros · " + m.size_mb + " MB" : "sin pesos"}</small></h3>
        <p class="paper">${a.paper}</p><div class="flow">${flow}</div><p class="flow-note">${a.flow[0][1]}</p><div class="kv">` +
        kv("Épocas", tr.epochs_run) + kv("Mejor época", m && m.epoch) + kv("Tiempo/época", tr.mean_epoch_s ? tr.mean_epoch_s + " s" : null) +
        kv("Lote", t.batch_size) + kv("Recorte", t.crop ? t.crop + " px" : null) + kv("Tasa de aprendizaje", t.lr) +
        kv("Backbone", name === "Mask2Former" ? "Swin-B COCO" : (t.backbone || "EfficientNet-B5").replace(/^tf_|\..*$/g, "")) +
        kv("Optimizador", "AdamW + poly") + kv("Mejor mIoU val.", m && m.val_metrics ? fmt(m.val_metrics.mIoU) : null) + `</div></div>`;
    }).join("");
    $("#augfig").innerHTML = figButtons(d.figures.aumento, "Aumento de datos");
    $("#augfig").classList.add("wide");

    // ajuste
    const tn = Object.entries(d.tuning);
    if (tn.length) {
      let rows = "";
      for (const [model, stages] of tn) for (const [stage, v] of Object.entries(stages)) {
        const a = v.antes, b2 = v.despues;
        const dd = (k) => `<td>${fmt(a[k])} → <b>${fmt(b2[k])}</b>${b2[k] > a[k] + 1e-4 ? `<span class="delta">+${(b2[k] - a[k]).toFixed(3)}</span>` : ""}</td>`;
        const LBL = { scales: "escalas", sigma: "suavizado σ", bias: "factor por clase", split_erode: "erosión px",
          region_min_area: "área mín. px", score_mode: "puntaje" };
        const CLS = ["Fondo", "Caminable", "Construida", "Verde", "Carretera"];
        const show = (k, v) => k === "bias" ? (Object.keys(v).length ? Object.entries(v).map(([c, f]) => `${CLS[c]} ×${f}`).join(", ") : "ninguno")
          : k === "scales" ? v.join(" + ") : k === "score_mode" ? ({ mean: "confianza", mean_logarea: "confianza·log(área)", mean_sqrtarea: "confianza·√área" }[v] || v) : v;
        const params = Object.entries(b2).filter(([k]) => k in LBL).map(([k, v]) => `${LBL[k]}: ${show(k, v)}`).join(" · ");
        rows += `<tr><td>${model}</td><td>${stage}</td>${dd("mIoU")}${dd("mF1")}${b2.mAP != null ? dd("mAP") : "<td>—</td>"}<td style="text-align:left;font-size:.8rem;color:var(--muted)">${params}</td></tr>`;
      }
      $("#tuning").innerHTML = `<table class="mt"><thead><tr><th>Modelo</th><th>Etapa</th><th>mIoU val.</th><th>F1 val.</th><th>mAP val.</th><th style="text-align:left">Parámetros elegidos</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    // evidencias
    if (d.figures.confusion.length) $("#evfigs").innerHTML = figButtons(d.figures.confusion, "Matriz de confusión");
    $("#evwide").innerHTML = figButtons([...d.figures.comparacion, ...d.figures.cualitativo], "Figura de resultados");
  }

  // visor ampliado de figuras
  document.addEventListener("click", (e) => {
    const b = e.target.closest(".gallery-fig button");
    if (!b) return;
    const lb = $("#lightbox");
    lb.querySelector("img").src = b.dataset.src;
    lb.showModal();
  });
  $("#lbclose").onclick = () => $("#lightbox").close();
  $("#lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") e.target.close(); });

  // resaltar la sección visible en la barra de navegación
  const navLinks = [...document.querySelectorAll(".nav a")];
  const io = new IntersectionObserver((ents) => ents.forEach((en) => {
    if (en.isIntersecting) navLinks.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + en.target.id));
  }), { rootMargin: "-40% 0px -55% 0px" });
  ["mision", "datos", "entrenamiento", "analisis", "resultados"].forEach((id) => { const x = document.getElementById(id); if (x) io.observe(x); });

  // ------------------------------------------------------------------
  // Descargas
  // ------------------------------------------------------------------
  function download(href, name) {
    const a = el("a", { href, download: name });
    document.body.append(a); a.click(); a.remove();
  }

  $("#dlimg").onclick = () => {
    const c = document.createElement("canvas");
    c.width = S.w; c.height = S.h;
    const x = c.getContext("2d");
    x.drawImage($("#under"), 0, 0);
    x.save(); x.beginPath(); x.rect(0, 0, (S.w * S.split) / 100, S.h); x.clip();
    x.drawImage($("#over"), 0, 0); x.restore();
    download(c.toDataURL("image/png"), `geotraza_${S.source}_${S.view}_${S.scene.name}.png`.replace(/\s+/g, "_"));
  };

  $("#dljson").onclick = () => {
    const d = S.scene;
    const rep = { escena: d.name, tamaño: [d.width, d.height], modo: d.mode, con_ground_truth: !!d.gt,
      modelos: Object.fromEntries(Object.entries(d.results).map(([n, r]) => [n, { tiempo_ms: r.time_ms, estadisticas: r.stats, metricas: r.metrics, segmentos: r.segments }])),
      comparacion: d.comparison ? { acuerdo: d.comparison.agreement, iou_entre_modelos: d.comparison.class_agreement_iou } : null };
    download(URL.createObjectURL(new Blob([JSON.stringify(rep, null, 2)], { type: "application/json" })), `geotraza_reporte_${d.name}.json`);
  };

  // ------------------------------------------------------------------
  // Eventos
  // ------------------------------------------------------------------
  const drop = $("#drop"), file = $("#file");
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  file.onchange = () => file.files[0] && runFile(file.files[0]);
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) runFile(f); });

  $("#urlform").onsubmit = (e) => {
    e.preventDefault();
    const u = $("#imgurl").value.trim();
    if (!u) return;
    const fd = new FormData();
    fd.append("image_url", u);
    if (S.gtFile) fd.append("gt_file", S.gtFile);
    S.selected = null; drawGallery();
    predict(fd, "la imagen del enlace");
  };

  $("#gtfile").onchange = (e) => {
    S.gtFile = e.target.files[0] || null;
    const l = $(".gt-opt");
    l.classList.toggle("has", !!S.gtFile);
    $("#gtlabel").textContent = S.gtFile ? `Ground truth: ${S.gtFile.name}` : "Adjuntar ground truth (.npy, opcional)";
  };

  document.querySelectorAll(".seg-ctrl button").forEach((b) => (b.onclick = () => {
    S.mode = b.dataset.mode;
    document.querySelectorAll(".seg-ctrl button").forEach((x) => x.setAttribute("aria-checked", x === b));
    $("#modehint").textContent = S.mode === "tta"
      ? "Promedia 8 rotaciones/espejos de la imagen. Más preciso, unas 8 veces más lento."
      : "Cabeza panóptica propia de cada modelo. Una pasada por imagen.";
  }));

  document.querySelectorAll("#views button").forEach((b) => (b.onclick = () => {
    if (b.disabled) return;
    S.view = b.dataset.view; syncTabs(); render();
  }));

  $("#filter").onchange = drawGallery;
  $("#alpha").oninput = (e) => { S.alpha = e.target.value / 100; render(); };
  $("#swipe").oninput = (e) => { S.split = +e.target.value; applySplit(); };
  $("#frame").addEventListener("mousemove", onHover);

  loadStatus().then(() => { loadHero(); loadSamples(); loadBenchmark(); loadMission(); });
})();
