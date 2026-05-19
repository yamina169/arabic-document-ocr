"use strict";

/* ── Constants ── */
const MODEL_META = {
  groq: {
    fullName: "meta-llama/llama-4-scout-17b-16e-instruct",
    displayName: "llama-4-scout",
    pipeline: "Multimodal",
    color: "#3b82f6",
  },
  qwen: {
    fullName: "qwen/qwen2.5-vl-72b-instruct",
    pipeline: "VLM",
    color: "#22c55e",
  },
  paddleocr: {
    fullName: "PaddleOCR-VL-1.5 + qwen/qwen3-32b",
    pipeline: "OCR-LLM",
    color: "#ef4444",
  },
};
const PIPELINE_COLORS = {
  Multimodal: "#3b82f6",
  VLM: "#22c55e",
  "OCR-LLM": "#8b5cf6",
};
const MEDALS = ["🥇", "🥈", "🥉", "4th", "5th", "6th"];

/* ── Helpers ── */
function fmt(v) { return v == null ? "—" : (v * 100).toFixed(1) + "%"; }
function fmtT(s) { return s >= 60 ? (s / 60).toFixed(1) + "m" : s.toFixed(1) + "s"; }

function accuracyColor(v) {
  if (v == null) return "#1e3a5f";
  v = Math.max(0, Math.min(1, v));
  let r, g, b;
  if (v <= 0.5) {
    const t = v * 2;
    r = 239;
    g = Math.round(68 + (179 - 68) * t);
    b = Math.round(68 + (8 - 68) * t);
  } else {
    const t = (v - 0.5) * 2;
    r = Math.round(234 + (34 - 234) * t);
    g = Math.round(179 + (197 - 179) * t);
    b = Math.round(8 + (94 - 8) * t);
  }
  return `rgb(${r},${g},${b})`;
}

function pipelineBg(p) {
  return ({
    Multimodal: "rgba(59,130,246,.15)",
    VLM: "rgba(34,197,94,.15)",
    "OCR-LLM": "rgba(139,92,246,.15)",
  }[p] || "#1e3a5f");
}
function pipelineTextColor(p) { return PIPELINE_COLORS[p] || "#94a3b8"; }

function docLabel(img) {
  const m = img.match(/_(front|back)_(\d+)/i);
  if (!m) return img.slice(0, 5);
  return m[1][0].toUpperCase() + m[2].replace(/^0+/, "").padStart(2, "0");
}

/* ── Process data ── */
function processData(raw) {
  const models = raw.models;
  const docImageSet = new Set();
  for (const m of Object.values(models)) {
    m.documents.forEach((d) => docImageSet.add(d.image));
  }
  const docImages = [...docImageSet].sort();

  const processed = {};
  for (const [id, m] of Object.entries(models)) {
    const meta = MODEL_META[id] || { fullName: id, pipeline: "Unknown", color: "#94a3b8" };
    const docs = m.documents;
    const front = docs.filter((d) => d.doc_type === "CIN_FRONT");
    const back  = docs.filter((d) => d.doc_type === "CIN_BACK");
    const frontAcc = front.length ? front.reduce((s, d) => s + (d.accuracy ?? 0), 0) / front.length : null;
    const backAcc  = back.length  ? back.reduce((s, d)  => s + (d.accuracy ?? 0), 0) / back.length  : null;
    const docMap = {};
    docs.forEach((d) => { docMap[d.image] = d; });
    const avgTime = docs.length ? m.total_time / docs.length : null;
    processed[id] = {
      ...meta,
      id,
      displayName: meta.displayName || id,
      overallAcc: m.field_accuracy,
      docAcc: m.doc_accuracy ?? null,
      time: m.total_time,
      avgTime,
      errors: m.errors,
      frontAcc,
      backAcc,
      docs,
      docMap,
    };
  }
  const ranked = Object.values(processed).sort((a, b) => (b.docAcc ?? 0) - (a.docAcc ?? 0));
  const generatedAt = raw.generated_at || null;
  return { ranked, docImages, generatedAt };
}

/* ── SVG chart helpers ── */
function svgGroupedBars(models, W, H) {
  const PAD = { top: 10, right: 10, bottom: 30, left: 36 };
  const cw = W - PAD.left - PAD.right;
  const ch = H - PAD.top - PAD.bottom;
  const n = models.length;
  const gw = cw / n;
  const bw = gw * 0.28;
  const gap = gw * 0.08;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="display:block;width:100%;height:auto;overflow:visible">`;
  for (let t = 0; t <= 4; t++) {
    const y = PAD.top + ch - (ch * t) / 4;
    s += `<line x1="${PAD.left}" y1="${y}" x2="${PAD.left + cw}" y2="${y}" stroke="#1e3a5f" stroke-dasharray="4,3"/>`;
    s += `<text x="${PAD.left - 4}" y="${y + 4}" text-anchor="end" font-size="9" fill="#475569">${t * 25}%</text>`;
  }
  models.forEach((m, i) => {
    const cx = PAD.left + i * gw + gw / 2;
    if (m.frontAcc != null) {
      const bh = ch * m.frontAcc;
      const by = PAD.top + ch - bh;
      s += `<rect x="${cx - bw - gap / 2}" y="${by}" width="${bw}" height="${bh}" fill="${m.color}" rx="3" opacity=".9"/>`;
      if (bh > 14) s += `<text x="${cx - bw / 2 - gap / 2}" y="${by + bh - 5}" text-anchor="middle" font-size="9" font-weight="700" fill="rgba(0,0,0,.6)">${(m.frontAcc * 100).toFixed(0)}</text>`;
    }
    if (m.backAcc != null) {
      const bh = ch * m.backAcc;
      const by = PAD.top + ch - bh;
      s += `<rect x="${cx + gap / 2}" y="${by}" width="${bw}" height="${bh}" fill="${m.color}" rx="3" opacity=".45"/>`;
      if (bh > 14) s += `<text x="${cx + bw / 2 + gap / 2}" y="${by + bh - 5}" text-anchor="middle" font-size="9" font-weight="700" fill="rgba(0,0,0,.6)">${(m.backAcc * 100).toFixed(0)}</text>`;
    }
    s += `<text x="${cx}" y="${PAD.top + ch + 18}" text-anchor="middle" font-size="11" font-weight="600" fill="${m.color}">${m.displayName}</text>`;
  });
  return s + `</svg>`;
}

function svgSpeedBars(models, W) {
  const sorted = [...models].sort((a, b) => (a.avgTime ?? a.time) - (b.avgTime ?? b.time));
  const maxT = sorted[sorted.length - 1].avgTime ?? sorted[sorted.length - 1].time;
  const rowH = 36;
  const labelW = 90;
  const barW = W - labelW - 60;
  const H = sorted.length * rowH + 10;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="display:block;width:100%;height:auto;overflow:visible">`;
  sorted.forEach((m, i) => {
    const y = i * rowH + 6;
    const t = m.avgTime ?? m.time;
    const w = Math.max(4, (t / maxT) * barW);
    s += `<text x="${labelW - 8}" y="${y + 14}" text-anchor="end" font-size="11" font-weight="600" fill="${m.color}">${m.displayName}</text>`;
    s += `<rect x="${labelW}" y="${y + 2}" width="${w}" height="22" fill="${m.color}" rx="4" opacity=".85"/>`;
    s += `<text x="${labelW + w + 6}" y="${y + 14}" font-size="10" fill="#94a3b8">${fmtT(t)}/doc</text>`;
    if (m.errors > 0) s += `<text x="${labelW + w + 6}" y="${y + 24}" font-size="9" fill="#ef4444">${m.errors} err</text>`;
  });
  return s + `</svg>`;
}

/* ── Section builders ── */
function buildPipelineSection() {
  const pipes = [
    {
      type: "Multimodal", models: "llama-4-scout", color: "#3b82f6",
      steps: [{ label: "Image", accent: false }, { label: "LLM", accent: true }, { label: "JSON", accent: false }],
      calls: "1 API call",
    },
    {
      type: "VLM", models: "qwen", color: "#22c55e",
      steps: [{ label: "Image", accent: false }, { label: "Vision Model", accent: true }, { label: "JSON", accent: false }],
      calls: "1 API call",
    },
    {
      type: "OCR + LLM", models: "paddleocr", color: "#8b5cf6",
      steps: [
        { label: "Image", accent: false }, { label: "OCR", accent: true },
        { label: "Text", accent: false }, { label: "LLM", accent: true }, { label: "JSON", accent: false },
      ],
      calls: "2 API calls",
    },
  ];
  return `
  <div class="section">
    <div class="section-title"><span class="section-num">0</span>Pipeline Architecture</div>
    <div class="pipeline-grid">
    ${pipes.map((p) => `
      <div class="pipe-card">
        <div class="pipe-label" style="color:${p.color}">${p.type}</div>
        <div class="pipe-models">${p.models}</div>
        <div class="pipe-flow">
          ${p.steps.map((s, i) => `
            ${i > 0 ? `<div class="pipe-arrow">&#8594;</div>` : ""}
            <div class="pipe-box ${s.accent ? "accent" : ""}" style="--pc:${p.color}">${s.label}</div>
          `).join("")}
        </div>
        <div class="pipe-calls">${p.calls}</div>
      </div>
    `).join("")}
    </div>
  </div>`;
}

function buildMethodologyNote() {
  return `
  <div class="section" style="padding:14px 20px">
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px">How metrics are calculated</div>
    <div style="display:flex;gap:32px;flex-wrap:wrap">
      <div style="flex:1;min-width:200px">
        <div style="font-size:12px;font-weight:700;color:#94a3b8;margin-bottom:4px">Field Accuracy</div>
        <div style="font-size:12px;color:#64748b;line-height:1.6">
          Average <strong>Levenshtein similarity</strong> between extracted and expected values,
          computed per field then averaged across all fields of all documents.
          Score ∈ [0, 1] — higher is better.
        </div>
      </div>
      <div style="flex:1;min-width:200px">
        <div style="font-size:12px;font-weight:700;color:#94a3b8;margin-bottom:4px">Doc Accuracy</div>
        <div style="font-size:12px;color:#64748b;line-height:1.6">
          Percentage of documents whose <strong>average field score ≥ threshold (0.85)</strong>.
          A document counts as passed only if its overall field quality meets the threshold.
        </div>
      </div>
      <div style="flex:1;min-width:200px">
        <div style="font-size:12px;font-weight:700;color:#94a3b8;margin-bottom:4px">Threshold (0.85)</div>
        <div style="font-size:12px;color:#64748b;line-height:1.6">
          Minimum similarity to consider a document as usable.
          Tolerates minor OCR noise (diacritics, spacing) without penalising correct extractions.
        </div>
      </div>
    </div>
  </div>`;
}

function buildRankingSection(ranked) {
  const cards = ranked.map((m, i) => `
    <div class="rank-card" style="--mc:${m.color}">
      <div class="rank-top">
        <div class="medal">${MEDALS[i]}</div>
        <div class="rank-info">
          <div class="rank-id">${m.displayName}</div>
          <div class="rank-full">${m.fullName}</div>
        </div>
      </div>
      <div class="pipe-badge" style="background:${pipelineBg(m.pipeline)};color:${pipelineTextColor(m.pipeline)}">${m.pipeline}</div>
      <div class="big-acc">
        <div class="val">${m.docAcc != null ? (m.docAcc * 100).toFixed(1) : "—"}<span style="font-size:16px;opacity:.6">%</span></div>
        <div class="lbl">Doc Accuracy</div>
      </div>
      <div class="mrow"><span class="mrow-lbl">Field accuracy</span><span class="mrow-val">${fmt(m.overallAcc)}</span></div>
      <div class="mrow"><span class="mrow-lbl">Front accuracy</span><span class="mrow-val">${fmt(m.frontAcc)}</span></div>
      <div class="mrow"><span class="mrow-lbl">Back accuracy</span><span class="mrow-val">${fmt(m.backAcc)}</span></div>
      <div class="mrow"><span class="mrow-lbl">Avg time/doc</span><span class="mrow-val">${m.avgTime != null ? fmtT(m.avgTime) : "—"}</span></div>
      <div class="mrow"><span class="mrow-lbl">Total time</span><span class="mrow-val">${fmtT(m.time)}</span></div>
      <div class="mrow"><span class="mrow-lbl">Errors</span><span class="mrow-val ${m.errors > 0 ? "err-val" : ""}">${m.errors}</span></div>
    </div>`).join("");
  return `
  <div class="section">
    <div class="section-title"><span class="section-num">1</span>Model Ranking</div>
    <div class="rank-grid">${cards}</div>
  </div>`;
}

function buildAnalysisSection(ranked, docImages) {
  const docLabels = docImages.map(docLabel);
  const headerRow = `<div class="hm-doc-labels">${docLabels.map((l) => `<div class="hm-doc-lbl">${l}</div>`).join("")}</div>`;
  const heatRows = ranked.map((m) => {
    const cells = docImages.map((img) => {
      const doc = m.docMap[img];
      const acc = doc ? doc.accuracy : null;
      const bg = accuracyColor(acc);
      const label = acc != null ? (acc * 100).toFixed(0) : "–";
      const textColor = acc != null && acc > 0.5 ? "rgba(0,0,0,.65)" : "rgba(255,255,255,.8)";
      const dt = doc ? doc.doc_type.replace("CIN_", "") : "?";
      return `<div class="hm-cell"
        style="background:${bg};color:${textColor}"
        data-tip="${img}|${m.displayName}|${acc != null ? (acc * 100).toFixed(1) + "%" : "error"}|${doc ? fmtT(doc.elapsed) : "—"}|${dt}"
      >${label}</div>`;
    }).join("");
    return `<div class="hm-row">
      <div class="hm-model-lbl" style="color:${m.color}">${m.displayName}</div>
      <div class="hm-cells">${cells}</div>
    </div>`;
  }).join("");

  return `
  <div class="section">
    <div class="section-title"><span class="section-num">2</span>Analysis</div>
    <div style="margin-bottom:20px">
      <div style="font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px">Document Accuracy Heatmap</div>
      ${headerRow}
      ${heatRows}
      <div class="hm-legend">
        <span>0%</span>
        <div class="hm-legend-bar"></div>
        <span>100%</span>
        <span style="margin-left:16px;color:#475569">Hover cell for details</span>
      </div>
      <div class="hm-note">F=front, B=back &nbsp;·&nbsp; number = accuracy %</div>
    </div>
    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">Front vs Back Accuracy</div>
        ${svgGroupedBars(ranked, 600, 200)}
        <div class="chart-legend">
          ${ranked.map((m) => `
            <div class="legend-item">
              <div class="legend-dot" style="background:${m.color};opacity:.9"></div>
              <span style="color:${m.color};font-weight:600">${m.displayName}</span>&nbsp;front
            </div>
            <div class="legend-item">
              <div class="legend-dot" style="background:${m.color};opacity:.45"></div><span>back</span>
            </div>
          `).join("")}
        </div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Speed (avg time/doc, lower = faster)</div>
        ${svgSpeedBars(ranked, 400)}
      </div>
    </div>
  </div>`;
}

function buildTableSection(ranked) {
  const rows = ranked.map((m) => {
    const acc = (v) => {
      if (v == null) return "—";
      const c = accuracyColor(v);
      return `<span class="acc-pill" style="background:${c}22;color:${c};border:1px solid ${c}44">${(v * 100).toFixed(1)}%</span>`;
    };
    return `<tr>
      <td style="font-weight:700;color:${m.color}">${m.displayName}</td>
      <td style="font-size:11px;color:#64748b;max-width:200px;word-break:break-all">${m.fullName}</td>
      <td><span style="color:${pipelineTextColor(m.pipeline)};font-weight:600">${m.pipeline}</span></td>
      <td class="${m.errors > 0 ? "err-val" : ""}">${m.errors}</td>
      <td>${acc(m.frontAcc)}</td>
      <td>${acc(m.backAcc)}</td>
      <td>${acc(m.overallAcc)}</td>
      <td>${acc(m.docAcc)}</td>
      <td>${m.avgTime != null ? fmtT(m.avgTime) : "—"}</td>
      <td>${fmtT(m.time)}</td>
    </tr>`;
  }).join("");
  return `
  <div class="section">
    <div class="section-title"><span class="section-num">3</span>Raw Results</div>
    <div class="tbl-toolbar">
      <button class="btn-export" onclick="exportCSV()">&#8595; Export CSV</button>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th><th>Full Name</th><th>Pipeline</th><th>Errors</th>
            <th>Front Acc.</th><th>Back Acc.</th><th>Field Acc.</th><th>Doc Acc.</th><th>Avg/doc</th><th>Total Time</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

/* ── Live Test Section ── */
function buildLiveTestSection() {
  return `
  <div class="section">
    <div class="section-title"><span class="section-num">4</span>Live Extraction Test</div>
    <div class="live-grid">
      <div class="live-upload-col">
        <div class="drop-zone" id="dropZone">
          <div class="drop-icon">&#128247;</div>
          <div class="drop-text">Drop image here or <label class="drop-browse" for="fileInput">browse</label></div>
          <div class="drop-sub">JPG, PNG, JPEG — max 20 MB</div>
          <input type="file" id="fileInput" accept=".jpg,.jpeg,.png" style="display:none">
        </div>
        <div class="img-preview-wrap" id="imgPreviewWrap" style="display:none">
          <img id="imgPreview" class="img-preview" alt="preview">
          <button class="btn-change-img" onclick="changeImage()">&#8635; Change image</button>
        </div>
        <div class="live-controls">
          <div class="ctrl-row">
            <label class="ctrl-lbl">Model</label>
            <select id="liveModel" class="ctrl-select">
              <option value="groq">Groq — LLaMA 4 Scout</option>
              <option value="qwen">Qwen2.5-VL 72B</option>
            </select>
          </div>
          <div class="ctrl-row">
            <label class="ctrl-lbl">Document type</label>
            <select id="liveDocType" class="ctrl-select">
              <option value="cin-front">CIN Front</option>
              <option value="cin-back">CIN Back</option>
            </select>
          </div>
          <button class="btn-extract" id="extractBtn" onclick="runExtract()" disabled>
            <span id="extractLabel">&#9654; Extract</span>
          </button>
        </div>
      </div>
      <div class="live-result-col" id="liveResult">
        <div class="result-placeholder">
          <div style="font-size:36px;opacity:.2">&#128196;</div>
          <div style="color:#475569;font-size:12px;margin-top:10px">Upload an image and click Extract</div>
        </div>
      </div>
    </div>
  </div>`;
}

function initLiveTest() {
  const drop = document.getElementById("dropZone");
  const input = document.getElementById("fileInput");
  const btn   = document.getElementById("extractBtn");

  function setFile(f) {
    if (!f) return;
    const url = URL.createObjectURL(f);
    document.getElementById("imgPreview").src = url;
    document.getElementById("imgPreviewWrap").style.display = "block";
    drop.style.display = "none";
    btn.disabled = false;
    document.getElementById("liveResult").innerHTML =
      "<div class=\"result-placeholder\"><div style=\"font-size:36px;opacity:.2\">&#128196;</div>"
      + "<div style=\"color:#475569;font-size:12px;margin-top:10px\">Click Extract to analyse</div></div>";
  }

  input.addEventListener("change", () => setFile(input.files[0]));
  drop.addEventListener("dragover",  (e) => { e.preventDefault(); drop.classList.add("drag-over"); });
  drop.addEventListener("dragleave", ()  => drop.classList.remove("drag-over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("drag-over"); setFile(e.dataTransfer.files[0]); });
  drop.addEventListener("click", () => input.click());
}

function changeImage() {
  document.getElementById("imgPreviewWrap").style.display = "none";
  document.getElementById("dropZone").style.display = "block";
  document.getElementById("extractBtn").disabled = true;
  document.getElementById("fileInput").value = "";
  document.getElementById("liveResult").innerHTML =
    "<div class=\"result-placeholder\"><div style=\"font-size:36px;opacity:.2\">&#128196;</div>"
    + "<div style=\"color:#475569;font-size:12px;margin-top:10px\">Upload an image and click Extract</div></div>";
}

async function runExtract() {
  const file    = document.getElementById("fileInput").files[0]
               || (() => { const p = document.getElementById("imgPreview"); return p.src ? fetch(p.src).then(r => r.blob()) : null; })();
  const model   = document.getElementById("liveModel").value;
  const docType = document.getElementById("liveDocType").value;
  const btn     = document.getElementById("extractBtn");
  const label   = document.getElementById("extractLabel");
  const result  = document.getElementById("liveResult");

  const input = document.getElementById("fileInput");
  if (!input.files[0]) return;

  btn.disabled = true;
  label.innerHTML = `<span class="spin-inline">&#9696;</span> Extracting…`;
  result.innerHTML = `<div class="result-placeholder"><div class="spinner" style="width:32px;height:32px"></div></div>`;

  const form = new FormData();
  form.append("image", input.files[0]);

  const t0 = Date.now();
  try {
    const resp = await fetch("/process/" + model + "/" + docType, { method: "POST", body: form });
    const data = await resp.json();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(2);

    if (data.erreur) {
      result.innerHTML = "<div class=\"result-error\"><strong>Error:</strong> " + data.erreur + "</div>";
    } else {
      const skip = new Set(["champs_manquants", "extrait_le", "erreur", "brut"]);
      const missing = new Set(data.champs_manquants || []);
      const rows = Object.entries(data)
        .filter(([k]) => !skip.has(k))
        .map(([k, v]) => {
          const isMissing = missing.has(k) || v === "?";
          return "<div class=\"res-row" + (isMissing ? " res-missing" : "") + "\">"
            + "<span class=\"res-key\">" + k + "</span>"
            + "<span class=\"res-val\">" + (isMissing ? "?" : v) + "</span>"
            + "</div>";
        }).join("");
      result.innerHTML = "<div class=\"res-header\">"
        + "<span class=\"res-model-badge\">" + model + "</span>"
        + "<span class=\"res-time\">&#9201; " + elapsed + "s</span>"
        + "</div>"
        + "<div class=\"res-fields\">" + rows + "</div>"
        + (missing.size ? "<div class=\"res-warn\">Missing fields: " + [...missing].join(", ") + "</div>" : "");
    }
  } catch (e) {
    result.innerHTML = "<div class=\"result-error\">Network error: " + e.message + "</div>";
  }

  btn.disabled = false;
  label.innerHTML = "&#9654; Extract";
}

/* ── Main render ── */
let _ranked = [];
function render(raw) {
  const { ranked, docImages, generatedAt } = processData(raw);
  _ranked = ranked;
  const app = document.getElementById("app");
  app.innerHTML = `
  <div class="container">
    <div class="header">
      <h1>Document OCR &mdash; Benchmark Dashboard</h1>
      <p>Tunisian CIN documents &nbsp;&middot;&nbsp; front &amp; back sides${generatedAt ? ` &nbsp;&middot;&nbsp; <span style="color:#64748b;font-size:12px">${new Date(generatedAt).toLocaleString()}</span>` : ""}</p>
    </div>
    ${buildPipelineSection()}
    ${buildMethodologyNote()}
    ${buildRankingSection(ranked)}
    ${buildAnalysisSection(ranked, docImages)}
    ${buildTableSection(ranked)}
    ${buildLiveTestSection()}
  </div>`;
  app.style.display = "block";

  initLiveTest();

  const tip = document.getElementById("tooltip");
  document.querySelectorAll(".hm-cell").forEach((cell) => {
    cell.addEventListener("mouseenter", () => {
      const [img, model, acc, elapsed, dt] = cell.dataset.tip.split("|");
      tip.innerHTML = `<div class="tt-doc">${img}</div>
        <div class="tt-row">Model: <span>${model}</span></div>
        <div class="tt-row">Type: <span>${dt}</span></div>
        <div class="tt-row">Accuracy: <span>${acc}</span></div>
        <div class="tt-row">Time: <span>${elapsed}</span></div>`;
      tip.style.display = "block";
    });
    cell.addEventListener("mousemove", (e) => {
      tip.style.left = e.clientX + 14 + "px";
      tip.style.top  = e.clientY - 10 + "px";
    });
    cell.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  });
}

/* ── CSV export ── */
function exportCSV() {
  const header = ["Model", "Full Name", "Pipeline", "Errors", "Front Accuracy", "Back Accuracy", "Field Accuracy", "Doc Accuracy", "Avg Time/doc (s)", "Total Time (s)"];
  const rows = _ranked.map((m) => [
    m.displayName, m.fullName, m.pipeline, m.errors,
    m.frontAcc != null ? (m.frontAcc * 100).toFixed(2) : "",
    m.backAcc  != null ? (m.backAcc  * 100).toFixed(2) : "",
    (m.overallAcc * 100).toFixed(2),
    m.docAcc  != null ? (m.docAcc   * 100).toFixed(2) : "",
    m.avgTime != null ? m.avgTime.toFixed(2) : "",
    m.time.toFixed(2),
  ]);
  const csv = [header, ...rows]
    .map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(","))
    .join("\r\n");
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ocr_benchmark.csv";
  a.click();
}

/* ── Boot ── */
fetch("/api/results")
  .then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`);
    return r.json();
  })
  .then((data) => {
    document.getElementById("loading").style.display = "none";
    render(data);
  })
  .catch((err) => {
    document.getElementById("loading").style.display = "none";
    document.getElementById("error-msg").textContent = err.message;
    document.getElementById("error").style.display = "block";
  });
