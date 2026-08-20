/* Co-RL project page — behavior. Vanilla JS, no dependencies.
   Reads window.siteContent (content.js) and window.siteResults (results.js). */
(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const fmt = v => esc(v);            // values arrive as the paper's exact strings
  const nf = v => parseFloat(v);       // numeric view, for arithmetic and scales only

  const C = window.siteContent;
  const R = window.siteResults;

  /* ---------- helpers over the data model ---------- */
  const avgOf = row => row.vals[row.vals.length - 1];
  const avgN = row => nf(avgOf(row));
  const rowBy = (rows, name) => rows.find(r => r.method === name);

  /* =============== RESULTS EXPLORER =============== */
  function methodRowClass(m) {
    if (m === "Base" || m === "GT-Reward") return "ref-row";
    if (m.indexOf("Co-RL") === 0) return "ours-row";
    return "";
  }

  // Cell color encodes the change against the Base row in the same column.
  // Violet above Base, red below, alpha grows with the size of the move.
  function heatColor(delta) {
    const a = Math.min(Math.abs(delta) / 6, 1) * 0.42;
    if (a < 0.03) return "";
    return delta >= 0 ? `rgba(123,80,162,${a.toFixed(2)})` : `rgba(190,55,55,${a.toFixed(2)})`;
  }

  function renderResultsTable(headers, rows, label) {
    const base = rowBy(rows, "Base");
    const baseVals = base ? base.vals.map(nf) : null;
    let html = `<table class="data-table" aria-label="${esc(label)}"><thead><tr><th>Method</th>`;
    headers.forEach((h, i) => html += `<th class="${i === headers.length - 1 ? "avg-col" : ""}">${esc(h)}</th>`);
    html += `</tr></thead><tbody>`;
    rows.forEach(r => {
      html += `<tr class="${methodRowClass(r.method)}"><td>${esc(r.method)}</td>`;
      r.vals.forEach((v, i) => {
        const mk = r.marks[i] === "b" ? "mark-b" : r.marks[i] === "u" ? "mark-u" : "";
        const avg = i === r.vals.length - 1 ? " avg-col" : "";
        let bg = "";
        if (baseVals && r.method !== "Base") {
          const col = heatColor(nf(v) - baseVals[i]);
          if (col) bg = ` style="background:${col}"`;
        }
        html += `<td class="${mk}${avg}"${bg}>${fmt(v)}</td>`;
      });
      html += `</tr>`;
    });
    return html + "</tbody></table>";
  }

  function renderAvgLadder(rows) {
    // horizontal lollipop of the Avg column; references drawn as dashed rules
    const items = rows.filter(r => r.method !== "Base");
    const base = rowBy(rows, "Base");
    const vals = rows.map(avgN);
    const vmin = Math.min(...vals) - 1.2, vmax = Math.max(...vals) + 1.2;
    const W = 900, rowH = 26, L = 210, Rt = 60;
    const H = items.length * rowH + 34;
    const x = v => L + (v - vmin) / (vmax - vmin) * (W - L - Rt);
    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="DM Mono, monospace">`;
    if (base) {
      svg += `<line x1="${x(avgN(base))}" y1="8" x2="${x(avgN(base))}" y2="${H - 26}" stroke="#a9b2b9" stroke-dasharray="4 4"/>`;
      svg += `<text x="${x(avgN(base))}" y="${H - 10}" text-anchor="middle" font-size="11" fill="#67707b">Base ${fmt(avgOf(base))}</text>`;
    }
    items.forEach((r, i) => {
      const y = 20 + i * rowH;
      const isCo = r.method.indexOf("Co-RL") === 0;
      const isGT = r.method === "GT-Reward";
      const color = isCo ? "#7b50a2" : isGT ? "#1b222c" : "#a9b2b9";
      svg += `<text x="${L - 10}" y="${y + 4}" text-anchor="end" font-size="12.5" fill="${isCo ? "#5d3a80" : "#3c4450"}" font-family="DM Sans, sans-serif">${esc(r.method)}</text>`;
      if (isGT) {
        svg += `<line x1="${x(avgN(r))}" y1="${y - 9}" x2="${x(avgN(r))}" y2="${y + 9}" stroke="#1b222c" stroke-width="2.5" stroke-dasharray="3 3"/>`;
        const flip = x(avgN(r)) > W - 200;
        svg += `<text x="${x(avgN(r)) + (flip ? -8 : 8)}" y="${y + 4}" font-size="12" fill="#1b222c" text-anchor="${flip ? "end" : "start"}">${fmt(avgOf(r))} (labeled reference)</text>`;
      } else {
        svg += `<line x1="${x(vmin)}" y1="${y}" x2="${x(avgN(r))}" y2="${y}" stroke="${color}" stroke-width="2" opacity="0.55"/>`;
        svg += `<circle cx="${x(avgN(r))}" cy="${y}" r="6.5" fill="${color}"/>`;
        svg += `<text x="${x(avgN(r)) + 11}" y="${y + 4}" font-size="12" fill="#1b222c">${fmt(avgOf(r))}</text>`;
      }
    });
    svg += `</svg>`;
    return `<div>${svg}</div><p class="chart-note">Average over the suite. Dashed marks are references,
      excluded from the label-free ranking.</p>`;
  }

  const TEXT_NOTE = `<div class="callout honesty"><span class="tag">Stated plainly</span>
    Same family edges Different family on Qwen2.5-3B, so the ordering is directional rather than monotone on
    every backbone. Different family beats the strongest self-rewarding baseline on all four text backbones.
    Different family+ posts the best label-free average on all four, by <span class="num">0.8 to 2.0%</span>.</div>`;
  const VLM_NOTE = `<div class="callout honesty"><span class="tag">Stated plainly</span>
    TTRL wins one of the four small-pair settings, on InternVL-3.5-2B with MMR1. At 7B to 12B, Co-RL beats
    TTRL for all three families and beats the labeled reference on Gemma-3-12B.</div>`;
  const N3_NOTE = `<div class="callout"><span class="tag">One run, three improved models</span>
    Rows are per-agent from a single three-model run and are not read against the two-agent tables.</div>`;

  function resultsOptions(domain) {
    if (domain === "text") return R.llmMain.map((b, i) => ({ v: String(i), label: b.backbone + " (" + b.tier + ")" }));
    if (domain === "vlm") {
      const opts = R.vlmSmall.map((b, i) => ({ v: "s" + i, label: b.backbone + " · " + b.dataset }));
      return opts.concat(R.vlmLarge.map((b, i) => ({ v: "l" + i, label: b.backbone + " · " + b.dataset })));
    }
    return R.n3.map((b, i) => ({ v: String(i), label: b.model }));
  }

  function renderResults() {
    const domain = $("#results-domain").value;
    const sel = $("#results-backbone");
    const key = sel.value;
    let rows, headers, label, note;
    if (domain === "text") {
      const b = R.llmMain[Number(key) || 0];
      rows = b.rows; headers = R.benchText; label = "Text results for " + b.backbone; note = TEXT_NOTE;
    } else if (domain === "vlm") {
      const b = key && key[0] === "l" ? R.vlmLarge[Number(key.slice(1))] : R.vlmSmall[Number((key || "s0").slice(1))];
      rows = b.rows; headers = R.benchVLM; label = "Multimodal results for " + b.backbone + " on " + b.dataset; note = VLM_NOTE;
    } else {
      const b = R.n3[Number(key) || 0];
      rows = b.rows; headers = R.benchText; label = "Three-agent results for " + b.model; note = N3_NOTE;
    }
    $("#results-ladder").innerHTML = renderAvgLadder(rows);
    $("#results-table").innerHTML = renderResultsTable(headers, rows, label);
    $("#results-note").innerHTML = note;
  }

  function setupResults() {
    const dom = $("#results-domain"), sel = $("#results-backbone");
    const fill = () => {
      sel.innerHTML = resultsOptions(dom.value).map(o => `<option value="${o.v}">${esc(o.label)}</option>`).join("");
    };
    dom.addEventListener("change", () => { fill(); renderResults(); });
    sel.addEventListener("change", renderResults);
    fill(); renderResults();
  }

  /* =============== FURNITURE =============== */
  function setupLightbox() {
    const box = $("#lightbox"), img = $("#lightbox-image"), closeBtn = $("#lightbox-close");
    let lastTrigger = null;
    const close = () => {
      box.hidden = true; document.body.classList.remove("no-scroll"); img.src = "";
      if (lastTrigger) { lastTrigger.focus(); lastTrigger = null; }
    };
    $$("[data-lightbox]").forEach(btn => btn.addEventListener("click", () => {
      img.src = btn.dataset.lightbox;
      img.alt = btn.querySelector("img") ? btn.querySelector("img").alt : "";
      lastTrigger = btn;
      box.hidden = false; document.body.classList.add("no-scroll");
      closeBtn.focus();
    }));
    closeBtn.addEventListener("click", close);
    box.addEventListener("click", e => { if (e.target === box) close(); });
    document.addEventListener("keydown", e => {
      if (box.hidden) return;
      if (e.key === "Escape") close();
      if (e.key === "Tab") { e.preventDefault(); closeBtn.focus(); } // one focusable stop
    });
  }

  function setupNav() {
    const toggle = $(".nav-toggle"), links = $("#nav-links");
    toggle.addEventListener("click", () => {
      const open = links.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", open);
    });
    $$("a", links).forEach(a => a.addEventListener("click", () => {
      links.classList.remove("is-open"); toggle.setAttribute("aria-expanded", "false");
    }));
  }

  function setupBibtex() {
    const btn = $("#copy-bibtex");
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText($("#bibtex-code").textContent);
        btn.textContent = "Copied";
        setTimeout(() => { btn.textContent = "Copy"; }, 1400);
      } catch (e) { btn.textContent = "Select + copy"; }
    });
  }

  function init() {
    setupResults();
    setupLightbox();
    setupNav();
    setupBibtex();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
