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
  const CO_VARIANTS = ["Co-RL (Same family)", "Co-RL (Different family)", "Co-RL (Different family+)"];

  /* =============== VERDICT SCOREBOARD =============== */
  // One chip per backbone/training-set block that carries a GT-Reward row.
  // Values are read live from siteResults — nothing here is hand-typed.
  function verdictChips(dfOnly) {
    const chips = [];
    // text blocks
    R.llmMain.forEach(b => {
      const gt = rowBy(b.rows, "GT-Reward");
      const cands = b.rows.filter(r => dfOnly ? r.method === "Co-RL (Different family)" : CO_VARIANTS.includes(r.method));
      if (!gt || !cands.length) return;
      const best = cands.reduce((a, r) => avgN(r) > avgN(a) ? r : a);
      chips.push({ domain: "text", model: b.backbone, meta: "text · 7-benchmark avg",
                   variant: best.method, ours: avgOf(best), gt: avgOf(gt) });
    });
    // three-agent blocks (Different family by construction)
    R.n3.forEach(b => {
      const gt = rowBy(b.rows, "GT-Reward");
      const co = b.rows.find(r => r.method.indexOf("Co-RL") === 0);
      if (!gt || !co) return;
      chips.push({ domain: "text", model: b.model, meta: "text · three-agent ring · 7-benchmark avg",
                   variant: "Co-RL (Different family), N=3", ours: avgOf(co), gt: avgOf(gt) });
    });
    // vision-language blocks (all Different family)
    [...R.vlmSmall, ...R.vlmLarge].forEach(b => {
      const gt = rowBy(b.rows, "GT-Reward");
      const co = b.rows.find(r => r.method.indexOf("Co-RL") === 0);
      if (!gt || !co) return;
      chips.push({ domain: "vlm", model: b.backbone, meta: "multimodal · " + b.dataset + " · 4-benchmark avg",
                   variant: co.method, ours: avgOf(co), gt: avgOf(gt) });
    });
    return chips;
  }

  function renderVerdict() {
    const domain = $("#verdict-domain").value;
    const dfOnly = $("#verdict-df-only").classList.contains("is-active");
    const chips = verdictChips(dfOnly).filter(c => domain === "all" || c.domain === domain);
    $("#verdict-chips").innerHTML = chips.map(c => {
      const d = nf(c.ours) - nf(c.gt);
      const sign = d >= 0 ? "+" : "−";
      const cls = d >= 0 ? "pos" : "neg";
      const word = d >= 0 ? "" : " (approaches)";
      return `<div class="verdict-chip">
        <div class="vc-model">${esc(c.model)}</div>
        <div class="vc-meta">${esc(c.meta)}</div>
        <div class="vc-nums"><span class="vc-ours">${fmt(c.ours)}</span>
          <span class="vc-gt">vs ${fmt(c.gt)}</span>
          <span class="vc-delta ${cls}">${sign}${Math.abs(d).toFixed(c.domain === "vlm" ? 2 : 1)}</span></div>
        <div class="vc-variant">${esc(c.variant)}${word} · label-free avg vs labeled reference</div>
      </div>`;
    }).join("");
  }

  function setupVerdict() {
    $("#verdict-domain").addEventListener("change", renderVerdict);
    const t = $("#verdict-df-only");
    t.addEventListener("click", () => {
      t.classList.toggle("is-active");
      t.setAttribute("aria-pressed", t.classList.contains("is-active"));
      renderVerdict();
    });
    renderVerdict();
  }

  /* =============== LADDER =============== */
  function renderRungPanel(rung) {
    let html = `<span class="badge tone-${rung.badgeTone}">${esc(rung.badge)}</span>
      <h3>Rung ${esc(rung.index)} — ${esc(rung.label)}</h3>
      <p>${esc(rung.what)}</p>
      <span class="rp-label">Measured</span><p>${esc(rung.measured)}</p>
      <span class="rp-label">Trained</span><p>${esc(rung.trained)}</p>`;
    if (rung.result) html += `<span class="rp-label">Result</span><p>${esc(rung.result)}</p>`;
    if (rung.rephrase) {
      html += `<div class="rephrase-toggle">
        <div class="rephrase-tabs" aria-label="Original or rephrased prompt">
          <button type="button" class="is-active" data-pane="original" aria-pressed="true">Original prompt</button>
          <button type="button" data-pane="rephrased" aria-pressed="false">DeepSeek-V3 rewrite</button>
        </div>
        <div class="rephrase-body" id="rephrase-body">${esc(rung.rephrase.original)}</div>
      </div>`;
    }
    $("#rung-panel").innerHTML = html;
    if (rung.rephrase) {
      $$(".rephrase-tabs button").forEach(btn => btn.addEventListener("click", () => {
        $$(".rephrase-tabs button").forEach(b => { b.classList.remove("is-active"); b.setAttribute("aria-pressed", "false"); });
        btn.classList.add("is-active"); btn.setAttribute("aria-pressed", "true");
        $("#rephrase-body").textContent = rung.rephrase[btn.dataset.pane];
      }));
    }
  }

  function setupLadder() {
    const rail = $("#rung-rail");
    rail.innerHTML = C.rungs.map((r, i) => `
      <button class="rung-button${i === 2 ? " is-active" : ""}" type="button" data-rung="${i}" aria-pressed="${i === 2}">
        <span class="rung-idx">${esc(r.index)}</span>
        <span><span class="rung-name">${esc(r.label)}</span>
        <span class="rung-kappa">${esc(r.kappa)}</span></span>
      </button>`).join("");
    $$(".rung-button", rail).forEach(btn => btn.addEventListener("click", () => {
      $$(".rung-button", rail).forEach(b => { b.classList.remove("is-active"); b.setAttribute("aria-pressed", "false"); });
      btn.classList.add("is-active"); btn.setAttribute("aria-pressed", "true");
      renderRungPanel(C.rungs[Number(btn.dataset.rung)]);
    }));
    renderRungPanel(C.rungs[2]); // default: Different family, the main setting
  }

  /* =============== PROBE =============== */
  const LEVEL_COLOR = { "different family": "#7b50a2", "same family": "#0072b2", "seed only": "#a9b2b9" };
  // text needs AA contrast on white; the light gray stays for dots and bars only
  const LEVEL_TEXT = { "different family": "#5d3a80", "same family": "#005c8f", "seed only": "#5f6a72" };

  function renderKappaStrip() {
    const pts = R.decPool;
    const W = 900, H = 210, L = 40, Rt = 30, axisY = 150;
    const xmin = 0.28, xmax = 0.62;
    const x = k => L + (k - xmin) / (xmax - xmin) * (W - L - Rt);
    // shaded empty band between the max different-family kappa and min same/seed kappa
    const dfMax = Math.max(...pts.filter(p => p.level === "different family").map(p => p.kappa));
    const otMin = Math.min(...pts.filter(p => p.level !== "different family").map(p => p.kappa));
    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="DM Mono, monospace">`;
    svg += `<rect x="${x(dfMax)}" y="52" width="${x(otMin) - x(dfMax)}" height="${axisY - 52}" fill="#f8ecd4"/>`;
    svg += `<text x="${(x(dfMax) + x(otMin)) / 2}" y="44" text-anchor="middle" font-size="12" fill="#8a6d1f">κ ${dfMax.toFixed(2)}–${otMin.toFixed(2)}: no pair lands here</text>`;
    svg += `<line x1="${L}" y1="${axisY}" x2="${W - Rt}" y2="${axisY}" stroke="#1b222c" stroke-width="1.5"/>`;
    for (let t = 0.30; t <= 0.61; t += 0.05) {
      svg += `<line x1="${x(t)}" y1="${axisY}" x2="${x(t)}" y2="${axisY + 6}" stroke="#1b222c"/>`;
      svg += `<text x="${x(t)}" y="${axisY + 22}" text-anchor="middle" font-size="12" fill="#67707b">${t.toFixed(2)}</text>`;
    }
    svg += `<text x="${W - Rt}" y="${axisY + 42}" text-anchor="end" font-size="12" fill="#67707b">Cohen's κ (lower = more decoupled errors)</text>`;
    // jitter dots vertically per level to avoid overlap
    const lanes = { "different family": 96, "same family": 116, "seed only": 140 };
    const seen = {};
    pts.forEach(p => {
      const key = p.kappa.toFixed(2) + p.level;
      seen[key] = (seen[key] || 0) + 1;
      // duplicates within a level stack away from the neighboring lane
      const dir = p.level === "seed only" ? 1 : -1;
      const cy = lanes[p.level] + (seen[key] - 1) * 14 * dir;
      svg += `<circle cx="${x(p.kappa)}" cy="${cy}" r="7" fill="${LEVEL_COLOR[p.level]}" opacity="0.88">` +
             `<title>${esc(p.pair)} (${esc(p.tier)}): κ ${p.kappa}, c ${p.c}%, w ${p.w}%, u ${p.u}%</title></circle>`;
    });
    // legend
    let lx = L;
    Object.keys(LEVEL_COLOR).forEach(lv => {
      svg += `<circle cx="${lx}" cy="16" r="6" fill="${LEVEL_COLOR[lv]}"/>`;
      svg += `<text x="${lx + 12}" y="20" font-size="12.5" fill="#1b222c">${lv}</text>`;
      lx += 22 + lv.length * 7.6;
    });
    svg += `</svg>`;
    $("#kappa-strip").innerHTML = svg;
  }

  function renderPoolTable() {
    let html = `<table class="data-table"><thead><tr>
      <th>Pair</th><th>Level</th><th>κ ↓</th><th>c ↑ (%)</th><th>w ↓ (%)</th><th>u ↑ (%)</th></tr></thead><tbody>`;
    let tier = null;
    R.decPool.forEach(p => {
      if (p.tier !== tier) { tier = p.tier; html += `<tr class="group-row"><td colspan="6">${esc(tier)}</td></tr>`; }
      html += `<tr><td>${esc(p.pair)}</td><td style="font-family:var(--sans);font-size:13px;color:${LEVEL_TEXT[p.level]}">${esc(p.level)}</td>
        <td>${p.kappa.toFixed(2)}</td><td>${p.c.toFixed(1)}</td><td>${p.w.toFixed(1)}</td><td>${p.u.toFixed(1)}</td></tr>`;
    });
    html += "</tbody></table>";
    $("#probe-pool-table").innerHTML = html;
  }

  function renderAnchorCharts() {
    const anchors = [...new Set(R.decAnchor.map(r => r.anchor))];
    const html = anchors.map(a => {
      const rows = R.decAnchor.filter(r => r.anchor === a);
      const bars = rows.map(r => {
        const w = (r.kappa / 0.6) * 100;
        return `<tr><td style="text-align:left;font-family:var(--sans)">${esc(r.partner)}</td>
          <td style="text-align:left;font-family:var(--sans);font-size:12.5px;color:${LEVEL_TEXT[r.level]}">${esc(r.level)}</td>
          <td style="width:45%"><div style="background:${LEVEL_COLOR[r.level]};height:13px;border-radius:4px;width:${w}%"></div></td>
          <td>${r.kappa.toFixed(2)}</td><td>${r.c.toFixed(1)}</td><td>${r.w.toFixed(1)}</td></tr>`;
      }).join("");
      return `<h3 style="font-size:15px;margin:22px 0 6px">Anchor: ${esc(a)} <span style="color:var(--muted);font-weight:400">— partner varies, capability fixed</span></h3>
        <div class="data-table-wrap"><table class="data-table"><thead>
        <tr><th>Partner</th><th>Level</th><th>κ (bar)</th><th>κ ↓</th><th>c ↑ (%)</th><th>w ↓ (%)</th></tr></thead>
        <tbody>${bars}</tbody></table></div>`;
    }).join("");
    $("#anchor-charts").innerHTML = html;
  }

  function setupProbeTabs() {
    $$(".tab-button").forEach(btn => btn.addEventListener("click", () => {
      $$(".tab-button").forEach(b => { b.classList.remove("is-active"); b.setAttribute("aria-pressed", "false"); });
      btn.classList.add("is-active"); btn.setAttribute("aria-pressed", "true");
      $$(".tab-panel").forEach(p => p.hidden = p.id !== btn.dataset.tab);
    }));
  }

  /* =============== RESULTS EXPLORER =============== */
  function methodRowClass(m) {
    if (m === "Base" || m === "GT-Reward") return "ref-row";
    if (m.indexOf("Co-RL") === 0) return "ours-row";
    return "";
  }

  function renderResultsTable(headers, rows, label) {
    let html = `<table class="data-table" aria-label="${esc(label)}"><thead><tr><th>Method</th>`;
    headers.forEach((h, i) => html += `<th class="${i === headers.length - 1 ? "avg-col" : ""}">${esc(h)}</th>`);
    html += `</tr></thead><tbody>`;
    rows.forEach(r => {
      html += `<tr class="${methodRowClass(r.method)}"><td>${esc(r.method)}</td>`;
      r.vals.forEach((v, i) => {
        const mk = r.marks[i] === "b" ? "mark-b" : r.marks[i] === "u" ? "mark-u" : "";
        const avg = i === r.vals.length - 1 ? " avg-col" : "";
        html += `<td class="${mk}${avg}">${fmt(v)}</td>`;
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
    return `<div>${svg}</div><p class="chart-note">Average over the suite. Dashed marks are references
      (excluded from the label-free ranking).</p>`;
  }

  const TEXT_NOTE = `<div class="callout honesty"><span class="tag">Stated plainly</span>
    On Qwen2.5-3B, Same family (48.7) edges Different family (48.5): the downstream ladder is directional,
    not monotone on every backbone, even though the pre-RL κ ladder is clean. What holds on all four text
    backbones: Different family beats the strongest self-rewarding baseline everywhere, and Different family+
    posts the best label-free average everywhere (margin +0.8–2.0%).</div>`;
  const VLM_NOTE = `<div class="callout honesty"><span class="tag">Stated plainly</span>
    On InternVL-3.5-2B with MMR1, TTRL's 45.30 average beats Co-RL's 45.15 — the one setting of four the small
    pair loses. Base is graded once with the corrected multiple-choice grader, so it is identical across the
    two training sets. MMR1 blocks use the corrected grader and open-r1 blocks the legacy grader: the two are
    never compared with each other.</div>
    <p class="chart-note">Vision-language families pair different encoders with different backbones:
    Qwen2.5-VL uses a natively trained dynamic-resolution ViT, InternVL uses InternViT, Gemma 3 uses SigLIP.
    At 7B–12B (InternVL3.5-8B as the shared partner): +7.2% Qwen2.5-VL-7B, +6.3% InternVL3.5-8B,
    +5.8% Gemma-3-12B over base — beating TTRL for all three, and on Gemma-3-12B beating the labeled
    reference outright (47.56 vs 45.17).</p>`;
  const N3_NOTE = `<div class="callout"><span class="tag">One run, three improved models</span>
    Qwen2.5-3B, Llama-3.2-3B-Instruct, and Qwen3-1.7B trained together along the directed ring: average gains
    of +7.8, +6.0, and +8.2 over base, each agent matching or outperforming its own labeled reference.
    Rows are per-agent — do not read them against the two-agent tables.</div>`;

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

  /* =============== CONTROLS CARDS =============== */
  function miniTable(rows, cols) {
    let h = `<table class="mini-table">`;
    rows.forEach(r => {
      h += `<tr><td>${esc(r[0])}</td>` + r.slice(1).map(v => `<td>${esc(v)}</td>`).join("") + `</tr>`;
    });
    return h + `</table>`;
  }

  function renderControls() {
    const eLL = R.ensLLM, gv = (setting) => avgOf(eLL.find(r => r.setting === setting));
    const eV = (g, s) => avgOf(R.ensVLM.find(r => r.group === g && r.setting === s));
    const comAvg = m => avgOf(R.comas.rows.find(r => r.method === m));
    const n3gain = R.n3.map(b => (avgN(b.rows.find(r => r.method.indexOf("Co-RL") === 0)) - avgN(rowBy(b.rows, "Base"))).toFixed(1));

    $("#controls-cards").innerHTML = `
      <div class="control-card">
        <p class="eyebrow">Control 1 · Matched budget</p>
        <h3>Same two models, same rollouts — trained with TTRL instead</h3>
        <div class="big-num">${fmt(gv("Co-RL (ensemble)"))} <small>vs ${fmt(gv("TTRL (ensemble)"))} · text maj@8</small></div>
        <p>The same two base models are trained independently with TTRL; at inference both methods pool four
           rollouts from each model for majority voting, matching the training-model and test-time budgets.
           The individual agents stay visible — the weaker partner is the one that moves:</p>
        ${miniTable([
          ["TTRL — Qwen2.5-3B", fmt(gv("TTRL (Qwen2.5-3B)"))],
          ["TTRL — Llama-3.2-3B", fmt(gv("TTRL (Llama-3.2-3B)"))],
          ["TTRL — ensemble", fmt(gv("TTRL (ensemble)"))],
          ["Co-RL — Qwen2.5-3B", fmt(gv("Co-RL (Qwen2.5-3B)"))],
          ["Co-RL — Llama-3.2-3B", fmt(gv("Co-RL (Llama-3.2-3B)"))],
          ["Co-RL — ensemble", fmt(gv("Co-RL (ensemble)"))]])}
        <p>Multimodal, blocks kept apart: ${fmt(eV("open-r1", "Co-RL (ensemble)"))} vs
           ${fmt(eV("open-r1", "TTRL (ensemble)"))} on open-r1;
           ${fmt(eV("MMR1", "Co-RL (ensemble)"))} vs ${fmt(eV("MMR1", "TTRL (ensemble)"))} on MMR1.
           \u201c\u2026the advantage of Co-RL cannot be explained merely by training two models.\u201d</p>
        <p class="src-note">Protocol: GSM8K, MATH-500, AMC at maj@8, T = 0.6 — a different evaluation regime
          from the seven-benchmark tables, never mixed with them.</p>
      </div>
      <div class="control-card">
        <p class="eyebrow">Control 2 · Versus multi-agent RL</p>
        <h3>CoMAS's own setup, half the agents, no judge</h3>
        <div class="big-num">${fmt(comAvg("Co-RL (Different family)"))} <small>vs ${fmt(comAvg("CoMAS"))} avg</small></div>
        <p>Under CoMAS's setup, official implementation, and evaluation protocol (prior rows quoted from
           their paper): Co-RL leads five of the seven benchmarks and the average — using half as many
           agents, with no LLM judge and no learned reward model.</p>
        ${miniTable([
          ["Base", fmt(comAvg("Base"))],
          ["MAPoRL", fmt(comAvg("MAPoRL"))],
          ["TTRL", fmt(comAvg("TTRL"))],
          ["CoMAS", fmt(comAvg("CoMAS"))],
          ["Co-RL (Different family)", fmt(comAvg("Co-RL (Different family)"))]])}
        <p class="src-note">Caveat that travels with this claim: CoMAS's coding aggregation admits a
          pass@5-vs-pass@1 loophole (worth 7.3% to the untrained baseline, 2.4% to Co-RL); the authors keep
          the five-sample budget but replace the aggregation with majority voting over candidates clustered
          by execution behavior. This table's suite and graders are CoMAS's — its numbers never share an
          axis with the seven-benchmark tables.</p>
      </div>
      <div class="control-card">
        <p class="eyebrow">Control 3 · Three agents</p>
        <h3>One ring, three models, three wins</h3>
        <div class="big-num">+${n3gain[0]} / +${n3gain[1]} / +${n3gain[2]} <small>avg gains over base</small></div>
        <p>Qwen2.5-3B, Llama-3.2-3B-Instruct, and Qwen3-1.7B trained together in a single run along the
           directed ring — models of different families <em>and</em> sizes. Each agent matches or outperforms
           its own labeled reference, and beats TTRL on the two 3B models while tying it on Qwen3-1.7B.</p>
        ${miniTable(R.n3.map(b => [b.model,
          "base " + fmt(avgOf(rowBy(b.rows, "Base"))),
          "Co-RL " + fmt(avgOf(b.rows.find(r => r.method.indexOf("Co-RL") === 0))),
          "GT " + fmt(avgOf(rowBy(b.rows, "GT-Reward")))]))}
        <p class="src-note">Same training configuration as the two-agent language runs; full per-benchmark
          rows in the Benchmarks explorer under "Three agents".</p>
      </div>`;
  }

  /* =============== THEORY PHASE DIAGRAM =============== */
  function renderPhaseDiagram() {
    const S = 340, M = 44, P = S - 2 * M; // plot square
    const px = v => M + v * P, py = v => S - M - v * P;
    let svg = `<svg viewBox="0 0 ${S} ${S}" xmlns="http://www.w3.org/2000/svg" font-family="DM Mono, monospace">`;
    // basins
    svg += `<polygon points="${px(0)},${py(1)} ${px(1)},${py(1)} ${px(1)},${py(0)}" fill="#ece4f3"/>`;
    svg += `<polygon points="${px(0)},${py(1)} ${px(0)},${py(0)} ${px(1)},${py(0)}" fill="#f7e4e4"/>`;
    // frame
    svg += `<rect x="${M}" y="${M}" width="${P}" height="${P}" fill="none" stroke="#1b222c" stroke-width="1.5"/>`;
    // separatrix
    svg += `<line x1="${px(0)}" y1="${py(1)}" x2="${px(1)}" y2="${py(0)}" stroke="#1b222c" stroke-width="1.5" stroke-dasharray="6 4"/>`;
    svg += `<text x="${px(0.56)}" y="${py(0.56)}" font-size="10.5" fill="#1b222c" transform="rotate(-45 ${px(0.56)} ${py(0.56)})">p_A + p_B = 1</text>`;
    // basin labels
    svg += `<text x="${px(0.63)}" y="${py(0.82)}" font-size="11" fill="#5d3a80" text-anchor="middle">→ (1,1) correct</text>`;
    svg += `<text x="${px(0.32)}" y="${py(0.16)}" font-size="11" fill="#be3737" text-anchor="middle">→ (0,0) wrong</text>`;
    // fixed points
    svg += `<circle cx="${px(1)}" cy="${py(1)}" r="6" fill="#147832"/><text x="${px(1) - 10}" y="${py(1) - 8}" font-size="10.5" text-anchor="end" fill="#147832">(1,1) stable</text>`;
    svg += `<circle cx="${px(0)}" cy="${py(0)}" r="6" fill="#be3737"/><text x="${px(0) + 10}" y="${py(0) + 14}" font-size="10.5" fill="#be3737">(0,0) stable</text>`;
    svg += `<circle cx="${px(0.5)}" cy="${py(0.5)}" r="5.5" fill="#fff" stroke="#1b222c" stroke-width="2"/><text x="${px(0.5) + 9}" y="${py(0.5) + 14}" font-size="10.5" fill="#1b222c">saddle (½,½)</text>`;
    // worked example points
    [[0.9, 0.2], [0.2, 0.9]].forEach(pt => {
      svg += `<circle cx="${px(pt[0])}" cy="${py(pt[1])}" r="5" fill="#7b50a2"/>`;
      svg += `<text x="${px(pt[0]) + (pt[0] > 0.5 ? -8 : 8)}" y="${py(pt[1]) - 8}" font-size="10.5" fill="#5d3a80" text-anchor="${pt[0] > 0.5 ? "end" : "start"}">(${pt[0]}, ${pt[1]})</text>`;
    });
    // axes labels
    svg += `<text x="${px(0.5)}" y="${S - 8}" font-size="11" text-anchor="middle" fill="#67707b">p_A — P(agent A correct)</text>`;
    svg += `<text x="12" y="${py(0.5)}" font-size="11" text-anchor="middle" fill="#67707b" transform="rotate(-90 12 ${py(0.5)})">p_B — P(agent B correct)</text>`;
    svg += `</svg>`;
    $("#phase-diagram").innerHTML = svg;
  }

  /* =============== FURNITURE =============== */
  function setupAccordions() {
    $$(".acc-toggle").forEach(btn => btn.addEventListener("click", () => {
      const item = btn.closest(".acc-item");
      const open = item.classList.toggle("is-open");
      btn.setAttribute("aria-expanded", open);
    }));
  }

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
    setupVerdict();
    setupLadder();
    renderKappaStrip();
    renderPoolTable();
    renderAnchorCharts();
    setupProbeTabs();
    setupResults();
    renderControls();
    renderPhaseDiagram();
    setupAccordions();
    setupLightbox();
    setupNav();
    setupBibtex();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
