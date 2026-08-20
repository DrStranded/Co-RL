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
  // Values are read live from siteResults; nothing here is hand-typed.
  function verdictChips(dfOnly) {
    const chips = [];
    R.llmMain.forEach(b => {
      const gt = rowBy(b.rows, "GT-Reward"), base = rowBy(b.rows, "Base"), tt = rowBy(b.rows, "TTRL");
      const cands = b.rows.filter(r => dfOnly ? r.method === "Co-RL (Different family)" : CO_VARIANTS.includes(r.method));
      if (!gt || !base || !cands.length) return;
      const best = cands.reduce((a, r) => avgN(r) > avgN(a) ? r : a);
      chips.push({ domain: "text", model: b.backbone, meta: "text · 7-benchmark avg",
                   variant: best.method, ours: avgOf(best), gt: avgOf(gt), base: avgOf(base), ttrl: avgOf(tt) });
    });
    R.n3.forEach(b => {
      const gt = rowBy(b.rows, "GT-Reward"), base = rowBy(b.rows, "Base"), tt = rowBy(b.rows, "TTRL");
      const co = b.rows.find(r => r.method.indexOf("Co-RL") === 0);
      if (!gt || !base || !co) return;
      chips.push({ domain: "text", model: b.model, meta: "text · three-agent ring · 7-benchmark avg",
                   variant: "Co-RL (Different family), N=3", ours: avgOf(co), gt: avgOf(gt), base: avgOf(base), ttrl: avgOf(tt) });
    });
    [...R.vlmSmall, ...R.vlmLarge].forEach(b => {
      const gt = rowBy(b.rows, "GT-Reward"), base = rowBy(b.rows, "Base"), tt = rowBy(b.rows, "TTRL");
      const co = b.rows.find(r => r.method.indexOf("Co-RL") === 0);
      if (!gt || !base || !co) return;
      chips.push({ domain: "vlm", model: b.backbone, meta: "multimodal · " + b.dataset + " · 4-benchmark avg",
                   variant: co.method, ours: avgOf(co), gt: avgOf(gt), base: avgOf(base), ttrl: avgOf(tt) });
    });
    return chips;
  }

  function renderVerdict() {
    const domain = $("#verdict-domain").value;
    const dfOnly = $("#verdict-df-only").classList.contains("is-active");
    const chips = verdictChips(dfOnly).filter(c => domain === "all" || c.domain === domain);
    const groups = [
      { name: "Text · two-agent pairs", rows: chips.filter(c => c.domain === "text" && c.meta.indexOf("ring") < 0) },
      { name: "Text · three-agent ring", rows: chips.filter(c => c.meta.indexOf("ring") >= 0) },
      { name: "Vision-language", rows: chips.filter(c => c.domain === "vlm") },
    ].filter(g => g.rows.length);
    const deltas = c => ({ co: nf(c.ours) - nf(c.base), tt: nf(c.ttrl) - nf(c.base), gt: nf(c.gt) - nf(c.base) });
    const MAX = Math.max(...chips.map(c => { const d = deltas(c); return Math.max(d.co, d.tt, d.gt, 0); })) * 1.08;
    const pct = v => (Math.max(v, 0) / MAX * 100).toFixed(1);
    let html = "";
    groups.forEach(g => {
      html += `<div class="dr-group">${esc(g.name)}</div>`;
      g.rows.forEach(c => {
        const d = deltas(c);
        const meta = c.domain === "vlm" ? c.meta.split("·")[1].trim() : (c.meta.indexOf("ring") >= 0 ? "N=3" : c.variant.replace("Co-RL ", "").replace(/[()]/g, ""));
        html += `<div class="dr-row">
          <span class="dr-label"><b>${esc(c.model)}</b><i>${esc(meta)}</i></span>
          <span class="vb-track" title="Gains over Base ${fmt(c.base)}: Co-RL +${d.co.toFixed(2)}, TTRL +${d.tt.toFixed(2)}, GT-Reward +${d.gt.toFixed(2)}">
            <span class="vb-bar co" style="width:${pct(d.co)}%"></span>
            <span class="vb-bar tt" style="width:${pct(d.tt)}%"></span>
            <span class="vb-gt" style="left:${pct(d.gt)}%"></span>
          </span>
          <span class="dr-nums"><b class="n-co">${fmt(c.ours)}</b> <em>TTRL ${fmt(c.ttrl)} · GT ${fmt(c.gt)}</em></span>
        </div>`;
      });
    });
    $("#verdict-chart").innerHTML = html;
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
      <h3>Rung ${esc(rung.index)}: ${esc(rung.label)}</h3>
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
    const XMAX = 0.62, B0 = 0.42, B1 = 0.51;
    const pct = v => (v / XMAX * 100).toFixed(2);
    const order = ["different family", "same family", "seed only"];
    let html = "";
    order.forEach(lv => {
      const rows = R.decPool.filter(p => p.level === lv).sort((a, b) => a.kappa - b.kappa);
      const lo = Math.min(...rows.map(p => p.kappa)).toFixed(2);
      const hi = Math.max(...rows.map(p => p.kappa)).toFixed(2);
      html += `<div class="dr-group">${esc(lv)} · κ ${lo}–${hi}</div>`;
      rows.forEach(p => {
        html += `<div class="dr-row kb-row">
          <span class="dr-label"><b>${esc(p.pair)}</b><i>${esc(p.tier)}</i></span>
          <span class="kb-track" title="${esc(p.pair)}: κ ${p.kappa}, complementarity ${p.c}%, wrong-agreement ${p.w}%, oracle ${p.u}%">
            <span class="kb-band" style="left:${pct(B0)}%;width:${pct(B1 - B0)}%"></span>
            <span class="kb-bar" style="width:${pct(p.kappa)}%;background:${LEVEL_COLOR[p.level]}"></span>
          </span>
          <span class="dr-nums"><b>${p.kappa.toFixed(2)}</b></span>
        </div>`;
      });
    });
    html += `<p class="chart-note">Bars are the error overlap κ for each base-checkpoint pair, shorter is more
      decoupled. The shaded strip is κ ${B0}–${B1}: no pair ends inside it. Hover a bar for the pair's
      complementarity and wrong-agreement.</p>`;
    $("#kappa-strip").innerHTML = html;
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
    On Qwen2.5-3B, Same family (48.7) edges Different family (48.5). The downstream ladder is directional,
    not monotone on every backbone, even though the pre-RL κ ladder is clean. What holds on all four text
    backbones: Different family beats the strongest self-rewarding baseline everywhere, and Different family+
    posts the best label-free average everywhere, by 0.8 to 2.0 points.</div>`;
  const VLM_NOTE = `<div class="callout honesty"><span class="tag">Stated plainly</span>
    On InternVL-3.5-2B with MMR1, TTRL's 45.30 average beats Co-RL's 45.15. That is the one setting of four
    the small pair loses. Base is graded once with the corrected multiple-choice grader, so it is identical
    across the two training sets. MMR1 blocks use the corrected grader and open-r1 blocks the legacy grader,
    and the two are never compared with each other.</div>
    <p class="chart-note">Vision-language families pair different encoders with different backbones:
    Qwen2.5-VL uses a natively trained dynamic-resolution ViT, InternVL uses InternViT, Gemma 3 uses SigLIP.
    At 7B to 12B, with InternVL3.5-8B as the shared partner, Co-RL improves the base models by 7.2%, 6.3%,
    and 5.8%, beats TTRL for all three families, and on Gemma-3-12B beats the labeled reference outright
    (47.56 vs 45.17).</p>`;
  const N3_NOTE = `<div class="callout"><span class="tag">One run, three improved models</span>
    Qwen2.5-3B, Llama-3.2-3B-Instruct, and Qwen3-1.7B trained together along the directed ring, with average
    gains of 7.8, 6.0, and 8.2 points over base. Each agent matches or outperforms its own labeled reference.
    Rows are per-agent, so do not read them against the two-agent tables.</div>`;

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

  /* =============== OBJECTIONS CARDS =============== */
  // Evidence bars in the reference page's style: label, track, printed value.
  // Bars start at zero, so ranking is visual and precision stays in the number.
  function barRows(items, maxHint) {
    const max = maxHint || Math.max(...items.map(it => nf(it.value))) * 1.06;
    return `<div class="bar-rows">` + items.map(it => {
      const w = (nf(it.value) / max * 100).toFixed(1);
      return `<div class="bar-row">
        <span class="bar-label">${esc(it.label)}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${w}%;background:${it.color}"></span></span>
        <span class="bar-value">${fmt(it.value)}</span></div>`;
    }).join("") + `</div>`;
  }

  const BLUE = "#0072b2", VIOLET = "#7b50a2", GRAY = "#a9b2b9";

  function renderControls() {
    const eLL = R.ensLLM, gv = s => avgOf(eLL.find(r => r.setting === s));
    const eV = (g, s) => avgOf(R.ensVLM.find(r => r.group === g && r.setting === s));
    const comAvg = m => avgOf(R.comas.rows.find(r => r.method === m));
    const n3gain = R.n3.map(b => (avgN(b.rows.find(r => r.method.indexOf("Co-RL") === 0)) - avgN(rowBy(b.rows, "Base"))).toFixed(1));

    const ensBars = barRows([
      { label: "TTRL · Qwen2.5-3B", value: gv("TTRL (Qwen2.5-3B)"), color: BLUE },
      { label: "TTRL · Llama-3.2-3B", value: gv("TTRL (Llama-3.2-3B)"), color: BLUE },
      { label: "TTRL · ensemble", value: gv("TTRL (ensemble)"), color: BLUE },
      { label: "Co-RL · Qwen2.5-3B", value: gv("Co-RL (Qwen2.5-3B)"), color: VIOLET },
      { label: "Co-RL · Llama-3.2-3B", value: gv("Co-RL (Llama-3.2-3B)"), color: VIOLET },
      { label: "Co-RL · ensemble", value: gv("Co-RL (ensemble)"), color: VIOLET },
    ]);
    const comasBars = barRows([
      { label: "Base", value: comAvg("Base"), color: GRAY },
      { label: "MAPoRL", value: comAvg("MAPoRL"), color: BLUE },
      { label: "TTRL", value: comAvg("TTRL"), color: BLUE },
      { label: "CoMAS", value: comAvg("CoMAS"), color: BLUE },
      { label: "Co-RL (Different family)", value: comAvg("Co-RL (Different family)"), color: VIOLET },
    ]);
    const n3Bars = barRows(R.n3.flatMap(b => [
      { label: b.model + " · base", value: avgOf(rowBy(b.rows, "Base")), color: GRAY },
      { label: b.model + " · Co-RL", value: avgOf(b.rows.find(r => r.method.indexOf("Co-RL") === 0)), color: VIOLET },
      { label: b.model + " · GT-Reward", value: avgOf(rowBy(b.rows, "GT-Reward")), color: BLUE },
    ]));

    $("#controls-cards").innerHTML = `
      <div class="control-card">
        <p class="eyebrow">Objection 1 · You trained two models</p>
        <h3>Train the same two models with TTRL instead, and the gain does not appear.</h3>
        <div class="big-num">${fmt(gv("Co-RL (ensemble)"))} <small>vs ${fmt(gv("TTRL (ensemble)"))} · text maj@8</small></div>
        <p>The same two base models are trained independently with TTRL. At inference both methods pool four
           rollouts from each model for majority voting, matching the training-model and test-time budgets.
           The weaker partner is the one that moves:</p>
        ${ensBars}
        <p>Multimodal, blocks kept apart: ${fmt(eV("open-r1", "Co-RL (ensemble)"))} vs
           ${fmt(eV("open-r1", "TTRL (ensemble)"))} on open-r1, and
           ${fmt(eV("MMR1", "Co-RL (ensemble)"))} vs ${fmt(eV("MMR1", "TTRL (ensemble)"))} on MMR1.
           In the paper's words, “…the advantage of Co-RL cannot be explained merely by training two
           models.”</p>
        <p class="src-note">Protocol: GSM8K, MATH-500, AMC at maj@8, T = 0.6. This is a different evaluation
          regime from the seven-benchmark tables and is never mixed with them.</p>
      </div>
      <div class="control-card">
        <p class="eyebrow">Objection 2 · Multi-agent RL already does this</p>
        <h3>Under CoMAS's own setup, Co-RL wins with half the agents and no judge.</h3>
        <div class="big-num">${fmt(comAvg("Co-RL (Different family)"))} <small>vs ${fmt(comAvg("CoMAS"))} avg</small></div>
        <p>Same setup, official implementation, and evaluation protocol as CoMAS, with prior rows quoted from
           their paper. Co-RL leads five of the seven benchmarks and the average, with no LLM judge and no
           learned reward model.</p>
        ${comasBars}
        <p class="src-note">Caveat that travels with this claim: CoMAS's coding aggregation admits a
          pass@5-versus-pass@1 loophole, worth 7.3% to the untrained baseline and 2.4% to Co-RL. The authors
          keep the five-sample budget but replace the aggregation with majority voting over candidates
          clustered by execution behavior. This table's suite and graders are CoMAS's, and its numbers never
          share an axis with the seven-benchmark tables.</p>
      </div>
      <div class="control-card">
        <p class="eyebrow">Objection 3 · Does it scale past two agents?</p>
        <h3>One ring trains three models at once, and all three improve.</h3>
        <div class="big-num">+${n3gain[0]} / +${n3gain[1]} / +${n3gain[2]} <small>avg gains over base</small></div>
        <p>Qwen2.5-3B, Llama-3.2-3B-Instruct, and Qwen3-1.7B trained together in a single run along the
           directed ring: models of different families and sizes. Each agent matches or outperforms its own
           labeled reference, and beats TTRL on the two 3B models while tying it on Qwen3-1.7B.</p>
        ${n3Bars}
        <p class="src-note">Same training configuration as the two-agent language runs. Full per-benchmark
          rows are in the Benchmarks explorer under Three agents.</p>
      </div>`;
  }

  /* =============== THEORY PHASE DIAGRAM =============== */
  function renderPhaseDiagram() {
    const S = 340, M = 44, P = S - 2 * M;
    const px = v => M + v * P, py = v => S - M - v * P;
    let svg = `<svg viewBox="0 0 ${S} ${S}" xmlns="http://www.w3.org/2000/svg" font-family="DM Mono, monospace">`;
    svg += `<polygon points="${px(0)},${py(1)} ${px(1)},${py(1)} ${px(1)},${py(0)}" fill="#ece4f3"/>`;
    svg += `<polygon points="${px(0)},${py(1)} ${px(0)},${py(0)} ${px(1)},${py(0)}" fill="#f7e4e4"/>`;
    svg += `<rect x="${M}" y="${M}" width="${P}" height="${P}" fill="none" stroke="#1b222c" stroke-width="1.5"/>`;
    svg += `<line x1="${px(0)}" y1="${py(1)}" x2="${px(1)}" y2="${py(0)}" stroke="#1b222c" stroke-width="1.5" stroke-dasharray="6 4"/>`;
    svg += `<text x="${px(0.30)}" y="${py(0.66)}" font-size="10.5" fill="#1b222c" text-anchor="middle" transform="rotate(-45 ${px(0.30)} ${py(0.66)})">chances sum to 1</text>`;
    svg += `<text x="${px(0.63)}" y="${py(0.86)}" font-size="11" fill="#5d3a80" text-anchor="middle">end up right together</text>`;
    svg += `<text x="${px(0.32)}" y="${py(0.13)}" font-size="11" fill="#be3737" text-anchor="middle">end up wrong together</text>`;
    svg += `<circle cx="${px(1)}" cy="${py(1)}" r="6" fill="#5d3a80"/><text x="${px(1) - 10}" y="${py(1) - 8}" font-size="10.5" text-anchor="end" fill="#5d3a80">both right, stable</text>`;
    svg += `<circle cx="${px(0)}" cy="${py(0)}" r="6" fill="#be3737"/><text x="${px(0) + 10}" y="${py(0) + 14}" font-size="10.5" fill="#be3737">both wrong, stable</text>`;
    svg += `<circle cx="${px(0.5)}" cy="${py(0.5)}" r="5.5" fill="#fff" stroke="#1b222c" stroke-width="2"/><text x="${px(0.5) + 9}" y="${py(0.5) + 14}" font-size="10.5" fill="#1b222c">tipping point</text>`;
    [[0.9, 0.2], [0.2, 0.9]].forEach(pt => {
      svg += `<circle cx="${px(pt[0])}" cy="${py(pt[1])}" r="5" fill="#7b50a2"/>`;
      svg += `<text x="${px(pt[0]) + (pt[0] > 0.5 ? -8 : 8)}" y="${py(pt[1]) - 8}" font-size="10.5" fill="#5d3a80" text-anchor="${pt[0] > 0.5 ? "end" : "start"}">(${pt[0]}, ${pt[1]})</text>`;
    });
    svg += `<text x="${px(0.5)}" y="${S - 8}" font-size="11" text-anchor="middle" fill="#67707b">chance agent A is right</text>`;
    svg += `<text x="12" y="${py(0.5)}" font-size="11" text-anchor="middle" fill="#67707b" transform="rotate(-90 12 ${py(0.5)})">chance agent B is right</text>`;
    svg += `</svg>`;
    $("#phase-diagram").innerHTML = svg;
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
    setupVerdict();
    setupLadder();
    renderKappaStrip();
    renderPoolTable();
    setupResults();
    renderControls();
    renderPhaseDiagram();
    setupLightbox();
    setupNav();
    setupBibtex();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
