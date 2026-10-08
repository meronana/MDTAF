import { SITE, ENDPOINTS, PHASES, DOMAINS, RHO_BANDS } from './config.js';
import { navHTML, setupTheme } from './common.js';

const NS = 'http://www.w3.org/2000/svg';
const f2 = (x) => (x == null ? '—' : x.toFixed(2));
const f3 = (x) => (x == null ? '—' : x.toFixed(3));
const signed = (x) => (x == null ? '—' : (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(3));
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- tooltip
const tip = document.createElement('div');
tip.className = 'tip';
document.body.appendChild(tip);

// Labels come from CSVs, so they go in only via textContent
function showTip(evt, rows) {
  tip.replaceChildren(
    ...rows.map(([cls, txt]) => {
      const d = document.createElement('div');
      d.className = cls;
      d.textContent = txt;
      return d;
    }),
  );
  tip.classList.add('on');
  const r = tip.getBoundingClientRect();
  const x = Math.min(evt.clientX + 14, window.innerWidth - r.width - 8);
  const y = evt.clientY + r.height + 20 > window.innerHeight ? evt.clientY - r.height - 12 : evt.clientY + 16;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => tip.classList.remove('on');

// ---------------------------------------------------------------- svg helpers
function el(name, attrs = {}, parent) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function text(parent, x, y, str, cls, anchor = 'start', extra = {}) {
  const t = el('text', { x, y, class: cls, 'text-anchor': anchor, 'dominant-baseline': 'middle', ...extra }, parent);
  t.textContent = str;
  return t;
}
const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
function hitTarget(g, x, y, w, h, rows, mark) {
  const r = el('rect', { x, y, width: w, height: h, class: 'hit', tabindex: 0 }, g);
  const on = (e) => {
    mark?.classList.add('hover');
    const box = r.getBoundingClientRect();
    showTip(e.clientX != null ? e : { clientX: box.x + box.width / 2, clientY: box.y }, rows);
  };
  const off = () => {
    mark?.classList.remove('hover');
    hideTip();
  };
  r.addEventListener('pointermove', on);
  r.addEventListener('pointerleave', off);
  r.addEventListener('focus', on);
  r.addEventListener('blur', off);
}
function tableView(container, header, rows) {
  const d = document.createElement('details');
  d.className = 'table';
  const s = document.createElement('summary');
  s.textContent = 'Show data table';
  const wrap = document.createElement('div');
  wrap.className = 'tbl-wrap';
  const t = document.createElement('table');
  const tr = t.insertRow();
  for (const h of header) {
    const th = document.createElement('th');
    th.textContent = h;
    tr.appendChild(th);
  }
  for (const r of rows) {
    const row = t.insertRow();
    for (const c of r) row.insertCell().textContent = c;
  }
  wrap.appendChild(t);
  d.append(s, wrap);
  container.appendChild(d);
}
// Redraw to the container width (real px instead of a viewBox so text does not shrink on mobile)
function responsive(host, draw) {
  let last = 0;
  const update = () => {
    const w = Math.round(host.clientWidth);
    if (w && w !== last) {
      last = w;
      draw(w);
    }
  };
  // Render immediately the first time: in a background tab the ResizeObserver callback arrives late (or never)
  update();
  new ResizeObserver(update).observe(host);
}

// ---------------------------------------------------------------- ΔR² chart (bar or forest)
// rows: [{label, v, ci?, control?, tip}]
function deltaChart(host, rows, { domain, ticks, forest }) {
  responsive(host, (W) => {
    host.replaceChildren();
    const narrow = W < 560;
    const L = narrow ? 112 : 190, R = 70, rowH = 34, top = 6;
    const bottom = top + rows.length * rowH;
    const svg = el('svg', { class: 'chart', width: W, height: bottom + 46, role: 'img' }, host);
    const x = scale(domain[0], domain[1], L, W - R);
    const clamp = (v) => Math.max(domain[0], Math.min(domain[1], v));
    for (const t of ticks) {
      el('line', { x1: x(t), x2: x(t), y1: top, y2: bottom, class: t === 0 ? 'zero-line' : 'grid-line' }, svg);
      text(svg, x(t), bottom + 14, (t > 0 ? '+' : t < 0 ? '−' : '') + Math.abs(t).toFixed(2), 'ax-tick', 'middle');
    }
    text(svg, (x(ticks[0]) + x(ticks.at(-1))) / 2, bottom + 36, 'ΔR² vs target-only baseline', 'ax-label', 'middle');
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2;
      text(svg, L - 12, y, r.label, r.control ? 'row-label muted' : 'row-label', 'end');
      const g = el('g', {}, svg);
      let mark;
      if (forest) {
        const color = r.control ? 'var(--neutral-mark)' : 'var(--accent)';
        el('line', { x1: x(clamp(r.ci[0])), x2: x(clamp(r.ci[1])), y1: y, y2: y, stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round' }, g);
        mark = el('circle', { cx: x(clamp(r.v)), cy: y, r: 5.5, fill: r.control ? 'var(--surface)' : color, stroke: color, 'stroke-width': 2, class: 'mark' }, g);
      } else {
        const color = r.control ? 'var(--neutral-mark)' : r.v >= 0 ? 'var(--pos)' : 'var(--neg)';
        const x0 = x(0), x1 = x(clamp(r.v));
        mark = el('rect', { x: Math.min(x0, x1), y: y - 9, width: Math.max(1, Math.abs(x1 - x0)), height: 18, rx: 4, fill: color, class: 'mark' }, g);
      }
      const right = forest ? Math.max(r.v, r.ci[1]) : Math.max(r.v, 0);
      text(svg, Math.min(x(clamp(right)) + 8, W - R + 6), y, signed(r.v), 'ax-tick');
      hitTarget(g, L, y - rowH / 2, W - R - L, rowH, r.tip, mark);
    });
  });
}

// ---------------------------------------------------------------- pair resolution
// Selection → which entry of the precomputed results
function resolvePair(src, tgt) {
  if (src.mode === 'upload' || tgt.mode === 'upload') return { kind: 'upload' };
  if (src.ep !== tgt.ep) return { kind: 'mismatch' };
  if (src.domain === 'invitro' && tgt.domain === 'human') return { kind: 'main', ep: src.ep };
  if (src.ep === 'CL') {
    const name = `${src.domain === 'invitro' ? 'in vitro' : 'animal'} → ${tgt.domain}`;
    return { kind: 'cross', name };
  }
  return { kind: 'none' };
}
const domainName = (side, id) => DOMAINS[side].find((d) => d.id === id).name;

// ---------------------------------------------------------------- page
async function main() {
  const data = await (await fetch(`${import.meta.env.BASE_URL}data/site.json`)).json();

  const sidePanel = (side, title) => `
    <div class="pick card" id="pick-${side}">
      <div class="pick-head"><h3>${title}</h3>
        <div class="seg" role="group" aria-label="${title} data mode">
          <button type="button" data-mode="bench" aria-pressed="true">Benchmark data</button>
          <button type="button" data-mode="upload" aria-pressed="false">Upload CSV</button>
        </div>
      </div>
      <div class="mode-bench">
        <label class="field"><span>Domain</span>
          <select data-k="domain">${DOMAINS[side].map((d) => `<option value="${d.id}">${d.name}</option>`).join('')}</select>
        </label>
        <p class="hint" data-k="note"></p>
        <div class="field"><span>Endpoint</span><div class="chips" data-k="ep"></div></div>
      </div>
      <div class="mode-upload" hidden>
        <label class="drop">
          <input type="file" accept=".csv,text/csv" data-k="file" />
          <span>Choose a CSV with a <code>smiles</code> column and one numeric label column</span>
        </label>
        <p class="hint" data-k="filemsg">The file is read in your browser only; nothing is uploaded yet.</p>
      </div>
    </div>`;

  $('app').innerHTML = `
  ${navHTML('home')}

  <header class="hero wrap" id="top">
    <p class="eyebrow">Transfer learning · ADME / PK</p>
    <h1>${SITE.title}</h1>
    <p class="sub">${SITE.subtitle}</p>
    <p class="authors">${SITE.authors}</p>
    <p class="aff">${SITE.affiliation}</p>
    <div class="btns">${SITE.links.map((l) => `<a class="btn${l.soon ? ' soon' : ''}" href="${l.href}">${l.label}</a>`).join('')}
      <a class="btn" href="#assess">Assess a pair</a><a class="btn" href="data.html">Download data</a></div>
  </header>

  <main class="wrap">
    <section id="framework">
      <p class="eyebrow">Framework</p>
      <h2>Should you transfer from this source?</h2>
      <p class="lede">Before training a transfer model, the framework profiles a source–target pair along input-space and
        label-space dimensions, then checks that profile against what domain adaptation actually delivers.</p>
      <div class="card figure">
        <img src="figures/architecture.png" alt="Framework overview: (a) data processing and feature encoding, (b) model training and domain adaptation" loading="lazy" />
        <p class="caption"><strong>(a)</strong> Source and target SMILES are preprocessed and encoded by a frozen Graphormer.
          <strong>(b)</strong> An MLP feature extractor (768 → 256 → 128) maps both domains to a shared latent space; a domain
          adaptation loss (MMD / CORAL / DANN / CDAN / IW) and the target regression loss are combined as
          ℒ = ℒ<sub>y</sub><sup>T</sup> + λ ℒ<sub>DA</sub> to update the extractor.</p>
      </div>
      <div class="phases" style="margin-top:20px">${PHASES.map((p) => `
        <div class="phase${p.key ? ' key' : ''}"><span class="n">PHASE ${String(p.n).padStart(2, '0')}</span><span class="g">${p.group}</span>
        <h3>${p.title}</h3><p>${p.body}</p></div>`).join('')}</div>
    </section>

    <section id="assess">
      <p class="eyebrow">Assess</p>
      <h2>Pick a source and a target</h2>
      <p class="lede">Choose a domain and endpoint for each side, then run the assessment. Benchmark pairs show precomputed
        results; uploaded data will be assessed by the prediction server once it is online.</p>
      <div class="pair">
        ${sidePanel('source', 'Source')}
        <div class="arrow" aria-hidden="true">→</div>
        ${sidePanel('target', 'Target')}
      </div>
      <div class="run-row">
        <p class="hint" id="pair-msg"></p>
        <button class="btn run" id="run" type="button">Run assessment</button>
      </div>
      <div id="result" aria-live="polite"></div>
    </section>

    <section id="try">
      <p class="eyebrow">Try it</p>
      <h2>Predict f<sub>u</sub>, CL and t<sub>½</sub> for your molecule</h2>
      <div class="try">
        <input type="text" value="CC(=O)Oc1ccccc1C(=O)O" disabled aria-label="SMILES input (coming soon)" />
        <p>The prediction server is under construction. It will return all three endpoints together with a transferability flag.</p>
      </div>
    </section>

    <section id="cite">
      <p class="eyebrow">Citation</p>
      <h2>Cite this work</h2>
      <pre class="cite">@article{tbd,
  title   = {${SITE.title}},
  author  = {${SITE.authors}},
  year    = {2026}
}</pre>
    </section>
  </main>
  <footer>${SITE.affiliation}</footer>`;

  setupTheme();

  // ---------- selection state
  const state = {
    source: { mode: 'bench', domain: 'invitro', ep: 'fu', file: null },
    target: { mode: 'bench', domain: 'human', ep: 'fu', file: null },
  };

  function renderSide(side) {
    const s = state[side];
    const root = $(`pick-${side}`);
    const dom = DOMAINS[side].find((d) => d.id === s.domain);
    if (!dom.endpoints.includes(s.ep)) s.ep = dom.endpoints[0];
    root.querySelector('[data-k=note]').textContent = dom.note;
    const chips = root.querySelector('[data-k=ep]');
    chips.replaceChildren(
      ...Object.entries(ENDPOINTS).map(([k, e]) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'chip';
        b.innerHTML = e.label; // fixed label from config.js
        const ok = dom.endpoints.includes(k);
        b.disabled = !ok;
        b.title = ok ? e.name : `${e.name}: no ${dom.name} data`;
        b.setAttribute('aria-pressed', String(k === s.ep));
        b.onclick = () => {
          s.ep = k;
          // Align the other side to the same endpoint too (only if that domain has data for it)
          const other = side === 'source' ? 'target' : 'source';
          const od = DOMAINS[other].find((d) => d.id === state[other].domain);
          if (od.endpoints.includes(k)) state[other].ep = k;
          renderSide('source');
          renderSide('target');
          updateMsg();
        };
        return b;
      }),
    );
    root.querySelectorAll('.pick-head .seg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === s.mode)));
    root.querySelector('.mode-bench').hidden = s.mode !== 'bench';
    root.querySelector('.mode-upload').hidden = s.mode !== 'upload';
  }

  for (const side of ['source', 'target']) {
    const root = $(`pick-${side}`);
    root.querySelector('[data-k=domain]').value = state[side].domain;
    root.querySelector('[data-k=domain]').onchange = (e) => {
      state[side].domain = e.target.value;
      renderSide(side);
      updateMsg();
    };
    root.querySelectorAll('.pick-head .seg button').forEach((b) => {
      b.onclick = () => {
        state[side].mode = b.dataset.mode;
        renderSide(side);
        updateMsg();
      };
    });
    root.querySelector('[data-k=file]').onchange = async (e) => {
      const msg = root.querySelector('[data-k=filemsg]');
      const file = e.target.files[0];
      state[side].file = file ? await inspectCsv(file) : null;
      msg.textContent = state[side].file ? state[side].file.summary : 'No file selected.';
      msg.classList.toggle('bad', !!state[side].file && !state[side].file.ok);
      updateMsg();
    };
    renderSide(side);
  }

  function updateMsg() {
    const p = resolvePair(state.source, state.target);
    const msg = {
      main: 'Benchmark pair: full nine-phase results available.',
      cross: 'Benchmark pair: label correspondence and transfer results available.',
      mismatch: 'Source and target endpoints differ. The framework compares the same property across domains.',
      none: 'No precomputed results for this combination.',
      upload: 'Uploaded data needs the prediction server, which is not online yet.',
    }[p.kind];
    $('pair-msg').textContent = msg;
    $('run').disabled = p.kind === 'mismatch';
  }
  updateMsg();

  $('run').onclick = () => {
    const out = $('result');
    out.replaceChildren();
    const p = resolvePair(state.source, state.target);
    if (p.kind === 'main') renderMain(out, data, p.ep);
    else if (p.kind === 'cross') renderCross(out, data, p.name, state);
    else renderEmpty(out, p.kind, state);
    out.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
}

// ---------------------------------------------------------------- CSV check (read in the browser only)
async function inspectCsv(file) {
  const txt = await file.text();
  const lines = txt.split(/\r?\n/).filter((l) => l.trim());
  const head = (lines[0] || '').split(',').map((h) => h.trim().toLowerCase());
  const si = head.findIndex((h) => h === 'smiles' || h.includes('smiles'));
  const rows = lines.slice(1).map((l) => l.split(','));
  const li = head.findIndex((h, i) => i !== si && rows.slice(0, 50).every((r) => r[i] !== undefined && r[i].trim() !== '' && !isNaN(Number(r[i]))));
  const ok = si >= 0 && li >= 0 && rows.length > 0;
  return {
    ok,
    n: rows.length,
    summary: ok
      ? `${file.name}: ${rows.length.toLocaleString()} rows · smiles = "${head[si]}", label = "${head[li]}"`
      : `${file.name}: needs a smiles column and a numeric label column (found: ${head.join(', ') || 'nothing'})`,
  };
}

// ---------------------------------------------------------------- result views
function bandFor(rho) {
  return RHO_BANDS.find((b) => rho >= b.min);
}
function verdictCard(title, rho, ci, extra) {
  const b = bandFor(rho);
  return `
    <div class="verdict band-${b.label.toLowerCase()}">
      <div>
        <p class="eyebrow">Label correspondence</p>
        <div class="vbig">${b.label}</div>
        <p class="vtext">${b.text}</p>
      </div>
      <div class="vnum">
        <div class="v">ρ = ${f2(rho)}</div>
        <div class="k">${ci ? `95% CI [${f2(ci[0])}, ${f2(ci[1])}] · ` : ''}${extra}</div>
      </div>
    </div>`;
}
function resultHeader(src, tgt, epText) {
  return `<div class="res-head"><h3>${src} → ${tgt}</h3><span class="tag">${epText}</span></div>`;
}

function renderMain(out, data, ep) {
  const d = data.diagnostics[ep];
  const n = data.datasets[ep];
  const e = ENDPOINTS[ep];
  out.innerHTML = `
    <div class="result card">
      ${resultHeader('In vitro ADME', 'Human in vivo PK', e.label + ' · ' + e.name)}
      ${verdictCard('', d.nn_rho, d.nn_rho_ci, `${d.nn_pairs} nearest-neighbour pairs`)}
      <h4>Phase 1–5 · Domain profile</h4>
      <div class="stats">
        <div class="stat"><div class="v">${n.source.toLocaleString()}</div><div class="k">source compounds</div></div>
        <div class="stat"><div class="v">${n.target.toLocaleString()}</div><div class="k">target compounds</div></div>
        <div class="stat"><div class="v">${f2(d.tanimoto)}</div><div class="k">mean Tanimoto similarity</div></div>
        <div class="stat"><div class="v">${f2(d.mmd)}</div><div class="k">MMD, Graphormer space</div></div>
        <div class="stat"><div class="v">${f2(d.domain_auc)}</div><div class="k">domain-classifier AUC</div></div>
        <div class="stat"><div class="v">${f2(d.within_rho)}</div><div class="k">within-target ρ (ceiling)</div></div>
      </div>
      <h4>Phase 8 · Input-alignment DA</h4>
      <p class="hint">Best λ per method, scaffold 5-fold CV.</p>
      <div id="r-da"></div><div id="t-da"></div>
      <h4>Label-aware transfer</h4>
      <p class="hint">Bars show bootstrap 95% CI. Hollow points are shuffled-label controls.</p>
      ${'<div class="legend"><span><i class="line" style="background:var(--accent)"></i>method</span><span><i class="line" style="background:var(--neutral-mark)"></i>shuffled-label control</span></div>'}
      <div id="r-prop"></div><div id="t-prop"></div>
    </div>`;

  const base = data.da.find((r) => r.endpoint === ep && r.method === 'Baseline').r2;
  const daRows = data.da
    .filter((r) => r.endpoint === ep && r.method !== 'Baseline')
    .map((r) => {
      const v = r.r2 - base;
      return {
        label: r.method === 'Importance Weighting' ? (innerWidth < 560 ? 'IW' : 'Importance wt.') : r.method,
        v,
        tip: [['tv', `ΔR² ${signed(v)}`], ['tl', r.method], ['tr', `R² ${f3(r.r2)} vs ${f3(base)} · λ = ${r.lambda}`]],
      };
    });
  deltaChart($('r-da'), daRows, { domain: [-0.1, 0.1], ticks: [-0.1, -0.05, 0, 0.05, 0.1] });
  tableView(
    $('t-da'),
    ['Method', 'λ', 'R²', 'ρ', 'MAE', 'RMSE'],
    data.da.filter((r) => r.endpoint === ep).map((r) => [r.method, r.lambda, f3(r.r2), f3(r.rho), f3(r.mae), f3(r.rmse)]),
  );

  const order = ['PTFT', 'PTFT-shuffled', 'LCW', 'LCW-rel', 'LCW-corr', 'IVP', 'IVP-shuffled', 'CMMD'];
  const prop = order.map((m) => data.proposed.find((r) => r.endpoint === ep && r.method === m)).filter(Boolean);
  deltaChart(
    $('r-prop'),
    prop.map((r) => ({
      label: r.method,
      v: r.dr2,
      ci: r.ci,
      control: r.method.endsWith('shuffled'),
      tip: [['tv', `ΔR² ${signed(r.dr2)}  [${f3(r.ci[0])}, ${f3(r.ci[1])}]`], ['tl', r.method], ['tr', `p = ${f3(r.p)} · Holm p = ${f3(r.p_holm)}`]],
    })),
    { domain: [-0.06, 0.16], ticks: [-0.05, 0, 0.05, 0.1, 0.15], forest: true },
  );
  tableView(
    $('t-prop'),
    ['Method', 'R²', 'ρ', 'ΔR²', '95% CI', 'p', 'Holm p'],
    data.proposed.filter((r) => r.endpoint === ep).map((r) => [r.method, f3(r.r2), f3(r.rho), signed(r.dr2), r.dr2 == null ? '—' : `[${f3(r.ci[0])}, ${f3(r.ci[1])}]`, f3(r.p), f3(r.p_holm)]),
  );
}

function renderCross(out, data, name, state) {
  const c = data.cross_domain.find((r) => r.pair === name);
  const src = domainName('source', state.source.domain);
  const tgt = domainName('target', state.target.domain);
  out.innerHTML = `
    <div class="result card">
      ${resultHeader(src, tgt, 'CL · Clearance')}
      ${verdictCard('', c.nn_rho, null, 'cross-domain experiment, RDKit descriptors')}
      <div class="stats">
        <div class="stat"><div class="v">${c.n_source.toLocaleString()}</div><div class="k">source compounds</div></div>
        <div class="stat"><div class="v">${c.n_target.toLocaleString()}</div><div class="k">target compounds</div></div>
        <div class="stat"><div class="v">${f2(c.r2_target_only)}</div><div class="k">target-only R²</div></div>
      </div>
      <h4>Transfer gain over target-only</h4>
      <p class="hint">Mean ΔR² over seeds and folds; hover for p-values.</p>
      <div id="r-cross"></div><div id="t-cross"></div>
    </div>`;
  const rows = [
    ['PTFT', c.d_ptft, c.p_ptft],
    ['PTFT-shuffled', c.d_ptft_shuffled, c.p_ptft_shuffled, true],
    ['LCW', c.d_lcw, c.p_lcw],
    ['IVP', c.d_ivp, c.p_ivp],
  ];
  deltaChart(
    $('r-cross'),
    rows.map(([m, v, p, control]) => ({ label: m, v, control, tip: [['tv', `ΔR² ${signed(v)}`], ['tl', m], ['tr', `p = ${f3(p)}`]] })),
    { domain: [-0.1, 0.1], ticks: [-0.1, -0.05, 0, 0.05, 0.1] },
  );
  tableView($('t-cross'), ['Method', 'ΔR²', 'p'], rows.map(([m, v, p]) => [m, signed(v), f3(p)]));
}

function renderEmpty(out, kind, state) {
  const src = state.source.mode === 'upload' ? 'Your source data' : domainName('source', state.source.domain);
  const tgt = state.target.mode === 'upload' ? 'Your target data' : domainName('target', state.target.domain);
  const body =
    kind === 'upload'
      ? 'Assessing uploaded data runs the full pipeline (standardisation, Graphormer encoding, phases 1–7) on the prediction server. This will be enabled once the server is online.'
      : 'This combination has not been evaluated yet. It will be computed on demand once the prediction server is online.';
  out.innerHTML = `<div class="result card empty">${resultHeader(src, tgt, ENDPOINTS[state.source.ep].label)}<p>${body}</p></div>`;
}

main();
