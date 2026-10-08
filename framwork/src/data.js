import { SITE, ENDPOINTS, DATA_PAGE } from './config.js';
import { navHTML, setupTheme } from './common.js';

const $ = (id) => document.getElementById(id);
const mb = (b) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);
const { name: repo, branch } = DATA_PAGE.repo;
const REPO = `https://github.com/${repo}`;
const rawUrl = (path) => `https://raw.githubusercontent.com/${repo}/${branch}/${path}`;
const treeUrl = (path) => `${REPO}/tree/${branch}/${path}`;
const ROLE = {
  source: { name: 'In vitro source', domain: 'ChEMBL · TDC · Biogen' },
  target: { name: 'Human in vivo target', domain: 'PKSmart' },
};

// Manifest contents (column names, cell values) come from CSVs, so they go into the DOM only via textContent
function cell(tag, txt, cls) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  n.textContent = txt;
  return n;
}

function fileCard(f) {
  const card = document.createElement('article');
  card.className = 'dl card';
  card.dataset.ep = f.endpoint;
  const e = ENDPOINTS[f.endpoint];
  card.innerHTML = `
    <div class="dl-head">
      <div>
        <div class="dl-tags"><span class="tag">${e.label} · ${e.name}</span><span class="tag ${f.role}">${ROLE[f.role].name}</span></div>
        <h3 class="dl-name"></h3>
        <p class="hint">${ROLE[f.role].domain}</p>
      </div>
      <a class="btn">Download CSV</a>
    </div>
    <dl class="dl-meta">
      <div><dt>Compounds</dt><dd>${f.rows.toLocaleString()}</dd></div>
      <div><dt>Columns</dt><dd>${f.columns.length}</dd></div>
      <div><dt>Size</dt><dd>${mb(f.bytes)}</dd></div>
    </dl>
    <details class="table"><summary>Preview first rows</summary><div class="tbl-wrap"><table></table></div></details>
    <p class="sha" title="SHA-256"></p>`;
  card.querySelector('.dl-name').textContent = f.name;
  const a = card.querySelector('a.btn');
  a.href = rawUrl(f.path);
  a.download = f.name;
  a.setAttribute('aria-label', `Download ${f.name} from GitHub`);
  card.querySelector('.sha').textContent = `SHA-256 ${f.sha256}`;
  const t = card.querySelector('table');
  const hr = t.insertRow();
  f.preview_columns.forEach((c) => hr.appendChild(cell('th', c)));
  if (f.columns.length > f.preview_columns.length) hr.appendChild(cell('th', `+${f.columns.length - f.preview_columns.length} more`, 'muted'));
  f.preview.forEach((r) => {
    const row = t.insertRow();
    r.forEach((v) => row.appendChild(cell('td', v)));
  });
  return card;
}

async function main() {
  const m = await (await fetch(`${import.meta.env.BASE_URL}downloads/manifest.json`)).json();
  const cv = m.cv;
  const dataBytes = m.files.reduce((a, f) => a + f.bytes, 0) + cv.bytes;
  const totalRows = (role) => m.files.filter((f) => f.role === role).reduce((a, f) => a + f.rows, 0);

  $('app').innerHTML = `
  ${navHTML('data')}
  <header class="hero wrap page-hero">
    <p class="eyebrow">Data</p>
    <h1>Curated datasets</h1>
    <p class="sub">In vitro ADME source sets and human in vivo PK target sets for f<sub>u</sub>, CL and t<sub>½</sub>,
      deduplicated, log-transformed and split by scaffold, exactly as used in ${SITE.title}.</p>
    <div class="btns">
      <a class="btn" href="${treeUrl('data')}">Browse data on GitHub</a>
      <a class="btn ghost" href="${REPO}/archive/refs/heads/${branch}.zip">Download repository · zip</a>
    </div>
    <p class="hint" style="margin-top:12px">All files (${mb(dataBytes)}) live in the <code>data/</code> folder of
      <a href="${REPO}">${repo}</a>. Check the SHA-256 checksums below after downloading.</p>
    <div class="stats narrow">
      <div class="stat"><div class="v">${totalRows('source').toLocaleString()}</div><div class="k">in vitro source records</div></div>
      <div class="stat"><div class="v">${totalRows('target').toLocaleString()}</div><div class="k">human target compounds</div></div>
      <div class="stat"><div class="v">3</div><div class="k">endpoints</div></div>
      <div class="stat"><div class="v">5</div><div class="k">scaffold CV folds</div></div>
    </div>
  </header>

  <main class="wrap">
    <section id="files">
      <div class="chart-head">
        <div><p class="eyebrow">Files</p><h2>Per-endpoint datasets</h2></div>
        <div class="seg" id="filter" role="group" aria-label="Filter by endpoint">
          <button type="button" data-ep="all" aria-pressed="true">All</button>
          ${Object.entries(ENDPOINTS).map(([k, e]) => `<button type="button" data-ep="${k}" aria-pressed="false">${e.label}</button>`).join('')}
        </div>
      </div>
      <div class="dl-grid" id="grid"></div>
    </section>

    <section id="splits">
      <p class="eyebrow">Splits</p>
      <h2>Scaffold cross-validation splits</h2>
      <p class="lede">The exact folds behind every reported number. Each fold has the source set (with that fold's test
        compounds removed), the target train set and the target test set. Per-fold normalisation statistics are included.</p>
      <div class="card dl-head">
        <div>
          <h3 class="dl-name">${cv.path}/</h3>
          <p class="hint">${cv.n_files} files · ${mb(cv.bytes)}</p>
        </div>
        <a class="btn" href="${treeUrl(cv.path)}">Open on GitHub</a>
      </div>
      <details class="table"><summary>Show fold sizes</summary><div class="tbl-wrap"><table id="cv-table"></table></div></details>
    </section>

    <section id="processing">
      <p class="eyebrow">Processing</p>
      <h2>How the data were built</h2>
      <ol class="steps">${DATA_PAGE.pipeline.map(([t, b]) => `<li><strong>${t}</strong><span>${b}</span></li>`).join('')}</ol>
    </section>

    <section id="columns">
      <p class="eyebrow">Columns</p>
      <h2>Column reference</h2>
      <div class="tbl-wrap card"><table class="cols">
        <tr><th>Column</th><th>Meaning</th></tr>
        ${Object.entries(DATA_PAGE.columns).map(([k, v]) => `<tr><td><code>${k}</code></td><td>${v}</td></tr>`).join('')}
      </table></div>
      <p class="hint" style="margin-top:10px">The clearance source file also carries the Biogen assay columns (MDR1-MDCK, solubility, PPB, RLM CLint) for the Biogen records.</p>
    </section>

    <section id="license">
      <p class="eyebrow">Licence & attribution</p>
      <h2>Using these data</h2>
      <p class="callout"><strong>${DATA_PAGE.license}.</strong> These sets are derived from public sources;
        please also cite and respect the terms of each original source.</p>
      <div class="tbl-wrap card" style="margin-top:16px"><table class="cols">
        <tr><th>Source</th><th>Used for</th><th>Terms</th></tr>
        ${DATA_PAGE.sources.map((s) => `<tr><td>${s.name}</td><td>${s.role}</td><td>${s.note}</td></tr>`).join('')}
      </table></div>
    </section>
  </main>
  <footer>${SITE.affiliation}</footer>`;

  setupTheme();

  const grid = $('grid');
  const order = ['fu', 'CL', 't12'];
  m.files
    .slice()
    .sort((a, b) => order.indexOf(a.endpoint) - order.indexOf(b.endpoint) || (a.role === 'source' ? -1 : 1))
    .forEach((f) => grid.appendChild(fileCard(f)));

  $('filter').querySelectorAll('button').forEach((b) => {
    b.onclick = () => {
      $('filter').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      grid.querySelectorAll('.dl').forEach((c) => (c.hidden = b.dataset.ep !== 'all' && c.dataset.ep !== b.dataset.ep));
    };
  });

  const t = $('cv-table');
  const hr = t.insertRow();
  ['Endpoint', 'Fold', 'Source', 'Target train', 'Target test'].forEach((h) => hr.appendChild(cell('th', h)));
  m.cv_summary.forEach((r) => {
    const row = t.insertRow();
    [ENDPOINTS[r.endpoint].text, r.fold, r.source.toLocaleString(), r.train, r.test].forEach((v) => row.appendChild(cell('td', String(v))));
  });
}

main();
