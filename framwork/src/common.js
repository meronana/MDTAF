import { SITE } from './config.js';

// Top navigation shared by both pages (index, data). page = 'home' | 'data'
export function navHTML(page) {
  const home = page === 'home' ? '' : 'index.html';
  return `
  <nav class="nav"><div class="nav-inner">
    <a class="nav-brand" href="${page === 'home' ? '#top' : 'index.html'}">${SITE.title}</a>
    <div class="nav-links">
      <a href="${home}#framework">Framework</a>
      <a href="${home}#assess">Assess</a>
      <a href="data.html"${page === 'data' ? ' aria-current="page"' : ''}>Data</a>
      <a href="${home}#try">Try it</a>
    </div>
    <button class="theme-btn" id="theme" type="button" aria-label="Toggle color theme">◐</button>
  </div></nav>`;
}

// Theme choice is stored in this browser only (the page still works if storage is blocked)
export function setupTheme() {
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem('theme');
    if (saved) root.dataset.theme = saved;
  } catch {}
  document.getElementById('theme').onclick = () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try {
      localStorage.setItem('theme', root.dataset.theme);
    } catch {}
  };
}
