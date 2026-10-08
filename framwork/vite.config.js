import { resolve } from 'node:path';
import { defineConfig } from 'vite';

// base './' : build with relative paths so it also works under a GitHub Pages sub-path (/repo-name/)
export default defineConfig({
  base: './',
  build: {
    rollupOptions: {
      input: { main: resolve(__dirname, 'index.html'), data: resolve(__dirname, 'data.html') },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    // Prevent the watcher from crashing with EBUSY while large files are being copied (file locked). After replacing, just reload.
    watch: { ignored: ['**/public/figures/**', '**/public/downloads/**'] },
  },
});
