// The test page of tests/browser: the real engine, Worker and models code bundled with a page that exposes window.__t. Built by
// tests/browser/conftest.py into a temp folder (SELFTEST_OUT); nothing of it ships. Same ort handling as the UI build.
import path from 'node:path';
import { defineConfig } from 'vite';
import { ortPlugin } from '../vite.config';

export default defineConfig({
  root: __dirname,
  base: './',
  worker: { format: 'es', plugins: () => [ortPlugin()] },
  plugins: [ortPlugin()],
  build: { outDir: process.env.SELFTEST_OUT ?? path.resolve(__dirname, '../.selftest-dist'), emptyOutDir: true, assetsInlineLimit: 0, target: 'es2022', sourcemap: false },
});
