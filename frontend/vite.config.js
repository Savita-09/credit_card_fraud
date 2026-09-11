import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { rollupOptions: { output: { manualChunks: { charts: ['recharts'], react: ['react', 'react-dom', 'react-router-dom'] } } } },
  server: { port: 5173, strictPort: true, proxy: { '/api': { target: process.env.API_PROXY_TARGET || 'http://127.0.0.1:8000', changeOrigin: true, rewrite: path => path.replace(/^\/api/, '') }, '/openapi.json': { target: process.env.API_PROXY_TARGET || 'http://127.0.0.1:8000' } } },
  test: { environment: 'jsdom', setupFiles: ['./src/test-setup.js'] },
});
