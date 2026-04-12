import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  server: {
    port: 5173,
    proxy: {
      // Proxy /api calls to the Python backend during development.
      // Production deployments should configure their own reverse proxy.
      '/api': {
        target:       'http://localhost:8000',
        changeOrigin: true,
        // Rewrite is a no-op here — keeps /api prefix so FastAPI routes match.
        rewrite: (path) => path,
      },
    },
  },

  build: {
    // Output to dashboard/dist (default)
    outDir: 'dist',
    // Ensure Tamil characters survive minification
    charset: 'utf8',
  },
});
