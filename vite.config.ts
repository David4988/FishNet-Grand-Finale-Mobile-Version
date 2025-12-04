import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

export default defineConfig(({ mode }) => ({
  // 1. ASSETS: Tell Vite to treat these as files, not code
  assetsInclude: ['**/*.tflite', '**/*.bin', '**/*.wasm'],

  server: {
    // Keep your original host/port settings
    host: "localhost",
    port: 5000,
    strictPort: false,
    hmr: {
      host: "localhost",
      port: 5000,
      protocol: "ws",
    },
    // 2. SECURITY HEADERS: Critical for TensorFlow.js WASM backend
    
  },

  plugins: [
    react(),
    mode === "development" && componentTagger(),
  ].filter(Boolean),

  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },

  // Keep your original exclusions
  optimizeDeps: {
    exclude: ["sql.js"],
  },

  // Keep global polyfill
  define: {
    global: "globalThis",
  },
}));