import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  publicDir: "../media",
  server: {
    host: "127.0.0.1",
    port: 8766,
    allowedHosts: ["notes.shoalstone.net"],
  },
});
