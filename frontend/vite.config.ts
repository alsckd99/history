import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // GitHub Pages 배포시 레포지토리 이름을 base로 설정
  // 예: https://username.github.io/history/ 형태로 배포됨
  base: process.env.NODE_ENV === 'production' ? '/history/' : '/',
  server: {
    port: 5173,
    host: "0.0.0.0"
  },
  build: {
    outDir: 'dist',
    sourcemap: false
  }
});

