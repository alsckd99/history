import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import App from "./App";
import TestApp from "./TestApp";
import TestApp2 from "./TestApp2";

// ============================================================
// 라우팅 설정
// /video1 - 키워드 연결된 영상 (TestApp - example_video)
// /video2 - 키워드 연결 안된 영상 (TestApp2 - example_video1)
// /app    - 일반 모드 (영상 업로드 필요)
// ============================================================

// GitHub Pages에서는 /history/ 가 base path
const basename = import.meta.env.BASE_URL;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter basename={basename}>
      <Routes>
        <Route path="/" element={<Navigate to="/video1" replace />} />
        <Route path="/video1" element={<TestApp />} />
        <Route path="/video2" element={<TestApp2 />} />
        <Route path="/app" element={<App />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);

