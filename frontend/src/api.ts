const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8080";

// ngrok 무료 버전 경고 페이지를 건너뛰기 위한 헤더
const defaultHeaders: HeadersInit = {
  "ngrok-skip-browser-warning": "true"
};

export interface RegisterPayload {
  video_id: string;
  keyword_path: string;
}

export interface KeywordItem {
  term: string;
  score: number;
}

export interface EntityItem {
  name: string;
  score: number;
}

export interface KeywordWindow {
  video_id: string;
  start: number | null;
  end: number | null;
  keywords: KeywordItem[];
  entities: { name: string; score: number }[];
  mapped_entities: EntityItem[];
  keyword_path?: string;
  slice_count: number;
}

export interface EntityResponse {
  entity: Record<string, unknown>;
  neighbors: Record<string, unknown>;
  documents: { content: string; metadata: Record<string, unknown> }[];
}

export interface QueryPayload {
  query: string;
  video_id?: string;
  focus_keywords?: string[];
}

export interface QueryResponse {
  query: string;
  answer: string;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || res.statusText);
  }
  return res.json();
}

export function registerVideo(payload: RegisterPayload) {
  return fetch(`${API_BASE}/videos/register`, {
    method: "POST",
    headers: { ...defaultHeaders, "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  }).then(handleResponse);
}

export function fetchKeywords(
  videoId: string,
  start?: number,
  end?: number,
  topK = 5
) {
  const params = new URLSearchParams();
  if (start !== undefined) params.append("start", String(start));
  if (end !== undefined) params.append("end", String(end));
  params.append("top_k", String(topK));
  return fetch(`${API_BASE}/videos/${encodeURIComponent(videoId)}/keywords?${params.toString()}`, {
    headers: defaultHeaders
  }).then(
    handleResponse<KeywordWindow>
  );
}

export function fetchEntity(entityName: string, depth = 1) {
  return fetch(`${API_BASE}/entity/${encodeURIComponent(entityName)}?depth=${depth}`, {
    headers: defaultHeaders
  }).then(
    handleResponse<EntityResponse>
  );
}

export function runQuery(payload: QueryPayload) {
  return fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { ...defaultHeaders, "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  }).then(handleResponse<QueryResponse>);
}

export async function uploadVideo(
  videoId: string,
  videoFile: File
) {
  const formData = new FormData();
  formData.append("video_id", videoId);
  formData.append("file", videoFile);

  return fetch(`${API_BASE}/videos/upload`, {
    method: "POST",
    headers: defaultHeaders,
    body: formData
  }).then(handleResponse);
}

export function createKeywordEventSource(
  videoId: string,
  currentTime: number,
  window: number = 5
) {
  const params = new URLSearchParams({
    current_time: String(currentTime),
    window: String(window)
  });
  return new EventSource(
    `${API_BASE}/videos/${encodeURIComponent(videoId)}/stream-keywords?${params.toString()}`
  );
}

// 영상 키워드 프리로드 시작
export interface PreloadStatus {
  video_id: string;
  status: "not_started" | "started" | "loading" | "complete";
  total_slices: number;
  loaded_slices: number;
  is_loading: boolean;
}

export function preloadVideoKeywords(videoId: string, topK = 5) {
  return fetch(`${API_BASE}/videos/${encodeURIComponent(videoId)}/preload?top_k=${topK}`, {
    method: "POST",
    headers: defaultHeaders
  }).then(handleResponse<PreloadStatus>);
}

export function getPreloadStatus(videoId: string) {
  return fetch(`${API_BASE}/videos/${encodeURIComponent(videoId)}/preload-status`, {
    headers: defaultHeaders
  }).then(
    handleResponse<PreloadStatus>
  );
}

