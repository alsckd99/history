import { FormEvent, useState, useRef, useEffect } from "react";
import {
  registerVideo,
  fetchKeywords,
  fetchEntity,
  runQuery,
  uploadVideo,
  KeywordWindow,
  EntityResponse
} from "./api";
import GraphView from "./GraphView";
import MapView from "./MapView";
import "./styles.css";

// 마크다운 링크를 HTML로 변환하는 함수
function renderMarkdownLinks(text: string): JSX.Element[] {
  // [텍스트](url) 패턴 매칭
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  const parts: JSX.Element[] = [];
  let lastIndex = 0;
  let match;
  let keyIndex = 0;

  while ((match = linkRegex.exec(text)) !== null) {
    // 링크 앞의 일반 텍스트
    if (match.index > lastIndex) {
      parts.push(<span key={keyIndex++}>{text.slice(lastIndex, match.index)}</span>);
    }
    // 링크
    parts.push(
      <a 
        key={keyIndex++} 
        href={match[2]} 
        target="_blank" 
        rel="noopener noreferrer"
        style={{ color: '#667eea', textDecoration: 'underline' }}
      >
        {match[1]}
      </a>
    );
    lastIndex = match.index + match[0].length;
  }

  // 남은 텍스트
  if (lastIndex < text.length) {
    parts.push(<span key={keyIndex++}>{text.slice(lastIndex)}</span>);
  }

  return parts.length > 0 ? parts : [<span key={0}>{text}</span>];
}

function App() {
  const [videoUrl, setVideoUrl] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const [windowSize, setWindowSize] = useState(60);

  const [registerId, setRegisterId] = useState("");
  const [registerPath, setRegisterPath] = useState("");
  const [registerResult, setRegisterResult] = useState<string | null>(null);
  const [registerError, setRegisterError] = useState<string | null>(null);

  const [videoId, setVideoId] = useState("");
  const [keywordData, setKeywordData] = useState<KeywordWindow | null>(null);
  const [keywordError, setKeywordError] = useState<string | null>(null);
  const [loadingKeywords, setLoadingKeywords] = useState(false);

  const [selectedEntity, setSelectedEntity] = useState("");
  const [entityDepth, setEntityDepth] = useState(1);
  const [entityData, setEntityData] = useState<EntityResponse | null>(null);
  const [entityError, setEntityError] = useState<string | null>(null);

  const [question, setQuestion] = useState("");
  const [queryVideoId, setQueryVideoId] = useState("");
  const [focusKeywords, setFocusKeywords] = useState("");
  // 대화 기록 (질문-답변 쌍 배열)
  const [chatHistory, setChatHistory] = useState<Array<{question: string, answer: string}>>([]);
  const [currentQuestion, setCurrentQuestion] = useState<string | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);

  const [uploadVideoId, setUploadVideoId] = useState("");
  const [uploadVideoFile, setUploadVideoFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const [graphEntityData, setGraphEntityData] = useState<EntityResponse | null>(null);
  const [graphEntityName, setGraphEntityName] = useState<string | null>(null);

  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [showYoutubeInput, setShowYoutubeInput] = useState(false);
  
  // RAG 팝업 창 표시 여부
  const [showRagPopup, setShowRagPopup] = useState(false);

  // RAG 스크롤 참조
  const ragContentRef = useRef<HTMLDivElement>(null);
  
  // 지도에 표시할 지명 목록
  const [mapPlaceNames, setMapPlaceNames] = useState<string[]>([]);
  
  // YouTube 영상 시간 추적용
  const youtubeTimerRef = useRef<number | null>(null);
  const youtubeIframeRef = useRef<HTMLIFrameElement>(null);
  const isYoutubeVideo = videoUrl.includes('youtube.com') || videoUrl.includes('youtu.be');
  const [isPaused, setIsPaused] = useState(false);

  // 대화 기록이나 로딩 상태 변경 시 스크롤 아래로
  useEffect(() => {
    if (ragContentRef.current) {
      ragContentRef.current.scrollTop = ragContentRef.current.scrollHeight;
    }
  }, [chatHistory, currentQuestion, queryLoading]);

  // YouTube 플레이어 상태 변화 감지 (pause/play)
  useEffect(() => {
    const handleYoutubeMessage = (event: MessageEvent) => {
      if (!event.data || typeof event.data !== "string") return;
      if (!event.origin.includes("youtube.com")) return;
      try {
        const data = JSON.parse(event.data);
        if (data.event === "onStateChange") {
          if (data.info === 1) {
            setIsPaused(false); // playing
          } else if (data.info === 2 || data.info === 0) {
            setIsPaused(true); // paused or ended
          }
        }
      } catch {
        // ignore parsing errors
      }
    };
    window.addEventListener("message", handleYoutubeMessage);
    return () => window.removeEventListener("message", handleYoutubeMessage);
  }, []);

  // YouTube iframe에 이벤트 리스너 등록
  useEffect(() => {
    if (!isYoutubeVideo || !youtubeIframeRef.current) return;
    const iframe = youtubeIframeRef.current;

    const sendCommand = (command: any) => {
      iframe.contentWindow?.postMessage(JSON.stringify(command), "*");
    };

    const interval = window.setInterval(() => {
      sendCommand({ event: "listening", id: "yt-player" });
      sendCommand({ event: "command", func: "addEventListener", args: ["onStateChange"] });
    }, 1000);

    const timeout = window.setTimeout(() => {
      window.clearInterval(interval);
    }, 5000);

    return () => {
      window.clearInterval(interval);
      window.clearTimeout(timeout);
    };
  }, [isYoutubeVideo, videoUrl]);

  // YouTube 영상일 때 실제 시간 추적 (YouTube API 사용)
  useEffect(() => {
    if (!isYoutubeVideo || !videoId || !youtubeIframeRef.current) return;
    
    const iframe = youtubeIframeRef.current;
    
    // YouTube에서 현재 시간 요청
    const requestCurrentTime = () => {
      iframe.contentWindow?.postMessage(JSON.stringify({
        event: "command",
        func: "getCurrentTime",
        args: []
      }), "*");
    };
    
    // YouTube에서 시간 응답 수신
    const handleTimeMessage = (event: MessageEvent) => {
      if (!event.origin.includes("youtube.com")) return;
      if (!event.data || typeof event.data !== "string") return;
      
      try {
        const data = JSON.parse(event.data);
        // getCurrentTime 응답 처리
        if (data.event === "infoDelivery" && data.info && typeof data.info.currentTime === "number") {
          const newTime = Math.floor(data.info.currentTime);
          setCurrentTime(prev => {
            if (Math.abs(prev - newTime) >= 1) {
              console.log('[App] YouTube 시간 업데이트:', prev, '->', newTime);
              return newTime;
            }
            return prev;
          });
        }
      } catch {
        // ignore parsing errors
      }
    };
    
    window.addEventListener("message", handleTimeMessage);
    
    // 일시정지 상태가 아닐 때만 시간 요청
    let timer: number | null = null;
    if (!isPaused) {
      console.log('[App] YouTube 시간 추적 시작');
      // 1초마다 현재 시간 요청
      timer = window.setInterval(requestCurrentTime, 1000);
    }
    
    return () => {
      window.removeEventListener("message", handleTimeMessage);
      if (timer) {
        clearInterval(timer);
      }
    };
  }, [isYoutubeVideo, videoId, isPaused]);

  // YouTube iframe 일시정지/재생 함수
  const pauseYoutubeVideo = () => {
    if (youtubeIframeRef.current) {
      youtubeIframeRef.current.contentWindow?.postMessage(
        JSON.stringify({ event: 'command', func: 'pauseVideo' }),
        '*'
      );
    }
  };

  const playYoutubeVideo = () => {
    if (youtubeIframeRef.current) {
      youtubeIframeRef.current.contentWindow?.postMessage(
        JSON.stringify({ event: 'command', func: 'playVideo' }),
        '*'
      );
    }
  };

  const handleRegister = async (event: FormEvent) => {
    event.preventDefault();
    setRegisterResult(null);
    setRegisterError(null);
    try {
      const result = await registerVideo({
        video_id: registerId,
        keyword_path: registerPath
      });
      setRegisterResult(JSON.stringify(result, null, 2));
    } catch (err) {
      setRegisterError((err as Error).message);
    }
  };

  const handleFetchKeywords = async () => {
    if (!videoId) {
      setKeywordError("영상 ID를 입력하세요.");
      return;
    }
    setKeywordError(null);
    setLoadingKeywords(true);
    try {
      const start = Math.max(currentTime - windowSize / 2, 0);
      const end = start + windowSize;
      const data = await fetchKeywords(videoId, Math.floor(start), Math.floor(end));
      setKeywordData(data);
    } catch (err) {
      setKeywordError((err as Error).message);
      setKeywordData(null);
    } finally {
      setLoadingKeywords(false);
    }
  };

  const handleFetchEntity = async () => {
    if (!selectedEntity) {
      setEntityError("엔티티 이름을 입력하세요.");
      return;
    }
    setEntityError(null);
    try {
      const data = await fetchEntity(selectedEntity, entityDepth);
      setEntityData(data);
    } catch (err) {
      setEntityError((err as Error).message);
      setEntityData(null);
    }
  };

  const handleQuery = async (event: FormEvent) => {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setQueryError("질문을 입력하세요.");
      return;
    }
    setQueryError(null);
    setQueryLoading(true);
    setCurrentQuestion(trimmedQuestion);
    setQuestion(""); // 입력창 비우기
    try {
      const payload = {
        query: trimmedQuestion,
        video_id: queryVideoId || undefined,
        focus_keywords: focusKeywords
          ? focusKeywords.split(",").map((kw) => kw.trim()).filter(Boolean)
          : undefined
      };
      const data = await runQuery(payload);
      // 대화 기록에 추가
      setChatHistory(prev => [...prev, { question: trimmedQuestion, answer: data.answer }]);
      setCurrentQuestion(null);
    } catch (err) {
      setQueryError((err as Error).message);
      setCurrentQuestion(null);
    } finally {
      setQueryLoading(false);
    }
  };

  const handleUpload = async (event: FormEvent) => {
    event.preventDefault();
    if (!uploadVideoId || !uploadVideoFile) {
      setUploadError("영상 ID와 파일을 입력하세요.");
      return;
    }
    setUploadError(null);
    setUploading(true);
    try {
      const result = await uploadVideo(uploadVideoId, uploadVideoFile);
      setUploadResult(JSON.stringify(result, null, 2));
      // 자동으로 videoId 설정
      setVideoId(uploadVideoId);
    } catch (err) {
      setUploadError((err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  // 자동 질문 전송 함수 (타입과 상위 노드 정보 포함)
  const sendAutoQuery = async (entityName: string, nodeType?: string, rootKeyword?: string) => {
    // 타입에 따른 프롬프트 생성
    let typeLabel = '';
    if (nodeType) {
      if (nodeType === '인물') typeLabel = ' 인물';
      else if (nodeType === '사건') typeLabel = ' 사건';
      else if (nodeType === '지명') typeLabel = ' 지명(장소)';
      else if (nodeType !== '미분류') typeLabel = ` ${nodeType}`;
    }
    
    // 상위 노드(루트 키워드) 정보 포함
    let contextPrefix = '';
    if (rootKeyword && rootKeyword !== entityName) {
      contextPrefix = `${rootKeyword}에서의 `;
    }
    
    const autoQuestion = `${contextPrefix}${entityName}${typeLabel}에 대해 알려줘`;
    setQueryLoading(true);
    setCurrentQuestion(autoQuestion);
    setShowRagPopup(true); // RAG 팝업 열기
    
    try {
      const payload = {
        query: autoQuestion,
        video_id: queryVideoId || undefined,
        focus_keywords: rootKeyword ? [entityName, rootKeyword] : [entityName]
      };
      const data = await runQuery(payload);
      setChatHistory(prev => [...prev, { question: autoQuestion, answer: data.answer }]);
      setCurrentQuestion(null);
    } catch (err) {
      setQueryError((err as Error).message);
      setCurrentQuestion(null);
    } finally {
      setQueryLoading(false);
    }
  };

  const handleGraphNodeClick = async (entityName: string | null, entityData: any) => {
    // 같은 노드 다시 클릭하거나 null이면 선택 해제 (영상은 재생하지 않음)
    if (!entityName || entityName === graphEntityName) {
      setGraphEntityName(null);
      setGraphEntityData(null);
      // 영상은 일시정지 상태 유지 (사용자가 직접 재생 버튼 클릭해야 함)
      return;
    }
    
    // 노드 선택 시 일시정지
    setIsPaused(true);
    if (isYoutubeVideo) {
      pauseYoutubeVideo();
    }
    
    setGraphEntityName(entityName);
    
    // 관련 사료 가져오기
    try {
      const docs = await fetchEntity(entityName, 1);
      setGraphEntityData(docs);
    } catch (err) {
      console.error("엔티티 조회 오류:", err);
      setGraphEntityData(null);
    }
    
    // 자동 질문 전송 (타입과 루트 키워드 정보 포함)
    const nodeType = entityData?.nodeType;
    const rootKeyword = entityData?.rootKeyword;
    sendAutoQuery(entityName, nodeType, rootKeyword);
  };

  const extractYoutubeVideoId = (url: string): string | null => {
    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/,
      /youtube\.com\/embed\/([^&\n?#]+)/,
    ];
    
    for (const pattern of patterns) {
      const match = url.match(pattern);
      if (match) return match[1];
    }
    return null;
  };

  const handleYoutubeSubmit = async () => {
    if (!youtubeUrl.trim()) {
      setUploadError("유튜브 URL을 입력하세요.");
      return;
    }
    
    const videoId = extractYoutubeVideoId(youtubeUrl);
    if (!videoId) {
      setUploadError("유효한 유튜브 URL이 아닙니다.");
      return;
    }
    
    // 유튜브 URL을 video_id로 사용 (전체 URL)
    setVideoId(youtubeUrl);
    // embed URL로 변환하여 표시 (autoplay=1 추가)
    setVideoUrl(`https://www.youtube.com/embed/${videoId}?autoplay=1`);
    setShowYoutubeInput(false);
    setYoutubeUrl("");
    setUploadError(null);

    // // 키워드 파일 정보 가져오기
    // try {
    //   const res = await fetch(`http://localhost:8080/videos/${encodeURIComponent(youtubeUrl)}/keywords?start=0&end=60`);
    //   if (res.ok) {
    //     const data = await res.json();
    //     const keywordFile = data.keyword_path?.split('/').pop() || data.keyword_path?.split('\\').pop() || '연결된 파일 없음';
    //     setUploadResult(`영상 ID: ${youtubeUrl}\n키워드 파일: ${keywordFile}`);
    //   } else {
    //     const errorData = await res.json().catch(() => ({ detail: '알 수 없는 오류' }));
    //     console.error('키워드 조회 실패:', errorData);
    //     setUploadResult(`영상 ID: ${youtubeUrl}\n키워드 파일: 연결된 파일 없음 (${errorData.detail || res.statusText})`);
    //   }
    // } catch (e) {
    //   console.error('키워드 조회 오류:', e);
    //   setUploadResult(`영상 ID: ${youtubeUrl}\n키워드 파일: 확인 실패`);
    // }
  };

  const currentKeywords = keywordData?.keywords ?? [];
  const mappedEntities = keywordData?.mapped_entities ?? [];

  return (
    <div className="app-layout">
      {/* 왼쪽: 영상 플레이어 */}
      <div className="video-section">
        <div className="section-header">
          <h2>영상</h2>
        </div>

        {!videoUrl ? (
          /* 초기 상태: 업로드 안내 */
          <div className="video-empty-state">
            <div className="upload-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="upload-message">시작하려면 영상 추가</p>
            
            {!showYoutubeInput ? (
              <>
                <div className="upload-options">
                  <input 
                    type="file" 
                    id="video-upload-input"
                    accept="video/*" 
                    onChange={async (e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        setUploadVideoFile(file);
                        const generatedId = `video_${Date.now()}`;
                        setUploadVideoId(generatedId);
                        setVideoId(generatedId);
                        setVideoUrl(URL.createObjectURL(file));
                        setUploadError(null);
                        // 자동 업로드
                        try {
                          const result: any = await uploadVideo(generatedId, file);
                          const keywordFile = result.keyword_path?.split('/').pop() || result.keyword_path?.split('\\').pop() || '연결된 파일 없음';
                          setUploadResult(`영상 ID: ${generatedId}\n키워드 파일: ${keywordFile}`);
                        } catch (err: any) {
                          setUploadError(err.message);
                        }
                      }
                    }}
                    style={{ display: 'none' }}
                  />
                  <label htmlFor="video-upload-input" className="upload-button-primary">
                    📁 로컬 영상 업로드
                  </label>
                  <button 
                    className="upload-button-primary"
                    onClick={() => {
                      setShowYoutubeInput(true);
                      setUploadError(null);
                    }}
                  >
                    🔗 유튜브 링크 추가
                  </button>
                </div>
                {uploadError && <div className="error-msg">{uploadError}</div>}
              </>
            ) : (
              <div className="youtube-input-container">
                <input
                  type="text"
                  className="youtube-input"
                  placeholder="유튜브 URL을 입력하세요 (예: https://www.youtube.com/watch?v=...)"
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleYoutubeSubmit()}
                />
                {uploadError && <div className="error-msg">{uploadError}</div>}
                <div className="youtube-buttons">
                  <button className="youtube-submit-btn" onClick={handleYoutubeSubmit}>
                    확인
                  </button>
                  <button 
                    className="youtube-cancel-btn" 
                    onClick={() => {
                      setShowYoutubeInput(false);
                      setYoutubeUrl("");
                      setUploadError(null);
                    }}
                  >
                    취소
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* 영상 업로드 후 */
          <>
            <div className="video-player">
              {isYoutubeVideo ? (
                <iframe
                  ref={youtubeIframeRef}
                  key={videoUrl} // URL 변경 시에만 리렌더링
                  src={videoUrl.includes('enablejsapi') ? videoUrl : videoUrl + (videoUrl.includes('?') ? '&' : '?') + 'enablejsapi=1'}
                  title="YouTube video player"
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  style={{ width: '100%', height: '100%' }}
                />
              ) : (
                <video
                  controls
                  src={videoUrl}
                  onTimeUpdate={(e) => setCurrentTime((e.target as HTMLVideoElement).currentTime)}
                  onPause={() => setIsPaused(true)}
                  onPlay={() => setIsPaused(false)}
                />
              )}
            </div>

            {/* 영상 정보 표시 */}
            {uploadResult && (
              <div className="video-info-box">
                {uploadResult.split('\n').map((line, idx) => (
                  <div key={idx} className="info-line">{line}</div>
                ))}
              </div>
            )}

            {/* 영상 변경 버튼 */}
            <div className="upload-button-wrapper">
              <input 
                type="file" 
                id="video-upload-input-secondary"
                accept="video/*" 
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    setUploadVideoFile(file);
                    const generatedId = `video_${Date.now()}`;
                    setUploadVideoId(generatedId);
                    setVideoId(generatedId);
                    setVideoUrl(URL.createObjectURL(file));
                    // 자동 업로드
                    try {
                      const result: any = await uploadVideo(generatedId, file);
                      const keywordFile = result.keyword_path?.split('/').pop() || result.keyword_path?.split('\\').pop() || '연결된 파일 없음';
                      setUploadResult(`영상 ID: ${generatedId}\n키워드 파일: ${keywordFile}`);
                    } catch (err: any) {
                      setUploadError(err.message);
                    }
                  }
                }}
                style={{ display: 'none' }}
              />
              <label htmlFor="video-upload-input-secondary" className="upload-button">
                📁 다른 영상 선택
              </label>
              <button 
                className="upload-button"
                onClick={() => setShowYoutubeInput(true)}
              >
                🔗 유튜브 링크
              </button>
              <button 
                className="upload-button catalog-button"
                onClick={() => setShowRagPopup(prev => !prev)}
              >
                💬 대화
              </button>
            </div>
            
            {/* 유튜브 입력 모달 */}
            {showYoutubeInput && (
              <div className="youtube-modal">
                <div className="youtube-modal-content">
                  <input
                    type="text"
                    className="youtube-input"
                    placeholder="유튜브 URL을 입력하세요"
                    value={youtubeUrl}
                    onChange={(e) => setYoutubeUrl(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleYoutubeSubmit()}
                    autoFocus
                  />
                  <div className="youtube-buttons">
                    <button className="youtube-submit-btn" onClick={handleYoutubeSubmit}>
                      확인
                    </button>
                    <button 
                      className="youtube-cancel-btn" 
                      onClick={() => {
                        setShowYoutubeInput(false);
                        setYoutubeUrl("");
                      }}
                    >
                      취소
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* 오른쪽 컨테이너 */}
      <div className="right-container">
        {/* 그래프 시각화 (전체 높이) */}
        <div className="graph-section graph-section-full">
          <div className="section-header">
            <h2>지식 그래프</h2>
          </div>
          <div className="graph-content">
            {videoId ? (
              <GraphView 
                videoId={videoId} 
                currentTime={currentTime} 
                onNodeClick={handleGraphNodeClick}
                selectedNode={graphEntityName}
                onPlaceNamesExtracted={(places) => {
                  // 새 지명만 추가 (중복 방지)
                  setMapPlaceNames(prev => {
                    const newPlaces = places.filter(p => !prev.includes(p));
                    return newPlaces.length > 0 ? [...prev, ...newPlaces] : prev;
                  });
                }}
              />
            ) : (
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center', 
                height: '100%',
                color: '#6b7280'
              }}>
                영상을 업로드하면 그래프가 표시됩니다
              </div>
            )}
          </div>
        </div>

        {/* 지도 섹션 (관련 사료 + 지도) */}
        <div className="map-section">
          <div className="section-header">
            <h2>{graphEntityName ? `${graphEntityName} 관련 사료 & 지도` : '지도'}</h2>
          </div>
          <div className="map-content" style={{ display: 'flex', gap: '8px' }}>
            {/* 왼쪽: 관련 사료 (노드 선택 시) */}
            {graphEntityName && graphEntityData && (
              <div className="map-source-panel">
                <div className="map-source-scroll">
                  {(() => {
                    const entity = graphEntityData.entity as any;
                    const sources = entity?.sources;
                    const documents = (graphEntityData as any).documents || [];
                    const isFaissFallback = (graphEntityData as any).faiss_fallback;
                    
                    // 1. GraphDB sources가 있으면 사용
                    if (sources && Array.isArray(sources) && sources.length > 0) {
                      return sources.map((source: any, idx: number) => {
                        let sourceName = source.doc || source.type || '';
                        sourceName = sourceName.replace(/\.(json|pdf|txt)$/i, '');
                        if (sourceName.includes('_')) {
                          sourceName = sourceName.replace(/_/g, ' ');
                        }
                        
                        const sourceUrl = source.url || '';
                        
                        return (
                          <div key={idx} className="map-source-item">
                            <p>{source.snippet || source.제목 || ''}</p>
                            <small>
                              {sourceUrl ? (
                                <a 
                                  href={sourceUrl} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  style={{ color: '#667eea', textDecoration: 'underline' }}
                                >
                                  {sourceName}
                                </a>
                              ) : (
                                sourceName
                              )}
                            </small>
                          </div>
                        );
                      });
                    }
                    
                    // 2. FAISS 폴백: documents 사용
                    if (documents && documents.length > 0) {
                      return (
                        <>
                          {isFaissFallback && (
                            <div style={{ 
                              color: '#94a3b8', 
                              fontSize: '11px', 
                              marginBottom: '8px',
                              padding: '4px 8px',
                              background: 'rgba(100, 116, 139, 0.2)',
                              borderRadius: '4px'
                            }}>
                              📚 FAISS 검색 결과
                            </div>
                          )}
                          {documents.map((doc: any, idx: number) => {
                            const content = doc.content || '';
                            const metadata = doc.metadata || {};
                            let sourceName = metadata.doc || metadata.source || '알 수 없음';
                            // 파일 경로에서 파일명만 추출
                            if (sourceName.includes('/') || sourceName.includes('\\')) {
                              sourceName = sourceName.split('/').pop()?.split('\\').pop() || sourceName;
                            }
                            sourceName = sourceName.replace(/\.(json|pdf|txt)$/i, '');
                            if (sourceName.includes('_')) {
                              sourceName = sourceName.replace(/_/g, ' ');
                            }
                            
                            return (
                              <div key={idx} className="map-source-item">
                                <p>{content}</p>
                                <small>{sourceName}</small>
                              </div>
                            );
                          })}
                        </>
                      );
                    }
                    
                    // 3. 아무것도 없으면 메시지 표시
                    return (
                      <div style={{ color: '#6b7280', fontSize: '12px', textAlign: 'center', padding: '20px' }}>
                        관련 사료를 찾을 수 없습니다
                      </div>
                    );
                  })()}
                </div>
              </div>
            )}
            
            {/* 오른쪽: 지도 */}
            <div style={{ 
              flex: graphEntityName && graphEntityData ? 1 : '1 1 100%',
              height: '100%',
              minWidth: 0
            }}>
              {videoId ? (
                <MapView 
                  currentTime={currentTime}
                  videoId={videoId}
                />
              ) : (
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  height: '100%',
                  color: '#6b7280',
                  fontSize: '14px'
                }}>
                  영상을 업로드하면 지도가 표시됩니다
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 대화 팝업 (Gmail 스타일) */}
      {showRagPopup && (
        <div className="rag-popup">
          <div className="rag-popup-header">
            <span>대화</span>
            <div className="rag-popup-controls">
              <button 
                className="rag-popup-btn"
                onClick={() => setShowRagPopup(false)}
              >
                ✕
              </button>
            </div>
          </div>
          
          <div className="rag-popup-content" ref={ragContentRef}>
            {/* 이전 대화 기록 */}
            {chatHistory.map((chat, idx) => (
              <div key={idx} className="chat-item">
                <div className="question-bubble-wrapper">
                  <div className="question-bubble">
                    {chat.question}
                  </div>
                </div>
                <div className="answer-box">
                  <div className="answer-content">{renderMarkdownLinks(chat.answer)}</div>
                </div>
              </div>
            ))}
            
            {/* 현재 질문 (로딩 중) */}
            {currentQuestion && (
              <div className="chat-item">
                <div className="question-bubble-wrapper">
                  <div className="question-bubble">
                    {currentQuestion}
                  </div>
                </div>
                
                {queryLoading && (
                  <div className="loading-box">
                    <div className="loading-icon">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="loading-spinner">
                        <circle cx="12" cy="12" r="10" stroke="#667eea" strokeWidth="3" strokeLinecap="round" strokeDasharray="31.4 31.4" />
                      </svg>
                    </div>
                    <span className="loading-text">잠시만 기다려 주세요...</span>
                  </div>
                )}
              </div>
            )}
            
            {queryError && <div className="error-msg">{queryError}</div>}
          </div>
          
          <div className="rag-popup-input">
            <form onSubmit={handleQuery} className="rag-form">
              <div className="input-container">
                <textarea 
                  placeholder="질문을 입력하세요" 
                  value={question} 
                  onChange={(e) => setQuestion(e.target.value)} 
                  rows={1}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      if (question.trim() && !queryLoading) {
                        handleQuery(e as any);
                      }
                    }
                  }}
                  onInput={(e) => {
                    const target = e.target as HTMLTextAreaElement;
                    target.style.height = 'auto';
                    target.style.height = Math.min(target.scrollHeight, 80) + 'px';
                  }}
                />
                <button 
                  type="submit" 
                  disabled={queryLoading || !question.trim()}
                  className="send-button"
                >
                  ↑
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

