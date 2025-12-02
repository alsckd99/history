import { FormEvent, useState, useRef, useEffect } from "react";
import {
  fetchEntity,
  runQuery,
  fetchKeywords,
  EntityResponse
} from "./api";
import GraphView from "./GraphView";
import MapView from "./MapView";
import "./styles.css";

// ============================================================
// 테스트용 설정 - 여기서 영상 경로와 ID를 변경하세요
// ============================================================
const TEST_CONFIG = {
  // 로컬 영상 파일 경로 (public 폴더 기준)
  // 예: public/test_video.mp4 → "/test_video.mp4"
  VIDEO_PATH: "노량.mp4",
  
  // 영상 ID (키워드 파일과 매칭되는 ID)
  VIDEO_ID: "example_video",
  
  // 자동 재생 여부
  AUTO_PLAY: true,
};
// ============================================================

// 마크다운 링크를 HTML로 변환하는 함수
function renderMarkdownLinks(text: string): JSX.Element[] {
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  const parts: JSX.Element[] = [];
  let lastIndex = 0;
  let match;
  let keyIndex = 0;

  while ((match = linkRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={keyIndex++}>{text.slice(lastIndex, match.index)}</span>);
    }
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

  if (lastIndex < text.length) {
    parts.push(<span key={keyIndex++}>{text.slice(lastIndex)}</span>);
  }

  return parts.length > 0 ? parts : [<span key={0}>{text}</span>];
}

function TestApp() {
  // 테스트용: 로컬 영상 파일 경로와 ID가 미리 설정됨
  const [videoUrl] = useState(TEST_CONFIG.VIDEO_PATH);
  const [videoId] = useState(TEST_CONFIG.VIDEO_ID);
  const [currentTime, setCurrentTime] = useState(0);

  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<Array<{question: string, answer: string}>>([]);
  const [currentQuestion, setCurrentQuestion] = useState<string | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);

  const [graphEntityData, setGraphEntityData] = useState<EntityResponse | null>(null);
  const [graphEntityName, setGraphEntityName] = useState<string | null>(null);

  const [showRagPopup, setShowRagPopup] = useState(false);
  const ragContentRef = useRef<HTMLDivElement>(null);
  
  const [mapPlaceNames, setMapPlaceNames] = useState<string[]>([]);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPaused, setIsPaused] = useState(!TEST_CONFIG.AUTO_PLAY);
  
  // 키워드 파일 매칭 여부
  const [hasKeywords, setHasKeywords] = useState<boolean | null>(null);

  // 키워드 파일 매칭 확인
  useEffect(() => {
    if (!videoId) {
      setHasKeywords(false);
      return;
    }
    
    const checkKeywords = async () => {
      try {
        await fetchKeywords(videoId, 0, 10);
        setHasKeywords(true);
      } catch {
        console.log('[TestApp] 키워드 파일 매칭 실패:', videoId);
        setHasKeywords(false);
      }
    };
    
    checkKeywords();
  }, [videoId]);

  // 대화 기록 스크롤
  useEffect(() => {
    if (ragContentRef.current) {
      ragContentRef.current.scrollTop = ragContentRef.current.scrollHeight;
    }
  }, [chatHistory, currentQuestion, queryLoading]);

  // 자동 재생 시작
  useEffect(() => {
    if (TEST_CONFIG.AUTO_PLAY && videoRef.current) {
      videoRef.current.play().catch(err => {
        console.log('[TestApp] 자동 재생 실패 (브라우저 정책):', err);
      });
    }
  }, []);

  const pauseVideo = () => {
    if (videoRef.current) {
      videoRef.current.pause();
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
    setQuestion("");
    try {
      const payload = {
        query: trimmedQuestion,
        video_id: videoId || undefined,
        focus_keywords: undefined
      };
      const data = await runQuery(payload);
      setChatHistory(prev => [...prev, { question: trimmedQuestion, answer: data.answer }]);
      setCurrentQuestion(null);
    } catch (err) {
      setQueryError((err as Error).message);
      setCurrentQuestion(null);
    } finally {
      setQueryLoading(false);
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
    setShowRagPopup(true);
    
    try {
      const payload = {
        query: autoQuestion,
        video_id: videoId || undefined,
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
    if (!entityName || entityName === graphEntityName) {
      setGraphEntityName(null);
      setGraphEntityData(null);
      return;
    }
    
    setIsPaused(true);
    pauseVideo();
    
    setGraphEntityName(entityName);
    
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

  return (
    <div className="app-layout" style={hasKeywords === false ? { gridTemplateColumns: '1fr' } : undefined}>
      {/* 왼쪽: 영상 플레이어 */}
      <div className="video-section">
        <div className="section-header">
          <h2>영상</h2>
        </div>

        <div className="video-player">
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            autoPlay={TEST_CONFIG.AUTO_PLAY}
            onTimeUpdate={(e) => setCurrentTime((e.target as HTMLVideoElement).currentTime)}
            onPause={() => setIsPaused(true)}
            onPlay={() => setIsPaused(false)}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        </div>

        {hasKeywords && (
          <div className="upload-button-wrapper">
            <button 
              className="upload-button catalog-button"
              onClick={() => setShowRagPopup(prev => !prev)}
            >
              💬 대화
            </button>
          </div>
        )}
      </div>

      {/* 오른쪽 컨테이너 - 키워드 매칭 시에만 표시 */}
      {hasKeywords && (
        <div className="right-container">
          {/* 그래프 시각화 */}
          <div className="graph-section graph-section-full">
            <div className="section-header">
              <h2>지식 그래프</h2>
            </div>
            <div className="graph-content">
              <GraphView 
                videoId={videoId} 
                currentTime={currentTime} 
                onNodeClick={handleGraphNodeClick}
                selectedNode={graphEntityName}
                onPlaceNamesExtracted={(places) => {
                  setMapPlaceNames(prev => {
                    const newPlaces = places.filter(p => !prev.includes(p));
                    return newPlaces.length > 0 ? [...prev, ...newPlaces] : prev;
                  });
                }}
              />
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
                            {/* {isFaissFallback && (
                              <div style={{ 
                                color: '#94a3b8', 
                                fontSize: '11px', 
                                marginBottom: '8px',
                                padding: '4px 8px',
                                background: 'rgba(100, 116, 139, 0.2)',
                                borderRadius: '4px'
                              }}>
                              </div>
                            )} */}
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
                <MapView 
                  currentTime={currentTime}
                  videoId={videoId}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 대화 팝업 */}
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

export default TestApp;

