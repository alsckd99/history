import os
import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from video_keyword_store import VideoKeywordStore
from graph_context_service import GraphContextService
from video_registry import VideoRegistry

# ============================================
# LLM 요청 큐 시스템
# ============================================
class LLMRequestQueue:
    """LLM 추론 요청을 순차적으로 처리하는 큐 시스템"""
    
    def __init__(self, max_concurrent: int = 1):
        self.queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._started = False
    
    async def submit(self, func, *args, **kwargs):
        """LLM 요청을 큐에 제출하고 결과를 기다림"""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor, 
                lambda: func(*args, **kwargs)
            )
            return result
    
    def shutdown(self):
        self._executor.shutdown(wait=False)

# 전역 LLM 큐
_llm_queue: Optional[LLMRequestQueue] = None

def get_llm_queue() -> LLMRequestQueue:
    global _llm_queue
    if _llm_queue is None:
        _llm_queue = LLMRequestQueue(max_concurrent=2)
    return _llm_queue


# ============================================
# 프리로드 캐시
# ============================================
_preload_cache: dict = {}
_preload_status: dict = {}


# ============================================
# Pydantic 모델
# ============================================
class QueryRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    focus_keywords: Optional[List[str]] = None


class QueryResponse(BaseModel):
    query: str
    answer: str


class VideoRegisterRequest(BaseModel):
    video_id: str
    keyword_path: str


class VideoStreamRequest(BaseModel):
    video_id: str
    interval: float = 1.0


# ============================================
# 전역 싱글톤 인스턴스
# ============================================
_graph_service = None
_keyword_store = None
_registry = None
_init_lock = threading.Lock()

def get_services():
    """서비스 인스턴스를 한 번만 초기화하고 재사용 (thread-safe)"""
    global _graph_service, _keyword_store, _registry
    
    if _graph_service is None:
        with _init_lock:
            if _graph_service is None:
                print("\n[초기화] GraphRAG 서비스 최초 로드 중...")
                graphrag_config = {
                    "embedding_model_name": os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-large-instruct"),
                    "llm_model_name": os.environ.get("LLM_MODEL", "gemma3:12b"),
                    "arango_host": os.environ.get("ARANGO_HOST", "localhost"),
                    "arango_port": int(os.environ.get("ARANGO_PORT", "8530")),
                    "arango_password": os.environ.get("ARANGO_PASSWORD", ""),
                    "arango_db_name": os.environ.get("ARANGO_DB", "knowledge_graph"),
                    "arango_reset": False,
                    "global_arango_db_name": os.environ.get("GLOBAL_GRAPH_DB", "knowledge_graph"),
                    "global_arango_reset": False,
                    "use_reranker": True,
                    "use_tika": False
                }
                index_dir = os.environ.get("GRAPHRAG_INDEX_DIR", "graphrag_data/global")
                keyword_root = os.environ.get("VIDEO_KEYWORD_DIR", "video_keywords")
                
                _keyword_store = VideoKeywordStore(keyword_root)
                _registry = VideoRegistry(os.environ.get("VIDEO_REGISTRY_FILE", os.path.join(keyword_root, "registry.json")))
                _graph_service = GraphContextService(graphrag_config, index_dir=index_dir)
                print("[완료] GraphRAG 서비스 초기화 완료\n")
    
    return _graph_service, _keyword_store, _registry


# ============================================
# 비동기 헬퍼 함수들
# ============================================
_thread_pool = ThreadPoolExecutor(max_workers=8)

async def run_in_thread(func, *args, **kwargs):
    """동기 함수를 비동기로 실행"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, lambda: func(*args, **kwargs))


# ============================================
# FastAPI 앱 생성
# ============================================
def create_app() -> FastAPI:
    app = FastAPI(title="Real-time GraphRAG API (Async)")

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # ========================================
    # 비동기 엔드포인트들
    # ========================================
    
    @app.get("/videos/registry")
    async def list_registry():
        """Registry에 등록된 모든 영상 목록 조회"""
        graph_service, keyword_store, registry = get_services()
        videos = await run_in_thread(registry.list_videos)
        return {"videos": videos}

    @app.post("/videos/register")
    async def register_video(payload: VideoRegisterRequest):
        graph_service, keyword_store, registry = get_services()
        try:
            await run_in_thread(registry.register, payload.video_id, payload.keyword_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        
        resolved = await run_in_thread(registry.resolve, payload.video_id)
        return {
            "video_id": payload.video_id,
            "keyword_path": resolved
        }

    @app.get("/videos/{video_id:path}/keywords")
    async def get_keywords(
        video_id: str,
        start: Optional[float] = Query(None),
        end: Optional[float] = Query(None),
        top_k: int = Query(10, ge=1, le=20)
    ):
        graph_service, keyword_store, registry = get_services()
        keyword_path = await run_in_thread(registry.resolve, video_id)
        
        if not keyword_path:
            raise HTTPException(
                status_code=404, 
                detail=f"영상 '{video_id}'에 대한 키워드 JSON이 registry에 등록되지 않았습니다."
            )
        
        # 캐시 확인
        slice_duration = 15
        if start is not None:
            slice_index = int(start // slice_duration)
            cache_key = (video_id, slice_index)
            
            if cache_key in _preload_cache:
                cached = _preload_cache[cache_key]
                print(f"[Cache HIT] video={video_id}, slice={slice_index}")
                return {
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                    "keywords": cached["keywords"],
                    "entities": cached.get("entities", []),
                    "mapped_entities": cached["mapped_entities"],
                    "keyword_path": keyword_path,
                    "slice_count": 1,
                    "from_cache": True
                }
        
        try:
            # 비동기로 키워드 조회
            window = await run_in_thread(
                keyword_store.query,
                video_id,
                start,
                end,
                top_k,
                keyword_path
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=404, 
                detail=f"키워드 JSON 파일을 찾을 수 없습니다: {keyword_path}"
            )

        keywords = [item["term"] for item in window["keywords"]]
        
        # 엔티티 매핑 (비동기)
        mapped_entities = await run_in_thread(
            graph_service.map_keywords_to_entities, 
            keywords, 
            top_k
        )
        
        window["mapped_entities"] = mapped_entities
        window["keyword_path"] = keyword_path
        window["from_cache"] = False
        return window

    @app.post("/videos/{video_id:path}/preload")
    async def preload_video_keywords(
        video_id: str,
        top_k: int = Query(10, ge=1, le=20)
    ):
        """영상의 모든 키워드 슬라이스를 백그라운드에서 미리 로드"""
        graph_service, keyword_store, registry = get_services()
        keyword_path = await run_in_thread(registry.resolve, video_id)
        
        if not keyword_path:
            raise HTTPException(
                status_code=404,
                detail=f"영상 '{video_id}'에 대한 키워드 JSON이 등록되지 않았습니다."
            )
        
        # 이미 로딩 중이면 상태만 반환
        if video_id in _preload_status and _preload_status[video_id].get("is_loading"):
            return {
                "video_id": video_id,
                "status": "loading",
                **_preload_status[video_id]
            }
        
        try:
            slices = await run_in_thread(
                keyword_store.load_slices, 
                video_id, 
                keyword_path
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"키워드 파일을 찾을 수 없습니다: {e}")
        
        total_slices = len(slices)
        cached_count = sum(1 for i in range(total_slices) if (video_id, i) in _preload_cache)
        
        if cached_count == total_slices:
            return {
                "video_id": video_id,
                "status": "complete",
                "total_slices": total_slices,
                "loaded_slices": total_slices,
                "is_loading": False
            }
        
        _preload_status[video_id] = {
            "total_slices": total_slices,
            "loaded_slices": cached_count,
            "is_loading": True
        }
        
        # 백그라운드 태스크로 프리로드
        asyncio.create_task(_preload_slices_async(
            video_id, slices, graph_service, top_k
        ))
        
        return {
            "video_id": video_id,
            "status": "started",
            "total_slices": total_slices,
            "loaded_slices": cached_count,
            "is_loading": True
        }

    async def _preload_slices_async(video_id: str, slices, graph_service, top_k: int):
        """비동기 프리로드 처리"""
        print(f"[Preload] 시작: video={video_id}, 총 {len(slices)}개 슬라이스")
        
        for idx, sl in enumerate(slices):
            cache_key = (video_id, idx)
            
            if cache_key in _preload_cache:
                continue
            
            try:
                # children 포함하여 키워드 저장
                keywords = []
                for k in sl.keywords:
                    term = k.get("term") or k.get("keyword")
                    if not term:
                        continue
                    kw_item = {
                        "term": term,
                        "score": k.get("score", 1.0)
                    }
                    # children, start, end 포함
                    if "children" in k:
                        kw_item["children"] = k["children"]
                    if "start" in k:
                        kw_item["start"] = k["start"]
                    if "end" in k:
                        kw_item["end"] = k["end"]
                    keywords.append(kw_item)
                
                if keywords:
                    keyword_terms = [k["term"] for k in keywords]
                    mapped_entities = await run_in_thread(
                        graph_service.map_keywords_to_entities, 
                        keyword_terms, 
                        top_k
                    )
                else:
                    mapped_entities = []
                
                _preload_cache[cache_key] = {
                    "keywords": keywords,
                    "entities": [{"name": e.get("name"), "score": e.get("score", 1.0)} for e in sl.entities] if sl.entities else [],
                    "mapped_entities": mapped_entities,
                    "slice_start": sl.start,
                    "slice_end": sl.end
                }
                
                # children 포함 여부 로그
                for kw in keywords:
                    if "children" in kw:
                        print(f"[Preload] '{kw['term']}'의 children: {[c.get('term') for c in kw['children']]}")
                
                _preload_status[video_id]["loaded_slices"] = idx + 1
                print(f"[Preload] 완료: slice {idx+1}/{len(slices)}")
                
            except Exception as e:
                print(f"[Preload] 오류 slice {idx}: {e}")
        
        _preload_status[video_id]["is_loading"] = False
        print(f"[Preload] 완료: video={video_id}")

    @app.get("/videos/{video_id:path}/preload-status")
    async def get_preload_status(video_id: str):
        """프리로드 진행 상태 확인"""
        if video_id not in _preload_status:
            return {
                "video_id": video_id,
                "status": "not_started",
                "total_slices": 0,
                "loaded_slices": 0,
                "is_loading": False
            }
        
        status = _preload_status[video_id]
        return {
            "video_id": video_id,
            "status": "complete" if not status["is_loading"] and status["loaded_slices"] == status["total_slices"] else "loading",
            **status
        }

    @app.get("/entity/{entity_name}")
    async def get_entity(entity_name: str, depth: int = Query(1, ge=1, le=3)):
        graph_service, keyword_store, registry = get_services()
        
        # 비동기로 엔티티 컨텍스트 조회
        context = await run_in_thread(
            graph_service.get_entity_context, 
            entity_name, 
            depth
        )
        
        entity = context.get("entity")
        neighbors = context.get("neighbors", {})
        documents = []
        faiss_fallback = False
        
        # 엔티티가 있고 sources가 있는 경우
        if entity:
            sources = entity.get("sources", [])
            if sources and len(sources) > 0:
                # GraphDB sources 사용
                documents = await run_in_thread(
                    graph_service.get_documents_for_entity, 
                    entity_name, 
                    5
                )
            else:
                # sources가 없으면 FAISS 검색
                print(f"[entity] '{entity_name}' sources 없음 - FAISS 검색")
                faiss_fallback = True
        else:
            # 엔티티가 없으면 FAISS 검색
            print(f"[entity] '{entity_name}' GraphDB에 없음 - FAISS 검색")
            faiss_fallback = True
        
        # FAISS 폴백 검색
        if faiss_fallback:
            try:
                faiss_results = await run_in_thread(
                    graph_service.search_documents_faiss,
                    entity_name,
                    5
                )
                
                # FAISS 결과를 documents 형식으로 변환
                for doc in faiss_results:
                    content = doc.get("content", "")
                    source = doc.get("source", "")
                    doc_name = source.split("/")[-1].split("\\")[-1] if source else "알 수 없음"
                    
                    documents.append({
                        "content": content,
                        "metadata": {
                            "source": source,
                            "doc": doc_name,
                            "from_faiss": True
                        }
                    })
                
                print(f"[entity] FAISS에서 {len(documents)}개 문서 검색됨")
            except Exception as e:
                print(f"[entity] FAISS 검색 오류: {e}")
        
        # 엔티티가 없어도 FAISS 결과가 있으면 반환
        if not entity and not documents:
            raise HTTPException(status_code=404, detail="엔티티를 찾을 수 없습니다.")
        
        return {
            "entity": entity,
            "neighbors": neighbors,
            "documents": documents,
            "faiss_fallback": faiss_fallback
        }

    @app.post("/query", response_model=QueryResponse)
    async def run_query(payload: QueryRequest):
        """LLM 질의 - 큐 시스템을 통해 순차 처리"""
        graph_service, keyword_store, registry = get_services()
        llm_queue = get_llm_queue()
        
        # LLM 큐를 통해 순차 처리 (동시 요청 제한)
        try:
            answer = await llm_queue.submit(
                graph_service.answer_query, 
                payload.query
            )
            return QueryResponse(**answer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"질의 처리 오류: {str(e)}")

    @app.post("/videos/upload")
    async def upload_video(
        video_id: str,
        file: UploadFile = File(...)
    ):
        """영상 파일 업로드 및 키워드 JSON 자동 연결"""
        import hashlib
        
        graph_service, keyword_store, registry = get_services()
        
        upload_dir = os.environ.get("VIDEO_UPLOAD_DIR", "uploaded_videos")
        os.makedirs(upload_dir, exist_ok=True)

        if video_id.startswith('http'):
            safe_filename = hashlib.md5(video_id.encode()).hexdigest()
        else:
            safe_filename = video_id
        
        video_filename = f"{safe_filename}{os.path.splitext(file.filename)[1]}"
        video_path = os.path.join(upload_dir, video_filename)

        # 비동기 파일 쓰기
        content = await file.read()
        await run_in_thread(lambda: open(video_path, "wb").write(content))

        existing_keyword_path = await run_in_thread(registry.resolve, video_id)
        
        if existing_keyword_path and os.path.exists(existing_keyword_path):
            keyword_path = existing_keyword_path
            message = f"업로드 완료 및 키워드 JSON 연결됨: {keyword_path}"
        else:
            keyword_path = None
            message = "업로드 완료 (키워드 JSON 없음 - registry에 수동 등록 필요)"

        return {
            "video_id": video_id,
            "video_path": video_path,
            "keyword_path": keyword_path,
            "message": message
        }

    @app.get("/videos/{video_id:path}/stream-keywords")
    async def stream_keywords(
        video_id: str,
        current_time: float = Query(0),
        window: float = Query(5.0)
    ):
        """실시간 키워드 스트리밍 (Server-Sent Events)"""
        import json
        
        graph_service, keyword_store, registry = get_services()

        async def event_generator():
            keyword_path = await run_in_thread(registry.resolve, video_id)
            if not keyword_path:
                yield "data: {\"error\": \"영상이 등록되지 않았습니다\"}\n\n"
                return

            try:
                start = max(current_time - window / 2, 0)
                end = current_time + window / 2

                window_data = await run_in_thread(
                    keyword_store.query,
                    video_id,
                    start,
                    end,
                    5,
                    keyword_path
                )

                keywords = [item["term"] for item in window_data["keywords"]]

                mapped = await run_in_thread(
                    graph_service.map_keywords_to_entities,
                    keywords,
                    5
                )

                for entity in mapped:
                    entity_name = entity.get("name")
                    if entity_name:
                        neighbors = await run_in_thread(
                            graph_service.get_entity_context,
                            entity_name,
                            1
                        )

                        event_data = {
                            "time": current_time,
                            "entity": entity,
                            "neighbors": neighbors
                        }
                        data = json.dumps(event_data, ensure_ascii=False)
                        yield f"data: {data}\n\n"

            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    @app.get("/health")
    async def health_check():
        """서버 상태 확인"""
        return {
            "status": "healthy",
            "workers": 4,
            "llm_queue_max_concurrent": 2
        }

    @app.get("/map-keywords")
    async def get_map_keywords():
        """지도 타임라인 키워드 조회 (map_keyword.json)"""
        import json
        
        keyword_root = os.environ.get("VIDEO_KEYWORD_DIR", "video_keywords")
        map_keyword_path = os.path.join(keyword_root, "map_keyword.json")
        
        if not os.path.exists(map_keyword_path):
            return {"timelineData": []}
        
        try:
            with open(map_keyword_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"map_keyword.json 로드 오류: {e}")
            return {"timelineData": []}

    @app.post("/cache/clear")
    async def clear_cache():
        """키워드 캐시 클리어 (JSON 변경 후 호출)"""
        graph_service, keyword_store, registry = get_services()
        keyword_store.clear_cache()
        _preload_cache.clear()
        _preload_status.clear()
        return {"message": "캐시 클리어 완료"}

    @app.get("/entity/{entity_name}/info")
    async def get_entity_info(entity_name: str):
        """
        엔티티 정보 조회 - GraphDB에 있으면 엔티티 정보 반환,
        없으면 FAISS에서 관련 문서/정보 검색하여 반환
        """
        graph_service, keyword_store, registry = get_services()
        
        # 1. GraphDB에서 엔티티 조회 시도
        entity_data = await run_in_thread(
            graph_service.get_entity_context,
            entity_name,
            1
        )
        
        if entity_data and entity_data.get("entity"):
            # GraphDB에 있음 - 엔티티 정보와 sources 반환
            entity = entity_data["entity"]
            sources = entity.get("sources", [])
            
            return {
                "found_in": "graphdb",
                "entity_name": entity_name,
                "entity": entity,
                "sources": sources,
                "neighbors": entity_data.get("neighbors", {})
            }
        
        # 2. GraphDB에 없음 - FAISS에서 검색
        print(f"[entity_info] '{entity_name}' GraphDB에 없음 - FAISS 검색")
        
        try:
            # FAISS 벡터 검색으로 관련 문서 조회
            faiss_results = await run_in_thread(
                graph_service.search_documents_faiss,
                entity_name,
                5  # top_k
            )
            
            # FAISS 결과에서 관련 엔티티들의 sources 수집
            related_sources = []
            related_entities = []
            
            for doc in faiss_results:
                content = doc.get("content", "")[:500]
                source = doc.get("source", "")
                
                # 문서 정보 추가
                related_sources.append({
                    "doc": source.split("/")[-1].split("\\")[-1] if source else "알 수 없음",
                    "snippet": content,
                    "type": "faiss_document"
                })
                
                # 문서 내용에서 GraphDB 엔티티 찾기
                entities_in_doc = await run_in_thread(
                    graph_service.find_entities_in_text,
                    content,
                    3  # 최대 3개
                )
                
                for ent in entities_in_doc:
                    if ent.get("name") and ent["name"] not in [e.get("name") for e in related_entities]:
                        # 해당 엔티티의 sources도 가져오기
                        ent_context = await run_in_thread(
                            graph_service.get_entity_context,
                            ent["name"],
                            0
                        )
                        if ent_context and ent_context.get("entity"):
                            ent_sources = ent_context["entity"].get("sources", [])
                            related_entities.append({
                                "name": ent["name"],
                                "category": ent_context["entity"].get("category", ""),
                                "type": ent_context["entity"].get("type", ""),
                                "sources": ent_sources[:3]  # 최대 3개 sources
                            })
            
            return {
                "found_in": "faiss",
                "entity_name": entity_name,
                "entity": None,
                "sources": related_sources,
                "related_entities": related_entities,
                "message": f"'{entity_name}'은(는) GraphDB에 없어 FAISS에서 관련 문서를 검색했습니다."
            }
            
        except Exception as e:
            print(f"[entity_info] FAISS 검색 오류: {e}")
            return {
                "found_in": "none",
                "entity_name": entity_name,
                "entity": None,
                "sources": [],
                "message": f"'{entity_name}'에 대한 정보를 찾을 수 없습니다."
            }

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 GraphRAG 서비스 미리 초기화"""
    print(" 서버 시작 - GraphRAG 서비스 초기화 중...")
    print("   LLM 동시 처리: 2개")
    
    # 비동기로 서비스 초기화
    await run_in_thread(get_services)
    
    # 키워드 스토어 캐시 클리어 (JSON 변경 반영)
    graph_service, keyword_store, registry = get_services()
    keyword_store.clear_cache()
    print("   키워드 캐시 클리어 완료")
    
    # LLM 큐 초기화
    get_llm_queue()
    
    print("\n" + "="*50)
    print("✅ 서버 준비 완료 - API 요청 대기 중")
    print("="*50 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    global _llm_queue, _thread_pool
    
    if _llm_queue:
        _llm_queue.shutdown()
    
    _thread_pool.shutdown(wait=False)
    print("서버 종료 완료")


if __name__ == "__main__":
    import uvicorn
    import sys

    use_reload = "--reload" in sys.argv or "-r" in sys.argv
    
    if use_reload:
        print("⚠️  개발 모드 (reload=True) - 파일 변경 시 서버 재시작됨")
        print("   무거운 모델이 매번 재로드되므로 주의하세요!")
        # 개발 모드에서는 워커 1개
        uvicorn.run("api_server:app", host="0.0.0.0", port=8080, reload=True)
    else:
        # 프로덕션 모드: 워커 4개
        print("🚀 프로덕션 모드 - 워커 4개로 실행")
        uvicorn.run(
            "api_server:app", 
            host="0.0.0.0", 
            port=8080, 
            workers=1,
            reload=False
        )
