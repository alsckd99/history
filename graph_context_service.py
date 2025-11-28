from typing import List, Dict, Any, Optional

from rag_graph import GraphRAGSystem


class GraphContextService:
    """GraphRAG + ArangoDB 래퍼"""

    def __init__(
        self,
        graphrag_config: Dict[str, Any],
        index_dir: Optional[str] = None,
        skip_graph_import: bool = True  # 기본적으로 그래프 재import 건너뛰기
    ):
        self.system = GraphRAGSystem(**graphrag_config)
        self.additional_vectorstores = []  # 추가 인덱스들
        
        if index_dir:
            try:
                # 인덱스 로드 시 그래프 import 건너뛰기 옵션 추가
                if skip_graph_import:
                    # 벡터 인덱스만 로드하고 그래프는 이미 DB에 있다고 가정
                    self._load_all_indexes(index_dir)
                else:
                    self.system.load_indexes(index_dir)
            except Exception as exc:
                print(f"[경고] 인덱스 로드 실패: {exc}")
    
    def _load_all_indexes(self, load_dir: str):
        """모든 사료 인덱스를 로드 (graphrag_data 내 모든 하위 디렉토리)"""
        import os
        from langchain_community.vectorstores import FAISS
        
        # 부모 디렉토리 (graphrag_data)
        parent_dir = os.path.dirname(load_dir)
        if not os.path.exists(parent_dir):
            parent_dir = load_dir
        
        # 모든 하위 디렉토리에서 인덱스 로드
        loaded_count = 0
        all_doc_stores = []
        all_entity_stores = []
        
        subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(parent_dir, subdir)
            
            # 문서 인덱스 로드
            doc_index_file = os.path.join(subdir_path, 'documents', 'index.faiss')
            if os.path.exists(doc_index_file):
                try:
                    doc_path = os.path.join(subdir_path, 'documents')
                    doc_store = FAISS.load_local(
                        doc_path,
                        self.system.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    all_doc_stores.append((subdir, doc_store))
                    loaded_count += 1
                except Exception as e:
                    print(f"  ⚠ {subdir} 문서 인덱스 로드 실패: {e}")
            
            # 엔티티 인덱스 로드
            entity_index_file = os.path.join(subdir_path, 'entities', 'index.faiss')
            if os.path.exists(entity_index_file):
                try:
                    entity_path = os.path.join(subdir_path, 'entities')
                    entity_store = FAISS.load_local(
                        entity_path,
                        self.system.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    all_entity_stores.append((subdir, entity_store))
                except Exception as e:
                    print(f"  ⚠ {subdir} 엔티티 인덱스 로드 실패: {e}")
        
        # 첫 번째 인덱스를 메인으로 설정
        if all_doc_stores:
            self.system.vectorstore = all_doc_stores[0][1]
            # 나머지는 병합
            for name, store in all_doc_stores[1:]:
                try:
                    self.system.vectorstore.merge_from(store)
                except Exception as e:
                    print(f"  ⚠ {name} 문서 인덱스 병합 실패: {e}")
            print(f"  ✓ 문서 인덱스 {len(all_doc_stores)}개 로드 완료")
        
        if all_entity_stores:
            self.system.entity_vectorstore = all_entity_stores[0][1]
            # 나머지는 병합
            for name, store in all_entity_stores[1:]:
                try:
                    self.system.entity_vectorstore.merge_from(store)
                except Exception as e:
                    print(f"  ⚠ {name} 엔티티 인덱스 병합 실패: {e}")
            print(f"  ✓ 엔티티 인덱스 {len(all_entity_stores)}개 로드 완료")
        
        print(f"  ✓ 총 {loaded_count}개 사료 인덱스 로드 완료")
        print(f"  ✓ 그래프 DB는 기존 ArangoDB 데이터 사용")

    def map_keywords_to_entities(self, keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """키워드를 ArangoDB 그래프에서 실제 관계된 엔티티로 매핑
        
        지식 그래프에서 직접 연결된 엔티티만 가져옴 (FAISS 미사용)
        - 그래프 DB에 이미 LLM으로 추출한 트리플 기반 관계가 저장되어 있음
        - 벡터 검색은 의미적 유사도만 제공하므로 실제 관계 정보가 아님
        """
        unique_keywords = [kw for kw in keywords if kw]
        seen = set()
        results = []
        
        if not self.system.graph_db:
            return []
        
        for kw in unique_keywords:
            # 그래프 DB에서 관계된 엔티티 조회 (depth=1: 직접 연결된 것만)
            neighbors = self.system.graph_db.query_neighbors(kw, depth=1)
            
            # 관계된 엔티티 추가
            for entity in neighbors.get('entities', []):
                name = entity.get('name')
                if not name or name in seen or name == kw:
                    continue
                seen.add(name)
                
                # 관계 타입 찾기
                relation_type = None
                for rel in neighbors.get('relations', []):
                    # _from 또는 _to에서 엔티티 이름 추출
                    from_name = rel.get('triple', {}).get('subject', '')
                    to_name = rel.get('triple', {}).get('object', '')
                    if name in [from_name, to_name]:
                        relation_type = rel.get('type') or rel.get('triple', {}).get('predicate')
                        break
                
                results.append({
                    "name": name,
                    "type": entity.get('type', 'unknown'),
                    "relation": relation_type,
                    "score": 1.0  # 그래프에서 직접 연결된 엔티티
                })
            
            # 충분한 결과가 모이면 중단
            if len(results) >= top_k:
                break
        
        return results[:top_k]

    def get_entity_context(self, entity_name: str, depth: int = 1) -> Dict[str, Any]:
        neighbors = {}
        if self.system.graph_db:
            neighbors = self.system.graph_db.query_neighbors(entity_name, depth=depth)
        entity_doc = None
        if self.system.graph_db:
            entity_doc = self.system.graph_db.query_entity(entity_name)
        return {
            "entity": entity_doc,
            "neighbors": neighbors
        }

    def answer_query(self, query: str) -> Dict[str, Any]:
        answer = self.system.generate_answer(query, use_graph=True)
        return {"query": query, "answer": answer}

    def get_documents_for_entity(self, entity_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
        docs = self.system.search_documents(entity_name, k=top_k)
        return docs

