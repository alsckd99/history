from typing import List, Dict, Any, Optional
import re

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

    def map_keywords_to_entities(self, keywords: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
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

    def search_documents_faiss(self, query: str, top_k: int = 5, expand_context: bool = True) -> List[Dict[str, Any]]:
        """FAISS 벡터 검색으로 관련 문서 조회
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            expand_context: True면 앞뒤 청크를 합쳐 완전한 문장으로 만듦
        """
        if not self.system.vectorstore:
            return []
        
        try:
            # 더 많은 청크를 가져와서 문맥 확장에 사용 (5배로 증가)
            search_k = top_k * 5 if expand_context else top_k
            docs = self.system.vectorstore.similarity_search(query, k=search_k)
            
            if not expand_context:
                results = []
                for doc in docs[:top_k]:
                    results.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "metadata": doc.metadata
                    })
                return results
            
            # 문맥 확장: 같은 소스의 청크들을 그룹화하고 합침
            source_chunks: Dict[str, List[Any]] = {}
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                if source not in source_chunks:
                    source_chunks[source] = []
                source_chunks[source].append(doc)
            
            results = []
            seen_content = set()  # 중복 방지
            
            for source, chunks in source_chunks.items():
                if len(results) >= top_k:
                    break
                
                # 청크들의 내용을 합침
                combined_content = self._combine_chunks(chunks)
                
                # 중복 체크 (첫 100자로)
                content_key = combined_content[:100] if combined_content else ""
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)
                
                if combined_content:
                    results.append({
                        "content": combined_content,
                        "source": source,
                        "metadata": chunks[0].metadata if chunks else {}
                    })
            
            return results[:top_k]
            
        except Exception as e:
            print(f"[FAISS 검색 오류] {e}")
            return []
    
    def _combine_chunks(self, chunks: List[Any], max_length: int = 1500) -> str:
        """여러 청크를 하나의 완전한 문장으로 합침 (앞뒤 문맥 포함)"""
        if not chunks:
            return ""
        
        # 청크 내용 추출
        contents = [chunk.page_content for chunk in chunks]
        
        # 청크들을 순서대로 정렬 (메타데이터에 chunk_id나 순서 정보가 있으면 사용)
        # 없으면 그냥 순서대로
        
        # 중복 제거하면서 합침
        seen_sentences = set()
        combined_parts = []
        
        for content in contents:
            # 문장 단위로 분리
            sentences = self._split_into_sentences(content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # 중복 문장 제거 (앞 30자로 체크)
                sentence_key = sentence[:30]
                if sentence_key in seen_sentences:
                    continue
                seen_sentences.add(sentence_key)
                
                combined_parts.append(sentence)
        
        # 합친 내용이 너무 길면 자르기
        combined = ' '.join(combined_parts)
        
        if len(combined) > max_length:
            # 문장 단위로 자르기
            truncated = ""
            for part in combined_parts:
                if len(truncated) + len(part) + 1 > max_length:
                    break
                truncated += part + " "
            combined = truncated.strip()
            if combined and not combined.endswith(('.', '다', '요', '음')):
                combined += "..."
        
        return combined
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분리"""
        # 한국어 문장 종결 패턴
        # 마침표, 물음표, 느낌표 뒤에 공백이나 줄바꿈이 오면 문장 끝
        sentences = re.split(r'(?<=[.?!다요음])\s+', text)
        
        # 빈 문장 제거
        return [s.strip() for s in sentences if s.strip()]

    def find_entities_in_text(self, text: str, max_count: int = 5) -> List[Dict[str, Any]]:
        """텍스트에서 GraphDB에 존재하는 엔티티 찾기"""
        if not self.system.graph_db:
            return []
        
        # 텍스트에서 n-gram 추출 (2~6자)
        found_entities = []
        seen_names = set()
        
        # 한글 단어 추출 (간단한 방식)
        import re
        words = re.findall(r'[가-힣]{2,10}', text)
        
        for word in words:
            if word in seen_names or len(word) < 2:
                continue
            
            # GraphDB에서 엔티티 조회
            entity = self.system.graph_db.query_entity(word)
            if entity:
                seen_names.add(word)
                found_entities.append({
                    "name": word,
                    "category": entity.get("category", ""),
                    "type": entity.get("type", "")
                })
                
                if len(found_entities) >= max_count:
                    break
        
        return found_entities

