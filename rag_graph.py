import json
import os
import glob
import re
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch

# LangChain imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document
    except ImportError:
        from langchain.schema import Document

try:
    from langchain_community.llms import Ollama
except ImportError:
    from langchain.llms import Ollama

# FAISS
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

# PDF 로더
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    try:
        from langchain.document_loaders import PyPDFLoader
    except ImportError:
        PyPDFLoader = None

# Tika 문서 추출
try:
    from tika import parser as tika_parser
    TIKA_AVAILABLE = True
except ImportError:
    TIKA_AVAILABLE = False

# 리랭커 모델
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

# Graph DB
from graph_db import ArangoGraphDB, KnowledgeGraphExtractor


def _normalize_metadata(
    raw_metadata: Optional[Dict],
    collection: str,
    filename: str,
    file_path: str,
    index: int
) -> Tuple[Dict, List[Dict]]:
    """표준 메타데이터 스키마와 주석 분리"""
    metadata = {
        'source': filename,
        'file_path': file_path,
        'index': index,
        'type': 'json',
        'collection': collection,
    }

    annotations = []
    if raw_metadata and isinstance(raw_metadata, dict):
        # 한국민족문화대백과사전: '항목명' 사용
        title = (
            raw_metadata.get('항목명') or
            raw_metadata.get('제목') or
            raw_metadata.get('title')
        )
        author = raw_metadata.get('저자') or raw_metadata.get('author')
        category = raw_metadata.get('카테고리') or raw_metadata.get('category')
        topic = raw_metadata.get('주제분류') or raw_metadata.get('topic')
        keywords = raw_metadata.get('키워드') or raw_metadata.get('keywords')

        if title:
            metadata['title'] = title
        if author:
            metadata['author'] = author
        if category:
            metadata['category'] = category
        if topic:
            metadata['topic'] = topic
        if keywords:
            metadata['keywords'] = keywords

        # 한국민족문화대백과사전 추가 필드 (한글 키 지원)
        # 엔티티 유형 (문서 유형 'json'과 구분)
        entity_type = (
            raw_metadata.get('항목 유형') or
            raw_metadata.get('type')
        )
        if entity_type:
            metadata['entity_type'] = entity_type
        
        # 원어 (한자)
        hanja = raw_metadata.get('원어') or raw_metadata.get('hanja')
        if hanja:
            metadata['hanja'] = hanja
        
        # 시대
        era = raw_metadata.get('시대') or raw_metadata.get('era')
        if era:
            metadata['era'] = era
        
        # 항목 정의
        definition = raw_metadata.get('항목 정의') or raw_metadata.get('definition')
        if definition:
            metadata['definition'] = definition
        
        # 요약
        summary = raw_metadata.get('요약') or raw_metadata.get('summary')
        if summary:
            metadata['summary'] = summary
        
        # URL
        url = raw_metadata.get('url')
        if url:
            metadata['url'] = url
        
        # 항목 분야
        field = raw_metadata.get('항목 분야') or raw_metadata.get('field')
        if field:
            metadata['category'] = field  # category로 매핑

        # 🆕 온톨로지 구축용 필드 (관련항목, 본문 표) - 한글 키 지원
        related = raw_metadata.get('관련항목') or raw_metadata.get('related_articles')
        if related:
            metadata['related_articles'] = related
        
        tables = raw_metadata.get('본문 표') or raw_metadata.get('tables')
        if tables:
            metadata['tables'] = tables

        # 다양한 키 이름 지원
        annotations = raw_metadata.get('annotations') or raw_metadata.get('주석') or []

        # 원본 값을 잃지 않도록 보관
        metadata['raw_metadata_keys'] = list(raw_metadata.keys())

    doc_id = f"{collection}_{filename}_{index}"
    metadata['document_id'] = metadata.get('title') or doc_id

    if annotations:
        metadata['has_annotations'] = True
        metadata['annotation_ids'] = [
            ann.get('id') for ann in annotations if isinstance(ann, dict) and ann.get('id')
        ]

    return metadata, annotations


def _extract_ontology_from_encyclopedia(item: Dict) -> Dict:
    """한국민족문화대백과사전 항목에서 온톨로지 정보 추출
    
    Returns:
        {
            'entity': { 엔티티 정보 },
            'relations': [ 관계 리스트 ],
            'battle_triples': [ 전투 표에서 추출한 트리플 ]
        }
    """
    result = {
        'entity': None,
        'relations': [],
        'battle_triples': []
    }
    
    # 1. 메인 엔티티 추출
    if item.get('항목명'):
        entity_type = item.get('항목 유형', '')
        # 항목 유형에서 실제 타입 추출 (예: "사건/전쟁" → "사건")
        if '/' in entity_type:
            entity_type = entity_type.split('/')[0]
        
        result['entity'] = {
            'name': item['항목명'],
            'hanja': item.get('원어', ''),
            'type': entity_type,
            'category': item.get('항목 분야', ''),
            'era': item.get('시대', ''),
            'definition': item.get('항목 정의', ''),
            'summary': item.get('요약', ''),
            'url': item.get('url', ''),
            'source': '한국민족문화대백과사전'
        }
    
    # 2. 관련항목에서 관계 추출
    related_articles = item.get('관련항목', [])
    main_name = item.get('항목명', '')
    
    for related in related_articles:
        if not isinstance(related, dict):
            continue
        
        related_name = related.get('항목명', '')
        if not related_name:
            continue
        
        related_type = related.get('항목 유형', '')
        if '/' in related_type:
            related_type = related_type.split('/')[0]
        
        # 관계 타입 추론
        relation_type = '관련_항목'
        if '인물' in related_type:
            relation_type = '관련_인물'
        elif '사건' in related_type or '전쟁' in related_type:
            relation_type = '관련_사건'
        elif '장소' in related_type or '지리' in related_type:
            relation_type = '관련_장소'
        elif '작품' in related_type or '문학' in related_type:
            relation_type = '관련_문헌'
        
        result['relations'].append({
            'subject': main_name,
            'subject_type': result['entity']['type'] if result['entity'] else '',
            'predicate': relation_type,
            'object': related_name,
            'object_type': related_type,
            'object_hanja': related.get('원어', ''),
            'object_url': related.get('URL', ''),
            'object_definition': related.get('항목 정의', ''),
            'source': '한국민족문화대백과사전'
        })
    
    # 3. 본문 표에서 전투 트리플 추출 (임진왜란 대소전투 등)
    tables = item.get('본문 표', [])
    for table in tables:
        if not isinstance(table, dict):
            continue
        
        title = table.get('title', '')
        rows = table.get('rows', [])
        
        # 전투 표인 경우
        if '전투' in title or '대첩' in title:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                
                # 날짜, 장소, 조선측, 왜측, 결과 추출
                date = row.get('col_0', '')
                place = row.get('col_1', '')
                joseon_commander = row.get('col_2', '')
                japan_commander = row.get('col_3', '')
                outcome = row.get('col_4', '')
                
                if place and joseon_commander:
                    # 장소 엔티티와 전투 관계
                    battle_name = f"{place} 전투" if '전투' not in place and '대첩' not in place else place
                    
                    # 조선 지휘관 → 전투 참여
                    result['battle_triples'].append({
                        'subject': joseon_commander.replace('(', '').replace(')', ''),
                        'subject_type': '인물',
                        'predicate': '전투_참여',
                        'object': battle_name,
                        'object_type': '전투',
                        'date': date,
                        'outcome': outcome,
                        'side': '조선',
                        'source': main_name
                    })
                    
                    # 일본 지휘관 → 전투 참여
                    if japan_commander and japan_commander != '?':
                        result['battle_triples'].append({
                            'subject': japan_commander,
                            'subject_type': '인물',
                            'predicate': '전투_참여',
                            'object': battle_name,
                            'object_type': '전투',
                            'date': date,
                            'outcome': outcome,
                            'side': '일본',
                            'source': main_name
                        })
                    
                    # 전투 → 장소
                    result['battle_triples'].append({
                        'subject': battle_name,
                        'subject_type': '전투',
                        'predicate': '발생_장소',
                        'object': place,
                        'object_type': '장소',
                        'date': date,
                        'source': main_name
                    })
    
    return result


def _append_annotations_to_content(content: str, annotations: List[Dict]) -> str:
    if not annotations:
        return content

    annotation_lines = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        label = ann.get('label') or ann.get('id') or 'annotation'
        text = ann.get('text') or ''
        if not text:
            continue
        annotation_lines.append(f"- [{label}] {text}")

    if not annotation_lines:
        return content

    return f"{content.rstrip()}\n\n### 주석\n" + "\n".join(annotation_lines)


def _build_annotation_documents(
    annotations: List[Dict],
    base_metadata: Dict
) -> List[Dict]:
    annotation_docs = []
    for idx, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            continue
        text = ann.get('text')
        if not text:
            continue

        ann_meta = dict(base_metadata)
        ann_meta.update({
            'type': 'annotation',
            'annotation_id': ann.get('id') or f"{base_metadata.get('document_id')}_ann_{idx}",
            'annotation_label': ann.get('label'),
            'annotation_index': idx,
        })

        annotation_docs.append({
            'content': text,
            'metadata': ann_meta
        })
    return annotation_docs


def load_documents_from_output(output_dir='output'):
    """output 디렉토리에서 JSON 및 PDF 문서 로드
    
    Args:
        output_dir: 문서 디렉토리 경로
        
    Returns:
        문서 리스트
    """
    print(f"\n{output_dir} 디렉토리에서 문서 로드 중...")
    
    if not os.path.exists(output_dir):
        print(f"디렉토리가 없습니다: {output_dir}")
        return []
    
    documents = []
    
    # JSON 파일 로드
    json_files = glob.glob(os.path.join(output_dir, '**/*.json'), recursive=True)
    print(f"JSON 파일 {len(json_files)}개 발견")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = os.path.basename(json_file)
            collection_name = os.path.basename(os.path.dirname(json_file))
            
            # 리스트 형태의 JSON
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    # 다양한 content 필드명 지원
                    content = (
                        item.get('content') or 
                        item.get('항목 본문') or  # 한국민족문화대백과사전
                        item.get('body') or
                        ''
                    )
                    if content:
                        # 한국민족문화대백과사전 형식인 경우 메타데이터 구성
                        if '항목명' in item:
                            raw_metadata = {
                                'title': item.get('항목명'),
                                'url': item.get('url'),
                                'category': item.get('항목 분야'),
                                'type': item.get('항목 유형'),
                                'era': item.get('시대'),
                                'definition': item.get('항목 정의'),
                                'summary': item.get('요약'),
                                'keywords': item.get('키워드'),
                                'hanja': item.get('원어'),
                                'annotations': item.get('주석', []),
                                # 🆕 온톨로지 구축용 추가 필드
                                'related_articles': item.get('관련항목', []),
                                'tables': item.get('본문 표', []),
                            }
                        else:
                            raw_metadata = item.get('metadata', {})
                        
                        normalized_meta, annotations = _normalize_metadata(
                            raw_metadata,
                            collection_name,
                            filename,
                            json_file,
                            idx
                        )
                        
                        # URL 추가
                        if item.get('url'):
                            normalized_meta['url'] = item.get('url')
                        
                        content_with_notes = _append_annotations_to_content(content, annotations)
                        documents.append({
                            'content': content_with_notes,
                            'metadata': normalized_meta
                        })
                        if annotations:
                            documents.extend(_build_annotation_documents(annotations, normalized_meta))
            
            # 딕셔너리 형태의 JSON
            elif isinstance(data, dict):
                if 'documents' in data:
                    for idx, item in enumerate(data['documents']):
                        content = item.get('content', '')
                        if content:
                            normalized_meta, annotations = _normalize_metadata(
                                item.get('metadata', {}),
                                collection_name,
                                filename,
                                json_file,
                                idx
                            )
                            if item.get('title') and 'title' not in normalized_meta:
                                normalized_meta['title'] = item['title']
                            content_with_notes = _append_annotations_to_content(content, annotations)
                            documents.append({
                                'content': content_with_notes,
                                'metadata': normalized_meta
                            })
                            if annotations:
                                documents.extend(_build_annotation_documents(annotations, normalized_meta))
            
            print(f"  ✓ {filename}")
            
        except Exception as e:
            print(f"  ✗ {filename}: {e}")
    
    # PDF 파일 로드
    if PyPDFLoader:
        pdf_files = glob.glob(os.path.join(output_dir, '**/*.pdf'), recursive=True)
        print(f"\nPDF 파일 {len(pdf_files)}개 발견")
        
        for pdf_file in pdf_files:
            try:
                filename = os.path.basename(pdf_file)
                loader = PyPDFLoader(pdf_file)
                pdf_docs = loader.load()
                
                for page_num, doc in enumerate(pdf_docs):
                    if doc.page_content.strip():
                        collection_name = os.path.basename(os.path.dirname(pdf_file))
                        documents.append({
                            'content': doc.page_content,
                            'metadata': {
                                'source': filename,
                                'file_path': pdf_file,
                                'page': page_num + 1,
                                'type': 'pdf',
                                'collection': collection_name
                            }
                        })
                
                print(f"  ✓ {filename} ({len(pdf_docs)}페이지)")
                
            except Exception as e:
                print(f"  ✗ {filename}: {e}")
    else:
        print("\nPyPDFLoader를 사용할 수 없습니다. PDF 파일을 건너뜁니다.")
        print("설치: pip install pypdf")
    
    print(f"\n총 {len(documents)}개 문서 로드 완료")
    return documents


class GraphRAGSystem:
    """그래프 + 벡터 기반 하이브리드 RAG 시스템 (고급 기능 포함)"""
    
    def __init__(
        self,
        embedding_model_name='jhgan/ko-sroberta-multitask',
        llm_model_name='gemma3:12b',
        arango_host='localhost',
        arango_port=8529,
        arango_password='',
        arango_db_name='knowledge_graph',
        arango_reset=False,
        global_arango_db_name=None,
        global_arango_reset=False,
        use_reranker=True,
        reranker_model='BAAI/bge-reranker-v2-m3',
        use_tika=False
    ):
        """초기화
        
        Args:
            embedding_model_name: 임베딩 모델 이름
            llm_model_name: LLM 모델 이름
            arango_host: ArangoDB 호스트
            arango_port: ArangoDB 포트
            arango_password: ArangoDB 비밀번호
            arango_reset: True면 ArangoDB 기존 데이터 삭제
            use_reranker: 리랭커 사용 여부
            reranker_model: 리랭커 모델 이름
            use_tika: Apache Tika 사용 여부
        """
        print("\n고급 GraphRAG 시스템 초기화 중...")
        
        # GPU/CPU 자동 감지
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"디바이스: {device.upper()}")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        # 임베딩 모델
        print(f"임베딩 모델 로드: {embedding_model_name}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': device},  # GPU/CPU 자동 선택
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"  ✓ 임베딩 모델 로드 성공 ({device.upper()})")
        except Exception as e:
            print(f"  ✗ 임베딩 모델 로드 실패: {e}")
            print("  기본 모델로 대체합니다...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name='jhgan/ko-sroberta-multitask',
                model_kwargs={'device': device}
            )
        
        # LLM
        print(f"LLM 초기화: {llm_model_name}")
        try:
            self.llm = Ollama(
                model=llm_model_name,
                temperature=0.7
            )
            print("  ✓ LLM 초기화 성공")
        except Exception as e:
            print(f"  ✗ LLM 초기화 실패: {e}")
            self.llm = None
        
        # 리랭커 초기화
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker and RERANKER_AVAILABLE:
            self._init_reranker(reranker_model)
        elif use_reranker and not RERANKER_AVAILABLE:
            print("\n⚠️  리랭커를 사용할 수 없습니다")
            print("   설치: pip install sentence-transformers")
            self.use_reranker = False
        
        # Tika 설정
        self.use_tika = use_tika
        if use_tika and not TIKA_AVAILABLE:
            print("\n⚠️  Tika를 사용할 수 없습니다")
            print("   설치: pip install tika")
            self.use_tika = False
        
        # 그래프 데이터베이스
        self.graph_db = ArangoGraphDB(
            host=arango_host,
            port=arango_port,
            password=arango_password,
            db_name=arango_db_name,
            reset=arango_reset
        )
        self.global_graph_db = None
        if global_arango_db_name:
            if global_arango_db_name == arango_db_name and not global_arango_reset and arango_reset:
                # 동일 DB를 두 번 초기화하지 않도록 reset 우선순위 조정
                global_arango_reset = False
            self.global_graph_db = ArangoGraphDB(
                host=arango_host,
                port=arango_port,
                password=arango_password,
                db_name=global_arango_db_name,
                reset=global_arango_reset
            )
        
        # 벡터 스토어
        self.vectorstore = None
        self.entity_vectorstore = None
        
        # 엔티티 임베딩 캐시
        self.entity_embeddings = {}  # entity_name -> embedding vector
        
        print("\n✓ GraphRAG 시스템 초기화 완료")
        if self.use_reranker:
            print("  - 리랭커: 활성화")
        if self.use_tika:
            print("  - Tika: 활성화")
    
    def _init_reranker(self, model_name: str):
        """리랭커 모델 초기화"""
        print(f"\n리랭커 모델 로드: {model_name}")
        try:
            self.reranker = CrossEncoder(model_name, max_length=512)
            print("  ✓ 리랭커 모델 로드 성공")
        except Exception as e:
            print(f"  ✗ 리랭커 모델 로드 실패: {e}")
            self.reranker = None
            self.use_reranker = False
    
    def build_index(
        self,
        documents: List[Dict],
        extract_graph: bool = True,
        skip_vector_index: bool = False
    ):
        """인덱스 구축 (벡터 + 그래프)
        
        Args:
            documents: 문서 리스트
            extract_graph: 지식 그래프 추출 여부
            skip_vector_index: 벡터 인덱스 구축 건너뛰기 (구조화된 데이터용)
        """
        print(f"\nGraphRAG 인덱스 구축 시작 ({len(documents)}개 문서)")
        
        # 1. 벡터 인덱스 구축 (문서 기반) - 건너뛸 수 있음
        if skip_vector_index:
            print("\n[1/3] 벡터 인덱스 구축 건너뜀 (구조화된 데이터)")
        else:
        print("\n[1/3] 벡터 인덱스 구축 중...")
        self._build_vector_index(documents)
        
        # 2. 지식 그래프 추출 및 저장
        if extract_graph and self.graph_db.db:
            print("\n[2/3] 지식 그래프 추출 중...")
            self._build_knowledge_graph(documents)
        
        # 3. 엔티티 벡터 인덱스 구축
        print("\n[3/3] 엔티티 벡터 인덱스 구축 중...")
        self._build_entity_vector_index()
        
        print("\nGraphRAG 인덱스 구축 완료!")
    
    def _build_vector_index(self, documents: List[Dict]):
        """문서 벡터 인덱스 구축"""
        # Document 객체로 변환
        docs = []
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        split_docs = text_splitter.split_documents(docs)
        
        print(f"  {len(split_docs)}개 청크 생성")
        
        # FAISS 벡터 스토어 생성
        self.vectorstore = FAISS.from_documents(
            documents=split_docs,
            embedding=self.embeddings
        )
        
        print("  문서 벡터 인덱스 생성 완료")
    
    def _build_knowledge_graph(self, documents: List[Dict]):
        """지식 그래프 구축 (온톨로지 + LLM 하이브리드)"""
        all_entities = []
        all_relations = []
        
        # 1단계: 문서 분류 (한국민족문화대백과사전 vs 일반 사료)
        encyclopedia_docs = []
        other_docs = []
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            collection = metadata.get('collection', '')
            
            # 한국민족문화대백과사전 형식 확인 (다양한 조건)
            is_encyclopedia = (
                '한국민족문화대백과사전' in collection or
                'encykorea' in collection.lower() or
                metadata.get('hanja') or  # 원어(한자)가 있으면
                metadata.get('era') or    # 시대 정보가 있으면
                metadata.get('definition') or  # 항목 정의가 있으면
                metadata.get('related_articles') or
                metadata.get('tables')
            )
            
            if is_encyclopedia:
                encyclopedia_docs.append(doc)
            else:
                other_docs.append(doc)
        
        # 2단계: 한국민족문화대백과사전 온톨로지 구축 (LLM 불필요, 빠름)
        if encyclopedia_docs:
            print(f"  📚 한국민족문화대백과사전 온톨로지 구축 중 ({len(encyclopedia_docs)}개 항목)...")
            onto_entities, onto_relations = self._extract_encyclopedia_ontology(encyclopedia_docs)
            all_entities.extend(onto_entities)
            all_relations.extend(onto_relations)
            print(f"    → 엔티티 {len(onto_entities)}개, 관계 {len(onto_relations)}개 추출")
        
        # 3단계: 나머지 문서에서 LLM 기반 추출 (간양록, 난중일기 등 일반 사료)
        if other_docs:
            print(f"  🤖 LLM 기반 지식 추출 중 ({len(other_docs)}개 문서)...")
        extractor = KnowledgeGraphExtractor(
            llm_model=self.llm.model if self.llm else 'deepseek-r1:latest'
        )
        llm_entities, llm_relations = extractor.extract_entities_and_relations(other_docs)
        all_entities.extend(llm_entities)
        all_relations.extend(llm_relations)
        print(f"    → 엔티티 {len(llm_entities)}개, 관계 {len(llm_relations)}개 추출")
        
        # 4단계: 그래프 DB에 삽입
        self._insert_into_graphs(all_entities, all_relations)
    
    def _extract_encyclopedia_ontology(self, docs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """한국민족문화대백과사전에서 온톨로지 직접 추출 (LLM 불필요)
        
        추출 소스:
        1. 메인 엔티티 (항목명, 정의, 시대 등)
        2. 관련항목 → 명시적 관계
        """
        entities = []
        relations = []
        seen_entities = set()
        
        # 본문 내 링크 패턴: [표시텍스트](E0000000) 또는 [『책이름』](ID)
        import re
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        
        
        # 디버깅: title 없는 문서 수 추적
        no_title_count = 0
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            
            # 메인 엔티티 생성
            title = metadata.get('title')
            if not title:
                no_title_count += 1
                if no_title_count <= 5:
                    # 디버깅: metadata의 키 확인
                    keys = list(metadata.keys())[:10]
                    print(f"    [DEBUG] title 없음 - metadata keys: {keys}")
                continue
            
            # entity_type 사용 (문서 type='json'과 구분)
            entity_type = metadata.get('entity_type') or metadata.get('category') or '항목'
            if '/' in str(entity_type):
                entity_type = entity_type.split('/')[0]
            
            # category(항목 분야)는 항상 존재하므로 키 생성에 사용
            category = metadata.get('category', '')
            
            # 중복 체크: name + category 조합 (동음이의어 구분)
            entity_key = f"{title}_{category}" if category else f"{title}_{entity_type}"
            if entity_key in seen_entities:
                continue
            
            # 항목 본문 가져오기
            content = doc.get('content', '')
            
            entities.append({
                'name': title,
                'type': entity_type,
                'hanja': metadata.get('hanja', ''),
                'era': metadata.get('era', ''),
                'category': category,
                'definition': metadata.get('definition', ''),
                'summary': metadata.get('summary', ''),
                'content': content,  # 항목 본문 저장
                'url': metadata.get('url', ''),  # 대표 URL (호환성)
                'sources': [{
                    'type': '한국민족문화대백과사전',
                    'doc': '한국민족문화대백과사전',  # 사료명
                    'title': title,  # 항목명
                    'url': metadata.get('url', ''),  # 소스별 URL
                    'snippet': metadata.get('definition', '')[:200]
                }]
            })
            seen_entities.add(entity_key)
            
            # 관련항목에서 관계만 추출 (엔티티는 메인 문서에서만 생성)
            related_articles = metadata.get('related_articles', [])
            for related in related_articles:
                if not isinstance(related, dict):
                    continue
                
                related_name = related.get('항목명', '')
                if not related_name:
                    continue
                
                # 관계 타입 추론
                related_type = related.get('항목 유형', '')
                predicate = '관련_항목'
                if '인물' in related_type:
                    predicate = '관련_인물'
                elif '사건' in related_type or '전쟁' in related_type:
                    predicate = '관련_사건'
                elif '장소' in related_type or '지리' in related_type:
                    predicate = '관련_장소'
                elif '작품' in related_type or '문학' in related_type:
                    predicate = '관련_문헌'
                
                # 동명이의어 매칭용 추가 정보 포함
                # subject_field는 메인 엔티티의 category(항목 분야)
                relations.append({
                    'subject': title,
                    'subject_type': entity_type,
                    'subject_hanja': metadata.get('hanja', ''),
                    'subject_field': metadata.get('category', ''),
                    'predicate': predicate,
                    'object': related_name,
                    'object_type': related_type.split('/')[0] if related_type else '',
                    'object_hanja': related.get('원어', ''),
                    'object_field': related.get('항목 분야', ''),
                    'source': '한국민족문화대백과사전'
                })
            
            # 본문 내 링크에서 관계만 추출 (엔티티는 메인 문서에서만 생성)
            content = doc.get('content', '')
            if content and title:
                links = link_pattern.findall(content)
                seen_links = set()  # 동일 문서 내 중복 링크 방지
                for display_text, link_id in links:
                    # 동일 link_id는 한 번만 처리
                    if link_id in seen_links:
                        continue
                    seen_links.add(link_id)
                    
                    # 표시 텍스트에서 엔티티명 추출
                    entity_name = display_text.strip()
                    # 괄호 안의 한자 제거
                    if '(' in entity_name:
                        entity_name = entity_name.split('(')[0].strip()
                    # 『』 제거
                    entity_name = entity_name.replace('『', '').replace('』', '')
                    entity_name = entity_name.replace('[', '').replace(']', '')
                    
                    if not entity_name or len(entity_name) < 2:
                        continue
                    if entity_name == title:  # 자기 자신 참조 제외
                        continue
                    
                    # 관계 추가 (link_id로 정확한 엔티티 매칭 가능)
                    relations.append({
                        'subject': title,
                        'subject_type': entity_type,
                        'subject_hanja': metadata.get('hanja', ''),
                        'subject_field': category,
                        'predicate': '본문_언급',
                        'object': entity_name,
                        'object_type': '',  # 본문 링크에는 타입 정보 없음
                        'object_hanja': '',
                        'object_field': '',
                        'object_url_id': link_id,  # URL ID로 정확한 매칭 가능
                        'source': '한국민족문화대백과사전'
                    })
        
        return entities, relations
    
    def _insert_into_graphs(self, entities: List[Dict], relations: List[Dict]):
        targets = []
        if self.graph_db and self.graph_db.db:
            targets.append(('개별 그래프', self.graph_db))
        if self.global_graph_db and self.global_graph_db.db:
            targets.append(('통합 그래프', self.global_graph_db))
        
        if not targets:
            return
        
        # 엔티티 이름 → 키 매핑 생성 (관계 삽입용)
        # category(항목 분야)가 항상 있으므로 이를 키 생성에 사용
        entity_key_map = {}
        for entity in entities:
            name = entity.get('name', '')
            if name:
                # _key가 없으면 생성 (category 우선, 없으면 type 사용)
                category = entity.get('category', '')
                entity_type = entity.get('type', '')
                if category:
                    key = self._sanitize_key_for_entity(f"{name}_{category}")
                elif entity_type and entity_type != '미분류':
                    key = self._sanitize_key_for_entity(f"{name}_{entity_type}")
                else:
                    key = self._sanitize_key_for_entity(name)
                if name not in entity_key_map:
                    entity_key_map[name] = {}
                # 매핑 키도 category 기반으로
                map_key = category or entity_type or ''
                entity_key_map[name][map_key] = key
        
        print(f"  → 엔티티 매핑: {len(entity_key_map)}개")
        
        for label, db in targets:
            db.insert_entities(entities)
            # 엔티티 삽입 후, 실제 DB에서 매핑 로드 (병합된 키 반영)
            # entity_key_map은 예상 키, 실제 DB의 키가 다를 수 있음
            db.insert_relations(relations, entity_key_map=None)  # DB에서 로드하도록
            stats = db.get_statistics()
            print(f"  [{label}] 엔티티: {stats.get('entities_count', 0)}개, 관계: {stats.get('relations_count', 0)}개")
    
    def _sanitize_key_for_entity(self, text: str) -> str:
        """엔티티 키 생성 (graph_db.py와 동일한 로직 - SHA256 24자)"""
        import hashlib
        import re
        if not text or not isinstance(text, str):
            return 'unknown_' + hashlib.sha256(str(id(text)).encode()).hexdigest()[:8]
        normalized = text.replace(' ', '_')
        ascii_only = re.sub(r'[^a-zA-Z0-9_-]', '', normalized)
        # 실제 영숫자가 3자 이상인 경우만 ASCII 키 사용
        alphanumeric_only = re.sub(r'[^a-zA-Z0-9]', '', ascii_only)
        if alphanumeric_only and len(alphanumeric_only) >= 3:
            if not ascii_only[0].isalpha():
                ascii_only = 'K_' + ascii_only
            return ascii_only[:128]
        # SHA256 해시의 앞 24자 사용 (충돌 확률 극히 낮음)
        hash_part = hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]
        return f"K_{hash_part}"
    
    def _build_entity_vector_index(self):
        """엔티티 벡터 인덱스 구축 (KNN 검색용)"""
        if not self.graph_db.db:
            print("  그래프 DB가 없어 엔티티 벡터 인덱스를 건너뜁니다.")
            return
        
        try:
            # 모든 엔티티 가져오기
            entities_collection = self.graph_db.db.collection('entities')
            entities = list(entities_collection.all())
            
            if not entities:
                print("  엔티티가 없습니다.")
                return
            
            print(f"  {len(entities)}개 엔티티 임베딩 중...")
            
            # 엔티티를 Document로 변환
            entity_docs = []
            for entity in entities:
                name = entity.get('name', '')
                entity_type = entity.get('type', 'entity')
                sources = entity.get('sources', [])
                
                # 엔티티 설명 생성
                description = f"{name} (타입: {entity_type})"
                if sources:
                    doc_names = []
                    for src in sources[:3]:
                        if isinstance(src, dict):
                            doc_names.append(src.get('doc') or src.get('type') or 'unknown')
                        else:
                            doc_names.append(str(src))
                    description += f" [출처: {', '.join(doc_names)}]"
                
                entity_docs.append(Document(
                    page_content=description,
                    metadata={
                        'entity_name': name,
                        'entity_key': entity['_key'],
                        'type': entity_type,
                        'is_entity': True
                    }
                ))
            
            # 엔티티 벡터 스토어 생성
            self.entity_vectorstore = FAISS.from_documents(
                documents=entity_docs,
                embedding=self.embeddings
            )
            
            # 엔티티 임베딩 캐시 생성
            for doc, entity in zip(entity_docs, entities):
                embedding = self.embeddings.embed_query(doc.page_content)
                self.entity_embeddings[entity['name']] = np.array(embedding)
            
            print(f"  엔티티 벡터 인덱스 생성 완료")
            
        except Exception as e:
            print(f"  엔티티 벡터 인덱스 구축 오류: {e}")
    
    def search_entities_knn(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """엔티티 KNN 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 엔티티 수
            
        Returns:
            유사한 엔티티 리스트
        """
        if not self.entity_vectorstore:
            print("엔티티 벡터 인덱스가 없습니다.")
            return []
        
        print(f"\n엔티티 KNN 검색: '{query}' (k={k})")
        
        # 유사도 검색
        results = self.entity_vectorstore.similarity_search_with_score(query, k=k)
        
        # 결과 정리
        entities = []
        for doc, score in results:
            entity_name = doc.metadata.get('entity_name', '')
            similarity = float(1 / (1 + score))
            
            entities.append({
                'name': entity_name,
                'type': doc.metadata.get('type', 'entity'),
                'similarity_score': similarity,
                'description': doc.page_content
            })
        
        return entities
    
    def search_documents(
        self,
        query: str,
        k: int = 5,
        use_reranker: bool = None
    ) -> List[Dict]:
        """문서 벡터 검색 (리랭킹 옵션)
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            use_reranker: 리랭커 사용 여부 (None이면 기본 설정 따름)
            
        Returns:
            유사한 문서 리스트
        """
        if not self.vectorstore:
            print("문서 벡터 인덱스가 없습니다.")
            return []
        
        print(f"\n문서 벡터 검색: '{query}' (k={k})")
        
        # 리랭킹 사용 여부 결정
        if use_reranker is None:
            use_reranker = self.use_reranker
        
        # 초기 검색 (리랭킹 시 더 많이 가져옴)
        initial_k = k * 3 if use_reranker and self.reranker else k
        results = self.vectorstore.similarity_search_with_score(query, k=initial_k)
        
        print(f"  1단계: {len(results)}개 문서 검색")
        
        # 리랭킹 수행
        if use_reranker and self.reranker and len(results) > 0:
            results = self._rerank_results(query, results, k)
            print(f"  2단계: 리랭킹 완료, 상위 {len(results)}개 선택")
        
        documents = []
        for doc, score in results[:k]:
            documents.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(1 / (1 + score))
            })
        
        return documents
    
    def _rerank_results(self, query: str, results: list, top_k: int) -> list:
        """리랭커로 검색 결과 재순위화
        
        Args:
            query: 검색 쿼리
            results: 초기 검색 결과 [(doc, score), ...]
            top_k: 반환할 문서 수
            
        Returns:
            재순위화된 결과 [(doc, rerank_score), ...]
        """
        if not self.reranker or len(results) == 0:
            return results
        
        try:
            # 쿼리-문서 쌍 생성
            query_doc_pairs = [[query, doc.page_content[:512]] for doc, _ in results]
            
            # 리랭킹 점수 계산
            rerank_scores = self.reranker.predict(query_doc_pairs)
            
            # (문서, 리랭킹 점수) 쌍으로 재구성
            reranked = [(results[i][0], float(rerank_scores[i])) 
                        for i in range(len(results))]
            
            # 점수 기준 내림차순 정렬 (높은 점수가 더 관련성 높음)
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            print(f"  ⚠️  리랭킹 오류: {e}")
            return results[:top_k]
    
    def _document_signature(self, doc: Dict) -> str:
        metadata = doc.get('metadata') or {}
        source = str(metadata.get('source') or metadata.get('file_path') or '')
        locator = str(metadata.get('page') or metadata.get('index') or metadata.get('chunk_id') or '')
        if not source and not locator:
            snippet = doc.get('content', '')[:80]
            return hashlib.md5(snippet.encode('utf-8', errors='ignore')).hexdigest()
        return f"{source}:{locator}"
    
    def _blend_documents(self, primary: List[Dict], secondary: List[Dict]) -> List[Dict]:
        if not secondary:
            return primary
        merged = list(primary)
        seen = {self._document_signature(doc) for doc in primary}
        for doc in secondary:
            signature = self._document_signature(doc)
            if signature in seen:
                continue
            merged.append(doc)
            seen.add(signature)
        return merged
    
    def _collect_graph_terms(self, entities: List[Dict], graph_depth: int) -> Tuple[List[str], List[Dict]]:
        terms = []
        graph_context = []
        if not self.graph_db or not self.graph_db.db or not entities:
            return terms, graph_context
        
        for entity in entities[:3]:
            name = entity.get('name')
            if not name:
                continue
            terms.append(name)
            neighbors = self.graph_db.query_neighbors(
                name,
                depth=graph_depth
            )
            if neighbors.get('entities') or neighbors.get('relations'):
                graph_context.append({
                    'center_entity': name,
                    'neighbors': neighbors
                })
            for relation in neighbors.get('relations', [])[:5]:
                rel_type = relation.get('type')
                if rel_type:
                    terms.append(rel_type)
            for neighbor_entity in neighbors.get('entities', [])[:5]:
                neighbor_name = neighbor_entity.get('name') or neighbor_entity.get('display_name') or neighbor_entity.get('normalized_name')
                if neighbor_name and neighbor_name != name:
                    terms.append(neighbor_name)
        
        unique_terms = []
        seen_terms = set()
        for term in terms:
            if term and term not in seen_terms:
                unique_terms.append(term)
                seen_terms.add(term)
            if len(unique_terms) >= 12:
                break
        
        return unique_terms, graph_context
    
    def _graph_expanded_document_search(
        self,
        query: str,
        graph_terms: List[str],
        k_docs: int
    ) -> List[Dict]:
        if not graph_terms:
            return []
        expansion = " ".join(graph_terms)
        expanded_query = f"{query} {expansion}"
        return self.search_documents(expanded_query, k=k_docs, use_reranker=False)
    
    def graph_only_search(
        self,
        query: str,
        k_entities: int = 5,
        graph_depth: int = 1
    ) -> Dict:
        """하이브리드 검색: FAISS로 엔티티 찾고 → 그래프 DB로 관계 확장
        
        Args:
            query: 검색 쿼리
            k_entities: 반환할 엔티티 수
            graph_depth: 그래프 탐색 깊이
            
        Returns:
            검색 결과 (엔티티, 관계, 문서 출처)
        """
        print(f"\n[GraphRAG] 검색 시작: '{query}'")
        
        all_entities = []
        all_relations = []
        all_sources = set()
        graph_context = []
        seen_entities = set()
        
        # 0단계: 쿼리에서 핵심 키워드 추출
        # "X에 대해 알려줘", "X가 뭐야", "X란?" 등에서 X 추출
        import re
        query_keywords = []
        
        # 패턴 매칭으로 핵심 키워드 추출
        patterns = [
            r'^(.+?)(?:에 대해|에 관해|에대해|에관해|가 뭐|이 뭐|란\?|이란|가 무엇|이 무엇|은 무엇|는 무엇|을 알려|를 알려|에 대한|에 관한)',
            r'^(.+?)(?:이란|란|이라는|라는|이라고|라고).*(?:뭐|무엇|알려)',
            r'^(.+?)(?:설명|알려줘|알려주세요|알려 줘|알려 주세요)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                keyword = match.group(1).strip()
                # 불필요한 조사 제거
                keyword = re.sub(r'(은|는|이|가|을|를|의|와|과|에서|에게|한테|로|으로)$', '', keyword)
                if keyword and len(keyword) >= 2:
                    query_keywords.append(keyword)
                    print(f"[GraphRAG] 쿼리에서 핵심 키워드 추출: '{keyword}'")
                break
        
        # 키워드를 추출하지 못했을 때만 명사 분리 시도
        if not query_keywords:
            # 간단한 명사 추출: 조사/어미 제거 후 3글자 이상 단어만
            words = query.replace('?', '').replace('!', '').replace('.', '').split()
            for word in words:
                # 조사/어미 제거
                cleaned = re.sub(r'(은|는|이|가|을|를|의|와|과|에서|에게|한테|로|으로|에|도|만|까지|부터|라고|이라고|라는|이라는|란|이란|야|이야|요|이요|죠|지요|네|군|구나)$', '', word)
                # 3글자 이상만 (너무 짧은 단어 제외)
                if cleaned and len(cleaned) >= 3 and cleaned not in query_keywords:
                    query_keywords.append(cleaned)
            if query_keywords:
                print(f"[GraphRAG] 명사 추출 결과: {query_keywords}")
        
        # 1단계: 질문에서 추출한 키워드로 GraphDB 직접 검색
        print(f"[GraphRAG] 1단계: 질문 키워드로 GraphDB 직접 검색...")
        
        # 메인 키워드
        main_keyword = None
        main_keyword_url = ''
        
        # GraphDB에서 키워드와 정확히 일치하는 엔티티만 검색
        if self.graph_db and self.graph_db.db:
            for kw in query_keywords:
                print(f"[GraphRAG] GraphDB에서 '{kw}' 검색 중...")
                
                # 정확히 일치하는 엔티티만 검색 (부분 매칭 제외)
                try:
                    find_query = """
                    FOR e IN entities
                        FILTER e.name == @name
                        LIMIT 5
                        RETURN e
                    """
                    cursor = self.graph_db.db.aql.execute(find_query, bind_vars={'name': kw})
                    found_entities = list(cursor)
                    
                    # 정확히 일치하는 것이 없으면 키워드를 포함하는 엔티티 검색
                    # (단, 키워드 길이가 3자 이상일 때만)
                    if not found_entities and len(kw) >= 3:
                        find_query2 = """
                        FOR e IN entities
                            FILTER CONTAINS(e.name, @name)
                            SORT LENGTH(e.name) ASC
                            LIMIT 3
                            RETURN e
                        """
                        cursor = self.graph_db.db.aql.execute(find_query2, bind_vars={'name': kw})
                        found_entities = list(cursor)
                    
                    for ent in found_entities:
                        name = ent.get('name')
                        if name and name not in seen_entities:
                            seen_entities.add(name)
                            entity_data = {
                                'name': name,
                                'type': ent.get('type', 'unknown'),
                                'sources': ent.get('sources', []),
                                'definition': ent.get('definition', ''),
                                'summary': ent.get('summary', ''),
                                'url': ent.get('url', ''),
                                'category': ent.get('category', '')
                            }
                            all_entities.append(entity_data)
                            print(f"[GraphRAG] GraphDB에서 발견: '{name}' (type: {ent.get('type', 'unknown')})")
                except Exception as e:
                    print(f"[GraphRAG] GraphDB 검색 오류: {e}")
        
        if not all_entities:
            print(f"[GraphRAG] GraphDB에서 일치하는 엔티티 없음")
        
        # 2단계: 그래프 DB에서 관계 확장
        if not self.graph_db or not self.graph_db.db:
            print("[GraphRAG] 경고: 그래프 DB가 연결되지 않음!")
            return {
                'query': query,
                'main_keyword': None,
                'main_keyword_url': '',
                'entities': all_entities,
                'relations': [],
                'graph_context': [],
                'sources': []
            }
        
        print(f"[GraphRAG] 2단계: 그래프 DB 관계 확장...")
        
        # 1단계에서 찾은 엔티티들의 그래프 관계 조회
        # 첫 번째 엔티티를 메인 키워드로 설정
        for entity in all_entities[:5]:  # 상위 5개 엔티티
            name = entity['name']
            print(f"[GraphRAG] '{name}' 관계 조회 중...")
            
            # 메인 키워드 설정 (첫 번째 엔티티)
            if main_keyword is None:
                main_keyword = name
                # URL은 이미 1단계에서 가져온 경우
                main_keyword_url = entity.get('url', '')
                if not main_keyword_url:
                    # sources에서 백과사전 URL 찾기
                    for src in entity.get('sources', []):
                        if isinstance(src, dict):
                            if src.get('doc') == '한국민족문화대백과사전' and src.get('url'):
                                main_keyword_url = src.get('url')
                                break
                print(f"[GraphRAG] 메인 키워드 설정: '{main_keyword}' (URL: {main_keyword_url[:50] if main_keyword_url else 'None'}...)")
            
            neighbors = self.graph_db.query_neighbors(name, depth=graph_depth)
            
            if neighbors.get('entities') or neighbors.get('relations'):
                
                graph_context.append({
                    'center_entity': name,
                    'neighbors': neighbors
                })
                print(f"[GraphRAG] '{name}' → 이웃: {len(neighbors.get('entities', []))}개, 관계: {len(neighbors.get('relations', []))}개")
            
            # 관계된 엔티티 추가
            for ent in neighbors.get('entities', [])[:k_entities]:
                neighbor_name = ent.get('name')
                if neighbor_name and neighbor_name not in seen_entities:
                    seen_entities.add(neighbor_name)
                    all_entities.append({
                        'name': neighbor_name,
                        'type': ent.get('type', 'unknown'),
                        'sources': ent.get('sources', [])
                    })
                    for src in ent.get('sources', []):
                        if isinstance(src, dict) and src.get('doc'):
                            all_sources.add(src.get('doc'))
            
            # 관계 추가
            for rel in neighbors.get('relations', []):
                triple = rel.get('triple', {})
                if triple:
                    all_relations.append({
                        'subject': triple.get('subject'),
                        'predicate': triple.get('predicate'),
                        'object': triple.get('object'),
                        'source': rel.get('source', {}).get('doc') if isinstance(rel.get('source'), dict) else rel.get('source')
                    })
                    if isinstance(rel.get('source'), dict) and rel.get('source', {}).get('doc'):
                        all_sources.add(rel.get('source', {}).get('doc'))
        
        # 검색 결과 요약
        print(f"\n[GraphRAG] 검색 완료:")
        print(f"  - 메인 키워드: {main_keyword}")
        print(f"  - 엔티티: {len(all_entities)}개")
        print(f"  - 관계: {len(all_relations)}개")
        print(f"  - 출처: {len(all_sources)}개")
        if all_entities:
            print(f"  - 엔티티 목록: {[e['name'] for e in all_entities[:5]]}")
        if all_relations:
            print(f"  - 관계 예시: {all_relations[0] if all_relations else 'None'}")
        
        return {
            'query': query,
            'main_keyword': main_keyword,  # 메인 키워드 (FAISS 결과 중 GraphDB에 존재하는 첫 번째)
            'main_keyword_url': main_keyword_url,  # 한국민족문화대백과사전 URL
            'entities': all_entities[:k_entities],
            'relations': all_relations[:10],
            'graph_context': graph_context,
            'sources': list(all_sources)
        }
    
    def hybrid_search(
        self,
        query: str,
        k_docs: int = 3,
        k_entities: int = 3,
        graph_depth: int = 1
    ) -> Dict:
        """하이브리드 검색 (벡터 + 그래프)
        
        Args:
            query: 검색 쿼리
            k_docs: 반환할 문서 수
            k_entities: 반환할 엔티티 수
            graph_depth: 그래프 탐색 깊이
            
        Returns:
            검색 결과 (문서, 엔티티, 그래프 정보)
        """
        print(f"\n하이브리드 검색: '{query}'")
        
        # 1. 문서 벡터 검색
        documents = self.search_documents(query, k=k_docs)
        
        # 2. 엔티티 KNN 검색
        entities = self.search_entities_knn(query, k=k_entities)
        
        # 3. 그래프 키워드/문서 확장
        graph_documents = []
        graph_terms, graph_context = self._collect_graph_terms(entities, graph_depth)
        if graph_terms:
            graph_documents = self._graph_expanded_document_search(query, graph_terms, k_docs)
            documents = self._blend_documents(documents, graph_documents)
        
        return {
            'query': query,
            'documents': documents,
            'entities': entities,
            'graph_context': graph_context,
            'graph_terms': graph_terms,
            'graph_documents': graph_documents
        }
    
    def generate_answer(
        self,
        query: str,
        use_graph: bool = True
    ) -> str:
        """질문에 대한 답변 생성 (한국민족문화대백과사전 우선)
        
        Args:
            query: 질문
            use_graph: 그래프 정보 사용 여부 (항상 True로 동작)
            
        Returns:
            답변 텍스트
            
        우선순위:
            1. 한국민족문화대백과사전 정보 (definition, summary)
            2. 관련 사료 정보 (선조실록, 난중일기 등)
            3. 정보 부족 시 FAISS 벡터 검색 보완
        """
        if not self.llm:
            return "LLM이 초기화되지 않았습니다."
        
        # 그래프 DB 기반 검색
        results = self.graph_only_search(query, k_entities=10, graph_depth=1)
        
        # 컨텍스트 구성 (우선순위별로 분리)
        encyclopedia_parts = []  # 한국민족문화대백과사전
        historical_parts = []    # 역사 사료 (선조실록, 난중일기 등)
        relation_parts = []      # 지식 그래프 관계
        
        # 1순위: 한국민족문화대백과사전 정보 추출
        for entity in results.get('entities', []):
            entity_name = entity.get('name', '')
            entity_type = entity.get('type', '')
            definition = entity.get('definition', '')
            summary = entity.get('summary', '')
            sources = entity.get('sources', [])
            
            # 백과사전 정보 확인
            has_encyclopedia = False
            encyclopedia_snippet = ''
            historical_snippets = []
            
            for src in sources if isinstance(sources, list) else []:
                if not isinstance(src, dict):
                    continue
                src_type = src.get('type', '')
                src_doc = src.get('doc', '')
                snippet = src.get('snippet', '')
                
                if '한국민족문화대백과사전' in src_type:
                    has_encyclopedia = True
                    encyclopedia_snippet = snippet or definition or summary
                else:
                    # 일반 사료
                    if snippet:
                        # 출처명 정리
                        doc_name = src_doc
                        for ext in ['.pdf', '.json', '.txt']:
                            doc_name = doc_name.replace(ext, '')
                        if '_' in doc_name:
                            doc_name = doc_name.replace('_', ' ')
                        historical_snippets.append((doc_name, snippet))
            
            # 백과사전 정보 추가 (1순위)
            if has_encyclopedia and encyclopedia_snippet:
                info = f"### {entity_name} ({entity_type})\n"
                info += f"{encyclopedia_snippet[:500]}"
                if definition and definition not in encyclopedia_snippet:
                    info += f"\n정의: {definition[:300]}"
                encyclopedia_parts.append(info)
        
            # 사료 정보 추가 (2순위)
            for doc_name, snippet in historical_snippets[:3]:
                info = f"- [{doc_name}] {entity_name}: {snippet[:200]}"
                historical_parts.append(info)
        
        # 지식 그래프 관계 (3순위)
        for rel in results.get('relations', [])[:10]:
            subject = rel.get('subject', '')
            predicate = rel.get('predicate', '')
            obj = rel.get('object', '')
            if subject and predicate and obj:
                relation_parts.append(f"- {subject} --[{predicate}]--> {obj}")
        
        # 컨텍스트 조합
        context_parts = []
        
        if encyclopedia_parts:
            context_parts.append("## 한국민족문화대백과사전 정보 (신뢰도 높음):")
            context_parts.extend(encyclopedia_parts[:5])
        
        if historical_parts:
            context_parts.append("\n\n## 관련 역사 사료:")
            context_parts.extend(historical_parts[:10])
        
        if relation_parts:
            context_parts.append("\n\n## 지식 그래프 관계:")
            context_parts.extend(relation_parts[:10])
        
        # 정보가 부족하면 FAISS 벡터 검색으로 보완
        faiss_sources = []  # FAISS에서 찾은 출처 저장
        if len(encyclopedia_parts) < 2 and len(historical_parts) < 3:
            if self.vectorstore:
                print("[답변 생성] 정보 부족 - FAISS 벡터 검색으로 보완")
                try:
                    faiss_docs = self.vectorstore.similarity_search(query, k=3)
                    if faiss_docs:
                        context_parts.append("\n\n## 추가 참고 문서 (벡터 검색):")
                        for doc in faiss_docs:
                            content = doc.page_content[:300]
                            source = doc.metadata.get('source', '')
                            if source:
                                source_name = source.split('/')[-1].split('\\')[-1]
                                for ext in ['.pdf', '.json', '.txt']:
                                    source_name = source_name.replace(ext, '')
                                context_parts.append(f"- [{source_name}] {content}")
                                # FAISS 출처 저장 (나중에 참고 문서에 추가)
                                if source_name:
                                    faiss_sources.append(source_name)
                            else:
                                context_parts.append(f"- {content}")
                except Exception as e:
                    print(f"FAISS 검색 오류: {e}")
        
        context = "\n".join(context_parts)
        
        # 프롬프트 생성
        prompt = f"""당신은 한국 역사 전문가입니다.
주어진 정보를 바탕으로 질문에 정확하게 답변하세요.

**중요**: 한국민족문화대백과사전 정보를 최우선으로 참고하고, 
역사 사료의 내용을 보충 설명에 활용하세요.

{context}

질문: {query}

답변 작성 가이드:
1. 한국민족문화대백과사전의 정의와 설명을 기반으로 핵심 내용 작성
2. 역사 사료의 구체적인 기록을 인용하여 보충
3. 역사적 사실을 정확하게 기술
4. 5-7문장으로 상세하게 설명

답변:"""
        
        print(f"\n{self.llm.model} LLM으로 답변 생성 중...")
        
        try:
            response = self.llm.invoke(prompt)
            
            # 답변에서 마크다운 헤더 제거 (## 제목 등)
            import re
            cleaned_response = response.strip()
            # 첫 줄이 마크다운 헤더면 제거
            cleaned_response = re.sub(r'^#{1,6}\s+.+?\n+', '', cleaned_response)
            # 중간에 있는 마크다운 헤더도 제거
            cleaned_response = re.sub(r'\n#{1,6}\s+.+?\n', '\n', cleaned_response)
            
            # 답변 구성
            answer_parts = [cleaned_response.strip()]
            
            # 메인 키워드 (FAISS 검색 결과 중 GraphDB에 존재하는 첫 번째)
            main_keyword = results.get('main_keyword', '')
            main_keyword_url = results.get('main_keyword_url', '')
            
            # 관련 출처 수집 (엔티티의 sources에서)
            doc_sources = []  # [(출처명, url), ...]
            seen_sources = set()
            
            # 1. 한국민족문화대백과사전 (메인 키워드 URL로 하이퍼링크)
            if main_keyword_url:
                doc_sources.append(('한국민족문화대백과사전', main_keyword_url))
                seen_sources.add('한국민족문화대백과사전')
            
            # 2. 다른 사료들 수집 (난중일기, 선조실록 등)
            for entity in results.get('entities', []):
                sources = entity.get('sources', [])
                for src in sources if isinstance(sources, list) else []:
                    if not isinstance(src, dict):
                        continue
                    
                    src_doc = src.get('doc', '')
                    src_type = src.get('type', '')
                    
                    # 한국민족문화대백과사전은 이미 추가됨
                    if src_doc == '한국민족문화대백과사전' or '한국민족문화대백과사전' in str(src_type):
                        continue
                    
                    # 출처명 정리
                    source_name = src_doc or src_type or ''
                    for ext in ['.pdf', '.json', '.txt', '.md', '.docx']:
                        source_name = source_name.replace(ext, '')
                    if '_' in source_name:
                        source_name = source_name.replace('_', ' ')
                    
                    if source_name and source_name not in seen_sources:
                        seen_sources.add(source_name)
                        doc_sources.append((source_name, ''))  # 다른 사료는 URL 없음
            
            # 3. results.sources에서도 추가 (fallback)
            for source in results.get('sources', []):
                if source:
                    source_name = source.split('/')[-1].split('\\')[-1]
                    for ext in ['.pdf', '.json', '.txt', '.md', '.docx']:
                        source_name = source_name.replace(ext, '')
                    if '_' in source_name:
                        source_name = source_name.replace('_', ' ')
                    if source_name and source_name not in seen_sources:
                        seen_sources.add(source_name)
                        doc_sources.append((source_name, ''))
            
            # 4. FAISS 검색에서 찾은 출처 추가
            for faiss_source in faiss_sources:
                # 출처명 정리
                source_name = faiss_source
                for ext in ['.pdf', '.json', '.txt', '.md', '.docx']:
                    source_name = source_name.replace(ext, '')
                if '_' in source_name:
                    source_name = source_name.replace('_', ' ')  # _ → 띄어쓰기
                if source_name and source_name not in seen_sources:
                    seen_sources.add(source_name)
                    doc_sources.append((source_name, ''))
            
            # 참고 문서 포맷팅
            if doc_sources:
                ref_parts = []
                for name, url in doc_sources[:10]:  # 최대 10개
                    if url:
                        ref_parts.append(f"[{name}]({url})")
                    else:
                        ref_parts.append(name)
                answer_parts.append("\n\n참고 문서: " + ", ".join(ref_parts))
            
            return "\n".join(answer_parts)
            
        except Exception as e:
            print(f"답변 생성 오류: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {e}"
    
    def save_indexes(self, save_dir: str = 'graphrag_index'):
        """인덱스 저장
        
        Args:
            save_dir: 저장 디렉토리
        """
        os.makedirs(save_dir, exist_ok=True)
        saved_count = 0
        
        # 문서 벡터 스토어 저장
        if self.vectorstore:
            doc_path = os.path.join(save_dir, 'documents')
            self.vectorstore.save_local(doc_path)
            print(f"  ✓ 문서 인덱스: {doc_path}/")
            saved_count += 1
        
        # 엔티티 벡터 스토어 저장
        if self.entity_vectorstore:
            entity_path = os.path.join(save_dir, 'entities')
            self.entity_vectorstore.save_local(entity_path)
            print(f"  ✓ 엔티티 인덱스: {entity_path}/")
            saved_count += 1
        
        # 지식 그래프 저장 (JSON 백업)
        if self.graph_db and self.graph_db.db:
            graph_path = os.path.join(save_dir, 'knowledge_graph.json')
            success = self.graph_db.export_graph(graph_path)
            if success:
                print(f"  ✓ 지식 그래프: {graph_path}")
                saved_count += 1
        
        print(f"\n총 {saved_count}개 인덱스 저장 완료")
    
    def load_indexes(self, load_dir: str = 'graphrag_index'):
        """인덱스 로드 (벡터 + 지식 그래프)
        
        Args:
            load_dir: 로드 디렉토리
        """
        loaded_count = 0
        
        # 문서 벡터 스토어 로드
        doc_index_file = os.path.join(load_dir, 'documents', 'index.faiss')
        if os.path.exists(doc_index_file):
            doc_path = os.path.join(load_dir, 'documents')
            self.vectorstore = FAISS.load_local(
                doc_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"  ✓ 문서 인덱스 로드 완료")
            loaded_count += 1
        else:
            print(f"  ✗ 문서 인덱스 없음")
        
        # 엔티티 벡터 스토어 로드
        entity_index_file = os.path.join(load_dir, 'entities', 'index.faiss')
        if os.path.exists(entity_index_file):
            entity_path = os.path.join(load_dir, 'entities')
            self.entity_vectorstore = FAISS.load_local(
                entity_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"  ✓ 엔티티 인덱스 로드 완료")
            loaded_count += 1
        else:
            print(f"  ✗ 엔티티 인덱스 없음")
        
        # 지식 그래프 로드 (JSON → ArangoDB)
        graph_path = os.path.join(load_dir, 'knowledge_graph.json')
        if os.path.exists(graph_path) and self.graph_db and self.graph_db.db:
            # ArangoDB에 데이터 로드
            success = self.graph_db.import_graph(graph_path)
            if success:
                print(f"  ✓ 지식 그래프 로드 완료 (ArangoDB)")
                loaded_count += 1
            else:
                print(f"  ✗ 지식 그래프 로드 실패")
        else:
            if not os.path.exists(graph_path):
                print(f"  ✗ 지식 그래프 파일 없음")
            elif not self.graph_db or not self.graph_db.db:
                print(f"  ✗ ArangoDB 연결 없음 (지식 그래프 건너뜀)")
        
        if loaded_count == 0:
            raise Exception("로드할 인덱스 파일이 없습니다")


def _copy_encykorea_to_global(encykorea_db: str, global_db: str):
    """한국민족문화대백과사전 DB를 통합 그래프 DB로 복사
    
    Args:
        encykorea_db: 백과사전 DB 이름 (kg_encykorea)
        global_db: 통합 그래프 DB 이름 (knowledge_graph)
    """
    from arango import ArangoClient
    
    client = ArangoClient(hosts='http://localhost:8530')
    sys_db = client.db('_system', username='root', password='')
    
    # 백과사전 DB 존재 확인
    if not sys_db.has_database(encykorea_db):
        raise Exception(f"백과사전 DB '{encykorea_db}'가 존재하지 않습니다.")
    
    # 통합 DB 초기화 (삭제 후 재생성)
    if sys_db.has_database(global_db):
        sys_db.delete_database(global_db)
        print(f"  - 기존 통합 DB '{global_db}' 삭제")
    
    sys_db.create_database(global_db)
    print(f"  - 통합 DB '{global_db}' 생성")
    
    # 백과사전 DB 연결
    ency_db = client.db(encykorea_db, username='root', password='')
    global_db_conn = client.db(global_db, username='root', password='')
    
    # 컬렉션 복사
    for col_name in ['entities', 'relations']:
        if ency_db.has_collection(col_name):
            # 통합 DB에 컬렉션 생성
            is_edge = (col_name == 'relations')
            if is_edge:
                global_db_conn.create_collection(col_name, edge=True)
            else:
                global_db_conn.create_collection(col_name)
            
            # 데이터 복사
            src_col = ency_db.collection(col_name)
            dst_col = global_db_conn.collection(col_name)
            
            # 배치로 복사 (성능 최적화)
            batch_size = 1000
            cursor = src_col.all()
            batch = []
            total_copied = 0
            
            for doc in cursor:
                # _id, _rev 제거 (새 DB에서 자동 생성)
                doc_copy = {k: v for k, v in doc.items() if k not in ['_id', '_rev']}
                batch.append(doc_copy)
                
                if len(batch) >= batch_size:
                    try:
                        dst_col.insert_many(batch)
                        total_copied += len(batch)
                    except Exception as e:
                        # 개별 삽입 시도
                        for d in batch:
                            try:
                                dst_col.insert(d)
                                total_copied += 1
                            except:
                                pass
                    batch = []
            
            # 남은 배치 처리
            if batch:
                try:
                    dst_col.insert_many(batch)
                    total_copied += len(batch)
                except Exception as e:
                    for d in batch:
                        try:
                            dst_col.insert(d)
                            total_copied += 1
                        except:
                            pass
            
            print(f"  - {col_name}: {total_copied}개 복사")


def main():
    """GraphRAG 시스템 실행"""
    print("GraphRAG 시스템 - 지식 그래프 + 벡터 검색")
    BASE_DATA_DIR = 'graphrag_data'
    OUTPUT_DIR = 'output'
    
    print(f"\n입력 문서 루트: {OUTPUT_DIR}/")
    print(f"인덱스 저장 루트: {BASE_DATA_DIR}/")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"입력 디렉토리가 없습니다: {OUTPUT_DIR}")
        return
    
    def slugify(name: str) -> str:
        if not name:
            return 'source'
        ascii_only = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        ascii_only = re.sub(r'_+', '_', ascii_only).strip('_').lower()
        return ascii_only or 'source'
    
    # 사료명 → 영문 슬러그 매핑 (필요 시 확장)
    custom_slugs = {
        '징비록': 'jingbirok',
        '조선왕조실록': 'joseon',
        '재조번방지': 'jaejo',
        '연려실기술': 'yeollyeo',
        '난중잡록': 'japrok',
        '기재사초': 'gijae',
        '고대일록': 'godae',
        '간양록': 'ganyang',
        '난중일기': 'najung',
        '한국민족문화대백과사전': 'encykorea'
    }
    # 한국민족문화대백과사전은 이미 구축됨 (kg_encykorea) - 초기화 금지
    # 통합 그래프(knowledge_graph)는 kg_encykorea를 먼저 복사한 후 다른 소스 추가
    ENCYKOREA_DB = 'kg_encykorea'  # 백과사전 DB (절대 초기화 안 됨)
    
    TARGET_SOURCES = [
        # '한국민족문화대백과사전',  # 이미 구축됨 - 제외
        '연려실기술',
        '고대일록',
        # '난중잡록'
    ]
    # ============================================================

    # 하위 사료 디렉토리 탐색 (TARGET_SOURCES에 지정된 것만)
    candidate_dirs = []
    for source_name in TARGET_SOURCES:
        path = os.path.join(OUTPUT_DIR, source_name)
        if os.path.isdir(path):
            candidate_dirs.append(path)
            print(f"  ✓ 처리 대상: {source_name}")
        else:
            print(f"  ⚠ 폴더 없음: {source_name} ({path})")
    
    if not candidate_dirs:
        print(f"\n⚠️  처리할 사료가 없습니다. TARGET_SOURCES를 확인하세요.")
        print(f"   현재 설정: {TARGET_SOURCES}")
        return
    
    global_db_name = os.getenv('GLOBAL_GRAPH_DB', 'knowledge_graph')
    env_global_reset = os.getenv('GLOBAL_GRAPH_RESET')
    if env_global_reset is None:
        global_reset_remaining = True  # 첫 실행은 항상 초기화
    else:
        global_reset_remaining = env_global_reset.lower() in ('1', 'true', 'yes')
    
    # 통합 그래프 초기화 시 한국민족문화대백과사전 데이터 먼저 복사
    if global_reset_remaining:
        print(f"\n📚 통합 그래프 초기화: {ENCYKOREA_DB} → {global_db_name}")
        try:
            from graph_db import ArangoGraphDB
            # 백과사전 DB에서 통합 DB로 복사
            _copy_encykorea_to_global(ENCYKOREA_DB, global_db_name)
            global_reset_remaining = False  # 복사 완료 후 초기화 비활성화
            print(f"  ✓ 백과사전 데이터 복사 완료")
        except Exception as e:
            print(f"  ⚠ 백과사전 데이터 복사 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def build_source_db_name(slug: str) -> str:
        env_db = os.getenv('SOURCE_GRAPH_DB')
        if env_db:
            return env_db
        prefix = os.getenv('SOURCE_GRAPH_PREFIX', 'kg')
        return f"{prefix}_{slug}"
    
    last_graphrag = None
    processed_sources = 0
    
    for collection_path in candidate_dirs:
        collection_name = os.path.basename(collection_path.rstrip(os.sep))
        collection_slug = custom_slugs.get(collection_name)
        if not collection_slug:
            slug_candidate = slugify(collection_name)
            if slug_candidate == 'source':
                slug_candidate = f"source_{abs(hash(collection_name)) & 0xffffffff:08x}"
            collection_slug = slug_candidate
        source_db_name = build_source_db_name(collection_slug)
        data_dir = os.path.join(BASE_DATA_DIR, collection_slug)
        os.makedirs(data_dir, exist_ok=True)
        
        print("\n" + "=" * 80)
        print(f"사료 처리: {collection_name}")
        print(f"  - 원본 경로: {collection_path}")
        print(f"  - 개별 DB: {source_db_name}")
        print(f"  - 인덱스 경로: {data_dir}/")
        print(f"  - 통합 DB: {global_db_name} (reset={global_reset_remaining})")
        
        documents = load_documents_from_output(collection_path)
        if not documents:
            print("  ⚠️  문서를 찾을 수 없어 건너뜁니다.")
            continue
        
        graphrag = GraphRAGSystem(
            embedding_model_name='intfloat/multilingual-e5-large-instruct',
            llm_model_name='gemma3:12b',
            arango_host='localhost',
            arango_port=8530,  # 새 ArangoDB 포트 (기존 8529는 유지)
            arango_password='',
            arango_db_name=source_db_name,
            arango_reset=True,
            global_arango_db_name=global_db_name,
            global_arango_reset=global_reset_remaining,
            use_reranker=True,
            use_tika=False
        )
        global_reset_remaining = False
        last_graphrag = graphrag
        processed_sources += 1
        
        doc_index_path = os.path.join(data_dir, 'documents', 'index.faiss')
        entity_index_path = os.path.join(data_dir, 'entities', 'index.faiss')
        graph_json_path = os.path.join(data_dir, 'knowledge_graph.json')
        
        has_doc_index = os.path.exists(doc_index_path)
        has_entity_index = os.path.exists(entity_index_path)
        has_graph_json = os.path.exists(graph_json_path)
        
        # 강제 재구축 여부 (환경변수 또는 명령줄 인자로 설정 가능)
        force_rebuild = os.getenv('FORCE_REBUILD', 'false').lower() in ('1', 'true', 'yes')
        
        if not force_rebuild and (has_doc_index or has_entity_index or has_graph_json):
            print(f"\n✓ 기존 인덱스 발견: {data_dir}")
            if has_doc_index:
                print("  - 문서 인덱스: ✓")
            if has_entity_index:
                print("  - 엔티티 인덱스: ✓")
            if has_graph_json:
                print("  - 지식 그래프 백업: ✓")
            
            print("\n기존 인덱스 로드 시도...")
            try:
                graphrag.load_indexes(data_dir)
                print("✓ 인덱스 로드 성공 (재구축 생략)")
                sample_docs = []
            except Exception as e:
                print(f"✗ 인덱스 로드 실패: {e}")
                print("새로 구축합니다...")
                sample_docs = documents  # 전체 문서
        else:
            if force_rebuild:
                print(f"\n⚠ 강제 재구축 모드 (FORCE_REBUILD=true)")
            print("\n기존 인덱스가 없습니다. 새로 구축합니다.")
            sample_docs = documents  # 전체 문서
        
        if sample_docs:
            print("\n인덱스 구축 시작")
            print(f"📄 총 {len(sample_docs)}개 문서 처리")
            
            # 한국민족문화대백과사전은 구조화된 데이터이므로 벡터 인덱스 불필요
            is_encyclopedia = (
                '한국민족문화대백과사전' in collection_name or
                'encykorea' in collection_name.lower()
            )
            
            graphrag.build_index(
                documents=sample_docs,
                extract_graph=True,
                skip_vector_index=is_encyclopedia
            )
            
            print("\n인덱스 저장 중...")
            graphrag.save_indexes(data_dir)
            print(f"저장 완료: {data_dir}/")
    
    if processed_sources == 0:
        print("\n처리된 사료가 없습니다.")
        return
    
    print("\n모든 사료 처리가 완료되었습니다.")
    if not last_graphrag or not last_graphrag.llm:
        print("\nLLM이 초기화되지 않았거나 마지막 인스턴스가 없어 질의응답을 생략합니다.")
        print("Ollama 설정을 확인하세요.")
        return
    
    print("\n마지막 처리 사료 기준 예시 질의응답")
    questions = [
        "한산도 대첩에 대해 설명해주세요",
        "임진왜란은 언제 시작되었나요?",
        "이순신 장군의 주요 업적은 무엇인가요?"
    ]
    
    for idx, question in enumerate(questions, 1):
        print(f"\n[질문 {idx}/{len(questions)}]")
        print("-" * 70)
        
        answer = last_graphrag.generate_answer(question, use_graph=True)
        print(answer)

if __name__ == "__main__":
    main()

