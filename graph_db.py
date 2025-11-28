import json
import os
from typing import List, Dict, Tuple, Optional, Set
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
import subprocess
import psutil  # pip install psutil

try:
    from langchain_community.llms import Ollama
except ImportError:
    from langchain.llms import Ollama

try:
    from arango import ArangoClient
    ARANGO_AVAILABLE = True
except ImportError:
    print("ArangoDB 클라이언트가 설치되지 않았습니다.")
    print("설치: pip install python-arango")
    ARANGO_AVAILABLE = False


def _load_stopwords_from_file(file_path: str) -> Set[str]:
    """불용어 파일에서 불용어 로드"""
    stopwords = set()
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):  # 빈 줄과 주석 제외
                        stopwords.add(word)
            print(f"  ✓ 불용어 {len(stopwords)}개 로드: {file_path}")
        except Exception as e:
            print(f"  ⚠ 불용어 파일 로드 실패: {file_path} ({e})")
    return stopwords


def _load_all_stopwords() -> Set[str]:
    """모든 불용어 파일 로드 및 기본 불용어와 병합"""
    # 기본 불용어 (코드 내장)
    base_stopwords = {
        # 대명사/지시어
    '그리고', '그러나', '하지만', '또한', '및', '또', '이', '그', '저', '것', '등', '사실',
    '현재', '당시', '이는', '그는', '그녀', '그들의', '우리', '너희', '여기', '저기',
        '무엇', '누가', '어디', '때문', '때로', '동안', '위해', '사용', '관련', '기록', '내용',
        # 인칭대명사
        '나는', '나를', '나의', '내가', '너는', '너를', '너의', '네가', '저는', '저를', '저의',
        '그것', '이것', '저것', '무엇', '어느', '모든', '각', '매', '어떤',
        # 일반 동사/형용사
        '하다', '되다', '있다', '없다', '이다', '아니다', '같다', '다르다',
        '하였다', '되었다', '있었다', '없었다', '하였으며', '있으며', '하고', '하여',
        # 시간 표현 (단독으로 의미 없는 것들)
        '오늘', '내일', '어제', '지금', '그때', '이때', '당시', '이후', '이전',
        # 접속사/부사
        '그래서', '따라서', '그러므로', '왜냐하면', '만약', '비록', '물론', '결국',
        '매우', '아주', '너무', '정말', '진짜', '참', '꽤', '상당히'
    }
    
    # 불용어 파일 경로 (여러 위치에서 찾기)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stopword_files = [
        os.path.join(script_dir, 'crawling', 'stopwords-ko.txt'),
        os.path.join(script_dir, 'crawling', 'stopwords.txt'),
        'crawling/stopwords-ko.txt',
        'crawling/stopwords.txt',
    ]
    
    # 파일에서 불용어 로드
    file_stopwords = set()
    loaded_files = set()
    for file_path in stopword_files:
        if os.path.exists(file_path) and file_path not in loaded_files:
            file_stopwords.update(_load_stopwords_from_file(file_path))
            loaded_files.add(file_path)
    
    # 모든 불용어 병합
    all_stopwords = base_stopwords | file_stopwords
    
    if file_stopwords:
        print(f"  ✓ 총 불용어 수: {len(all_stopwords)}개 (기본 {len(base_stopwords)}개 + 파일 {len(file_stopwords)}개)")
    
    return all_stopwords


# 불용어 세트 초기화 (모듈 로드 시 한 번만 실행)
print("\n[불용어 초기화]")
KOREAN_STOPWORDS = _load_all_stopwords()

ENGLISH_STOPWORDS = {
    'and', 'but', 'however', 'also', 'etc', 'this', 'that', 'these', 'those', 'it',
    'they', 'them', 'him', 'her', 'its', 'their', 'whose', 'which', 'what', 'when', 'where'
}

# 단독으로 사용시 의미 없는 시간/숫자 패턴
TIME_PATTERNS = {
    '1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월',
    '정월', '윤월', '초하루', '그믐', '보름',
    '월', '일', '시', '분', '초',
    '하루', '이틀', '사흘', '나흘', '닷새', '엿새', '이레', '여드레', '아흐레', '열흘'
}


class KnowledgeGraphExtractor:
    """Ollama를 사용한 지식 트리플 추출기"""
    
    def __init__(self, llm_model='gemma3:12b'):
        """초기화
        
        Args:
            llm_model: Ollama 모델 이름
        """
        print(f"\n지식 그래프 추출기 초기화 중... (모델: {llm_model})")
        
        try:
            self.llm = Ollama(
                model=llm_model,
                temperature=0.3,  # 더 결정적인 출력을 위해 낮은 temperature
            )
            print("Ollama LLM 초기화 성공")
        except Exception as e:
            print(f"Ollama LLM 초기화 실패: {e}")
            print("'ollama serve' 실행 및 모델 다운로드 확인 필요")
            self.llm = None
    
    def _check_system_resources(self) -> Dict:
        """시스템 리소스 확인 (GPU, RAM, Swap)"""
        resources = {}
        
        try:
            # 1. GPU 메모리 (docker exec로 nvidia-smi 실행)
            result = subprocess.run(
                ['docker', 'exec', 'history-ollama', 'nvidia-smi', 
                 '--query-gpu=memory.used,memory.total,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(',')
                if len(gpu_info) >= 3:
                    resources['gpu_mem_used'] = int(gpu_info[0].strip())
                    resources['gpu_mem_total'] = int(gpu_info[1].strip())
                    resources['gpu_util'] = int(gpu_info[2].strip())
        except Exception:
            pass
        
        # 2. 시스템 메모리
        try:
            mem = psutil.virtual_memory()
            resources['ram_used_gb'] = mem.used / (1024**3)
            resources['ram_total_gb'] = mem.total / (1024**3)
            resources['ram_percent'] = mem.percent
            
            # Swap
            swap = psutil.swap_memory()
            resources['swap_used_gb'] = swap.used / (1024**3)
            resources['swap_percent'] = swap.percent
        except Exception:
            pass
        
        return resources
    
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """텍스트에서 지식 트리플(주어-술어-목적어) 추출
        
        Args:
            text: 입력 텍스트
            
        Returns:
            (주어, 술어, 목적어) 튜플 리스트
        """
        if not self.llm:
            print("LLM이 초기화되지 않았습니다.")
            return []
        
        prompt = f"""다음 한국 역사 텍스트에서 의미 있는 지식 트리플(주어, 술어, 목적어)을 추출하세요.

텍스트:
{text}

지식 트리플을 다음 형식으로 추출하세요:
주어 | 술어 | 목적어

좋은 예시:
이순신 | 직책 | 조선 수군 통제사
이순신 | 참여 | 한산도 대첩
임진왜란 | 시작 연도 | 1592년
선조 | 즉위 연도 | 1567년

나쁜 예시 (추출하지 마세요):
1 | 있다 | 2
10 | 이다 | 15
그 | 하다 | 것

규칙:
1. 의미 있는 엔티티만 추출 (인물, 장소, 사건, 연도 등)
2. 순수 숫자만 있는 엔티티는 제외 (예: "1", "10" ❌)
3. 연도나 날짜는 단위와 함께 (예: "1592년" ✅, "1592" ❌)
4. 한 줄에 하나의 트리플만 작성
5. 각 요소는 ' | '로 구분
6. 명확하고 구체적인 관계만 추출
7. 역사적 사실과 인물, 사건, 장소 간의 관계에 집중

트리플:"""

        try:
            response = self.llm.invoke(prompt)
            
            # 응답 파싱
            triples = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 3:
                        subject, predicate, obj = parts
                        if subject and predicate and obj:
                            triples.append((subject, predicate, obj))
            
            return triples
            
        except Exception as e:
            print(f"트리플 추출 오류: {e}")
            return []
    
    def _process_single_document(self, doc: Dict, idx: int, total: int) -> Tuple[List[Dict], List[Dict], Dict]:
        """단일 문서에서 엔티티와 관계 추출 (병렬 처리용)"""
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        if not content:
            return [], [], {}
        
        # URL 추출 (문서 최상위 또는 metadata에서)
        doc_url = doc.get('url') or metadata.get('url')
        
        # annotations가 있으면 metadata에 추가
        annotations = doc.get('annotations')
        if annotations:
            metadata['annotations'] = annotations
        
        print(f"[{idx}/{total}] 문서 처리 중...")
        
        # 트리플 추출
        triples = self.extract_triples(content[:2000])
        
        local_entities = {}
        local_relations = []
        
        for subject, predicate, obj in triples:
            # 의미 없는 엔티티 필터링
            if not self._is_valid_entity(subject) or not self._is_valid_entity(obj):
                continue
            
            relation_context = f"{subject} {predicate} {obj}"
            relation_source = self._build_source_entry(metadata, content, subject, predicate, obj, url=doc_url)
            
            # 엔티티 추가 (URL 전달)
            if subject not in local_entities:
                local_entities[subject] = self._build_entity_record(
                    subject, metadata, content, predicate, obj, url=doc_url
                )
            else:
                self._add_source_entry(local_entities[subject]['sources'], relation_source)
                self._merge_aliases(
                    local_entities[subject],
                    self._build_entity_record(subject, metadata, content, predicate, obj, url=doc_url)
                )
            
            if obj not in local_entities:
                local_entities[obj] = self._build_entity_record(
                    obj, metadata, content, predicate, subject, subject_flag=False, url=doc_url
                )
            else:
                self._add_source_entry(local_entities[obj]['sources'], relation_source)
                self._merge_aliases(
                    local_entities[obj],
                    self._build_entity_record(obj, metadata, content, predicate, subject, subject_flag=False, url=doc_url)
                )
            
            # 관계 추가
            local_relations.append({
                '_from': f"entities/{self._sanitize_key(subject)}",
                '_to': f"entities/{self._sanitize_key(obj)}",
                'type': predicate,
                'source': relation_source,
                'context': relation_context,  # 트리플 자체
                'triple': {
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj
                }
            })
        
        return list(local_entities.values()), local_relations, local_entities
    
    def _process_document_batch(self, batch_docs: List[Tuple[int, Dict]], total: int) -> List[Tuple[List[Dict], List[Dict], Dict]]:
        """문서 배치를 한 번에 처리 """
        import time
        
        if not batch_docs or not self.llm:
            return []
        
        batch_size = len(batch_docs)
        indices = [idx for idx, _ in batch_docs]
        batch_start_time = time.time()
        
        print(f"\n[배치 {indices[0]}-{indices[-1]}/{total}] 진행률: {indices[-1]/total*100:.1f}%")
        
        # 배치 프롬프트 생성
        print(f"  [1/4] 프롬프트 생성 중...", end='', flush=True)
        prompt_start = time.time()
        
        batch_prompts = []
        for idx, doc in batch_docs:
            content = doc.get('content', '')[:2000]
            batch_prompts.append(f"[문서 {idx}]\n{content}")
        
        combined_text = "\n\n---구분선---\n\n".join(batch_prompts)
        prompt_time = time.time() - prompt_start
        print(f" OK ({prompt_time:.2f}초)")
        
        prompt = f"""다음은 {batch_size}개의 한국 역사 문서입니다. 각 문서에서 의미 있는 지식 트리플(주어, 술어, 목적어)을 추출하세요.

{combined_text}

각 문서별로 지식 트리플을 다음 형식으로 추출하세요:
[문서 번호] 주어 | 술어 | 목적어

좋은 예시:
[문서 1] 이순신 | 직책 | 조선 수군 통제사
[문서 1] 이순신 | 참여 | 한산도 대첩
[문서 2] 임진왜란 | 시작 연도 | 1592년
[문서 2] 선조 | 즉위 연도 | 1567년

나쁜 예시 (추출하지 마세요):
[문서 1] 1 | 있다 | 2
[문서 1] 10 | 이다 | 15
[문서 1] 그 | 하다 | 것

규칙:
1. 의미 있는 엔티티만 추출 (인물, 장소, 사건, 연도 등)
2. 순수 숫자만 있는 엔티티는 제외 (예: "1", "10", "240" ❌)
3. 연도나 날짜는 단위와 함께 (예: "1592년" ✅, "1592" ❌)
4. 각 트리플 앞에 [문서 번호] 필수
5. 한 줄에 하나의 트리플만 작성
6. 각 요소는 ' | '로 구분
7. 명확하고 구체적인 관계만 추출

트리플:"""
        
        try:
            # LLM 호출 전 리소스 확인
            resources_before = self._check_system_resources()
            
            print(f"  [2/4] LLM 처리 중 ({batch_size}개 문서)...", end='', flush=True)
            llm_start = time.time()
            
            response = self.llm.invoke(prompt)
            
            llm_time = time.time() - llm_start
            
            # LLM 호출 후 리소스 확인
            resources_after = self._check_system_resources()
            
            print(f" OK ({llm_time:.2f}초, {llm_time/batch_size:.2f}초/문서)")
            
            # 리소스 정보 출력
            if resources_after:
                status_parts = []
                if 'gpu_mem_used' in resources_after:
                    gpu_mem = resources_after['gpu_mem_used']
                    status_parts.append(f"GPU: {gpu_mem}MB")
                if 'ram_percent' in resources_after:
                    ram_pct = resources_after['ram_percent']
                    status_parts.append(f"RAM: {ram_pct:.0f}%")
                if 'swap_percent' in resources_after and resources_after['swap_percent'] > 1:
                    swap_pct = resources_after['swap_percent']
                    status_parts.append(f"Swap: {swap_pct:.0f}%")
                
                if status_parts:
                    print(f"       리소스: {' | '.join(status_parts)}")
            
            # 배치 응답 파싱
            print(f"  [3/4] 트리플 추출 중...", end='', flush=True)
            parse_start = time.time()
            
            results = []
            doc_triples = {idx: [] for idx, _ in batch_docs}
            
            for line in response.strip().split('\n'):
                line = line.strip()
                if '|' not in line or '[문서' not in line:
                    continue
                
                # [문서 N] 추출
                import re
                doc_match = re.search(r'\[문서\s*(\d+)\]', line)
                if not doc_match:
                    continue
                
                doc_num = int(doc_match.group(1))
                if doc_num not in doc_triples:
                    continue
                
                # 트리플 추출
                triple_part = line.split(']', 1)[1].strip()
                parts = [p.strip() for p in triple_part.split('|')]
                
                if len(parts) == 3:
                    subject, predicate, obj = parts
                    if subject and predicate and obj:
                        doc_triples[doc_num].append((subject, predicate, obj))
            
            parse_time = time.time() - parse_start
            total_triples = sum(len(t) for t in doc_triples.values())
            print(f" OK ({parse_time:.2f}초, 트리플 {total_triples}개)")
            
            # 트리플이 0개면 디버깅 정보 출력
            if total_triples == 0:
                print(f"  [경고] 트리플이 추출되지 않았습니다. LLM 응답 샘플:")
                print(f"  {response[:300]}...")
            
            # 각 문서별로 엔티티/관계 생성
            print(f"  [4/4] 엔티티/관계 생성 중...", end='', flush=True)
            entity_start = time.time()
            
            for idx, doc in batch_docs:
                triples = doc_triples.get(idx, [])
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                # URL 추출 (문서 최상위 또는 metadata에서)
                doc_url = doc.get('url') or metadata.get('url')
                
                # annotations가 있으면 metadata에 추가
                annotations = doc.get('annotations')
                if annotations:
                    metadata['annotations'] = annotations
                
                local_entities = {}
                local_relations = []
                relation_signatures = set()
                
                for subject, predicate, obj in triples:
                    # 의미 없는 엔티티 필터링
                    if not self._is_valid_entity(subject) or not self._is_valid_entity(obj):
                        continue
                    if not predicate or len(predicate.strip()) == 0 or len(predicate) > 60:
                        continue
                    
                    subject_sig = self._entity_signature(subject)
                    object_sig = self._entity_signature(obj)
                    if not subject_sig or not object_sig:
                        continue
                    
                    relation_context = f"{subject} {predicate} {obj}"
                    relation_source = self._build_source_entry(metadata, content, subject, predicate, obj, url=doc_url)
                    
                    # 엔티티 추가 (URL 전달)
                    if subject_sig not in local_entities:
                        local_entities[subject_sig] = self._build_entity_record(
                            subject, metadata, content, predicate, obj, url=doc_url
                        )
                    else:
                        self._add_source_entry(local_entities[subject_sig]['sources'], relation_source)
                        self._merge_aliases(
                            local_entities[subject_sig],
                            self._build_entity_record(subject, metadata, content, predicate, obj, url=doc_url)
                        )
                    
                    if object_sig not in local_entities:
                        local_entities[object_sig] = self._build_entity_record(
                            obj, metadata, content, predicate, subject, subject_flag=False, url=doc_url
                        )
                    else:
                        self._add_source_entry(local_entities[object_sig]['sources'], relation_source)
                        self._merge_aliases(
                            local_entities[object_sig],
                            self._build_entity_record(obj, metadata, content, predicate, subject, subject_flag=False, url=doc_url)
                        )
                    
                    # 관계 추가
                    relation_record = self._build_relation_record(
                        subject, predicate, obj, relation_source, relation_context
                    )
                    if relation_record['_key'] in relation_signatures:
                        continue
                    relation_signatures.add(relation_record['_key'])
                    local_relations.append(relation_record)
                
                results.append((list(local_entities.values()), local_relations, local_entities))
            
            entity_time = time.time() - entity_start
            print(f" OK ({entity_time:.2f}초)")
            
            # 전체 배치 통계
            batch_total_time = time.time() - batch_start_time
            total_entities = sum(len(r[0]) for r in results)
            total_relations = sum(len(r[1]) for r in results)
            
            print(f"   배치 완료: {batch_total_time:.2f}초 (평균 {batch_total_time/batch_size:.2f}초/문서)")
            print(f"   추출 결과: 엔티티 {total_entities}개, 관계 {total_relations}개")
            
            return results
            
        except Exception as e:
            batch_total_time = time.time() - batch_start_time
            print(f"\n   배치 처리 오류: {e} ({batch_total_time:.2f}초)")
            # 실패 시 빈 결과 반환
            return [([], [], {}) for _ in batch_docs]
    
    def extract_entities_and_relations(
        self, 
        documents: List[Dict],
        max_workers: int = 1,  # 병렬 처리 워커 수 (1로 줄임)
        batch_size: int = 4    # 배치 크기 (4로 줄임 - 안정성 우선)
    ) -> Tuple[List[Dict], List[Dict]]:
        """문서들에서 엔티티와 관계 추출 (배치 병렬 처리)
        
        Args:
            documents: 문서 리스트 (각 문서는 'content'와 'metadata' 포함)
            max_workers: 병렬 처리 워커 수 (기본 8개)
            batch_size: 한 번에 처리할 문서 수 (기본 8개)
            
        Returns:
            (엔티티 리스트, 관계 리스트) 튜플
        """
        import time
        overall_start = time.time()
        
        print(f" 지식 그래프 추출 시작")
        print(f"  문서 수: {len(documents)}개")
        print(f"  배치 크기: {batch_size}개 문서/배치")
        print(f"  병렬 워커: {max_workers}개")
        
        entities = {}  # entity_name -> entity_dict
        relations = []
        relation_signatures = set()
        lock = Lock()
        
        # 빈 문서 필터링
        valid_docs = [(idx, doc) for idx, doc in enumerate(documents, 1) if doc.get('content', '')]
        
        # 배치로 그룹화
        batches = []
        for i in range(0, len(valid_docs), batch_size):
            batch = valid_docs[i:i+batch_size]
            batches.append(batch)
        
        print(f"  총 {len(batches)}개 배치 생성")
        print(f"  예상 시간: 약 {len(batches) * 15 / max_workers:.0f}-{len(batches) * 30 / max_workers:.0f}초 ({len(batches) * 15 / max_workers / 60:.1f}-{len(batches) * 30 / max_workers / 60:.1f}분)\n")
        
        completed_batches = 0
        completed_lock = Lock()
        
        # 성능 추적
        batch_times = []
        batch_gpu_mems = []
        batch_ram_percents = []
        baseline_time = None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 각 배치를 병렬로 처리
            futures = {
                executor.submit(self._process_document_batch, batch, len(documents)): i 
                for i, batch in enumerate(batches)
            }
            
            # 결과 수집
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    
                    # 스레드 안전하게 결과 병합
                    with lock:
                        for local_entities_list, local_relations, local_entities_dict in batch_results:
                            # 엔티티 병합
                            for entity_signature, entity_data in local_entities_dict.items():
                                if not entity_signature:
                                    continue
                                if entity_signature not in entities:
                                    entities[entity_signature] = entity_data
                                else:
                                    entities[entity_signature] = self._merge_entity_records(
                                        entities[entity_signature], entity_data
                                    )
                            
                            # 관계 추가 (중복 제거)
                            for relation in local_relations:
                                relation_key = relation.get('_key')
                                if relation_key and relation_key in relation_signatures:
                                    continue
                                if relation_key:
                                    relation_signatures.add(relation_key)
                                relations.append(relation)
                    
                    # 진행 상황 업데이트
                    with completed_lock:
                        completed_batches += 1
                        elapsed = time.time() - overall_start
                        progress = completed_batches / len(batches) * 100
                        avg_time = elapsed / completed_batches
                        remaining = (len(batches) - completed_batches) * avg_time
                        
                        # 성능 데이터 수집
                        batch_time = elapsed / completed_batches if completed_batches > 0 else 0
                        batch_times.append(batch_time)
                        
                        # 리소스 데이터 수집
                        current_resources = self._check_system_resources()
                        if 'gpu_mem_used' in current_resources:
                            batch_gpu_mems.append(current_resources['gpu_mem_used'])
                        if 'ram_percent' in current_resources:
                            batch_ram_percents.append(current_resources['ram_percent'])
                        
                        print(f"\n[진행] {completed_batches}/{len(batches)} 배치 완료 ({progress:.1f}%)")
                        print(f"  경과: {elapsed:.0f}초 | 평균: {avg_time:.1f}초/배치 | 남은 시간: 약 {remaining:.0f}초 ({remaining/60:.1f}분)")
                        print(f"  누적: 엔티티 {len(entities)}개, 관계 {len(relations)}개")
                        
                        # 초기 baseline 설정 (처음 3개 배치)
                        if completed_batches == 3:
                            baseline_time = sum(batch_times[:3]) / 3
                        
                        # 성능 경고 (5개 배치 이상 처리 후)
                        if completed_batches >= 5 and baseline_time:
                            recent_avg = sum(batch_times[-3:]) / min(3, len(batch_times))
                            
                            # 속도 저하 감지 (baseline 대비 50% 이상 느려짐)
                            if recent_avg > baseline_time * 1.5:
                                slowdown_ratio = recent_avg / baseline_time
                                print(f"  [경고] 속도 저하 감지 ({slowdown_ratio:.1f}x 느림)")
                                
                                # 원인 분석
                                warnings = []
                                if len(batch_gpu_mems) >= 3:
                                    recent_gpu = sum(batch_gpu_mems[-3:]) / 3
                                    if recent_gpu > 25000:  # 25GB 이상
                                        warnings.append("GPU VRAM 부족 (25GB+ 사용)")
                                
                                if len(batch_ram_percents) >= 3:
                                    recent_ram = sum(batch_ram_percents[-3:]) / 3
                                    if recent_ram > 85:
                                        warnings.append(f"RAM 부족 ({recent_ram:.0f}% 사용)")
                                
                                if current_resources.get('swap_percent', 0) > 10:
                                    warnings.append(f"메모리 스와핑 발생 ({current_resources['swap_percent']:.0f}%)")
                                
                                if warnings:
                                    print(f"  원인: {' / '.join(warnings)}")
                                    print(f"  해결: batch_size 줄이기 또는 max_workers 줄이기")
                        
                except Exception as e:
                    print(f"\n[오류] 배치 처리 실패: {e}")
        
        overall_time = time.time() - overall_start
        
        print(f"\n[완료] 지식 그래프 추출 완료!")
        print(f"  총 소요 시간: {overall_time:.1f}초 ({overall_time/60:.1f}분)")
        print(f"  최종 결과:")
        print(f"    - 엔티티: {len(entities)}개")
        print(f"    - 관계: {len(relations)}개")
        print(f"    - 평균 처리 속도: {len(documents)/overall_time:.1f}개 문서/초")
        
        # 성능 분석
        if len(batch_times) > 5:
            first_3_avg = sum(batch_times[:3]) / 3
            last_3_avg = sum(batch_times[-3:]) / 3
            
            print(f"\n  [성능 분석]")
            print(f"    초기 평균 속도: {first_3_avg:.1f}초/배치")
            print(f"    최종 평균 속도: {last_3_avg:.1f}초/배치")
            
            if last_3_avg > first_3_avg * 1.3:
                slowdown = (last_3_avg / first_3_avg - 1) * 100
                print(f"    [경고] 후반부 {slowdown:.0f}% 속도 저하 발생")
                print(f"    원인: GPU VRAM 부족, KV Cache 조각화, 또는 메모리 스와핑")
                print(f"    다음 실행 시: batch_size={batch_size//2} 또는 max_workers=1 권장")
            else:
                print(f"    [양호] 안정적인 처리 속도 유지")
        
        print()
        
        return list(entities.values()), relations
    
    def _is_stopword(self, entity: str) -> bool:
        token = entity.strip().lower()
        return token in KOREAN_STOPWORDS or token in ENGLISH_STOPWORDS
    
    def _entity_signature(self, entity: str) -> Optional[str]:
        if not entity:
            return None
        normalized = self._normalize_entity_name(entity)
        if not normalized:
            return None
        return normalized.lower()
    
    def _is_valid_entity(self, entity: str) -> bool:
        """엔티티 유효성 검증
        
        다음을 필터링:
        - 너무 짧음 (1-2글자)
        - 순수 숫자 (단위 없는 숫자)
        - 의미 없는 문자
        - 단독 시간/날짜 표현 (9월, 하루 등)
        - 일반적인 불용어
        """
        if not entity or not isinstance(entity, str):
            return False
        
        entity = entity.strip()
        
        # 불용어 체크
        if self._is_stopword(entity):
            return False
        
        # 시간/날짜 패턴 체크 (단독 사용시 의미 없음)
        if entity in TIME_PATTERNS:
            return False
        
        # 1. 길이 체크 (최소 2글자, 한글은 1글자도 OK)
        if len(entity) < 2:
            # 한글 1글자는 허용 (예: 왕, 군)
            if not ('가' <= entity <= '힣'):
                return False
        
        # 2. 순수 숫자만 있으면 제외 (연도, 날짜는 단위와 함께 있어야 함)
        if entity.isdigit():
            return False
        
        # 3. 숫자+월/일 패턴 (예: "5월", "15일") - 연도 없이 단독 사용시 제외
        if re.match(r'^\d{1,2}[월일]$', entity):
            return False
        
        # 4. 특수문자만 있으면 제외
        if re.match(r'^[^a-zA-Z0-9가-힣]+$', entity):
            return False
        
        # 5. 너무 긴 엔티티 제외 (100자 이상)
        if len(entity) > 100:
            return False
        
        # 6. 불용어 및 일반명사 필터
        if len(entity) <= 2 and not ('가' <= entity <= '힣'):
            return False
        
        # 7. 조사가 붙은 대명사 패턴 제외
        pronoun_patterns = [
            r'^[나너저그이]가$', r'^[나너저그이]는$', r'^[나너저그이]를$', r'^[나너저그이]의$',
            r'^[나너저그이]에게$', r'^그것[이을를의]?$', r'^이것[이을를의]?$', r'^저것[이을를의]?$'
        ]
        for pattern in pronoun_patterns:
            if re.match(pattern, entity):
                return False
        
        return True
    
    def _normalize_entity_name(self, entity: str) -> str:
        if not entity:
            return 'unknown'
        normalized = entity.strip()
        suffixes = [
            ' 장군', ' 대첩', ' 전투', ' 장군님', ' 장상', ' 공', ' 대사', ' 대왕', ' 대감',
            ' 장군과', ' 사건', ' 전쟁', ' 해전', ' 축'
        ]
        for suffix in suffixes:
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
                normalized = normalized[:-len(suffix)].strip()
        return normalized or entity.strip()
    
    def _classify_entity(self, entity: str) -> str:
        if not entity:
            return 'entity'
        text = entity.strip()
        if re.search(r'\d{2,4}\s*년', text) or any(token in text for token in ['월', '일']):
            return 'date'
        dynasty_keywords = ['조선', '고려', '고구려', '백제', '신라', '발해', '대한제국']
        if any(token in text for token in dynasty_keywords):
            return 'dynasty'
        if any(token in text for token in ['대첩', '전투', '전쟁', '사건', '해전', '봉기']):
            return 'event'
        if any(token in text for token in ['성', '포', '도', '섬', '산', '강', '현', '군']) and len(text) <= 6:
            return 'location'
        if any(text.endswith(suffix) for suffix in ['장군', '공', '대감', '장관', '후', '왕', '황제']):
            return 'person'
        if any(token in text for token in ['수군', '군사', '부대', '관청', '정부', '사령부']):
            return 'organization'
        return 'entity'
    
    def _extract_snippet(self, content: str, target: Optional[str] = None, window: int = 200, include_surrounding_sentences: bool = True) -> str:
        """문장 단위로 snippet 추출 (앞뒤 문장 포함)
        
        Args:
            content: 전체 본문
            target: 찾을 대상 텍스트
            window: 최대 글자 수 (기본 200)
            include_surrounding_sentences: 앞뒤 문장 포함 여부
            
        Returns:
            target을 포함하는 문장 + 앞뒤 문장
        """
        if not content:
            return ''
        
        # 문장 분리 (마침표, 물음표, 느낌표, 【】 기준)
        # 한문 사료의 경우 【】로 주석이 구분됨
        sentence_pattern = r'(?<=[.!?。])\s+|(?<=】)\s*|(?=【)'
        sentences = re.split(sentence_pattern, content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return content[:window].strip()
        
        if target:
            # target이 포함된 문장 찾기
            target_idx = -1
            for i, sentence in enumerate(sentences):
                if target in sentence:
                    target_idx = i
                    break
            
            if target_idx != -1:
                if include_surrounding_sentences:
                    # 앞뒤 문장 포함 (최대 3문장)
                    start_idx = max(0, target_idx - 1)
                    end_idx = min(len(sentences), target_idx + 2)
                    snippet_sentences = sentences[start_idx:end_idx]
                else:
                    snippet_sentences = [sentences[target_idx]]
                
                snippet = ' '.join(snippet_sentences)
                
                # 너무 길면 자르기 (window 기준)
                if len(snippet) > window:
                    # target 위치를 중심으로 자르기
                    target_pos = snippet.find(target)
                    if target_pos != -1:
                        start = max(0, target_pos - window // 3)
                        end = min(len(snippet), target_pos + len(target) + window * 2 // 3)
                        snippet = snippet[start:end].strip()
                
                return snippet
        
        # target을 못 찾은 경우 첫 문장들 반환
        result = ''
        for sentence in sentences:
            if len(result) + len(sentence) > window:
                break
            result += sentence + ' '
        
        return result.strip() if result else content[:window].strip()
    
    def _build_source_entry(
        self,
        metadata: Dict,
        content: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        url: Optional[str] = None
    ) -> Dict:
        """소스 엔트리 생성 (metadata 전체 저장, url 포함, 문장 단위 snippet)
        
        Args:
            metadata: 문서 메타데이터 (서명, 제목, 날짜, 인물, 장소 등)
            content: 본문 내용
            subject: 주어 엔티티
            predicate: 술어
            obj: 목적어 엔티티
            url: 원본 문서 URL
        """
        metadata = metadata or {}
        triple_text = " ".join(filter(None, [subject, predicate, obj]))
        
        # 문장 단위 snippet 추출 (앞뒤 문장 포함)
        snippet = self._extract_snippet(content, subject or obj, window=300, include_surrounding_sentences=True)
        
        source_entry = {
            # 기본 문서 정보
            'doc': metadata.get('source') or metadata.get('file_path', 'unknown'),
            'url': url or metadata.get('url'),  # URL 저장
            'page': metadata.get('page'),
            'index': metadata.get('index'),
            'type': metadata.get('type'),
            
            # 문서 메타데이터 (사료 정보)
            '서명': metadata.get('서명'),
            '제목': metadata.get('제목') or metadata.get('title'),
            '날짜': metadata.get('날짜') or metadata.get('date'),
            '주제분류': metadata.get('주제분류') or metadata.get('topic'),
            '카테고리': metadata.get('카테고리') or metadata.get('category'),
            '키워드': metadata.get('키워드') or metadata.get('keywords'),
            
            # 추출된 엔티티 정보
            '인물': metadata.get('인물') or metadata.get('persons'),
            '장소': metadata.get('장소') or metadata.get('locations'),
            
            # 트리플 정보
            'snippet': snippet,
            'triple': triple_text,
            
            # 시스템 메타데이터
            'timestamp': metadata.get('timestamp'),
            'collection': metadata.get('collection'),
            'document_id': metadata.get('document_id'),
        }
        
        # annotation 정보 (주석)
        annotation_id = metadata.get('annotation_id')
        if annotation_id:
            source_entry['annotation_id'] = annotation_id
            source_entry['annotation_label'] = metadata.get('annotation_label')
        
        annotation_ids = metadata.get('annotation_ids')
        if annotation_ids:
            source_entry['annotation_ids'] = annotation_ids
        
        # annotations 배열이 있는 경우 (문서에 직접 포함된 주석들)
        annotations = metadata.get('annotations')
        if annotations and isinstance(annotations, list):
            source_entry['annotations'] = annotations
        
        # None 값 제거 (저장 공간 절약)
        source_entry = {k: v for k, v in source_entry.items() if v is not None}
        
        return source_entry
    
    def _relation_signature(self, from_key: str, predicate: str, to_key: str) -> str:
        raw = f"{from_key}|{predicate.strip()}|{to_key}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()
    
    def _build_relation_record(
        self,
        subject: str,
        predicate: str,
        obj: str,
        relation_source: Dict,
        relation_context: str
    ) -> Dict:
        normalized_subject = self._normalize_entity_name(subject) or subject
        normalized_object = self._normalize_entity_name(obj) or obj
        subject_key = self._sanitize_key(normalized_subject)
        object_key = self._sanitize_key(normalized_object)
        signature = self._relation_signature(subject_key, predicate, object_key)
        return {
            '_key': signature,
            '_from': f"entities/{subject_key}",
            '_to': f"entities/{object_key}",
            'type': predicate,
            'source': relation_source,
            'context': relation_context,
            'signature': signature,
            'triple': {
                'subject': subject,
                'predicate': predicate,
                'object': obj
            }
        }
    
    def _build_entity_record(
        self,
        name: str,
        metadata: Dict,
        content: str,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        subject_flag: bool = True,
        url: Optional[str] = None
    ) -> Dict:
        """엔티티 레코드 생성
        
        Args:
            name: 엔티티 이름
            metadata: 문서 메타데이터
            content: 본문 내용
            predicate: 술어
            obj: 관련 엔티티
            subject_flag: 주어 여부 (True면 name이 주어, False면 obj가 주어)
            url: 원본 문서 URL
        """
        normalized = self._normalize_entity_name(name)
        signature = self._entity_signature(name)
        category = self._classify_entity(name)
        
        # URL 전달하여 source_entry 생성
        source_entry = self._build_source_entry(
            metadata, content, 
            name if subject_flag else obj, 
            predicate, 
            obj if subject_flag else name,
            url=url or metadata.get('url')
        )
        sources = [source_entry] if source_entry else []
        aliases = []
        if normalized and normalized != name:
            aliases.append(normalized)
        key_target = normalized or name
        return {
            '_key': self._sanitize_key(key_target),
            'name': name,
            'display_name': name,
            'normalized_name': normalized,
            'canonical_name': normalized or name,
            'signature': signature,
            'type': category,
            'category': category,
            'aliases': aliases,
            'sources': sources
        }
    
    def _add_source_entry(self, sources: List[Dict], new_entry: Dict):
        if not new_entry:
            return
        signature = (new_entry.get('doc'), new_entry.get('page'), new_entry.get('snippet'))
        for existing in sources:
            existing_sig = (existing.get('doc'), existing.get('page'), existing.get('snippet'))
            if existing_sig == signature:
                return
        sources.append(new_entry)
    
    def _merge_aliases(self, entity: Dict, incoming: Dict):
        aliases = set(entity.get('aliases', []) or [])
        incoming_aliases = incoming.get('aliases', []) if incoming else []
        if not isinstance(incoming_aliases, list):
            incoming_aliases = [incoming_aliases]
        aliases.update(filter(None, incoming_aliases))
        if entity.get('normalized_name') and entity.get('name') != entity.get('normalized_name'):
            aliases.add(entity['normalized_name'])
        entity['aliases'] = list(aliases)
    
    def _merge_entity_records(self, existing: Dict, incoming: Dict) -> Dict:
        """엔티티 레코드 병합 (KnowledgeGraphExtractor용)"""
        merged = existing.copy()
        for key in ['name', 'display_name', 'normalized_name', 'canonical_name', 'signature', 'category', 'type']:
            if incoming.get(key):
                merged[key] = incoming[key]
        existing_aliases = existing.get('aliases', [])
        incoming_aliases = incoming.get('aliases', [])
        if not isinstance(existing_aliases, list):
            existing_aliases = [existing_aliases]
        if not isinstance(incoming_aliases, list):
            incoming_aliases = [incoming_aliases]
        merged['aliases'] = sorted(
            set(existing_aliases) | set(filter(None, incoming_aliases))
        )
        merged['sources'] = self._merge_sources(
            existing.get('sources', []),
            incoming.get('sources', [])
        )
        return merged
    
    def _merge_sources(self, existing: List[Dict], incoming: List[Dict]) -> List[Dict]:
        """소스 리스트 병합 (중복 제거)"""
        if not isinstance(existing, list):
            existing = [existing] if existing else []
        if not isinstance(incoming, list):
            incoming = [incoming] if incoming else []
        combined = list(existing)
        signatures = {
            (src.get('doc'), src.get('page'), src.get('snippet'))
            for src in combined
        }
        for source in incoming:
            signature = (source.get('doc'), source.get('page'), source.get('snippet'))
            if signature not in signatures:
                combined.append(source)
                signatures.add(signature)
        return combined
    
    def _sanitize_key(self, text: str) -> str:
        """ArangoDB 키로 사용 가능하도록 텍스트 정제
        
        ArangoDB 키 규칙:
        - 영문자, 숫자, 언더스코어, 하이픈만 허용
        - 한글/한자 등 유니코드 문자는 해시로 변환
        """
        if not text or not isinstance(text, str):
            return 'unknown'
        
        # 1. 공백을 언더스코어로 변환
        normalized = text.replace(' ', '_')
        
        # 2. ASCII만 남기고 나머지는 제거 (임시)
        ascii_only = re.sub(r'[^a-zA-Z0-9_-]', '', normalized)
        
        # 3. ASCII만으로 충분한 경우 (영문 엔티티)
        if ascii_only and len(ascii_only) >= 3:
            # 영문자로 시작하도록
            if not ascii_only[0].isalpha():
                ascii_only = 'K_' + ascii_only
            return ascii_only[:128]
        
        # 4. 한글/특수문자 포함 시 → 해시 기반 키 생성
        # MD5 해시의 앞 12자 사용 (충돌 확률 극히 낮음)
        hash_part = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        
        # 원본 텍스트의 일부를 포함 (가독성)
        prefix = ascii_only[:8] if ascii_only else 'entity'
        
        return f"K_{prefix}_{hash_part}"


class ArangoGraphDB:
    """ArangoDB 기반 지식 그래프 데이터베이스"""
    
    def __init__(
        self,
        host='localhost',
        port=8529,
        username='root',
        password='',
        db_name='knowledge_graph',
        reset=False
    ):
        """초기화
        
        Args:
            host: ArangoDB 호스트
            port: ArangoDB 포트
            username: 사용자명
            password: 비밀번호
            db_name: 데이터베이스 이름
            reset: True면 기존 컬렉션/그래프를 삭제하고 새로 생성
        """
        if not ARANGO_AVAILABLE:
            self.db = None
            return
        
        print(f"\nArangoDB 연결 중... ({host}:{port}, DB: {db_name})")
        
        try:
            # 클라이언트 생성
            self.client = ArangoClient(hosts=f'http://{host}:{port}')
            
            # 시스템 데이터베이스 연결
            sys_db = self.client.db('_system', username=username, password=password)
            
            # 데이터베이스 생성 (존재하지 않으면)
            if not sys_db.has_database(db_name):
                sys_db.create_database(db_name)
                print(f"  ✓ 데이터베이스 '{db_name}' 생성됨")
            else:
                print(f"  ✓ 데이터베이스 '{db_name}' 존재함")
            
            # 데이터베이스 연결
            self.db = self.client.db(db_name, username=username, password=password)
            
            # 컬렉션 생성
            self._setup_collections(reset=reset)
            
            print(f"  ✓ ArangoDB '{db_name}' 연결 성공")
            
        except Exception as e:
            print(f"ArangoDB 연결 실패: {e}")
            print("\nArangoDB 설치 및 실행 방법:")
            print("  1. https://www.arangodb.com/download 에서 다운로드")
            print("  2. 설치 후 서비스 시작")
            print("  3. 기본 설정: localhost:8529, 사용자: root")
            self.db = None
    
    def _setup_collections(self, reset: bool = False):
        """컬렉션 및 그래프 설정
        
        Args:
            reset: True면 기존 컬렉션/그래프를 삭제하고 새로 생성
        """
        if not self.db:
            return
        
        # 리셋 모드: 기존 데이터 삭제
        if reset:
            print("\n기존 데이터 삭제 중...")
            
            # 그래프 삭제
            if self.db.has_graph('knowledge_graph'):
                self.db.delete_graph('knowledge_graph', drop_collections=False)
                print("  ✓ 그래프 정의 삭제됨")
            
            # 컬렉션 삭제
            if self.db.has_collection('entities'):
                self.db.delete_collection('entities')
                print("  ✓ entities 컬렉션 삭제됨")
            
            if self.db.has_collection('relations'):
                self.db.delete_collection('relations')
                print("  ✓ relations 컬렉션 삭제됨")
        
        # 엔티티 컬렉션 (문서 컬렉션)
        entities_created = False
        if not self.db.has_collection('entities'):
            self.db.create_collection('entities')
            print("'entities' 컬렉션 생성됨")
            entities_created = True
        
        # 관계 컬렉션 (엣지 컬렉션)
        relations_created = False
        if not self.db.has_collection('relations'):
            self.db.create_collection('relations', edge=True)
            print("'relations' 컬렉션 생성됨")
            relations_created = True
        
        # 인덱스 설정 (검색 성능 최적화)
        self._setup_indexes(entities_created, relations_created)
        
        # 그래프 정의 생성 (Web UI에서 보이도록)
        if not self.db.has_graph('knowledge_graph'):
            graph = self.db.create_graph('knowledge_graph')
            # 엣지 정의: relations 컬렉션이 entities를 연결
            graph.create_edge_definition(
                edge_collection='relations',
                from_vertex_collections=['entities'],
                to_vertex_collections=['entities']
            )
            print("'knowledge_graph' 그래프 정의 생성됨")
    
    def _setup_indexes(self, entities_new: bool = False, relations_new: bool = False):
        """검색 성능 최적화를 위한 인덱스 설정
        
        인덱스가 있으면 검색 속도가 O(n) → O(log n)으로 개선됨
        (예: 0.5초 → 0.001초)
        """
        try:
            entities_col = self.db.collection('entities')
            relations_col = self.db.collection('relations')
            
            # 기존 인덱스 확인
            entity_indexes = {idx['fields'][0] for idx in entities_col.indexes() 
                            if idx.get('fields') and len(idx['fields']) == 1}
            relation_indexes = {idx['fields'][0] for idx in relations_col.indexes() 
                              if idx.get('fields') and len(idx['fields']) == 1}
            
            indexes_added = 0
            
            # === Entities 컬렉션 인덱스 ===
            # 1. name 필드 (가장 중요: 엔티티 검색에 사용)
            if 'name' not in entity_indexes:
                entities_col.add_persistent_index(
                    fields=['name'],
                    unique=False,
                    name='idx_entity_name'
                )
                indexes_added += 1
            
            # 2. type 필드 (필터링에 사용)
            if 'type' not in entity_indexes:
                entities_col.add_persistent_index(
                    fields=['type'],
                    unique=False,
                    name='idx_entity_type'
                )
                indexes_added += 1
            
            # 3. canonical_name 필드 (한자 포함 검색)
            if 'canonical_name' not in entity_indexes:
                entities_col.add_persistent_index(
                    fields=['canonical_name'],
                    unique=False,
                    name='idx_entity_canonical'
                )
                indexes_added += 1
            
            # === Relations 컬렉션 인덱스 ===
            # _from, _to는 엣지 컬렉션에서 자동 인덱싱됨
            
            # 1. type 필드 (관계 유형 필터링)
            if 'type' not in relation_indexes:
                relations_col.add_persistent_index(
                    fields=['type'],
                    unique=False,
                    name='idx_relation_type'
                )
                indexes_added += 1
            
            # 2. predicate 필드 (관계명 검색)
            if 'predicate' not in relation_indexes:
                relations_col.add_persistent_index(
                    fields=['predicate'],
                    unique=False,
                    name='idx_relation_predicate'
                )
                indexes_added += 1
            
            if indexes_added > 0:
                print(f"  ✓ {indexes_added}개 인덱스 생성됨 (검색 성능 최적화)")
            
        except Exception as e:
            print(f"  ⚠ 인덱스 설정 중 오류: {e}")
    
    def insert_entities(self, entities: List[Dict]) -> int:
        """엔티티 삽입 (유사도 매칭 포함)
        
        Args:
            entities: 엔티티 리스트
            
        Returns:
            삽입된 엔티티 수
        """
        if not self.db:
            return 0
        
        collection = self.db.collection('entities')
        inserted = 0
        linked = 0
        
        for entity in entities:
            try:
                entity_name = entity.get('name', '')
                
                # type 처리 로직:
                # 1. LLM이 추론한 type이 있으면 그것을 우선 사용
                # 2. type이 없으면 백과사전에서 가져오기 시도
                entity_type = entity.get('type', '')
                llm_inferred_type = entity_type  # LLM이 추론한 타입 보존
                
                if not entity_type or entity_type == '미분류':
                    # 이름으로 기존 엔티티 검색하여 type 가져오기
                    existing_entity = self._find_entity_by_name(collection, entity_name)
                    if existing_entity and existing_entity.get('type'):
                        entity_type = existing_entity.get('type')
                        entity['type'] = entity_type
                elif entity_type and entity_type != '미분류':
                    # LLM이 type을 추론했고, 백과사전에 같은 이름이 있는 경우
                    # type이 다르면 별개 엔티티로 처리
                    existing_entity = self._find_entity_by_name(collection, entity_name)
                    if existing_entity:
                        existing_type = existing_entity.get('type', '')
                        # type이 다르면 → 동음이의어, 별개 엔티티로 유지
                        if existing_type and existing_type != entity_type:
                            # LLM이 추론한 type 유지 (별개 엔티티)
                            pass
                        else:
                            # type이 같으면 → 동일 엔티티, 병합 대상
                            pass
                
                # _key가 없으면 자동 생성 (name + type 조합으로 동음이의어 구분)
                if '_key' not in entity or not entity['_key']:
                    # 동음이의어 구분을 위해 name_type 형태로 키 생성
                    if entity_type and entity_type != '미분류':
                        key_source = f"{entity_name}_{entity_type}"
                    else:
                        key_source = entity_name
                    entity['_key'] = self._sanitize_key(key_source)
                entity_key = entity['_key']
                
                # 1. 정확히 일치하는 엔티티가 있는지 확인
                if collection.has(entity_key):
                    existing = collection.get(entity_key)
                    merged = self._merge_entity_records(existing, entity)
                    collection.update(merged)
                    inserted += 1
                    continue
                
                # 2. 별칭(aliases)으로 매칭 시도
                existing_by_alias = self._find_entity_by_alias(
                    collection, entity_name
                )
                if existing_by_alias:
                    # 기존 엔티티에 별칭 추가 및 병합
                    merged = self._merge_entity_records(existing_by_alias, entity)
                    # 새 이름을 별칭에 추가
                    aliases = set(merged.get('aliases', []) or [])
                    aliases.add(entity_name)
                    merged['aliases'] = list(aliases)
                    collection.update(merged)
                    linked += 1
                    continue
                
                # 3. 유사도 매칭 (한글 엔티티만, threshold 0.8)
                if len(entity_name) >= 2:
                    similar = self._find_similar_entity_in_db(
                        collection, entity_name, threshold=0.8
                    )
                    if similar:
                        merged = self._merge_entity_records(similar, entity)
                        aliases = set(merged.get('aliases', []) or [])
                        aliases.add(entity_name)
                        merged['aliases'] = list(aliases)
                        collection.update(merged)
                        linked += 1
                        continue
                
                # 4. 새 엔티티 삽입
                collection.insert(entity)
                inserted += 1
                
            except Exception as e:
                print(f"엔티티 삽입 오류 ({entity.get('name', 'unknown')}): {e}")
        
        if linked > 0:
            print(f"{inserted}개 삽입, {linked}개 기존 엔티티에 연결됨")
        else:
            print(f"{inserted}개 엔티티 삽입 완료")
        return inserted + linked
    
    def _find_entity_by_name(self, collection, name: str) -> Optional[Dict]:
        """이름으로 엔티티 찾기 (정확히 일치)"""
        try:
            query = """
            FOR entity IN @@collection
                FILTER entity.name == @name
                RETURN entity
            """
            cursor = self.db.aql.execute(
                query,
                bind_vars={'@collection': 'entities', 'name': name}
            )
            results = list(cursor)
            return results[0] if results else None
        except Exception:
            return None
    
    def _find_entity_by_alias(self, collection, name: str) -> Optional[Dict]:
        """별칭으로 엔티티 찾기"""
        try:
            # AQL로 aliases 배열에서 검색
            query = """
            FOR entity IN @@collection
                FILTER @name IN entity.aliases
                RETURN entity
            """
            cursor = self.db.aql.execute(
                query,
                bind_vars={'@collection': 'entities', 'name': name}
            )
            results = list(cursor)
            return results[0] if results else None
        except Exception:
            return None
    
    def _find_similar_entity_in_db(
        self, collection, name: str, threshold: float = 0.8
    ) -> Optional[Dict]:
        """유사도 기반 엔티티 찾기 (DB에서)"""
        try:
            # 공백 제거 버전으로도 검색
            name_no_space = name.replace(' ', '')
            
            # 1. 공백 제거 후 정확히 일치하는지 확인
            query = """
            FOR entity IN @@collection
                LET name_no_space = SUBSTITUTE(entity.name, ' ', '')
                FILTER name_no_space == @name_no_space
                RETURN entity
            """
            cursor = self.db.aql.execute(
                query,
                bind_vars={'@collection': 'entities', 'name_no_space': name_no_space}
            )
            results = list(cursor)
            if results:
                return results[0]
            
            # 2. 부분 문자열 매칭 (한산대첩 ⊂ 한산도 대첩)
            if len(name) >= 3:
                query = """
                FOR entity IN @@collection
                    FILTER CONTAINS(entity.name, @name) OR CONTAINS(@name, entity.name)
                    RETURN entity
                """
                cursor = self.db.aql.execute(
                    query,
                    bind_vars={'@collection': 'entities', 'name': name}
                )
                results = list(cursor)
                if results:
                    # 가장 짧은 이름 차이를 가진 것 선택
                    best = min(results, key=lambda x: abs(len(x['name']) - len(name)))
                    return best
            
            return None
        except Exception:
            return None
    
    def _merge_entity_records(self, existing: Dict, incoming: Dict) -> Dict:
        merged = existing.copy()
        for key in ['name', 'display_name', 'normalized_name', 'canonical_name', 'signature', 'category', 'type']:
            if incoming.get(key):
                merged[key] = incoming[key]
        existing_aliases = existing.get('aliases', [])
        incoming_aliases = incoming.get('aliases', [])
        if not isinstance(existing_aliases, list):
            existing_aliases = [existing_aliases]
        if not isinstance(incoming_aliases, list):
            incoming_aliases = [incoming_aliases]
        merged['aliases'] = sorted(
            set(existing_aliases) | set(filter(None, incoming_aliases))
        )
        merged['sources'] = self._merge_sources(
            existing.get('sources', []),
            incoming.get('sources', [])
        )
        return merged
    
    def _merge_sources(self, existing: List[Dict], incoming: List[Dict]) -> List[Dict]:
        if not isinstance(existing, list):
            existing = [existing] if existing else []
        if not isinstance(incoming, list):
            incoming = [incoming] if incoming else []
        combined = list(existing)
        signatures = {
            (src.get('doc'), src.get('page'), src.get('snippet'))
            for src in combined
        }
        for source in incoming:
            signature = (source.get('doc'), source.get('page'), source.get('snippet'))
            if signature not in signatures:
                combined.append(source)
                signatures.add(signature)
        return combined
    
    def insert_relations(self, relations: List[Dict], entity_key_map: Dict[str, str] = None) -> int:
        """관계 삽입
        
        Args:
            relations: 관계 리스트
            entity_key_map: 엔티티 이름 → _key 매핑 (없으면 DB 검색)
            
        Returns:
            삽입된 관계 수
        """
        if not self.db:
            return 0
        
        collection = self.db.collection('relations')
        entities_col = self.db.collection('entities')
        inserted = 0
        skipped = 0
        skipped_no_subject = 0
        skipped_no_object = 0
        errors = 0
        
        # 엔티티 이름 → 키 매핑이 없으면 DB에서 로드
        if entity_key_map is None:
            print("  엔티티 키 매핑 로드 중...")
            entity_key_map = self._load_entity_key_map()
            print(f"  → {len(entity_key_map)}개 엔티티 매핑 로드됨")
        
        total = len(relations)
        print(f"  관계 삽입 시작: {total}개")
        
        for idx, relation in enumerate(relations):
            try:
                subject = relation.get('subject', '')
                obj = relation.get('object', '')
                predicate = relation.get('predicate', '')
                
                if not subject or not obj:
                    skipped += 1
                    continue
                
                # 매핑에서 키 찾기 (빠름!)
                from_key = entity_key_map.get(subject)
                to_key = entity_key_map.get(obj)
                
                # 매핑에 없으면 DB 검색 (느림)
                if not from_key:
                    from_key = self._find_entity_key(entities_col, subject)
                if not to_key:
                    to_key = self._find_entity_key(entities_col, obj)
                
                if not from_key:
                    skipped_no_subject += 1
                    skipped += 1
                    # 처음 5개만 샘플 출력
                    if skipped_no_subject <= 5:
                        print(f"    [subject없음] '{subject}' → '{obj}'")
                    continue
                if not to_key:
                    skipped_no_object += 1
                    skipped += 1
                    # 처음 5개만 샘플 출력
                    if skipped_no_object <= 5:
                        print(f"    [object없음] '{subject}' → '{obj}'")
                    continue
                
                # 진행 상황 출력 (1,000개마다)
                if (idx + 1) % 1000 == 0:
                    skip_detail = f"subject없음:{skipped_no_subject}, object없음:{skipped_no_object}"
                    print(f"    진행: {idx+1}/{total} (삽입: {inserted}, 건너뜀: {skipped} - {skip_detail})")
                
                # 엣지 문서 생성
                edge_doc = {
                    '_from': f"entities/{from_key}",
                    '_to': f"entities/{to_key}",
                    'type': predicate,
                    'predicate': predicate,
                    'subject': subject,
                    'object': obj,
                    'source': relation.get('source', ''),
                    'date': relation.get('date', ''),
                    'context': relation.get('context', ''),
                }
                
                # 중복 체크용 키 생성
                edge_key = self._sanitize_key(f"{subject}_{predicate}_{obj}")
                edge_doc['_key'] = edge_key
                
                if collection.has(edge_key):
                    skipped += 1
                    continue
                
                collection.insert(edge_doc)
                inserted += 1
                
            except Exception as e:
                errors += 1
                if errors <= 5:  # 처음 5개 오류만 출력
                    print(f"  관계 삽입 오류: {e}")
        
        if skipped_no_subject > 0 or skipped_no_object > 0:
            print(f"{inserted}개 관계 삽입 (건너뜀: {skipped} - subject없음:{skipped_no_subject}, object없음:{skipped_no_object})")
        else:
            print(f"{inserted}개 관계 삽입 완료")
        return inserted
    
    def _load_entity_key_map(self) -> Dict[str, str]:
        """DB에서 모든 엔티티의 이름 → _key 매핑 로드"""
        entity_map = {}
        try:
            query = """
            FOR e IN entities
                RETURN {name: e.name, key: e._key}
            """
            cursor = self.db.aql.execute(query)
            sample_names = []
            for doc in cursor:
                if doc.get('name') and doc.get('key'):
                    entity_map[doc['name']] = doc['key']
                    if len(sample_names) < 5:
                        sample_names.append(doc['name'])
            
            # 디버깅: 샘플 엔티티 이름 출력
            if sample_names:
                print(f"    [샘플 엔티티] {sample_names}")
            
            # '속동문선' 확인
            if '속동문선' in entity_map:
                print(f"    ✓ '속동문선' 있음: {entity_map['속동문선']}")
            else:
                print(f"    ✗ '속동문선' 없음")
            if '유민탄' in entity_map:
                print(f"    ✓ '유민탄' 있음: {entity_map['유민탄']}")
            else:
                print(f"    ✗ '유민탄' 없음")
                
        except Exception as e:
            print(f"  ⚠ 엔티티 매핑 로드 오류: {e}")
        return entity_map
    
    def _find_entity_key(self, collection, name: str) -> Optional[str]:
        """엔티티 이름으로 _key 찾기"""
        try:
            query = """
            FOR e IN @@collection
                FILTER e.name == @name
                LIMIT 1
                RETURN e._key
            """
            cursor = self.db.aql.execute(
                query,
                bind_vars={'@collection': 'entities', 'name': name}
            )
            results = list(cursor)
            return results[0] if results else None
        except Exception:
            return None
    
    def fetch_keywords_by_category(
        self,
        source: Optional[str] = None,
        limit_per_category: int = 60
    ) -> Dict[str, List[str]]:
        """그래프에서 카테고리별 대표 키워드 추출"""
        if not self.db:
            return {}
        
        filter_clause = ""
        bind_vars = {'limit': limit_per_category}
        if source:
            filter_clause = """
            LET matched = (
                FOR s IN entity.sources ? entity.sources : []
                    FILTER s.doc == @source
                    RETURN 1
            )
            FILTER LENGTH(matched) > 0
            """
            bind_vars['source'] = source
        
        query = f"""
        FOR entity IN entities
            {filter_clause}
            COLLECT category = entity.category INTO grouped = entity
            LET sorted_docs = (
                FOR doc IN grouped
                    LET e = doc.entity
                    LET name = e.name
                    FILTER name != NULL AND name != ""
                    LET weight = LENGTH(e.sources)
                    SORT weight DESC, LENGTH(name) ASC, name ASC
                    RETURN name
            )
            LET unique_names = UNIQUE(sorted_docs)
            LET top_keywords = SLICE(unique_names, 0, @limit)
            RETURN {{
                category: category,
                keywords: top_keywords
            }}
        """
        
        try:
            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            keyword_map = {}
            for row in cursor:
                category = row.get('category') or 'entity'
                keywords = [kw for kw in row.get('keywords', []) if kw]
                if keywords:
                    keyword_map[category] = keywords
            return keyword_map
        except Exception as e:
            print(f"카테고리 키워드 조회 오류: {e}")
            return {}
    
    def fetch_relation_types(
        self,
        source: Optional[str] = None,
        limit: int = 40
    ) -> List[Dict]:
        """그래프에서 관계 타입 빈도 조회"""
        if not self.db:
            return []
        
        bind_vars = {'limit': limit}
        filter_clause = ""
        if source:
            filter_clause = """
            LET source_doc = (
                IS_OBJECT(rel.source) ? rel.source.doc :
                (IS_STRING(rel.source) ? rel.source : null)
            )
            FILTER source_doc == @source
            """
            bind_vars['source'] = source
        else:
            filter_clause = ""
        
        query = f"""
        FOR rel IN relations
            {filter_clause}
            COLLECT rel_type = rel.type WITH COUNT INTO count
            SORT count DESC
            LIMIT @limit
            RETURN {{
                type: rel_type,
                count: count
            }}
        """
        
        try:
            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            return [row for row in cursor if row.get('type')]
        except Exception as e:
            print(f"관계 타입 조회 오류: {e}")
            return []
    
    def query_entity(self, entity_name: str) -> Optional[Dict]:
        """엔티티 조회 (해시 키 사용)
        
        Args:
            entity_name: 엔티티 이름 (한글 가능)
            
        Returns:
            엔티티 정보 또는 None
            
        Note:
            entity_name을 해시로 변환하여 검색합니다.
            같은 이름 → 항상 같은 해시 (MD5의 결정론적 특성)
        """
        if not self.db:
            return None
        
        try:
            collection = self.db.collection('entities')
            key = self._sanitize_key(entity_name)
            return collection.get(key)
        except Exception:
            return None
    
    def search_entities_by_name(self, entity_name: str) -> List[Dict]:
        """엔티티를 name 필드로 직접 검색 (대안 방법)
        
        Args:
            entity_name: 엔티티 이름 (한글)
            
        Returns:
            매칭되는 엔티티 리스트
            
        Note:
            해시 대신 name 필드를 직접 검색합니다.
            부분 매칭도 가능 (LIKE 검색)
        """
        if not self.db:
            return []
        
        try:
            # AQL: name 필드로 검색
            query = """
            FOR entity IN entities
                FILTER entity.name == @name
                RETURN entity
            """
            cursor = self.db.aql.execute(query, bind_vars={'name': entity_name})
            return list(cursor)
        except Exception as e:
            print(f"name 검색 오류: {e}")
            return []
    
    def query_neighbors(self, entity_name: str, depth: int = 1) -> Dict:
        """엔티티의 이웃 조회
        
        Args:
            entity_name: 엔티티 이름 (한글 가능)
            depth: 탐색 깊이
            
        Returns:
            이웃 정보 (엔티티와 관계)
            
        Note:
            entity_name을 해시로 변환하여 검색합니다.
            같은 이름 → 항상 같은 해시 (MD5의 결정론적 특성)
        """
        if not self.db:
            return {'entities': [], 'relations': []}
        
        # 방법 1: 해시 키로 직접 조회 (빠름)
        key = self._sanitize_key(entity_name)
        
        # AQL 쿼리
        query = f"""
        FOR v, e, p IN 1..{depth} ANY 'entities/{key}' relations
            RETURN {{
                entity: v,
                relation: e,
                path: p
            }}
        """
        
        try:
            cursor = self.db.aql.execute(query)
            results = list(cursor)
            
            entities = []
            relations = []
            
            for result in results:
                if result['entity']:
                    entities.append(result['entity'])
                if result['relation']:
                    relations.append(result['relation'])
            
            return {
                'entities': entities,
                'relations': relations
            }
            
        except Exception as e:
            print(f"이웃 조회 오류: {e}")
            return {'entities': [], 'relations': []}
    
    def query_path(self, start_entity: str, end_entity: str) -> List[Dict]:
        """두 엔티티 간의 경로 찾기
        
        Args:
            start_entity: 시작 엔티티
            end_entity: 종료 엔티티
            
        Returns:
            경로 리스트
        """
        if not self.db:
            return []
        
        start_key = self._sanitize_key(start_entity)
        end_key = self._sanitize_key(end_entity)
        
        query = f"""
        FOR v, e IN OUTBOUND SHORTEST_PATH 
            'entities/{start_key}' TO 'entities/{end_key}'
            relations
            RETURN {{vertex: v, edge: e}}
        """
        
        try:
            cursor = self.db.aql.execute(query)
            return list(cursor)
        except Exception as e:
            print(f"경로 조회 오류: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """그래프 통계 조회"""
        if not self.db:
            return {}
        
        try:
            entities_count = self.db.collection('entities').count()
            relations_count = self.db.collection('relations').count()
            
            return {
                'entities_count': entities_count,
                'relations_count': relations_count
            }
        except Exception:
            return {}
    
    def export_graph(self, output_file: str = 'knowledge_graph_export.json'):
        """지식 그래프를 JSON 파일로 저장 (ArangoDB → JSON)"""
        if not self.db:
            return False
        
        try:
            entities_collection = self.db.collection('entities')
            relations_collection = self.db.collection('relations')
            
            entities = list(entities_collection.all())
            relations = list(relations_collection.all())
            
            export_data = {
                'entities': entities,
                'relations': relations,
                'metadata': {
                    'exported_at': str(datetime.now()),
                    'entities_count': len(entities),
                    'relations_count': len(relations)
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"    오류: {e}")
            return False
    
    def import_graph(self, input_file: str = 'knowledge_graph_export.json'):
        """JSON 파일에서 지식 그래프 로드 → ArangoDB에 삽입"""
        if not self.db:
            return False
        
        if not os.path.exists(input_file):
            return False
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                export_data = json.load(f)
            
            entities = export_data.get('entities', [])
            relations = export_data.get('relations', [])
            
            # 기존 데이터 확인
            stats_before = self.get_statistics()
            before_entities = stats_before.get('entities_count', 0)
            before_relations = stats_before.get('relations_count', 0)
            
            # 데이터 삽입/업데이트
            self.insert_entities(entities)
            self.insert_relations(relations)
            
            # 최종 통계
            stats_after = self.get_statistics()
            after_entities = stats_after.get('entities_count', 0)
            after_relations = stats_after.get('relations_count', 0)
            
            # 결과 출력 (간단히)
            if before_entities > 0:
                # 병합
                pass  # 메시지는 상위에서 출력
            
            return True
            
        except Exception as e:
            print(f"    오류: {e}")
            return False
    
    def _sanitize_key(self, text: str) -> str:
        """ArangoDB 키로 사용 가능하도록 텍스트 정제
        
        ArangoDB 키 규칙:
        - 영문자, 숫자, 언더스코어, 하이픈만 허용
        - 한글/한자 등 유니코드 문자는 해시로 변환
        """
        if not text or not isinstance(text, str):
            return 'unknown'
        
        # 1. 공백을 언더스코어로 변환
        normalized = text.replace(' ', '_')
        
        # 2. ASCII만 남기고 나머지는 제거 (임시)
        ascii_only = re.sub(r'[^a-zA-Z0-9_-]', '', normalized)
        
        # 3. ASCII만으로 충분한 경우 (영문 엔티티)
        if ascii_only and len(ascii_only) >= 3:
            # 영문자로 시작하도록
            if not ascii_only[0].isalpha():
                ascii_only = 'K_' + ascii_only
            return ascii_only[:128]
        
        # 4. 한글/특수문자 포함 시 → 해시 기반 키 생성
        # MD5 해시의 앞 12자 사용 (충돌 확률 극히 낮음)
        hash_part = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        
        # 원본 텍스트의 일부를 포함 (가독성)
        prefix = ascii_only[:8] if ascii_only else 'entity'
        
        return f"K_{prefix}_{hash_part}"


class GoldenDataLoader:
    """한국민족문화대백과사전 JSON에서 골든 데이터(Ontology) 직접 구축
    
    LLM 없이 직접 파싱하여 Canonical 엔티티와 관계를 생성합니다.
    이 데이터가 지식 그래프의 '뼈대'가 됩니다.
    """
    
    # 엔티티 유형 매핑 (한국민족문화대백과사전 → 표준 유형)
    ENTITY_TYPE_MAP = {
        '인물/전통 인물': 'person',
        '인물/근현대 인물': 'person',
        '인물': 'person',
        '사건/전쟁': 'event',
        '사건/사건': 'event',
        '사건': 'event',
        '장소/유적지': 'location',
        '장소/지명': 'location',
        '장소': 'location',
        '제도/법제/관직': 'position',
        '제도/법제': 'institution',
        '제도': 'institution',
        '문화재/유물': 'artifact',
        '문화재': 'artifact',
        '저술/전적': 'document',
        '저술': 'document',
    }
    
    def __init__(self, graph_db: ArangoGraphDB = None):
        """초기화
        
        Args:
            graph_db: ArangoDB 인스턴스 (없으면 나중에 설정)
        """
        self.graph_db = graph_db
        self._sanitize_key = self._create_sanitize_key()
    
    def _create_sanitize_key(self):
        """키 정규화 함수 생성"""
        def sanitize(text: str) -> str:
            if not text or not isinstance(text, str):
                return 'unknown'
            normalized = text.replace(' ', '_')
            ascii_only = re.sub(r'[^a-zA-Z0-9_-]', '', normalized)
            if ascii_only and len(ascii_only) >= 3:
                if not ascii_only[0].isalpha():
                    ascii_only = 'K_' + ascii_only
                return ascii_only[:128]
            hash_part = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
            prefix = ascii_only[:8] if ascii_only else 'entity'
            return f"K_{prefix}_{hash_part}"
        return sanitize
    
    def _normalize_entity_type(self, raw_type: str) -> str:
        """엔티티 유형 정규화"""
        if not raw_type:
            return 'entity'
        
        # 직접 매핑 확인
        if raw_type in self.ENTITY_TYPE_MAP:
            return self.ENTITY_TYPE_MAP[raw_type]
        
        # 부분 매칭
        for key, value in self.ENTITY_TYPE_MAP.items():
            if key in raw_type or raw_type in key:
                return value
        
        return 'entity'
    
    def _extract_links_from_body(self, body: str) -> List[Dict]:
        """본문에서 링크된 엔티티 추출
        
        형식: [이이(李珥)](E0045546)
        """
        if not body:
            return []
        
        links = []
        # 마크다운 링크 패턴: [텍스트](ID)
        pattern = r'\[([^\]]+)\]\(([A-Z]\d+)\)'
        
        for match in re.finditer(pattern, body):
            display_text = match.group(1)
            entity_id = match.group(2)
            
            # 한자 병기 분리: "이이(李珥)" → name="이이", hanja="李珥"
            hanja_match = re.match(r'(.+?)\(([^)]+)\)', display_text)
            if hanja_match:
                name = hanja_match.group(1).strip()
                hanja = hanja_match.group(2).strip()
            else:
                name = display_text.strip()
                hanja = None
            
            links.append({
                'name': name,
                'hanja': hanja,
                'entity_id': entity_id,
                'display_text': display_text
            })
        
        return links
    
    def load_golden_data(self, json_path: str) -> Tuple[List[Dict], List[Dict]]:
        """한국민족문화대백과사전 JSON 로드 및 파싱
        
        Args:
            json_path: JSON 파일 경로
            
        Returns:
            (엔티티 리스트, 관계 리스트)
        """
        print(f"\n[골든 데이터 로드] {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        entities = []
        relations = []
        entity_map = {}  # entity_id -> entity_data
        
        print(f"  총 {len(data)}개 항목 처리 중...")
        
        for idx, article in enumerate(data, 1):
            # 1. 주 엔티티 생성 (항목 자체)
            entity_name = article.get('항목명', '')
            if not entity_name:
                continue
            
            entity_id = self._extract_entity_id(article.get('url', ''))
            entity_type = self._normalize_entity_type(article.get('항목 유형', ''))
            
            entity = {
                '_key': self._sanitize_key(entity_name),
                'name': entity_name,
                'display_name': entity_name,
                'hanja': article.get('원어'),
                'canonical_name': f"{entity_name}({article.get('원어', '')})" if article.get('원어') else entity_name,
                'type': entity_type,
                'category': entity_type,
                'field': article.get('항목 분야'),
                'era': article.get('시대'),
                'definition': article.get('항목 정의'),
                'summary': article.get('요약'),
                'url': article.get('url'),
                'entity_id': entity_id,  # 백과사전 고유 ID
                'is_canonical': True,  # 정본 데이터 표시
                'source': '한국민족문화대백과사전',
                'aliases': [],
                'sources': [{
                    'doc': '한국민족문화대백과사전',
                    'url': article.get('url'),
                    'type': 'golden_data'
                }]
            }
            
            # 한자가 있으면 aliases에 추가
            if article.get('원어'):
                entity['aliases'].append(article.get('원어'))
            
            entities.append(entity)
            entity_map[entity_id] = entity
            
            # 2. 본문에서 링크된 엔티티 → 관계 생성
            body = article.get('항목 본문', '')
            links = self._extract_links_from_body(body)
            
            for link in links:
                # 관계 타입 추론 (본문 컨텍스트 기반)
                relation_type = self._infer_relation_type(entity_type, link)
                
                relation = {
                    '_key': hashlib.md5(
                        f"{entity_name}|{relation_type}|{link['name']}".encode()
                    ).hexdigest()[:16],
                    '_from': f"entities/{self._sanitize_key(entity_name)}",
                    '_to': f"entities/{self._sanitize_key(link['name'])}",
                    'type': relation_type,
                    'source': {
                        'doc': '한국민족문화대백과사전',
                        'url': article.get('url'),
                        'type': 'golden_data',
                        'linked_entity_id': link['entity_id']
                    },
                    'context': f"{entity_name} - {relation_type} - {link['name']}",
                    'is_canonical': True,
                    'triple': {
                        'subject': entity_name,
                        'predicate': relation_type,
                        'object': link['name']
                    }
                }
                relations.append(relation)
            
            # 3. 관련항목에서 관계 생성
            related_articles = article.get('관련항목', [])
            for related in related_articles:
                related_name = related.get('항목명', '')
                if not related_name:
                    continue
                
                related_type = self._normalize_entity_type(related.get('항목 유형', ''))
                relation_type = self._infer_relation_type_from_types(entity_type, related_type)
                
                # 관련 엔티티도 생성 (간략 버전)
                related_entity = {
                    '_key': self._sanitize_key(related_name),
                    'name': related_name,
                    'display_name': related_name,
                    'hanja': related.get('원어'),
                    'canonical_name': f"{related_name}({related.get('원어', '')})" if related.get('원어') else related_name,
                    'type': related_type,
                    'category': related_type,
                    'field': related.get('항목 분야'),
                    'era': related.get('시대'),
                    'definition': related.get('항목 정의'),
                    'url': f"https://encykorea.aks.ac.kr{related.get('URL', '')}",
                    'is_canonical': True,
                    'source': '한국민족문화대백과사전',
                    'aliases': [related.get('원어')] if related.get('원어') else [],
                    'sources': [{
                        'doc': '한국민족문화대백과사전',
                        'type': 'golden_data'
                    }]
                }
                entities.append(related_entity)
                
                # 관계 생성
                relation = {
                    '_key': hashlib.md5(
                        f"{entity_name}|{relation_type}|{related_name}".encode()
                    ).hexdigest()[:16],
                    '_from': f"entities/{self._sanitize_key(entity_name)}",
                    '_to': f"entities/{self._sanitize_key(related_name)}",
                    'type': relation_type,
                    'source': {
                        'doc': '한국민족문화대백과사전',
                        'type': 'golden_data'
                    },
                    'context': f"{entity_name} - {relation_type} - {related_name}",
                    'is_canonical': True,
                    'triple': {
                        'subject': entity_name,
                        'predicate': relation_type,
                        'object': related_name
                    }
                }
                relations.append(relation)
        
        # 중복 제거
        entities = self._deduplicate_entities(entities)
        relations = self._deduplicate_relations(relations)
        
        print(f"  ✓ 엔티티 {len(entities)}개, 관계 {len(relations)}개 추출 완료")
        
        return entities, relations
    
    def _extract_entity_id(self, url: str) -> str:
        """URL에서 엔티티 ID 추출"""
        if not url:
            return ''
        match = re.search(r'/([A-Z]\d+)/?$', url)
        return match.group(1) if match else ''
    
    def _infer_relation_type(self, source_type: str, link: Dict) -> str:
        """본문 링크에서 관계 타입 추론"""
        # 기본값
        return '관련_항목'
    
    def _infer_relation_type_from_types(self, source_type: str, target_type: str) -> str:
        """엔티티 타입 기반 관계 타입 추론"""
        type_relations = {
            ('event', 'person'): '관련_인물',
            ('person', 'event'): '참여',
            ('event', 'location'): '발생_장소',
            ('person', 'location'): '활동_지역',
            ('person', 'position'): '역임',
            ('person', 'document'): '저술',
            ('event', 'event'): '관련_사건',
            ('person', 'person'): '관련_인물',
        }
        
        key = (source_type, target_type)
        return type_relations.get(key, '관련_항목')
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """엔티티 중복 제거 (병합)"""
        entity_map = {}
        for entity in entities:
            key = entity.get('_key')
            if key in entity_map:
                # 기존 엔티티와 병합
                existing = entity_map[key]
                # aliases 병합
                existing_aliases = set(existing.get('aliases', []) or [])
                new_aliases = set(entity.get('aliases', []) or [])
                existing['aliases'] = list(existing_aliases | new_aliases)
                # sources 병합
                existing_sources = existing.get('sources', [])
                new_sources = entity.get('sources', [])
                existing['sources'] = existing_sources + new_sources
                # 더 상세한 정보로 업데이트
                for field in ['definition', 'summary', 'era', 'field']:
                    if not existing.get(field) and entity.get(field):
                        existing[field] = entity[field]
            else:
                entity_map[key] = entity
        
        return list(entity_map.values())
    
    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """관계 중복 제거"""
        relation_map = {}
        for relation in relations:
            key = relation.get('_key')
            if key not in relation_map:
                relation_map[key] = relation
        return list(relation_map.values())
    
    def seed_to_arango(self, json_path: str, graph_db: ArangoGraphDB = None) -> Dict:
        """골든 데이터를 ArangoDB에 직접 삽입
        
        Args:
            json_path: JSON 파일 경로
            graph_db: ArangoDB 인스턴스
            
        Returns:
            통계 정보
        """
        if graph_db:
            self.graph_db = graph_db
        
        if not self.graph_db or not self.graph_db.db:
            print("  ⚠ ArangoDB 연결이 필요합니다.")
            return {}
        
        entities, relations = self.load_golden_data(json_path)
        
        print(f"\n[ArangoDB 삽입]")
        inserted_entities = self.graph_db.insert_entities(entities)
        inserted_relations = self.graph_db.insert_relations(relations)
        
        stats = {
            'entities_loaded': len(entities),
            'relations_loaded': len(relations),
            'entities_inserted': inserted_entities,
            'relations_inserted': inserted_relations
        }
        
        print(f"  ✓ 골든 데이터 시딩 완료!")
        return stats


class EntityLinker:
    """엔티티 연결/해소 (Entity Linking/Resolution)
    
    추출된 엔티티를 기존 DB의 Canonical 엔티티와 매칭합니다.
    - 정확 매칭: 이름이 정확히 일치
    - 별칭 매칭: aliases에 포함
    - 한자 매칭: 한자 표기 일치
    - 유사도 매칭: 편집 거리 기반
    """
    
    def __init__(self, graph_db: ArangoGraphDB):
        self.graph_db = graph_db
        self._entity_cache = {}  # 캐시 (name -> entity)
        self._alias_index = {}   # 별칭 인덱스 (alias -> canonical_name)
    
    def build_index(self):
        """엔티티 인덱스 구축 (시작 시 한 번 호출)"""
        if not self.graph_db or not self.graph_db.db:
            return
        
        print("[엔티티 인덱스 구축]")
        
        try:
            query = """
            FOR entity IN entities
                FILTER entity.is_canonical == true
                RETURN {
                    key: entity._key,
                    name: entity.name,
                    hanja: entity.hanja,
                    aliases: entity.aliases,
                    type: entity.type
                }
            """
            cursor = self.graph_db.db.aql.execute(query)
            
            for entity in cursor:
                name = entity.get('name', '')
                self._entity_cache[name] = entity
                
                # 별칭 인덱스
                for alias in (entity.get('aliases') or []):
                    if alias:
                        self._alias_index[alias] = name
                
                # 한자도 별칭으로 처리
                hanja = entity.get('hanja')
                if hanja:
                    self._alias_index[hanja] = name
            
            print(f"  ✓ {len(self._entity_cache)}개 Canonical 엔티티, {len(self._alias_index)}개 별칭 인덱싱")
            
        except Exception as e:
            print(f"  ⚠ 인덱스 구축 실패: {e}")
    
    def link_entity(self, name: str, entity_type: str = None) -> Optional[Dict]:
        """엔티티 연결 시도
        
        Args:
            name: 추출된 엔티티 이름 (예: "충무공", "통제사 영감")
            entity_type: 엔티티 타입 힌트
            
        Returns:
            매칭된 Canonical 엔티티 또는 None
        """
        if not name:
            return None
        
        name = name.strip()
        
        # 1. 정확 매칭
        if name in self._entity_cache:
            return {
                'match_type': 'exact',
                'canonical': self._entity_cache[name],
                'confidence': 1.0
            }
        
        # 2. 별칭 매칭
        if name in self._alias_index:
            canonical_name = self._alias_index[name]
            return {
                'match_type': 'alias',
                'canonical': self._entity_cache.get(canonical_name),
                'confidence': 0.95
            }
        
        # 3. 부분 매칭 (접미사 제거)
        suffixes = ['장군', '공', '대감', '대왕', '선생', '영감', '통제사']
        for suffix in suffixes:
            if name.endswith(suffix):
                base_name = name[:-len(suffix)].strip()
                if base_name in self._entity_cache:
                    return {
                        'match_type': 'suffix_removal',
                        'canonical': self._entity_cache[base_name],
                        'confidence': 0.85
                    }
                if base_name in self._alias_index:
                    canonical_name = self._alias_index[base_name]
                    return {
                        'match_type': 'suffix_removal_alias',
                        'canonical': self._entity_cache.get(canonical_name),
                        'confidence': 0.80
                    }
        
        # 4. 유사도 매칭 (편집 거리 기반)
        best_match = self._find_similar_entity(name)
        if best_match:
            return best_match
        
        # 매칭 실패
        return None
    
    def _find_similar_entity(self, name: str, threshold: float = 0.7) -> Optional[Dict]:
        """유사도 기반 엔티티 찾기"""
        if len(name) < 2:
            return None
        
        best_score = 0
        best_entity = None
        
        for canonical_name, entity in self._entity_cache.items():
            score = self._similarity(name, canonical_name)
            if score > best_score and score >= threshold:
                best_score = score
                best_entity = entity
        
        if best_entity:
            return {
                'match_type': 'similar',
                'canonical': best_entity,
                'confidence': best_score
            }
        
        return None
    
    def _similarity(self, s1: str, s2: str) -> float:
        """두 문자열의 유사도 계산 (0~1)"""
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        # Jaccard similarity (문자 단위)
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def resolve_and_link(self, extracted_entity: Dict, graph_db: ArangoGraphDB = None) -> Dict:
        """추출된 엔티티를 해소하고 DB에 연결
        
        Args:
            extracted_entity: LLM이 추출한 엔티티 정보
            graph_db: ArangoDB 인스턴스
            
        Returns:
            연결된 엔티티 정보 (canonical_key 포함)
        """
        name = extracted_entity.get('name', '')
        entity_type = extracted_entity.get('type', '')
        
        # 연결 시도
        link_result = self.link_entity(name, entity_type)
        
        if link_result:
            # 매칭 성공: Canonical 엔티티 키 사용
            canonical = link_result['canonical']
            return {
                **extracted_entity,
                '_key': canonical.get('key'),
                'canonical_name': canonical.get('name'),
                'linked': True,
                'link_confidence': link_result['confidence'],
                'link_type': link_result['match_type']
            }
        else:
            # 매칭 실패: 신규 엔티티로 생성
            return {
                **extracted_entity,
                'linked': False,
                'is_new': True
            }


class InContextKnowledgeExtractor(KnowledgeGraphExtractor):
    """In-Context Learning 기반 지식 추출기
    
    골든 데이터의 스키마와 예시를 프롬프트에 포함하여
    더 정확한 트리플을 추출합니다.
    """
    
    # 엔티티 분류 체계 (골든 데이터에서 추출)
    ENTITY_SCHEMA = {
        'person': {
            'description': '인물 (장군, 왕, 신하, 학자 등)',
            'examples': ['이순신', '선조', '이이', '곽재우', '김시민'],
            'attributes': ['직책', '생몰년', '시대', '본관']
        },
        'event': {
            'description': '사건/전쟁 (전투, 전쟁, 사화, 반란 등)',
            'examples': ['임진왜란', '한산도대첩', '행주대첩', '진주성싸움'],
            'attributes': ['발생일', '장소', '결과']
        },
        'location': {
            'description': '장소 (지명, 성, 산, 포구 등)',
            'examples': ['한산도', '평양성', '부산포', '진주성'],
            'attributes': ['지역', '유형']
        },
        'position': {
            'description': '관직/직책',
            'examples': ['통제사', '영의정', '좌의정', '병마절도사'],
            'attributes': ['소속', '품계']
        },
        'document': {
            'description': '문헌/저술',
            'examples': ['징비록', '난중일기', '선조실록'],
            'attributes': ['저자', '시기']
        }
    }
    
    # 관계 유형 체계
    RELATION_SCHEMA = {
        '역임': '인물이 관직을 역임함',
        '참여': '인물이 사건에 참여함',
        '지휘': '인물이 전투/군대를 지휘함',
        '발생_장소': '사건이 장소에서 발생함',
        '시기': '사건의 발생 시기',
        '결과': '사건의 결과',
        '저술': '인물이 문헌을 저술함',
        '관련_인물': '항목과 관련된 인물',
        '관련_사건': '항목과 관련된 사건',
    }
    
    def __init__(self, llm_model='gemma3:12b', entity_linker: EntityLinker = None):
        super().__init__(llm_model)
        self.entity_linker = entity_linker
    
    def _build_icl_prompt(self, text: str) -> str:
        """In-Context Learning 프롬프트 생성"""
        
        # 스키마 설명 생성
        schema_desc = "\n".join([
            f"- {etype}: {info['description']} (예: {', '.join(info['examples'][:3])})"
            for etype, info in self.ENTITY_SCHEMA.items()
        ])
        
        # 관계 설명 생성
        relation_desc = "\n".join([
            f"- {rtype}: {desc}"
            for rtype, desc in self.RELATION_SCHEMA.items()
        ])
        
        # 허용된 type 목록 (백과사전 기반)
        allowed_types = """
- person: 인물 (왕, 장군, 신하, 문인 등)
- event: 사건/전쟁 (전투, 난, 사화 등)
- location: 장소 (지명, 성, 산, 강 등)
- position: 관직 (통제사, 영의정 등)
- institution: 제도/기관 (조직, 법제 등)
- artifact: 문화재/유물 (거북선, 도자기 등)
- document: 저술/문서 (서적, 기록 등)
- concept: 개념/추상명사 (조상, 충의 등)"""

        prompt = f"""당신은 한국 역사 지식 그래프 구축을 위한 전문 AI입니다.
입력된 텍스트에서 엔티티와 관계를 추출하여 지정된 형식으로 출력하세요.

[중요: 엔티티 유형 - 반드시 아래 목록에서만 선택]
{allowed_types}

[참고: 엔티티 분류 체계]
{schema_desc}

[참고: 관계 유형]
{relation_desc}

[추출 규칙]
1. 이미 존재하는 엔티티 형식을 따를 것 (예: "이순신", "임진왜란")
2. 한자(Hanja)가 병기된 경우 그대로 포함할 것 (예: "선조(宣祖)")
3. 날짜는 구체적으로 명시할 것 (예: "1592년 4월 14일")
4. 의미 없는 일반 명사는 제외할 것 (예: "사람", "전쟁", "일")
5. 숫자만 있는 엔티티는 제외할 것 (예: "1", "10")
6. 엔티티 유형은 반드시 괄호 안에 명시할 것

[출력 형식]
엔티티명(type) | 관계 | 엔티티명(type)

[예시 1]
입력: "이순신은 한산도에서 왜군을 격파했다."
출력:
이순신(person) | 지휘 | 한산도대첩(event)
한산도대첩(event) | 발생_장소 | 한산도(location)

[예시 2]
입력: "선조가 의주로 피난하였다."
출력:
선조(person) | 피난 | 의주(location)

[예시 3]
입력: "권율은 행주산성에서 왜군을 물리쳤다."
출력:
권율(person) | 지휘 | 행주대첩(event)
행주대첩(event) | 발생_장소 | 행주산성(location)

[예시 4]
입력: "선조의 뜻을 이어받아 후손들이 충의를 지켰다."
출력:
선조(concept) | 관련 | 충의(concept)

[입력 텍스트]
{text}

[트리플 추출 결과]
"""
        return prompt
    
    # 허용된 type 목록 (백과사전 기반)
    ALLOWED_TYPES = {
        'person', 'event', 'location', 'position', 
        'institution', 'artifact', 'document', 'concept', 'entity'
    }
    
    def _parse_entity_with_type(self, entity_str: str) -> Tuple[str, str]:
        """엔티티 문자열에서 이름과 type 추출
        
        예: "이순신(person)" → ("이순신", "person")
            "이순신" → ("이순신", "")
        """
        # 패턴: 이름(type)
        match = re.match(r'^(.+?)\(([^)]+)\)$', entity_str.strip())
        if match:
            name = match.group(1).strip()
            entity_type = match.group(2).strip().lower()
            
            # 허용된 type인지 확인
            if entity_type in self.ALLOWED_TYPES:
                return name, entity_type
            else:
                # 허용되지 않은 type이면 빈 문자열 반환
                return name, ''
        
        return entity_str.strip(), ''
    
    def extract_triples_icl(self, text: str) -> List[Tuple[str, str, str, str, str]]:
        """In-Context Learning 기반 트리플 추출
        
        Returns:
            List of (subject, subject_type, predicate, object, object_type)
        """
        if not self.llm:
            print("LLM이 초기화되지 않았습니다.")
            return []
        
        prompt = self._build_icl_prompt(text[:2000])
        
        try:
            response = self.llm.invoke(prompt)
            
            triples = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 3:
                        subject_raw, predicate, obj_raw = parts
                        
                        # type 추출
                        subject, subj_type = self._parse_entity_with_type(subject_raw)
                        obj, obj_type = self._parse_entity_with_type(obj_raw)
                        
                        if subject and predicate and obj:
                            # 유효성 검증
                            if self._is_valid_entity(subject) and self._is_valid_entity(obj):
                                triples.append((subject, subj_type, predicate, obj, obj_type))
            
            return triples
            
        except Exception as e:
            print(f"ICL 트리플 추출 오류: {e}")
            return []
    
    def extract_with_linking(self, documents: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """문서에서 추출 + 엔티티 연결
        
        1. ICL 프롬프트로 트리플 추출 (with type)
        2. 추출된 엔티티를 Canonical 엔티티와 연결
        3. 연결된 엔티티 키로 관계 생성
        """
        print("\n[ICL + Entity Linking 기반 추출]")
        
        entities = {}
        relations = []
        
        for idx, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            if not content:
                continue
            
            print(f"  [{idx}/{len(documents)}] 처리 중...")
            
            # 1. ICL로 트리플 추출 (type 포함)
            triples = self.extract_triples_icl(content)
            
            for subject, subj_type, predicate, obj, obj_type in triples:
                # 2. 엔티티 연결 (type 힌트 제공)
                if self.entity_linker:
                    subj_link = self.entity_linker.link_entity(subject, subj_type)
                    obj_link = self.entity_linker.link_entity(obj, obj_type)
                    
                    # Canonical 키 사용 (있으면)
                    subj_key = subj_link['canonical']['key'] if subj_link else None
                    obj_key = obj_link['canonical']['key'] if obj_link else None
                    
                    # 연결된 엔티티에서 type 가져오기
                    if subj_link and not subj_type:
                        subj_type = subj_link['canonical'].get('type', '')
                    if obj_link and not obj_type:
                        obj_type = obj_link['canonical'].get('type', '')
                else:
                    subj_key = None
                    obj_key = None
                
                # type 기반 키 생성 (동음이의어 구분)
                if not subj_key:
                    key_base = f"{subject}_{subj_type}" if subj_type else subject
                    subj_key = self._sanitize_key(key_base)
                if not obj_key:
                    key_base = f"{obj}_{obj_type}" if obj_type else obj
                    obj_key = self._sanitize_key(key_base)
                
                # 엔티티 생성/업데이트 (type 포함)
                if subject not in entities:
                    entities[subject] = self._build_entity_record(
                        subject, metadata, content, predicate, obj,
                        url=doc.get('url') or metadata.get('url')
                    )
                    entities[subject]['_key'] = subj_key
                    entities[subject]['type'] = subj_type or 'entity'
                
                if obj not in entities:
                    entities[obj] = self._build_entity_record(
                        obj, metadata, content, predicate, subject,
                        subject_flag=False,
                        url=doc.get('url') or metadata.get('url')
                    )
                    entities[obj]['_key'] = obj_key
                    entities[obj]['type'] = obj_type or 'entity'
                    entities[obj]['_key'] = obj_key
                
                # 관계 생성
                relation = {
                    '_key': hashlib.md5(f"{subj_key}|{predicate}|{obj_key}".encode()).hexdigest()[:16],
                    '_from': f"entities/{subj_key}",
                    '_to': f"entities/{obj_key}",
                    'type': predicate,
                    'source': self._build_source_entry(metadata, content, subject, predicate, obj),
                    'context': f"{subject} {predicate} {obj}",
                    'triple': {'subject': subject, 'predicate': predicate, 'object': obj}
                }
                relations.append(relation)
        
        print(f"  ✓ 엔티티 {len(entities)}개, 관계 {len(relations)}개 추출")
        
        return list(entities.values()), relations


def build_knowledge_graph_from_json(
    json_path: str,
    llm_model: str = 'gemma3:12b',
    arango_host: str = 'localhost',
    arango_port: int = 8529,
    arango_password: str = ''
) -> Tuple[ArangoGraphDB, Dict]:
    """JSON 파일에서 지식 그래프 구축
    
    Args:
        json_path: JSON 파일 경로
        llm_model: Ollama 모델 이름
        arango_host: ArangoDB 호스트
        arango_port: ArangoDB 포트
        arango_password: ArangoDB 비밀번호
        
    Returns:
        (ArangoGraphDB 인스턴스, 통계 정보)
    """
    print(f"\n지식 그래프 구축 시작: {json_path}")
    
    # JSON 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 문서 형식으로 변환
    documents = []
    if isinstance(data, list):
        documents = data
    elif isinstance(data, dict) and 'documents' in data:
        documents = data['documents']
    
    if not documents:
        print("문서를 찾을 수 없습니다.")
        return None, {}
    
    print(f"{len(documents)}개 문서 발견")
    
    # 지식 트리플 추출
    extractor = KnowledgeGraphExtractor(llm_model=llm_model)
    entities, relations = extractor.extract_entities_and_relations(documents)
    
    # ArangoDB에 저장
    graph_db = ArangoGraphDB(
        host=arango_host,
        port=arango_port,
        password=arango_password
    )
    
    if graph_db.db:
        graph_db.insert_entities(entities)
        graph_db.insert_relations(relations)
        
        stats = graph_db.get_statistics()
        print(f"\n지식 그래프 구축 완료!")
        print(f"  엔티티: {stats.get('entities_count', 0)}개")
        print(f"  관계: {stats.get('relations_count', 0)}개")
        
        return graph_db, stats
    else:
        print("\nArangoDB 연결 실패. 그래프 저장을 건너뜁니다.")
        return graph_db, {'entities': len(entities), 'relations': len(relations)}


def build_hybrid_knowledge_graph(
    golden_data_path: str,
    saryeo_paths: List[str],
    llm_model: str = 'gemma3:12b',
    arango_host: str = 'localhost',
    arango_port: int = 8529,
    arango_password: str = '',
    arango_db_name: str = 'knowledge_graph',
    reset: bool = False
) -> Tuple[ArangoGraphDB, Dict]:
    """하이브리드 지식 그래프 구축 (골든 데이터 + 사료)
    
    단계 1: 골든 데이터(한국민족문화대백과사전)로 뼈대 구축
    단계 2: 엔티티 링커 인덱스 구축
    단계 3: 사료에서 ICL 기반 추출 + 엔티티 연결
    
    Args:
        golden_data_path: 골든 데이터 JSON 경로
        saryeo_paths: 사료 JSON 경로 리스트
        llm_model: Ollama 모델
        arango_host: ArangoDB 호스트
        arango_port: ArangoDB 포트
        arango_password: ArangoDB 비밀번호
        arango_db_name: 데이터베이스 이름
        reset: 기존 데이터 삭제 여부
        
    Returns:
        (ArangoGraphDB 인스턴스, 통계 정보)
    """
    print("\n" + "="*60)
    print(" 하이브리드 지식 그래프 구축")
    print("="*60)
    
    # 1. ArangoDB 연결
    graph_db = ArangoGraphDB(
        host=arango_host,
        port=arango_port,
        password=arango_password,
        db_name=arango_db_name,
        reset=reset
    )
    
    if not graph_db.db:
        print("ArangoDB 연결 실패")
        return None, {}
    
    stats = {'phases': {}}
    
    # 2. 단계 1: 골든 데이터 시딩
    print("\n[단계 1/3] 골든 데이터(Ontology) 구축")
    print("-" * 40)
    
    loader = GoldenDataLoader(graph_db)
    golden_stats = loader.seed_to_arango(golden_data_path)
    stats['phases']['golden_data'] = golden_stats
    
    # 3. 단계 2: 엔티티 링커 인덱스 구축
    print("\n[단계 2/3] 엔티티 링커 구축")
    print("-" * 40)
    
    linker = EntityLinker(graph_db)
    linker.build_index()
    
    # 4. 단계 3: 사료 처리 (ICL + 엔티티 연결)
    print("\n[단계 3/3] 사료 처리 (ICL + Entity Linking)")
    print("-" * 40)
    
    extractor = InContextKnowledgeExtractor(llm_model=llm_model, entity_linker=linker)
    
    saryeo_stats = {'total_entities': 0, 'total_relations': 0}
    
    for saryeo_path in saryeo_paths:
        print(f"\n처리: {saryeo_path}")
        
        try:
            with open(saryeo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 문서 형식으로 변환
            if isinstance(data, list):
                documents = data
            elif isinstance(data, dict):
                documents = data.get('documents', [data])
            else:
                documents = []
            
            if not documents:
                continue
            
            # ICL + 엔티티 연결 추출
            entities, relations = extractor.extract_with_linking(documents)
            
            # ArangoDB에 삽입
            graph_db.insert_entities(entities)
            graph_db.insert_relations(relations)
            
            saryeo_stats['total_entities'] += len(entities)
            saryeo_stats['total_relations'] += len(relations)
            
        except Exception as e:
            print(f"  ⚠ 처리 실패: {e}")
    
    stats['phases']['saryeo'] = saryeo_stats
    
    # 5. 최종 통계
    final_stats = graph_db.get_statistics()
    stats['final'] = final_stats
    
    print("\n" + "="*60)
    print(" 하이브리드 지식 그래프 구축 완료!")
    print("="*60)
    print(f"  골든 데이터: 엔티티 {golden_stats.get('entities_loaded', 0)}개, 관계 {golden_stats.get('relations_loaded', 0)}개")
    print(f"  사료 추출: 엔티티 {saryeo_stats['total_entities']}개, 관계 {saryeo_stats['total_relations']}개")
    print(f"  최종 DB: 엔티티 {final_stats.get('entities_count', 0)}개, 관계 {final_stats.get('relations_count', 0)}개")
    
    return graph_db, stats

