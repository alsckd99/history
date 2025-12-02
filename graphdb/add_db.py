#!/usr/bin/env python3
"""
엔티티의 sources snippet에서 다른 엔티티를 찾아 '본문_언급' 관계 추가

동작:
1. 대상 엔티티들의 sources에서 snippet 추출
2. snippet을 분석하여 DB에 존재하는 엔티티 찾기
3. 찾은 엔티티와 '본문_언급' 관계 추가
"""

from arango import ArangoClient
from arango.http import DefaultHTTPClient
import hashlib
import re

# 설정
ARANGO_HOST = "http://localhost:8530"
ARANGO_USER = "root"
ARANGO_PASSWORD = ""
DATABASE = "knowledge_graph"

# 타임아웃 설정 (초)
REQUEST_TIMEOUT = 300

# 분석할 엔티티 목록
TARGET_ENTITIES = ["노량해전", "명나라", "진린"]

# 제외할 일반 단어 (관계 생성 제외)
EXCLUDE_WORDS = {
    "것", "수", "등", "때", "중", "후", "년", "월", "일", "명", "개", "곳",
    "이", "그", "저", "이것", "그것", "저것", "여기", "거기", "저기",
    "하다", "되다", "있다", "없다", "같다", "보다", "오다", "가다",
    "또한", "그리고", "하지만", "그러나", "따라서", "때문에",
    "대해", "통해", "위해", "의해", "관해"
}


def generate_key(name: str, category: str) -> str:
    """엔티티 키 생성"""
    key_source = f"{name}_{category}"
    hash_value = hashlib.sha256(key_source.encode('utf-8')).hexdigest()[:24]
    return f"K_{hash_value}"


def generate_relation_key(from_key: str, to_key: str, rel_type: str) -> str:
    """관계 키 생성"""
    key_source = f"{from_key}_{to_key}_{rel_type}"
    hash_value = hashlib.sha256(key_source.encode('utf-8')).hexdigest()[:24]
    return f"R_{hash_value}"


def find_entity_by_name(db, name: str):
    """이름으로 엔티티 찾기 (가장 sources가 많은 것 우선)"""
    
    query = """
    FOR e IN entities
        FILTER e.name == @name
        LET src_count = LENGTH(e.sources || [])
        SORT src_count DESC
        LIMIT 1
        RETURN e
    """
    
    cursor = db.aql.execute(query, bind_vars={'name': name})
    results = list(cursor)
    return results[0] if results else None


def find_entities_in_text_batch(db, text: str, exclude_names: set):
    """텍스트에서 DB에 존재하는 엔티티 찾기 (배치 쿼리로 최적화)"""
    
    if not text:
        return []
    
    # 2글자 이상 단어 추출 (한글, 한자)
    words = set()
    
    # 한글 단어 추출 (2~10글자)
    korean_words = re.findall(r'[가-힣]{2,10}', text)
    words.update(korean_words)
    
    # 한자 단어 추출 (2~10글자)
    hanja_words = re.findall(r'[一-龥]{2,10}', text)
    words.update(hanja_words)
    
    # 제외 단어 및 대상 엔티티 제거
    words = words - EXCLUDE_WORDS - exclude_names
    words = [w for w in words if len(w) >= 2]
    
    if not words:
        return []
    
    # 배치 쿼리로 한 번에 검색 (category가 '미분류'인 것 제외)
    query = """
    FOR name IN @names
        FOR e IN entities
            FILTER e.name == name
            FILTER e.category != '미분류' AND e.category != null AND e.category != ''
            COLLECT entity_name = e.name INTO groups = e
            LET best = FIRST(
                FOR g IN groups 
                    SORT LENGTH(g.sources || []) DESC 
                    RETURN g
            )
            RETURN best
    """
    
    try:
        cursor = db.aql.execute(query, bind_vars={'names': list(words)})
        return [e for e in cursor if e is not None]
    except Exception as e:
        print(f"    ⚠️ 배치 쿼리 오류: {e}")
        return []


def check_relation_exists(db, from_key: str, to_key: str, rel_type: str):
    """관계가 이미 존재하는지 확인"""
    
    from_id = f"entities/{from_key}"
    to_id = f"entities/{to_key}"
    
    query = """
    FOR r IN relations
        FILTER r._from == @from_id AND r._to == @to_id AND r.type == @rel_type
        LIMIT 1
        RETURN r
    """
    
    cursor = db.aql.execute(query, bind_vars={
        'from_id': from_id,
        'to_id': to_id,
        'rel_type': rel_type
    })
    results = list(cursor)
    return len(results) > 0


def add_relation(db, from_entity: dict, to_entity: dict, rel_type: str, source_doc: str = None):
    """관계 추가"""
    
    from_key = from_entity['_key']
    to_key = to_entity['_key']
    
    # 자기 자신 관계 제외
    if from_key == to_key:
        return False
    
    # 이미 존재하는 관계 제외
    if check_relation_exists(db, from_key, to_key, rel_type):
        return False
    
    rel_key = generate_relation_key(from_key, to_key, rel_type)
    
    relation = {
        '_key': rel_key,
        '_from': f"entities/{from_key}",
        '_to': f"entities/{to_key}",
        'type': rel_type,
        'predicate': rel_type
    }
    
    if source_doc:
        relation['source'] = source_doc
    
    try:
        db.collection('relations').insert(relation)
        return True
    except Exception as e:
        # 중복 키 등의 오류는 무시
        return False


def extract_relations_from_snippets(db, dry_run: bool = True):
    """엔티티의 sources snippet에서 다른 엔티티를 찾아 관계 추가"""
    
    print(f"\n📊 대상 엔티티: {TARGET_ENTITIES}")
    
    total_relations_added = 0
    total_snippets_processed = 0
    
    # 대상 엔티티 이름들 (자기 자신 제외용)
    exclude_names = set(TARGET_ENTITIES)
    
    for target_name in TARGET_ENTITIES:
        print(f"\n{'=' * 50}")
        print(f"🔍 '{target_name}' 처리 중...")
        
        # 엔티티 찾기
        target_entity = find_entity_by_name(db, target_name)
        
        if not target_entity:
            print(f"  ❌ 엔티티 '{target_name}' 찾을 수 없음")
            continue
        
        target_key = target_entity['_key']
        sources = target_entity.get('sources', [])
        
        print(f"  - key: {target_key}")
        print(f"  - type: {target_entity.get('type')}")
        print(f"  - category: {target_entity.get('category')}")
        print(f"  - sources 개수: {len(sources)}")
        
        if not sources:
            print(f"  ⚠️ sources 없음")
            continue
        
        # snippet에서 엔티티 추출
        found_entities_all = {}  # name -> entity
        snippet_count = 0
        total_sources = len(sources)
        
        for idx, src in enumerate(sources):
            snippet = src.get('snippet', '')
            if not snippet:
                continue
            
            snippet_count += 1
            source_doc = src.get('doc', '')
            
            # 진행 상황 표시 (10개마다)
            if snippet_count % 10 == 0:
                print(f"    진행: {snippet_count}/{total_sources} snippets...")
            
            # snippet에서 엔티티 찾기 (배치 쿼리 사용)
            found = find_entities_in_text_batch(db, snippet, exclude_names)
            
            for ent in found:
                ent_name = ent.get('name')
                if ent_name and ent_name not in found_entities_all:
                    found_entities_all[ent_name] = {
                        'entity': ent,
                        'source_doc': source_doc
                    }
        
        total_snippets_processed += snippet_count
        
        print(f"\n  📝 처리한 snippet: {snippet_count}개")
        print(f"  🔗 발견된 엔티티: {len(found_entities_all)}개")
        
        if found_entities_all:
            print(f"\n  발견된 엔티티 목록 (처음 20개):")
            for i, (name, info) in enumerate(list(found_entities_all.items())[:20]):
                ent = info['entity']
                print(f"    {i+1}. {name} (type: {ent.get('type')}, category: {ent.get('category', '')[:20]})")
        
        if dry_run:
            print(f"\n  ⚠️ DRY RUN - 관계 추가하지 않음")
            continue
        
        # 관계 추가
        relations_added = 0
        
        for name, info in found_entities_all.items():
            found_entity = info['entity']
            source_doc = info['source_doc']
            
            # 본문_언급 관계 추가
            if add_relation(db, target_entity, found_entity, '본문_언급', source_doc):
                relations_added += 1
        
        print(f"\n  ✅ 추가된 관계: {relations_added}개")
        total_relations_added += relations_added
    
    print(f"\n{'=' * 60}")
    print(f"📊 전체 결과:")
    print(f"  - 처리한 엔티티: {len(TARGET_ENTITIES)}개")
    print(f"  - 처리한 snippet: {total_snippets_processed}개")
    
    if dry_run:
        print(f"\n⚠️ DRY RUN 모드 - 실제 관계 추가하지 않음")
        print(f"실제 수정하려면: python add_db.py --execute")
    else:
        print(f"  - 추가된 관계: {total_relations_added}개")


def main():
    import sys
    
    dry_run = "--execute" not in sys.argv
    
    print("=" * 60)
    print("엔티티 snippet에서 관계 추출 스크립트")
    print("=" * 60)
    print(f"\n대상 엔티티: {TARGET_ENTITIES}")
    print(f"\n동작:")
    print(f"  1. 대상 엔티티의 sources에서 snippet 추출")
    print(f"  2. snippet에서 DB에 존재하는 엔티티 찾기")
    print(f"  3. 찾은 엔티티와 '본문_언급' 관계 추가")
    
    if dry_run:
        print("\n⚠️  DRY RUN 모드 (실제 수정 안함)")
        print("    실제 수정하려면: python add_db.py --execute\n")
    else:
        print("\n🔴 EXECUTE 모드 - 실제 데이터 수정됨!\n")
        confirm = input("계속하시겠습니까? (yes/no): ")
        if confirm.lower() != 'yes':
            print("취소됨")
            return
    
    # 타임아웃 설정된 HTTP 클라이언트
    http_client = DefaultHTTPClient(request_timeout=REQUEST_TIMEOUT)
    client = ArangoClient(hosts=ARANGO_HOST, http_client=http_client)
    
    try:
        db = client.db(DATABASE, username=ARANGO_USER, password=ARANGO_PASSWORD)
        print(f"\n[DB] {DATABASE} 연결됨")
        
        extract_relations_from_snippets(db, dry_run=dry_run)
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
