#!/usr/bin/env python3
"""
ArangoDB 엔티티 sources 병합 스크립트

특정 엔티티의 sources를 다른 엔티티로 이동/병합

예시:
- 원본: "노량 해전 승리" (type: 사건)
- 대상: "노량해전" (category: 역사/조선시대사)

원본 엔티티의 sources를 대상 엔티티로 병합
"""

from arango import ArangoClient

# 설정
ARANGO_HOST = "http://localhost:8530"
ARANGO_USER = "root"
ARANGO_PASSWORD = ""
DATABASE = "knowledge_graph"

# 이동할 엔티티 정보
SOURCE_ENTITY = {
    "name": "이번",
    "category": "역사/조선시대사"  # 또는 None으로 설정하면 category 무시
}

TARGET_ENTITY = {
    "name": "이번",
    "category": "종교·철학/유교"  # 또는 None으로 설정하면 category 무시
}


def find_entity(db, name: str, type_or_category: str = None, field: str = "type"):
    """엔티티 검색"""
    if type_or_category:
        query = f"""
        FOR e IN entities
            FILTER e.name == @name AND e.{field} == @value
            RETURN e
        """
        cursor = db.aql.execute(query, bind_vars={'name': name, 'value': type_or_category})
    else:
        query = """
        FOR e IN entities
            FILTER e.name == @name
            RETURN e
        """
        cursor = db.aql.execute(query, bind_vars={'name': name})
    
    results = list(cursor)
    return results


def merge_sources(db, source_entity: dict, target_entity: dict, dry_run: bool = True):
    """원본 엔티티의 sources를 대상 엔티티로 병합
    
    주의: 한국민족문화대백과사전 sources는 제외하고 병합
    """
    
    source_key = source_entity['_key']
    target_key = target_entity['_key']
    
    source_sources = source_entity.get('sources', [])
    target_sources = target_entity.get('sources', [])
    
    print(f"\n  원본 엔티티: {source_entity.get('name')} (key: {source_key})")
    print(f"    - type: {source_entity.get('type')}")
    print(f"    - category: {source_entity.get('category')}")
    print(f"    - sources 개수: {len(source_sources)}")
    
    print(f"\n  대상 엔티티: {target_entity.get('name')} (key: {target_key})")
    print(f"    - type: {target_entity.get('type')}")
    print(f"    - category: {target_entity.get('category')}")
    print(f"    - sources 개수: {len(target_sources)}")
    
    if not source_sources:
        print(f"\n  ⚠️ 원본 엔티티에 sources가 없음")
        return False
    
    # 한국민족문화대백과사전 sources 분리
    encyclopedia_sources = []
    other_sources = []
    
    for src in source_sources:
        src_type = src.get('type', '') or ''
        src_doc = src.get('doc', '') or ''
        
        # 한국민족문화대백과사전 판별
        if '한국민족문화대백과사전' in src_type or '한국민족문화대백과사전' in src_doc:
            encyclopedia_sources.append(src)
        else:
            other_sources.append(src)
    
    print(f"\n  원본 sources 분류:")
    print(f"    - 한국민족문화대백과사전: {len(encyclopedia_sources)}개 (제외됨)")
    print(f"    - 기타 sources: {len(other_sources)}개 (이동 대상)")
    
    # sources 샘플 출력
    if other_sources:
        print(f"\n  이동할 sources 샘플 (처음 3개):")
        for src in other_sources[:3]:
            print(f"    - doc: {src.get('doc', '(없음)')}")
            print(f"      type: {src.get('type', '(없음)')}")
            snippet = src.get('snippet', '')
            if snippet:
                print(f"      snippet: {snippet[:50]}...")
    
    if encyclopedia_sources:
        print(f"\n  ⚠️ 원본에 남을 한국민족문화대백과사전 sources:")
        for src in encyclopedia_sources[:2]:
            print(f"    - doc: {src.get('doc', '(없음)')}")
    
    if not other_sources:
        print(f"\n  ⚠️ 이동할 sources가 없음 (모두 한국민족문화대백과사전)")
        return False
    
    # 중복 제거하면서 병합
    existing_docs = set()
    for src in target_sources:
        doc = src.get('doc', '')
        snippet = src.get('snippet', '')[:50] if src.get('snippet') else ''
        existing_docs.add((doc, snippet))
    
    new_target_sources = target_sources.copy()
    added_count = 0
    
    for src in other_sources:
        doc = src.get('doc', '')
        snippet = src.get('snippet', '')[:50] if src.get('snippet') else ''
        
        if (doc, snippet) not in existing_docs:
            new_target_sources.append(src)
            existing_docs.add((doc, snippet))
            added_count += 1
    
    print(f"\n  📊 병합 결과:")
    print(f"    - 대상 기존 sources: {len(target_sources)}개")
    print(f"    - 추가될 sources: {added_count}개")
    print(f"    - 대상 최종 sources: {len(new_target_sources)}개")
    print(f"    - 원본 남을 sources: {len(encyclopedia_sources)}개 (한국민족문화대백과사전)")
    
    if dry_run:
        print(f"\n  ⚠️ DRY RUN 모드 - 실제 수정하지 않음")
        return True
    
    # 실제 수정
    try:
        # 1. 대상 엔티티에 sources 병합
        db.collection('entities').update({
            '_key': target_key,
            'sources': new_target_sources
        })
        print(f"\n  ✅ 대상 엔티티 sources 업데이트 완료")
        
        # 2. 원본 엔티티의 sources를 한국민족문화대백과사전만 남김
        db.collection('entities').update({
            '_key': source_key,
            'sources': encyclopedia_sources
        })
        print(f"  ✅ 원본 엔티티 sources 업데이트 (한국민족문화대백과사전만 유지)")
        
        # 3. 원본 엔티티는 삭제하지 않음 (한국민족문화대백과사전 정보 유지)
        print(f"  ℹ️ 원본 엔티티는 삭제하지 않음 (백과사전 정보 유지)")
        
        return True
        
    except Exception as e:
        print(f"\n  ❌ 오류: {e}")
        return False


def main():
    import sys
    
    dry_run = "--execute" not in sys.argv
    
    print("=" * 60)
    print("엔티티 Sources 병합 스크립트")
    print("=" * 60)
    
    print(f"\n원본: '{SOURCE_ENTITY['name']}' (category: {SOURCE_ENTITY.get('category', 'any')})")
    print(f"대상: '{TARGET_ENTITY['name']}' (category: {TARGET_ENTITY.get('category', 'any')})")
    
    if dry_run:
        print("\n⚠️  DRY RUN 모드 (실제 수정 안함)")
        print("    실제 수정하려면: python fix_db.py --execute\n")
    else:
        print("\n🔴 EXECUTE 모드 - 실제 데이터 수정됨!\n")
        confirm = input("계속하시겠습니까? (yes/no): ")
        if confirm.lower() != 'yes':
            print("취소됨")
            return
    
    client = ArangoClient(hosts=ARANGO_HOST)
    
    try:
        db = client.db(DATABASE, username=ARANGO_USER, password=ARANGO_PASSWORD)
        print(f"\n[DB] {DATABASE} 연결됨")
        
        # 원본 엔티티 검색
        print(f"\n{'=' * 40}")
        print("원본 엔티티 검색 중...")
        source_results = find_entity(
            db, 
            SOURCE_ENTITY['name'], 
            SOURCE_ENTITY.get('category'),
            'category'
        )
        
        if not source_results:
            print(f"  ❌ 원본 엔티티 '{SOURCE_ENTITY['name']}' 찾을 수 없음")
            
            # 유사한 이름 검색
            print(f"\n  유사한 이름 검색:")
            query = """
            FOR e IN entities
                FILTER CONTAINS(e.name, @name) OR CONTAINS(@name, e.name)
                LIMIT 5
                RETURN { name: e.name, type: e.type, category: e.category, key: e._key }
            """
            cursor = db.aql.execute(query, bind_vars={'name': SOURCE_ENTITY['name']})
            for r in cursor:
                print(f"    - {r['name']} (type: {r['type']}, category: {r['category']})")
            return
        
        if len(source_results) > 1:
            print(f"  ⚠️ 원본 엔티티가 {len(source_results)}개 발견됨:")
            for r in source_results:
                print(f"    - {r['name']} (type: {r.get('type')}, category: {r.get('category')}, key: {r['_key']})")
            print(f"  첫 번째 엔티티 사용")
        
        source_entity = source_results[0]
        
        # 대상 엔티티 검색
        print(f"\n{'=' * 40}")
        print("대상 엔티티 검색 중...")
        target_results = find_entity(
            db,
            TARGET_ENTITY['name'],
            TARGET_ENTITY.get('category'),
            'category'
        )
        
        if not target_results:
            print(f"  ❌ 대상 엔티티 '{TARGET_ENTITY['name']}' 찾을 수 없음")
            
            # 유사한 이름 검색
            print(f"\n  유사한 이름 검색:")
            query = """
            FOR e IN entities
                FILTER CONTAINS(e.name, @name) OR CONTAINS(@name, e.name)
                LIMIT 5
                RETURN { name: e.name, type: e.type, category: e.category, key: e._key }
            """
            cursor = db.aql.execute(query, bind_vars={'name': TARGET_ENTITY['name']})
            for r in cursor:
                print(f"    - {r['name']} (type: {r['type']}, category: {r['category']})")
            return
        
        if len(target_results) > 1:
            print(f"  ⚠️ 대상 엔티티가 {len(target_results)}개 발견됨:")
            for r in target_results:
                print(f"    - {r['name']} (type: {r.get('type')}, category: {r.get('category')}, key: {r['_key']})")
            print(f"  첫 번째 엔티티 사용")
        
        target_entity = target_results[0]
        
        # Sources 병합
        print(f"\n{'=' * 40}")
        print("Sources 병합")
        print('=' * 40)
        
        success = merge_sources(db, source_entity, target_entity, dry_run=dry_run)
        
        if success:
            print(f"\n{'=' * 60}")
            if dry_run:
                print("✅ DRY RUN 완료 - 실제 수정하려면 --execute 옵션 사용")
            else:
                print("✅ 병합 완료!")
            print("=" * 60)
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
