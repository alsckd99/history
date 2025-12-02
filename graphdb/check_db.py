#!/usr/bin/env python3
"""ArangoDB 엔티티 디버깅 스크립트 - 누락 원인 추적"""

import sys

# ============================================================
# 설정
# ============================================================
ARANGO_PORT = 8530  # 포트 설정
DB_NAME = "knowledge_graph"

# 검색할 키워드 (명령줄 인자 또는 기본값)
if len(sys.argv) > 1:
    SEARCH_KEYWORD = sys.argv[1]
else:
    SEARCH_KEYWORD = input("검색할 키워드를 입력하세요: ").strip() or "이순신"

print(f"\n🔍 검색 키워드: '{SEARCH_KEYWORD}'")
print("=" * 60)

# ============================================================
# 1단계: relations 컬렉션 및 관련 노드 확인
# ============================================================
print("\n" + "=" * 60)
print("[1] Relations 컬렉션 및 관련 노드 확인")
print("=" * 60)

try:
    from arango import ArangoClient
    
    client = ArangoClient(hosts=f'http://localhost:{ARANGO_PORT}')
    db = client.db(DB_NAME, username='root', password='')
    print(f"DB '{DB_NAME}' (포트 {ARANGO_PORT}) 연결됨")
    
    # relations 컬렉션 확인
    if db.has_collection('relations'):
        relations = db.collection('relations')
        print(f"\nrelations 컬렉션: {relations.count()}개")
        
        # 샘플 relation 확인
        sample = list(db.aql.execute('FOR r IN relations LIMIT 5 RETURN {from: r._from, to: r._to, type: r.type}'))
        print('\n샘플 relations:')
        for r in sample:
            print(f"  {r}")
    else:
        print('\n⚠️ relations 컬렉션 없음!')
    
    # 키워드 관련 엔티티 확인
    print('\n' + '-' * 40)
    print(f"'{SEARCH_KEYWORD}' 관련 노드 검색")
    print('-' * 40)
    
    query = f"""
    FOR e IN entities
        FILTER e.name == '{SEARCH_KEYWORD}' OR e.canonical_name == '{SEARCH_KEYWORD}'
        LIMIT 1
        RETURN e
    """
    result = list(db.aql.execute(query))
    
    if result:
        entity = result[0]
        entity_key = entity['_key']
        print(f"✅ '{SEARCH_KEYWORD}' 엔티티 찾음: _key={entity_key}")
        print(f"   name: {entity.get('name')}")
        print(f"   type: {entity.get('type')}")
        
        # 관계 검색 (나가는 관계)
        rel_out_query = f"""
        FOR r IN relations
            FILTER r._from == 'entities/{entity_key}'
            LIMIT 10
            RETURN {{to: r._to, type: r.type, predicate: r.predicate}}
        """
        rels_out = list(db.aql.execute(rel_out_query))
        print(f"\n나가는 관계 ({SEARCH_KEYWORD} -> ?): {len(rels_out)}개")
        for r in rels_out[:5]:
            print(f"  -> {r}")
        
        # 관계 검색 (들어오는 관계)
        rel_in_query = f"""
        FOR r IN relations
            FILTER r._to == 'entities/{entity_key}'
            LIMIT 10
            RETURN {{from: r._from, type: r.type, predicate: r.predicate}}
        """
        rels_in = list(db.aql.execute(rel_in_query))
        print(f"\n들어오는 관계 (? -> {SEARCH_KEYWORD}): {len(rels_in)}개")
        for r in rels_in[:5]:
            print(f"  <- {r}")
        
        # 이웃 노드 직접 조회
        neighbor_query = f"""
        FOR v, e IN 1..1 ANY 'entities/{entity_key}' relations
            LIMIT 10
            RETURN {{neighbor_name: v.name, neighbor_type: v.type, relation: e.type}}
        """
        neighbors = list(db.aql.execute(neighbor_query))
        print(f"\n이웃 노드 (graph traversal): {len(neighbors)}개")
        for n in neighbors[:10]:
            print(f"  - {n}")
            
    else:
        print(f'❌ \'{SEARCH_KEYWORD}\' 엔티티 없음!')
        
        # 비슷한 이름 검색
        similar_query = f"""
        FOR e IN entities
            FILTER CONTAINS(LOWER(e.name), LOWER('{SEARCH_KEYWORD}'))
            LIMIT 10
            RETURN {{name: e.name, type: e.type, key: e._key}}
        """
        similar = list(db.aql.execute(similar_query))
        if similar:
            print('\n비슷한 이름의 엔티티:')
            for s in similar:
                print(f"  - {s}")
    
    # graph_db.py의 _sanitize_key 방식으로 키 생성 테스트
    print('\n' + '-' * 40)
    print('graph_db.py 키 생성 방식 테스트')
    print('-' * 40)
    
    import hashlib
    import re
    
    def sanitize_key_sha256(name: str) -> str:
        """graph_db.py의 _sanitize_key 방식 (SHA256 24자)"""
        if not name or not isinstance(name, str):
            return 'unknown_' + hashlib.sha256(str(id(name)).encode()).hexdigest()[:8]
        normalized = name.replace(' ', '_')
        ascii_only = re.sub(r'[^a-zA-Z0-9_-]', '', normalized)
        alphanumeric_only = re.sub(r'[^a-zA-Z0-9]', '', ascii_only)
        if alphanumeric_only and len(alphanumeric_only) >= 3:
            if not ascii_only[0].isalpha():
                ascii_only = 'K_' + ascii_only
            return ascii_only[:128]
        hash_part = hashlib.sha256(name.encode('utf-8')).hexdigest()[:24]
        return f"K_{hash_part}"
    
    def sanitize_key_md5(name: str) -> str:
        """MD5 24자 방식"""
        hash_part = hashlib.md5(name.encode('utf-8')).hexdigest()[:24]
        return f"K_{hash_part}"
    
    # 현재 사용할 함수
    sanitize_key = sanitize_key_sha256
    
    test_name = SEARCH_KEYWORD
    
    # 실제 DB에서 키워드의 키와 category 확인
    actual_query = f"""
    FOR e IN entities
        FILTER e.name == '{SEARCH_KEYWORD}'
        LIMIT 5
        RETURN {{key: e._key, name: e.name, category: e.category, type: e.type}}
    """
    actual_entities = list(db.aql.execute(actual_query))
    
    if actual_entities:
        print(f"'{test_name}' 엔티티 목록 ({len(actual_entities)}개):")
        for ent in actual_entities:
            actual_key = ent['key']
            category = ent.get('category', '')
            ent_type = ent.get('type', '')
            
            # 키 생성 시 category 또는 type 포함 여부에 따라 테스트
            if category:
                expected_key = sanitize_key(f"{test_name}_{category}")
                key_source = f"{test_name}_{category}"
            elif ent_type and ent_type != '미분류':
                expected_key = sanitize_key(f"{test_name}_{ent_type}")
                key_source = f"{test_name}_{ent_type}"
            else:
                expected_key = sanitize_key(test_name)
                key_source = test_name
            
            print(f"\n  - category: '{category}', type: '{ent_type}'")
            print(f"    실제 키: {actual_key}")
            print(f"    예상 키: {expected_key} (from: '{key_source}')")
            
            if expected_key == actual_key:
                print("    ✅ 일치!")
            else:
                print("    ❌ 불일치!")
                
                # 단순 이름으로만 키 생성했을 경우도 테스트
                simple_key = sanitize_key(test_name)
                print(f"    (참고) 단순 이름 키: {simple_key}")
    else:
        print(f"'{test_name}' 엔티티 없음!")
    
    # 다양한 키 생성 방식 테스트
    print('\n' + '-' * 40)
    print('다양한 키 생성 방식 비교')
    print('-' * 40)
    
    # 실제 DB에서 해당 키워드의 키 가져오기
    actual_key = actual_entities[0]['key'] if actual_entities else "없음"
    
    test_inputs = [
        SEARCH_KEYWORD,
        f"{SEARCH_KEYWORD}_인물",
        f"{SEARCH_KEYWORD}_역사/조선시대사", 
        f"{SEARCH_KEYWORD}_문헌",
        f"{SEARCH_KEYWORD}_사건",
    ]
    
    print(f"실제 DB 키: {actual_key}")
    print()
    
    for test_input in test_inputs:
        sha256_key = sanitize_key_sha256(test_input)
        md5_key = sanitize_key_md5(test_input)
        
        sha256_match = "✅" if sha256_key == actual_key else ""
        md5_match = "✅" if md5_key == actual_key else ""
        
        print(f"'{test_input}':")
        print(f"  SHA256: {sha256_key} {sha256_match}")
        print(f"  MD5:    {md5_key} {md5_match}")

except Exception as e:
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 2단계: 통합 그래프에서 비-백과사전 엔티티 확인
# ============================================================
print("\n" + "=" * 60)
print("[2] 통합 그래프에서 소스별 엔티티 확인")
print("=" * 60)

try:
    from arango import ArangoClient
    
    # 통합 그래프 DB 연결
    client = ArangoClient(hosts=f'http://localhost:{ARANGO_PORT}')
    sys_db = client.db('_system', username='root', password='')
    
    global_db_name = DB_NAME
    if sys_db.has_database(global_db_name):
        global_db = client.db(global_db_name, username='root', password='')
        print(f"통합 그래프 DB '{global_db_name}' 연결됨")
        
        if global_db.has_collection('entities'):
            entities_col = global_db.collection('entities')
            total_count = entities_col.count()
            print(f"총 엔티티 수: {total_count}")
            
            # 소스별 엔티티 수 집계
            source_query = """
            FOR e IN entities
                LET source_types = (
                    FOR s IN (e.sources || [])
                        RETURN s.type
                )
                LET primary_source = LENGTH(source_types) > 0 ? source_types[0] : "unknown"
                COLLECT source = primary_source WITH COUNT INTO cnt
                SORT cnt DESC
                RETURN {source: source, count: cnt}
            """
            source_stats = list(global_db.aql.execute(source_query))
            
            print("\n소스별 엔티티 수:")
            ency_count = 0
            non_ency_count = 0
            for stat in source_stats:
                source = stat['source']
                count = stat['count']
                if '한국민족문화대백과사전' in str(source):
                    ency_count += count
                    marker = "📚"
                else:
                    non_ency_count += count
                    marker = "📄"
                print(f"  {marker} {source}: {count}개")
            
            print("\n요약:")
            print(f"  - 한국민족문화대백과사전: {ency_count}개")
            print(f"  - 기타 소스: {non_ency_count}개")
            
            # 비-백과사전 엔티티 샘플 조회
            if non_ency_count > 0:
                non_ency_query = """
                FOR e IN entities
                    LET source_types = (
                        FOR s IN (e.sources || [])
                            RETURN s.type
                    )
                    LET has_ency = LENGTH(
                        FOR t IN source_types
                            FILTER CONTAINS(t, "한국민족문화대백과사전")
                            RETURN 1
                    ) > 0
                    FILTER !has_ency
                    LIMIT 20
                    RETURN {
                        name: e.name,
                        type: e.type,
                        sources: e.sources,
                        key: e._key
                    }
                """
                non_ency_samples = list(global_db.aql.execute(non_ency_query))
                
                print(f"\n비-백과사전 엔티티 샘플 ({len(non_ency_samples)}개):")
                for ent in non_ency_samples:
                    sources = ent.get('sources', [])
                    source_info = []
                    for s in sources[:2]:
                        if isinstance(s, dict):
                            source_info.append(f"{s.get('type', '?')}:{s.get('doc', '?')[:20]}")
                    print(f"  - {ent['name']} ({ent.get('type', '?')}) ← {', '.join(source_info)}")
        else:
            print("entities 컬렉션 없음")
    else:
        print(f"통합 그래프 DB '{global_db_name}' 없음")
        
except Exception as e:
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()


# ============================================================
# 3단계: 키워드 엔티티의 category와 관련 노드 확인
# ============================================================
print("\n" + "=" * 60)
print(f"[3] '{SEARCH_KEYWORD}' 엔티티의 category와 관련 노드 확인")
print("=" * 60)

try:
    # 키워드 이름을 가진 모든 엔티티 조회 (정확히 일치 + 포함)
    keyword_query = f"""
    FOR e IN entities
        FILTER e.name == '{SEARCH_KEYWORD}' OR CONTAINS(e.name, '{SEARCH_KEYWORD}')
        RETURN e
    """
    keyword_entities = list(db.aql.execute(keyword_query))
    
    # 정확히 키워드인 것만 먼저 분리
    exact_match = [e for e in keyword_entities if e.get('name') == SEARCH_KEYWORD]
    contains_match = [e for e in keyword_entities if e.get('name') != SEARCH_KEYWORD]
    
    print(f"\n정확히 '{SEARCH_KEYWORD}' 이름의 엔티티: {len(exact_match)}개")
    print(f"'{SEARCH_KEYWORD}' 포함 엔티티: {len(contains_match)}개")
    
    if contains_match:
        print(f"\n'{SEARCH_KEYWORD}' 포함 엔티티 목록:")
        for e in contains_match[:10]:
            print(f"  - {e.get('name')} (category: {e.get('category')}, type: {e.get('type')}, key: {e.get('_key')})")
    
    # 정확히 일치하는 것만 상세 분석
    keyword_entities_exact = exact_match
    
    print(f"\n'{SEARCH_KEYWORD}' 이름의 엔티티: {len(keyword_entities_exact)}개")
    
    for idx, entity in enumerate(keyword_entities_exact):
        entity_key = entity.get('_key', '')
        name = entity.get('name', '')
        category = entity.get('category', '없음')
        ent_type = entity.get('type', '없음')
        sources = entity.get('sources', [])
        source_count = len(sources) if isinstance(sources, list) else 0
        
        print(f"\n[{idx + 1}] '{SEARCH_KEYWORD}' 엔티티")
        print(f"    _key: {entity_key}")
        print(f"    category: {category}")
        print(f"    type: {ent_type}")
        print(f"    sources 개수: {source_count}")
        
        # sources 상세 확인 (어떤 소스에서 왔는지)
        if sources:
            print("\n    📚 sources 상세 (처음 10개):")
            # 소스 타입별로 그룹화
            source_types = {}
            for src in sources:
                if isinstance(src, dict):
                    src_type = src.get('type', 'unknown')
                    if src_type not in source_types:
                        source_types[src_type] = []
                    source_types[src_type].append(src.get('doc', '?'))
            
            for src_type, docs in source_types.items():
                print(f"      [{src_type}]: {len(docs)}개")
                for doc in docs[:3]:
                    print(f"        - {doc}")
                if len(docs) > 3:
                    print(f"        ... 외 {len(docs) - 3}개")
            
            # 백과사전 소스가 있는지 확인
            has_encyclopedia = any('한국민족문화대백과사전' in str(src.get('type', '')) 
                                   for src in sources if isinstance(src, dict))
            if has_encyclopedia:
                print("\n    ✅ 한국민족문화대백과사전 소스 있음!")
                # 백과사전 소스의 category 확인
                for src in sources:
                    if isinstance(src, dict) and '한국민족문화대백과사전' in str(src.get('type', '')):
                        src_category = src.get('주제분류', src.get('category', '없음'))
                        print(f"       백과사전 원본 카테고리: {src_category}")
                        break
            else:
                print("\n    ⚠️ 한국민족문화대백과사전 소스 없음")
        
        # 해당 엔티티의 관련 노드(이웃) 조회
        neighbor_query = f"""
        FOR v, e IN 1..1 ANY 'entities/{entity_key}' relations
            LIMIT 20
            RETURN {{
                neighbor_name: v.name, 
                neighbor_category: v.category,
                neighbor_type: v.type, 
                neighbor_sources_count: LENGTH(v.sources || []),
                relation_type: e.type
            }}
        """
        neighbors = list(db.aql.execute(neighbor_query))
        
        print(f"    관련 노드: {len(neighbors)}개")
        
        if neighbors:
            # 관련 노드를 sources 개수 기준으로 정렬 (미분류 후순위)
            def sort_key(n):
                is_unclassified = (n.get('neighbor_category') == '미분류' or 
                                   n.get('neighbor_type') == '미분류')
                source_count = n.get('neighbor_sources_count', 0)
                return (1 if is_unclassified else 0, -source_count)
            
            sorted_neighbors = sorted(neighbors, key=sort_key)
            
            print("    (정렬 기준: 미분류 아님 + sources 많은 순)")
            for n in sorted_neighbors[:10]:
                unclass_mark = "⚪" if (n.get('neighbor_category') == '미분류' or 
                                        n.get('neighbor_type') == '미분류') else "🔵"
                print(f"      {unclass_mark} {n['neighbor_name']} "
                      f"(category: {n.get('neighbor_category', '없음')}, "
                      f"type: {n.get('neighbor_type', '없음')}, "
                      f"sources: {n.get('neighbor_sources_count', 0)}개, "
                      f"relation: {n.get('relation_type', '?')})")

except Exception as e:
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 4단계: 모든 category 종류 확인
# ============================================================
print("\n" + "=" * 60)
print("[4] 모든 category 종류 확인")
print("=" * 60)

try:
    # category 종류 및 개수 집계
    category_query = """
    FOR e IN entities
        COLLECT category = (e.category || "없음") WITH COUNT INTO cnt
        SORT cnt DESC
        RETURN {category: category, count: cnt}
    """
    category_stats = list(db.aql.execute(category_query))
    
    print(f"\n총 {len(category_stats)}개 category 종류:")
    print()
    
    for stat in category_stats:
        category = stat['category']
        count = stat['count']
        print(f"  {count:>6}개  |  {category}")

except Exception as e:
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 5단계: 모든 type 종류 확인
# ============================================================
print("\n" + "=" * 60)
print("[5] 모든 type 종류 확인")
print("=" * 60)

try:
    # type 종류 및 개수 집계
    type_query = """
    FOR e IN entities
        COLLECT type = (e.type || "없음") WITH COUNT INTO cnt
        SORT cnt DESC
        RETURN {type: type, count: cnt}
    """
    type_stats = list(db.aql.execute(type_query))
    
    print(f"\n총 {len(type_stats)}개 type 종류:")
    print()
    
    for stat in type_stats:
        ent_type = stat['type']
        count = stat['count']
        print(f"  {count:>6}개  |  {ent_type}")

except Exception as e:
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("완료!")
print("=" * 60)
