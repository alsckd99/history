"""현재 ArangoDB 상태 확인 스크립트"""
from arango import ArangoClient

# ArangoDB 연결
client = ArangoClient(hosts='http://localhost:8530')
db = client.db('kg_encykorea', username='root', password='')

# 엔티티 컬렉션 확인
entities = db.collection('entities')
print(f"\n=== 엔티티 통계 ===")
print(f"총 엔티티 수: {entities.count()}")

# 샘플 엔티티 확인
print(f"\n=== 샘플 엔티티 (처음 10개) ===")
cursor = db.aql.execute("FOR e IN entities LIMIT 10 RETURN {name: e.name, type: e.type, key: e._key}")
for doc in cursor:
    print(f"  - {doc['name']} ({doc['type']}) → key: {doc['key']}")

# 특정 엔티티 검색
print(f"\n=== 특정 엔티티 검색 ===")
test_names = ['허균', '속동문선', '임진왜란', '이순신', '유민탄']
for name in test_names:
    cursor = db.aql.execute(
        "FOR e IN entities FILTER e.name == @name RETURN e",
        bind_vars={'name': name}
    )
    results = list(cursor)
    if results:
        e = results[0]
        print(f"  ✓ '{name}' 있음 → key: {e['_key']}, type: {e.get('type', 'N/A')}")
    else:
        # 부분 일치로 검색
        cursor2 = db.aql.execute(
            "FOR e IN entities FILTER CONTAINS(e.name, @name) LIMIT 3 RETURN e.name",
            bind_vars={'name': name}
        )
        similar = list(cursor2)
        if similar:
            print(f"  ✗ '{name}' 없음. 유사: {similar}")
        else:
            print(f"  ✗ '{name}' 없음")

# type별 엔티티 분포
print(f"\n=== type별 엔티티 수 (상위 15개) ===")
cursor = db.aql.execute("""
    FOR e IN entities
        COLLECT t = e.type WITH COUNT INTO cnt
        SORT cnt DESC
        LIMIT 15
        RETURN {type: t, count: cnt}
""")
for doc in cursor:
    t = doc['type'] if doc['type'] else '(없음/빈값)'
    print(f"  {t}: {doc['count']}개")

# 관계 컬렉션 확인
relations = db.collection('relations')
print(f"\n=== 관계 통계 ===")
print(f"총 관계 수: {relations.count()}")

# 샘플 관계 확인
if relations.count() > 0:
    print(f"\n=== 샘플 관계 (처음 5개) ===")
    cursor = db.aql.execute("FOR r IN relations LIMIT 5 RETURN {from: r.subject, rel: r.predicate, to: r.object}")
    for doc in cursor:
        print(f"  - {doc['from']} → {doc['rel']} → {doc['to']}")

print("\n완료!")

