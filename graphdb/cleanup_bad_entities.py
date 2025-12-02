#!/usr/bin/env python3
"""
잘못 추출된 엔티티 정리 스크립트

문제:
- LLM이 "이순신과 함께 진을 쳤음" 등 문장을 엔티티 이름으로 잘못 추출함
- 실제 엔티티 정보는 sources.snippet에 있음

해결:
1. 문장 형태의 잘못된 엔티티 식별
2. sources.snippet에서 실제 엔티티 추출 시도
3. DB에 존재하는 엔티티면 → sources 병합
4. DB에 없으면 → 해당 잘못된 엔티티 삭제
"""

from arango import ArangoClient
import re

# 설정
ARANGO_HOST = "http://localhost:8530"
ARANGO_USER = "root"
ARANGO_PASSWORD = ""
DATABASE = "knowledge_graph"


def find_bad_entities(db):
    """잘못 추출된 엔티티 찾기"""
    
    query = """
    FOR e IN entities
        LET name_len = LENGTH(e.name)
        LET space_count = LENGTH(e.name) - LENGTH(SUBSTITUTE(e.name, ' ', ''))
        LET is_sentence = (
            REGEX_TEST(e.name, "(했|됐|함|됨|음|임|다|요|죠|네|군|나|라|고|며|서|니|면|까)$") OR
            REGEX_TEST(e.name, "(하였|되었|이다|였다|있다|없다|한다)") OR
            space_count >= 3 OR
            (name_len >= 10 AND space_count >= 2) OR
            STARTS_WITH(e.name, '"') OR
            STARTS_WITH(e.name, "'")
        )
        FILTER e.type == '미분류' AND is_sentence
        SORT space_count DESC, name_len DESC
        RETURN e
    """
    
    cursor = db.aql.execute(query)
    return list(cursor)


def extract_entities_from_snippet(snippet: str) -> list:
    """snippet에서 엔티티 후보 추출
    
    한국 역사 관련 고유명사 패턴:
    - 인물: 2-4글자 한글 이름 (이순신, 류성룡, 권율)
    - 지명: 2-6글자 (한산도, 부산포, 평양성)
    - 관직: 통제사, 영의정, 병마절도사 등
    """
    if not snippet:
        return []
    
    candidates = []
    
    # 1. 인물명 패턴 (성+이름, 2-4글자)
    # 한국 성씨 목록
    surnames = '김이박최정강조윤장임한오서신권황안송류홍전고문양손배백허남심노정하곽성차주우구신임나전민유진지엄채원천방공강현변함태'
    person_pattern = rf'([{surnames}][가-힣]{{1,3}})'
    for match in re.finditer(person_pattern, snippet):
        name = match.group(1)
        if 2 <= len(name) <= 4:
            candidates.append(('인물', name))
    
    # 2. 지명 패턴 (X도, X산, X성, X포, X진 등)
    location_suffixes = ['도', '산', '성', '포', '진', '관', '루', '정', '원', '궁', '사', '읍', '군', '현', '부']
    for suffix in location_suffixes:
        pattern = rf'([가-힣]{{1,5}}{suffix})'
        for match in re.finditer(pattern, snippet):
            loc = match.group(1)
            if 2 <= len(loc) <= 6:
                candidates.append(('지명', loc))
    
    # 3. 관직명
    positions = ['통제사', '영의정', '좌의정', '우의정', '병마절도사', '수군절도사', 
                 '도원수', '부원수', '판서', '참판', '참의', '대사헌', '대사간',
                 '좌상', '우상', '정승', '감사', '목사', '부사', '현감', '현령']
    for pos in positions:
        if pos in snippet:
            candidates.append(('제도', pos))
    
    # 4. 특정 사건/전투
    events = ['임진왜란', '정유재란', '한산도대첩', '명량해전', '노량해전', 
              '행주대첩', '진주성전투', '부산포해전', '사천해전', '당포해전']
    for event in events:
        if event in snippet:
            candidates.append(('사건', event))
    
    # 중복 제거
    seen = set()
    unique = []
    for type_, name in candidates:
        if name not in seen:
            seen.add(name)
            unique.append((type_, name))
    
    return unique


def find_matching_entity(db, name: str, entity_type: str = None):
    """DB에서 일치하는 엔티티 찾기"""
    
    if entity_type:
        query = """
        FOR e IN entities
            FILTER e.name == @name AND e.type != '미분류'
            SORT LENGTH(e.sources || []) DESC
            LIMIT 1
            RETURN e
        """
    else:
        query = """
        FOR e IN entities
            FILTER e.name == @name AND e.type != '미분류'
            SORT LENGTH(e.sources || []) DESC
            LIMIT 1
            RETURN e
        """
    
    cursor = db.aql.execute(query, bind_vars={'name': name})
    results = list(cursor)
    return results[0] if results else None


def cleanup_entities(db, dry_run: bool = True):
    """잘못된 엔티티 정리"""
    
    bad_entities = find_bad_entities(db)
    print(f"\n📊 잘못 추출된 엔티티: {len(bad_entities)}개")
    
    if not bad_entities:
        print("✅ 정리할 엔티티 없음")
        return
    
    # 분석 결과
    can_merge = []  # 병합 가능
    to_delete = []  # 삭제 대상 (매칭 엔티티 없음)
    
    print(f"\n🔍 snippet에서 엔티티 추출 중...")
    
    for ent in bad_entities:
        sources = ent.get('sources', [])
        extracted = []
        
        # 모든 sources의 snippet에서 엔티티 추출
        for src in sources:
            snippet = src.get('snippet', '')
            candidates = extract_entities_from_snippet(snippet)
            extracted.extend(candidates)
        
        # 추출된 엔티티 중 DB에 존재하는 것 찾기
        matched = None
        for entity_type, name in extracted:
            target = find_matching_entity(db, name, entity_type)
            if target:
                matched = target
                break
        
        if matched:
            can_merge.append({
                'bad': ent,
                'target': matched,
                'extracted': extracted[:3]  # 상위 3개만
            })
        else:
            to_delete.append({
                'bad': ent,
                'extracted': extracted[:3]
            })
    
    # 결과 출력
    print(f"\n📊 분석 결과:")
    print(f"  - 병합 가능: {len(can_merge)}개")
    print(f"  - 삭제 대상 (매칭 없음): {len(to_delete)}개")
    
    # 병합 가능 샘플
    print(f"\n✅ 병합 가능 샘플 (처음 10개):")
    for item in can_merge[:10]:
        bad = item['bad']
        target = item['target']
        print(f"  - '{bad['name'][:30]}...'")
        print(f"    → 병합 대상: '{target['name']}' (type: {target.get('type')})")
    
    # 삭제 대상 샘플
    print(f"\n❌ 삭제 대상 샘플 (처음 10개):")
    for item in to_delete[:10]:
        bad = item['bad']
        extracted = item['extracted']
        print(f"  - '{bad['name'][:40]}...'")
        if extracted:
            print(f"    추출된 후보: {[n for _, n in extracted]}")
        else:
            print(f"    추출된 후보: 없음")
    
    if dry_run:
        print(f"\n⚠️ DRY RUN 모드 - 실제 수정하지 않음")
        print(f"실제 수정하려면: python cleanup_bad_entities.py --execute")
        return
    
    # 실제 정리
    print(f"\n🔧 정리 중...")
    merged_count = 0
    deleted_count = 0
    
    # 1. 병합 처리
    for item in can_merge:
        try:
            bad = item['bad']
            target = item['target']
            
            # sources 병합
            bad_sources = bad.get('sources', [])
            target_sources = target.get('sources', [])
            
            if bad_sources:
                new_sources = target_sources + bad_sources
                db.collection('entities').update({
                    '_key': target['_key'],
                    'sources': new_sources
                })
            
            # 잘못된 엔티티 삭제
            db.collection('entities').delete(bad['_key'])
            merged_count += 1
            
            if merged_count % 100 == 0:
                print(f"  병합 진행: {merged_count}/{len(can_merge)}")
                
        except Exception as e:
            print(f"  ❌ 병합 오류 ({bad['name'][:20]}): {e}")
    
    # 2. 삭제 처리
    for item in to_delete:
        try:
            bad = item['bad']
            db.collection('entities').delete(bad['_key'])
            deleted_count += 1
            
            if deleted_count % 100 == 0:
                print(f"  삭제 진행: {deleted_count}/{len(to_delete)}")
                
        except Exception as e:
            print(f"  ❌ 삭제 오류: {e}")
    
    print(f"\n✅ 완료:")
    print(f"  - 병합 완료: {merged_count}개")
    print(f"  - 삭제 완료: {deleted_count}개")


def main():
    import sys
    
    dry_run = "--execute" not in sys.argv
    
    print("=" * 60)
    print("잘못 추출된 엔티티 정리 스크립트")
    print("=" * 60)
    print("\n동작 방식:")
    print("  1. 문장 형태의 잘못된 엔티티 찾기")
    print("  2. sources.snippet에서 실제 엔티티 추출")
    print("  3. DB에 존재하면 → sources 병합 후 삭제")
    print("  4. DB에 없으면 → 그냥 삭제")
    
    if dry_run:
        print("\n⚠️  DRY RUN 모드 (실제 수정 안함)")
        print("    실제 수정하려면: python cleanup_bad_entities.py --execute\n")
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
        
        cleanup_entities(db, dry_run=dry_run)
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
