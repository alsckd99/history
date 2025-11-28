"""
ITKC 통합 크롤링 시스템

3가지 수집 모드:
1. 일반 검색 (BT_AA, JT_AA): 검색어로 고전번역서/조선왕조실록 검색 후 본문 수집
2. 고전번역서 한 권 (BT_SJ): 특정 고전번역서의 전체 내용 수집
3. 정보 검색 (PR_AA, PR_BB, TR_AA): 인물/사건/용어 정보 검색 후 상세 수집
"""

import os
import sys
from typing import Optional

# 현재 디렉토리를 시스템 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 모듈 import
from ITKC_List import search_and_save
from database_1 import collect_content_from_search
from database_2 import collect_event_detail


class ITKCCrawlerMain:
    """ITKC 통합 크롤러"""
    
    # 카테고리 코드 정의
    CATEGORY_INFO = {
        'BT_AA': '고전번역서 (기사)',
        'BT_SJ': '고전번역서 (한 권 전체)',
        'JT_AA': '조선왕조실록',
        'PR_AA': '인물정보',
        'PR_BB': '사건정보',
        'TR_AA': '용어정보',
    }
    
    def __init__(self, output_dir: str = "output"):
        """초기화"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def mode_1_general_search(
        self,
        keyword: str,
        categories: list = ['BT_AA', 'JT_AA'],
        max_items: Optional[int] = None,
        num_workers: int = 4
    ):
        """모드 1: 일반 검색 (고전번역서 + 조선왕조실록)"""
        print(f"\n[모드 1] 키워드: {keyword}, 카테고리: {', '.join([self.CATEGORY_INFO[c] for c in categories])}")
        
        print("[1/2] 검색 중...")
        search_and_save(keyword, categories, self.output_dir, True, num_workers, True)
        
        search_file = os.path.join(self.output_dir, f"itkc_search_{keyword}.json")
        if os.path.exists(search_file):
            print("[2/2] 본문 수집 중...")
            collect_content_from_search(search_file, self.output_dir, max_items, num_workers)
        else:
            print(f"검색 결과 파일 없음: {search_file}")
        
        print("모드 1 완료\n")
    
    def mode_2_book_full_content(
        self,
        keyword: str,
        max_items: Optional[int] = None,
        num_workers: int = 4
    ):
        """모드 2: 고전번역서 한 권 전체 내용 수집"""
        print(f"\n[모드 2] 고전번역서: {keyword}")
        
        print("[1/2] 검색 중 (BT_SJ)...")
        crawler = search_and_save(keyword, ['BT_SJ'], self.output_dir, True, num_workers, False, save_to_file=False)
        
        print("[2/2] 본문 수집 중...")
        collect_content_from_search(crawler, self.output_dir, max_items, num_workers)
        
        print("모드 2 완료\n")
    
    def mode_3_info_search(
        self,
        keyword: str,
        categories: list = ['PR_BB', 'TR_AA'],
        max_items: Optional[int] = None,
        num_workers: int = 4
    ):
        """모드 3: 인물/사건/용어 정보 검색 및 상세 수집"""
        print(f"\n[모드 3] 키워드: {keyword}, 카테고리: {', '.join([self.CATEGORY_INFO[c] for c in categories])}")
        
        print("[1/2] 검색 중...")
        search_and_save(keyword, categories, self.output_dir, True, num_workers, True)
        
        if any(c.startswith('PR') for c in categories):
            print("[2/2] 상세 정보 수집 중...")
            
            if 'PR_BB' in categories:
                event_file = os.path.join(self.output_dir, f"사건정보_{keyword}.json")
                if os.path.exists(event_file):
                    collect_event_detail(event_file, self.output_dir, max_items, num_workers, True)
                else:
                    print(f"사건정보 파일 없음: {event_file}")
            
            if 'PR_AA' in categories:
                person_file = os.path.join(self.output_dir, f"인물정보_{keyword}.json")
                if os.path.exists(person_file):
                    collect_event_detail(person_file, self.output_dir, max_items, num_workers, True)
                else:
                    print(f"인물정보 파일 없음: {person_file}")
        else:
            print("[2/2] 용어정보는 검색 결과만 저장")
        
        print("모드 3 완료\n")


def interactive_mode():
    """대화형 모드"""
    print("\nITKC 통합 크롤링 시스템")
    print("수집 모드 선택:")
    print("  1. 일반 검색 (고전번역서 + 조선왕조실록)")
    print("  2. 고전번역서 한 권 전체")
    print("  3. 인물/사건/용어 정보")
    print("  0. 종료")
    
    crawler = ITKCCrawlerMain()
    
    while True:
        choice = input("\n모드 선택 (0-3): ").strip()
        
        if choice == '0':
            print("프로그램 종료")
            break
        
        elif choice == '1':
            keyword = input("검색 키워드: ").strip()
            if not keyword:
                print("키워드를 입력해주세요")
                continue
            
            print("카테고리: 1.고전번역서 2.조선왕조실록 3.둘다")
            cat_choice = input("선택 (1-3): ").strip()
            
            if cat_choice == '1':
                categories = ['BT_AA']
            elif cat_choice == '2':
                categories = ['JT_AA']
            else:
                categories = ['BT_AA', 'JT_AA']
            
            max_items_input = input("최대 수집 개수 (Enter=전체): ").strip()
            max_items = int(max_items_input) if max_items_input else None
            
            workers = input("워커 수 (기본=4): ").strip()
            num_workers = int(workers) if workers else 4
            
            crawler.mode_1_general_search(keyword, categories, max_items, num_workers)
        
        elif choice == '2':
            keyword = input("고전번역서 제목: ").strip()
            if not keyword:
                print("제목을 입력해주세요")
                continue
            
            max_items_input = input("최대 수집 개수 (Enter=전체): ").strip()
            max_items = int(max_items_input) if max_items_input else None
            
            workers = input("워커 수 (기본=4): ").strip()
            num_workers = int(workers) if workers else 4
            
            crawler.mode_2_book_full_content(keyword, max_items, num_workers)
        
        elif choice == '3':
            keyword = input("검색 키워드: ").strip()
            if not keyword:
                print("키워드를 입력해주세요")
                continue

            categories = ['PR_AA', 'PR_BB', 'TR_AA']
            
            max_items_input = input("최대 수집 개수 (Enter=전체): ").strip()
            max_items = int(max_items_input) if max_items_input else None
            
            workers = input("워커 수 (기본=4): ").strip()
            num_workers = int(workers) if workers else 4
            
            crawler.mode_3_info_search(keyword, categories, max_items, num_workers)
        
        else:
            print("잘못된 선택입니다")


def batch_mode():
    """배치 모드 예제"""
    print("\nITKC 배치 크롤링 예제")
    crawler = ITKCCrawlerMain()
    
    print("\n[예제 1] 임진왜란 - 조선왕조실록")
    crawler.mode_1_general_search("임진왜란", ['JT_AA'], 50, 4)
    
    crawler.mode_2_book_full_content("징비록", None, 4)
    
    print("\n[예제 3] 이순신 - 인물/사건 정보")
    crawler.mode_3_info_search("이순신", ['PR_AA', 'PR_BB'], 30, 4)


def main():
    """메인 함수"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        batch_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
