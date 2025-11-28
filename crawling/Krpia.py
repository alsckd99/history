import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from xml.dom import minidom
import time
import json
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime


class KrpiaCrawler:
    def __init__(self, base_url, debug=False, use_selenium=True):
        """
        KRPIA 웹사이트에서 콘텐츠를 크롤링하는 클래스
        
        Args:
            base_url: 기본 URL (예: https://www.krpia.co.kr/viewer/open?plctId=PLCT00004800&nodeId=NODE03884955&medaId=MEDA03976653)
            debug: 디버그 모드 활성화 여부
            use_selenium: Selenium 사용 여부 (JavaScript 렌더링 필요)
        """
        self.base_url = base_url
        self.debug = debug
        self.use_selenium = use_selenium
        
        if use_selenium:
            # Selenium 설정
            chrome_options = Options()
            if not debug:
                chrome_options.add_argument('--headless')  # 브라우저 창 숨기기
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            # ChromeDriver 자동 설치 및 실행
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            
            if self.debug:
                print("✓ Selenium WebDriver 초기화 완료")
        else:
            # requests 사용
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
    
    def __del__(self):
        """소멸자: Selenium 드라이버 종료"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
                if self.debug:
                    print("\n✓ Selenium WebDriver 종료")
            except:
                pass
        
    def get_toc_items(self):
        """
        목차(Table of Contents) 항목들을 가져옵니다.
        
        Returns:
            list: [{'node_id': str, 'seq': str, 'title': str}]
        """
        if not self.use_selenium:
            print("목차 크롤링은 Selenium 필요")
            return []
        
        try:
            # 초기 페이지 로드
            self.driver.get(self.base_url)
            
            if self.debug:
                print(f"✓ [DEBUG] 페이지 로드 완료")
            
            # 목차 트리가 로드될 때까지 대기
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "ul.treeview"))
            )
            
            time.sleep(2)  # 추가 대기
            
            if self.debug:
                print(f"✓ [DEBUG] 목차 트리 로드 완료, rootNode 펼치기 시작...")
            
            # 모든 rootNode를 펼치기 (접혀있는 목차 항목 표시)
            try:
                # rootNode의 토글 버튼 찾기
                root_toggles = self.driver.find_elements(By.CSS_SELECTOR, "ul.treeview li.rootNode > button.tvIcon")
                
                if self.debug:
                    print(f"  - rootNode {len(root_toggles)}개 발견")
                
                for idx, toggle in enumerate(root_toggles):
                    try:
                        # 버튼이 보이는 경우에만 클릭
                        if toggle.is_displayed():
                            # 접혀있는지 확인 (클래스 체크)
                            parent_li = toggle.find_element(By.XPATH, "..")
                            if parent_li:
                                toggle.click()
                                time.sleep(0.3)  # 펼쳐지는 시간 대기
                                if self.debug:
                                    print(f"  - rootNode {idx+1} 펼침")
                    except Exception as e:
                        if self.debug:
                            print(f"  - rootNode {idx+1} 펼치기 실패: {e}")
                        continue
                
                time.sleep(1)  # 모든 항목이 펼쳐진 후 대기
                
            except Exception as e:
                if self.debug:
                    print(f"rootNode 펼치기 오류: {e}")
            
            # 목차 항목 찾기 (모든 leafNode)
            leaf_nodes = self.driver.find_elements(By.CSS_SELECTOR, "ul.treeview li.leafNode")
            
            toc_items = []
            for node in leaf_nodes:
                node_id = node.get_attribute('id')
                seq = node.get_attribute('seq')
                
                # a 태그에서 제목 추출
                try:
                    link = node.find_element(By.CSS_SELECTOR, "a")
                    title = link.text.strip()
                except:
                    title = ""
                
                if node_id and seq:
                    toc_items.append({
                        'node_id': node_id,
                        'seq': seq,
                        'title': title
                    })
            
            if self.debug:
                print(f"\n✓ [DEBUG] 목차 항목 {len(toc_items)}개 발견")
                for idx, item in enumerate(toc_items[:5], 1):
                    print(f"  {idx}. {item['node_id']} (seq={item['seq']}): {item['title'][:30]}")
            
            return toc_items
            
        except Exception as e:
            print(f"목차 가져오기 실패: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return []
    
    def click_and_get_content(self, node_id):
        """
        목차 항목을 클릭하고 콘텐츠를 가져옵니다.
        
        Args:
            node_id: 노드 ID (예: "NODE03884957")
            
        Returns:
            dict: {'node_id': str, 'name': str, 'title': str, 'content': str}
        """
        try:
            if self.debug:
                print(f"\n{node_id} 클릭 중...")
            
            # li 태그의 하위 a 태그 클릭
            try:
                link = self.driver.find_element(By.CSS_SELECTOR, f"li#{node_id} > a")
                link.click()
                if self.debug:
                    print(f"✓ [DEBUG] a 태그 클릭 완료")
            except Exception as e:
                if self.debug:
                    print(f"a 태그 클릭 실패, 다른 방법 시도: {e}")
                # a 태그가 없으면 li 자체를 클릭
                li_element = self.driver.find_element(By.ID, node_id)
                li_element.click()
            
            if self.debug:
                print(f"✓ [DEBUG] 클릭 완료, 콘텐츠 로딩 대기 중...")
            
            # 콘텐츠가 로드될 때까지 대기
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda driver: len(driver.find_element(By.CSS_SELECTOR, "div#translate div.content").text.strip()) > 0
                )
            except:
                if self.debug:
                    print("콘텐츠 로딩 타임아웃")
            
            time.sleep(1)  # 추가 대기
            
            # HTML 가져오기
            html = self.driver.page_source
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # title 추출 (a id="titleNow")
            title_element = soup.find('a', {'id': 'titleNow'})
            if title_element:
                title = title_element.get_text(strip=True)
                if not title:
                    title = title_element.get('title', '')
            else:
                title = ''
            
            # content 추출 (div id="translate" name="...")
            content_element = soup.find('div', {'id': 'translate'})
            
            if content_element:
                # name 속성 가져오기
                name = content_element.get('name', '')
                
                # name이 없으면 node_id 사용
                if not name:
                    name = node_id
                
                # content 텍스트 추출
                content_div = content_element.find('div', {'class': 'content'})
                
                if content_div:
                    content_text = content_div.get_text(separator='\n', strip=True)
                else:
                    content_text = content_element.get_text(separator='\n', strip=True)
                
                if self.debug:
                    print(f"  ✓ title: {title}")
                    print(f"  ✓ name: {name}")
                    print(f"  ✓ 콘텐츠 길이: {len(content_text)} 문자")
                    print(f"  - 미리보기: {content_text[:100]}...")
                
                return {
                    'title': title,
                    'content': content_text
                }
            else:
                print(f"{node_id}: content 없음")
                return None
                
        except Exception as e:
            print(f"{node_id} 크롤링 실패: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def crawl_all_toc(self, delay=1, max_items=None):
        """
        목차의 모든 항목을 순서대로 크롤링합니다.
        
        Args:
            delay: 각 요청 사이의 대기 시간(초)
            max_items: 최대 크롤링 항목 수 (None이면 전체)
            
        Returns:
            list: 크롤링된 데이터 리스트
        """
        # 목차 항목 가져오기
        toc_items = self.get_toc_items()
        
        if not toc_items:
            print("목차 항목 없음")
            return []
        
        # max_items 제한
        if max_items:
            toc_items = toc_items[:max_items]
        
        print(f"\n총 {len(toc_items)}개 항목 크롤링 시작\n")
        
        results = []
        
        for i, item in enumerate(toc_items, 1):
            print(f"[{i}/{len(toc_items)}] {item['node_id']} 크롤링 중...")
            print(f"  제목: {item['title']}")
            
            data = self.click_and_get_content(item['node_id'])
            if data:
                # 목차 정보도 추가
                data['toc_title'] = item['title']
                data['seq'] = item['seq']
                results.append(data)
                print(f"✓ 완료\n")
            else:
                print("실패\n")
            
            # 서버 부하 방지를 위한 대기
            if i < len(toc_items):
                time.sleep(delay)
        
        return results
    
    def save_to_json(self, data_list, output_file='output.json', 
                      서명="", 카테고리="", 키워드=""):
        """
        크롤링된 데이터를 JSON 파일로 저장합니다.
        
        Args:
            data_list: 크롤링된 데이터 리스트
            output_file: 출력 파일명
            서명: 문서 서명 (예: "징비록")
            카테고리: 카테고리 (예: "고전번역서")
            키워드: 키워드
        """
        # 새로운 형식으로 문서 변환
        documents = []
        for data in data_list:
            document = {
                "url": self.base_url,
                "content": data.get('content', ''),
                "metadata": {
                    "서명": 서명,
                    "제목": data.get('title', ''),
                    "주제분류": "",
                    "카테고리": 카테고리,
                    "키워드": 키워드,
                    "날짜": "",
                    "인물": [],
                    "장소": []
                },
                "annotations": []
            }
            documents.append(document)
        
        # JSON 구조 생성
        json_data = {
            'document_count': len(documents),
            'documents': documents
        }
        
        # 파일로 저장 (한글 유지, 들여쓰기 적용)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nJSON 파일 저장: {output_file}")
        print(f"   총 {len(documents)}개 문서 저장됨")
    
    def save_to_xml(self, data_list, output_file='output.xml'):
        """
        크롤링된 데이터를 XML 파일로 저장합니다.
        
        Args:
            data_list: 크롤링된 데이터 리스트
            output_file: 출력 파일명
        """
        # XML 루트 생성
        root = ET.Element('documents')
        
        for data in data_list:
            # 각 문서를 document 태그로 추가
            doc = ET.SubElement(root, 'document')
            doc.set('name', data['name'])
            
            # title 태그 추가
            title_elem = ET.SubElement(doc, 'title')
            title_elem.text = data['title']
            
            # content 태그 추가
            content_elem = ET.SubElement(doc, 'content')
            content_elem.text = data['content']
        
        # XML을 보기 좋게 포맷팅
        xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ", encoding='utf-8')
        
        # 파일로 저장
        with open(output_file, 'wb') as f:
            f.write(xml_str)
        
        print(f"\nXML 파일 저장: {output_file}")
        print(f"   총 {len(data_list)}개 문서 저장됨")
    
    def generate_meda_ids(self, start_id, count):
        """
        연속된 medaId를 생성합니다.
        
        Args:
            start_id: 시작 ID (예: "MEDA03976653" 또는 3976653)
            count: 생성할 ID 개수
            
        Returns:
            list: medaId 리스트
        """
        # 숫자 부분 추출
        if isinstance(start_id, str) and start_id.startswith('MEDA'):
            num_part = int(start_id[4:])
        else:
            num_part = int(start_id)
        
        # 연속된 ID 생성
        ids = []
        for i in range(count):
            meda_id = f"MEDA{num_part + i:08d}"
            ids.append(meda_id)
        
        return ids


# 사용 예시
if __name__ == "__main__":
    # 기본 URL 설정
    base_url = "https://www.krpia.co.kr/viewer/open?plctId=PLCT00004800&nodeId=NODE03884955&medaId=MEDA03976653&keyword=%EC%A7%95%EB%B9%84%EB%A1%9D%20%EB%B2%88%EC%97%AD&affectType=searchNew#none"
    
    # 크롤러 생성 (디버그 모드 활성화, Selenium 사용)
    crawler = KrpiaCrawler(base_url, debug=True, use_selenium=True)
    
    # 목차 기반 크롤링
    print("=" * 60)
    print("KRPIA 목차 기반 크롤링 시작")
    print("=" * 60)
    
    # 전체 크롤링 (max_items=None) 또는 일부만 크롤링 (max_items=5)
    results = crawler.crawl_all_toc(delay=1, max_items=None)  # 테스트용: 10개만 크롤링
    
    # JSON 파일로 저장
    if results:
        json_file = f'output/징비록.json'
        crawler.save_to_json(
            results, 
            json_file,
            서명="징비록",
            카테고리="고전번역서",
            키워드="징비록"
        )
        
        # XML도 원하면 주석 해제
        # xml_file = f'output/징비록_크롤링.xml'
        # crawler.save_to_xml(results, xml_file)
    else:
        print("\n크롤링 데이터 없음")
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)

