import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
import json
import os
import time

def convert_id(level5_id):
    """level5 id를 변환 (wna_12504013_001 -> kna_12504013_001)"""
    parts = level5_id.split('_')
    if len(parts) >= 3:
        # wna를 kna로 변환하고, 전체 구조 유지
        return f"k{parts[0][1:]}_{parts[1]}_{parts[2]}"
    return level5_id

def scrape_sillok_content(session, url):
    """조선왕조실록 웹사이트에서 내용 스크래핑"""
    try:
        response = session.get(url, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            print(f"  - 페이지 로드 실패: {response.status_code}")
            return {"내용": [], "인물": [], "장소": [], "주석": []}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # ins_view_pd 클래스를 가진 div 찾기
        content_div = soup.find('div', class_='ins_view_pd')
        
        if not content_div:
            print("  - ins_view_pd div를 찾을 수 없음")
            return {"내용": [], "인물": [], "장소": [], "주석": []}
        
        # 분리된 데이터 저장
        content_text = []  # p 태그 내용
        people = []  # 인물
        places = []  # 장소
        annotations = []  # 주석 (id, label, text 형식)
        annotation_counter = 1
        
        # p 태그들의 텍스트만 추출 (span 태그 제외)
        for p_tag in content_div.find_all('p', recursive=False):
            text = p_tag.get_text(strip=True)
            if text and text not in ['"', '']:
                content_text.append(text)
        
        # span 태그들을 분류하여 추출
        for span in content_div.find_all('span', class_='idx_wrap'):
            classes = span.get('class', [])
            text = span.get_text(strip=True)
            
            if not text or text in ['"', '']:
                continue
            
            if 'idx_person' in classes:
                if text not in people:  # 중복 제거
                    people.append(text)
            elif 'idx_place' in classes:
                if text not in places:  # 중복 제거
                    places.append(text)
            elif 'idx_annotation03' in classes:
                # 주석을 id, label, text 형식으로 변환
                # label과 text 분리 시도 (일반적으로 첫 문장이 label)
                parts = text.split('\n', 1)
                if len(parts) == 2:
                    label = parts[0].strip()
                    annotation_text = parts[1].strip()
                else:
                    # 분리가 안되면 전체를 text로
                    label = text[:50] + "..." if len(text) > 50 else text
                    annotation_text = text
                
                # 중복 체크
                existing_texts = [a["text"] for a in annotations]
                if annotation_text not in existing_texts:
                    annotations.append({
                        "id": f"D{annotation_counter:03d}",
                        "label": label,
                        "text": annotation_text
                    })
                    annotation_counter += 1
        
        return {
            "내용": content_text,
            "인물": people,
            "장소": places,
            "주석": annotations
        }
        
    except Exception as e:
        print(f"  - 스크래핑 오류: {str(e)}")
        return {"내용": [], "인물": [], "장소": [], "주석": []}

def parse_xml_and_create_json(xml_file_path, output_json_path, sillok_name="조선왕조실록"):
    """XML 파일을 파싱하여 JSON 생성"""
    
    # 세션 생성 (연결 재사용)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # XML 파일을 스트림으로 파싱
    context = ET.iterparse(xml_file_path, events=('start', 'end'))
    context = iter(context)
    
    documents = []
    
    for event, elem in context:
        if event == 'end' and elem.tag == 'level4':
            # level4 데이터 추출
            # biblioData에서 date 정보 추출 (서기 날짜)
            biblio = elem.find('.//front/biblioData[@type="L"]')
            날짜 = None
            
            if biblio is not None:
                date_elem = biblio.find('date')
                if date_elem is not None:
                    for date_occurred in date_elem.findall('dateOccured'):
                        date_type = date_occurred.get('type')
                        if date_type == '서기':
                            # date 속성에서 날짜 추출 (예: "1592-04-13L0" -> "1592-04-13")
                            raw_date = date_occurred.get('date', '')
                            if raw_date:
                                # "L0" 등 suffix 제거
                                날짜 = raw_date.split('L')[0] if 'L' in raw_date else raw_date
                            break
            
            # level5 데이터 추출
            for level5 in elem.findall('.//level5'):
                level5_id = level5.get('id')
                
                # mainTitle 추출
                title_elem = level5.find('.//front/biblioData[@type="T"]/title/mainTitle')
                main_title = title_elem.text if title_elem is not None else ""
                
                # subjectClass 추출
                subject_classes = level5.findall('.//front/biblioData[@type="T"]/subjectClass')
                분류_list = [sc.text for sc in subject_classes if sc.text]
                분류 = " / ".join(분류_list)
                
                # id 변환
                converted_id = convert_id(level5_id)
                url = f"https://sillok.history.go.kr/id/{converted_id}"
                print(url)
                print(f"\n처리중: {main_title}")
                print(f"  URL: {url}")
                
                # 웹 스크래핑 (세션 전달)
                content_data = scrape_sillok_content(session, url)
                
                # 새로운 형식으로 문서 생성
                document = {
                    "url": url,
                    "content": "\n".join(content_data["내용"]),
                    "metadata": {
                        "서명": sillok_name,
                        "제목": main_title,
                        "주제분류": 분류,
                        "카테고리": "조선왕조실록",
                        "키워드": sillok_name,
                        "날짜": 날짜,
                        "인물": content_data["인물"],
                        "장소": content_data["장소"]
                    },
                    "annotations": content_data["주석"]
                }
                
                documents.append(document)
                
                # 서버 부하 방지를 위한 대기 (세션 재사용으로 속도 향상)
                time.sleep(0.3)
            
            # 메모리 절약을 위해 처리한 요소 제거
            elem.clear()
    
    # 새로운 형식으로 결과 데이터 구성
    result_data = {
        "document_count": len(documents),
        "documents": documents
    }
    
    # JSON 파일로 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    # 세션 종료
    session.close()
    
    print(f"\n\n완료! 결과가 {output_json_path}에 저장되었습니다.")
    print(f"총 {len(documents)}개의 문서가 처리되었습니다.")

if __name__ == "__main__":
    # 스크립트 파일 위치 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # history 디렉토리
    
    # 처리할 디렉토리 목록 (crawling 폴더 기준)
    directories = [
        ("조선왕조실록/선조실록", "선조실록"),
        ("조선왕조실록/선조수정실록", "선조수정실록")
    ]
    
    # output 디렉토리가 없으면 생성 (history/output 폴더)
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    total_files_count = 0
    
    for xml_directory, directory_name in directories:
        # crawling 폴더 기준 경로
        xml_dir_path = os.path.join(script_dir, xml_directory)
        
        # 디렉토리가 존재하는지 확인
        if not os.path.exists(xml_dir_path):
            print(f"'{xml_dir_path}' 디렉토리가 존재하지 않습니다. 건너뜁니다.")
            continue
        
        # 디렉토리 내의 모든 XML 파일 찾기
        xml_files = [f for f in os.listdir(xml_dir_path) if f.endswith('.xml')]
        
        if not xml_files:
            print(f"'{xml_dir_path}' 디렉토리에 XML 파일이 없습니다.")
            continue

        
        for idx, xml_filename in enumerate(xml_files, 1):
            xml_file_path = os.path.join(xml_dir_path, xml_filename)
            
            # 출력 파일명 생성 (예: 2nd_wna_125.xml -> 선조실록_25년.json)
            base_name = os.path.splitext(xml_filename)[0]
            parts = base_name.split('_')
            if len(parts) >= 3 and parts[-1].isdigit():
                year_num = parts[-1][1:]  # 125 -> 25
                output_file = os.path.join(output_dir, f"{directory_name}_{year_num}년.json")
            else:
                output_file = os.path.join(output_dir, f"{directory_name}_{base_name}.json")
            
            print(f"\n[{idx}/{len(xml_files)}] 처리 중...")
            print(f"입력 파일: {xml_file_path}")
            print(f"출력 파일: {output_file}")
            print("-"*60)
            
            try:
                parse_xml_and_create_json(xml_file_path, output_file, directory_name)
                total_files_count += 1
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
                continue
    print(f" 모든 처리 완료! 총 {total_files_count}개의 파일이 처리되었습니다.")

