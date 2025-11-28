import re
from bs4 import BeautifulSoup
import json
import time
from typing import Dict, Optional
import os
from urllib.parse import urlparse, parse_qs
import multiprocessing as mp
import atexit

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False


class EventDetailCrawler:
    """사건정보 상세 크롤러"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_event_detail(self, url: str) -> Dict:
        """URL에서 사건/인물 상세정보 추출"""
        result = {
            'url': url,
            'event_name': '',
            'details': [],
            'related_people': [],
            'related_events': [],
            'success': False
        }
        
        try:
            if not SELENIUM_AVAILABLE:
                result['error'] = 'Selenium not available'
                return result
            
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--lang=ko-KR')
            chrome_options.add_argument(f'--user-agent={self.headers["User-Agent"]}')
            
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), 
                options=chrome_options
            )
            
            try:
                driver.get(url)
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
                # 사건명 추출
                event_name_elem = soup.select_one('h3 span.gosrch')
                if event_name_elem:
                    result['event_name'] = event_name_elem.get_text(strip=True)
                
                # 테이블 정보 추출
                tables = soup.find_all('tbody')
                if len(tables) >= 2:
                    tbody = tables[1]
                    rows = tbody.find_all('tr')
                    
                    for row in rows:
                        th = row.find('th')
                        td = row.find('td')
                        
                        if th and td:
                            result['details'].append({
                                'subtitle': th.get_text(strip=True),
                                'content': td.get_text(strip=True)
                            })
                
                # 관련 인물/사건 추출
                wtables = soup.select('table.wtable')
                
                for idx, wtable in enumerate(wtables):
                    tbody = wtable.find('tbody')
                    if not tbody:
                        continue
                    
                    tbody_text = tbody.get_text(strip=True)
                    if '조회' in tbody_text and '목록이' in tbody_text and '존재하지' in tbody_text:
                        continue
                    
                    headers = []
                    thead = wtable.find('thead')
                    
                    if thead:
                        th_tags = thead.find_all('th')
                        for th in th_tags:
                            a_tag = th.find('a')
                            if a_tag:
                                headers.append(a_tag.get_text(strip=True))
                            else:
                                headers.append(th.get_text(strip=True))
                    else:
                        first_row = tbody.find('tr')
                        if first_row:
                            th_tags = first_row.find_all('th')
                            if th_tags:
                                for th in th_tags:
                                    a_tag = th.find('a')
                                    if a_tag:
                                        headers.append(a_tag.get_text(strip=True))
                                    else:
                                        headers.append(th.get_text(strip=True))
                    
                    if not headers:
                        continue
                    
                    rows = tbody.find_all('tr')
                    
                    headers_str = ' '.join(headers).lower()
                    is_event_table = any(keyword in headers_str for keyword in ['사건명', '사건일', '사건인물', '관련사건'])
                    
                    start_idx = 0
                    if not thead and rows:
                        first_row_ths = rows[0].find_all('th')
                        if first_row_ths:
                            start_idx = 1
                    
                    for row in rows[start_idx:]:
                        td_tags = row.find_all('td')
                        if len(td_tags) == len(headers):
                            data = {}
                            row_has_link = False
                            row_link = ''
                            
                            for i, td in enumerate(td_tags):
                                a_tag = td.find('a')
                                if a_tag:
                                    data[headers[i]] = a_tag.get_text(strip=True)
                                    if not row_has_link and a_tag.get('href'):
                                        href = a_tag.get('href')
                                        if not href.startswith('http'):
                                            row_link = f"https://db.itkc.or.kr/people/{href}"
                                        else:
                                            row_link = href
                                        row_has_link = True
                                else:
                                    data[headers[i]] = td.get_text(strip=True)
                            
                            if data:
                                if row_has_link:
                                    data['DCI_s'] = row_link
                                
                                if is_event_table:
                                    result['related_events'].append(data)
                                else:
                                    result['related_people'].append(data)
                
                result['success'] = bool(result['event_name'] or result['details'] or result['related_people'] or result['related_events'])
                
            finally:
                driver.quit()
        
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
        
        return result


def _clean_text(value: str, subtitle: str = '') -> str:
    """텍스트 정리"""
    if not isinstance(value, str):
        return value
    cleaned = value.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if subtitle in ('원문정보', '참고자료'):
        cleaned = cleaned.replace('바로가기', ' ')
        cleaned = re.sub(r'전체\s*\d+개\s*더보기', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _extract_category_and_normalize_url(original_url: str, preferred_category: Optional[str] = None) -> Dict[str, str]:
    """URL 카테고리 판별 및 표준화"""
    try:
        parsed = urlparse(original_url)
        query_params = parse_qs(parsed.query)
        fragment = parsed.fragment or ""
        frag_query = fragment.split("?", 1)[1] if "?" in fragment else ""
        frag_params = parse_qs(frag_query)

        if preferred_category in ("사건", "인물"):
            category = preferred_category
        else:
            gubun = (query_params.get("gubun", [None])[0]
                     or frag_params.get("gubun", [None])[0])
            if gubun == "evnt" or "viewEvnt" in fragment:
                category = "사건"
            elif gubun == "person" or "#view" in fragment:
                category = "인물"
            else:
                category = "사건"

        data_id = (frag_params.get("dataId", [None])[0]
                   or query_params.get("dataId", [None])[0])

        if category == "사건" and data_id:
            normalized = (
                "https://db.itkc.or.kr/people/item?gubun=evnt#"
                "/viewEvnt?gubun=evntcate&cate1=Z&cate2=&dataId="
                f"{data_id}"
            )
        elif category == "인물" and data_id:
            normalized = (
                "https://db.itkc.or.kr/people/item?gubun=person"
                "#view?gubun=person&cate1=N&cate2=&dataId="
                f"{data_id}"
            )
        else:
            normalized = original_url

        return {"category": category, "normalized_url": normalized}
    except Exception:
        return {"category": "", "normalized_url": original_url}


g_driver = None


def cleanup_worker():
    """워커 정리"""
    global g_driver
    try:
        if g_driver is not None:
            g_driver.quit()
    except Exception:
        pass
    finally:
        g_driver = None


def init_worker():
    """워커 초기화"""
    if not SELENIUM_AVAILABLE:
        return
    global g_driver
    if g_driver is None:
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--lang=ko-KR')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        g_driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        atexit.register(cleanup_worker)


def _worker_process(task: Dict) -> Dict:
    """워커 프로세스: 상세정보 수집"""
    url = task['url']
    category = task['category']
    
    result = {
        'url': url,
        'event_name': '',
        'details': [],
        'related_people': [],
        'related_events': [],
        'success': False,
        'category': category
    }
    
    if not SELENIUM_AVAILABLE:
        result['error'] = 'Selenium not available'
        return result
    
    try:
        global g_driver
        
        old_title = g_driver.title if hasattr(g_driver, 'title') else ''
        g_driver.get(url)
        time.sleep(1)
        
        max_wait = 1
        start = time.time()
        while time.time() - start < max_wait:
            current_title = g_driver.title
            if current_title != old_title:
                break
            time.sleep(0.3)
        
        html = g_driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        event_name_elem = soup.select_one('h3 span.gosrch')
        if event_name_elem:
            result['event_name'] = event_name_elem.get_text(strip=True)
        
        tables = soup.find_all('tbody')
        if len(tables) >= 2:
            tbody = tables[1]
            rows = tbody.find_all('tr')
            
            for row in rows:
                th = row.find('th')
                td = row.find('td')
                
                if th and td:
                    result['details'].append({
                        'subtitle': th.get_text(strip=True),
                        'content': td.get_text(strip=True)
                    })
        
        wtables = soup.select('table.wtable')
        
        for idx, wtable in enumerate(wtables):
            tbody = wtable.find('tbody')
            if not tbody:
                continue
            
            tbody_text = tbody.get_text(strip=True)
            if '조회' in tbody_text and '목록이' in tbody_text and '존재하지' in tbody_text:
                continue
            
            headers = []
            thead = wtable.find('thead')
            
            if thead:
                th_tags = thead.find_all('th')
                for th in th_tags:
                    a_tag = th.find('a')
                    if a_tag:
                        headers.append(a_tag.get_text(strip=True))
                    else:
                        headers.append(th.get_text(strip=True))
            else:
                first_row = tbody.find('tr')
                if first_row:
                    th_tags = first_row.find_all('th')
                    if th_tags:
                        for th in th_tags:
                            a_tag = th.find('a')
                            if a_tag:
                                headers.append(a_tag.get_text(strip=True))
                            else:
                                headers.append(th.get_text(strip=True))
            
            if not headers:
                continue
            
            rows = tbody.find_all('tr')
            
            headers_str = ' '.join(headers).lower()
            is_event_table = any(keyword in headers_str for keyword in ['사건명', '사건일', '사건인물', '관련사건'])
            
            start_idx = 0
            if not thead and rows:
                first_row_ths = rows[0].find_all('th')
                if first_row_ths:
                    start_idx = 1
            
            for row in rows[start_idx:]:
                td_tags = row.find_all('td')
                if len(td_tags) == len(headers):
                    data = {}
                    row_has_link = False
                    row_link = ''
                    
                    for i, td in enumerate(td_tags):
                        a_tag = td.find('a')
                        if a_tag:
                            data[headers[i]] = a_tag.get_text(strip=True)
                            if not row_has_link and a_tag.get('href'):
                                href = a_tag.get('href')
                                if not href.startswith('http'):
                                    row_link = f"https://db.itkc.or.kr/people/{href}"
                                else:
                                    row_link = href
                                row_has_link = True
                        else:
                            data[headers[i]] = td.get_text(strip=True)
                    
                    if data:
                        if row_has_link:
                            data['DCI_s'] = row_link
                        
                        if is_event_table:
                            result['related_events'].append(data)
                        else:
                            result['related_people'].append(data)
        
        result['success'] = bool(result['event_name'] or result['details'] or result['related_people'] or result['related_events'])
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
    
    return result


def process_event_info_json_parallel(input_file: str, output_dir: str = "output", max_items: Optional[int] = None, num_workers: int = 4):
    """사건/인물정보 병렬 처리"""
    print(f"\n사건정보 상세 크롤링 시작 (병렬)")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    if max_items:
        events = events[:max_items]
    
    print(f"항목: {len(events)}개 | 워커: {num_workers}개\n")
    
    tasks = []
    for event in events:
        url = event.get('DCI_s', '')
        if not url:
            continue
        
        input_category_raw = event.get('category') or event.get('카테고리')
        if isinstance(input_category_raw, str):
            input_category_raw = input_category_raw.strip()
        
        if input_category_raw in ("사건정보", "사건", "event", "evnt"):
            preferred = "사건"
        elif input_category_raw in ("인간정보", "인물정보", "인물", "person"):
            preferred = "인물"
        else:
            preferred = None
        
        url_info = _extract_category_and_normalize_url(url, preferred_category=preferred)
        category = url_info.get("category", "사건")
        normalized_url = url_info.get("normalized_url", url)
        
        tasks.append({
            'url': normalized_url,
            'category': category,
            'original_event': event
        })
    
    os.makedirs(output_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    results = []
    start_time = time.time()
    
    with ctx.Pool(processes=num_workers, initializer=init_worker) as pool:
        for i, detail in enumerate(pool.imap_unordered(_worker_process, tasks, chunksize=1), 1):
            사건정보 = {}
            for item in detail.get('details', []):
                subtitle = item.get('subtitle', '')
                content = item.get('content', '')
                if subtitle:
                    사건정보[subtitle] = _clean_text(content, subtitle)
            
            관련인물 = []
            for person in detail.get('related_people', []):
                cleaned_person = {}
                for key, value in person.items():
                    cleaned_person[key] = _clean_text(value) if isinstance(value, str) else value
                관련인물.append(cleaned_person)
            
            관련사건 = []
            for event in detail.get('related_events', []):
                cleaned_event = {}
                for key, value in event.items():
                    cleaned_event[key] = _clean_text(value) if isinstance(value, str) else value
                관련사건.append(cleaned_event)
            
            category = detail.get('category', '사건')
            name_key = '사건명' if category == '사건' else '인물명'
            info_key = '사건정보' if category == '사건' else '인물정보'
            
            result_item = {
                'DCI_s': detail.get('url', ''),
                name_key: detail.get('event_name', ''),
                info_key: 사건정보,
                '관련인물': 관련인물,
                '관련사건': 관련사건,
                '카테고리': category
            }
            
            is_duplicate = any(
                r.get('DCI_s') == result_item['DCI_s'] and 
                r.get(name_key) == result_item[name_key] 
                for r in results
            )
            
            if not is_duplicate:
                results.append(result_item)
                status_mark = 'OK'
            else:
                status_mark = 'DUP'
            
            status = status_mark if detail.get('success') else 'FAIL'
            event_name = detail.get('event_name', '(없음)')[:30]
            print(f"[{i}/{len(tasks)}] {status} {name_key}: {event_name}, 상세: {len(사건정보)}개, 관련인물: {len(관련인물)}명, 관련사건: {len(관련사건)}건")
    
    elapsed = time.time() - start_time
    print(f"\n소요시간: {elapsed:.1f}초")
    
    keyword = events[0].get('keyword', '사건정보') if events else '사건정보'
    output_file = os.path.join(output_dir, f'사건정보_상세_{keyword}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    duplicates_removed = len(tasks) - len(results)
    success_count = sum(1 for r in results if r.get('사건명') or r.get('인물명'))
    print(f"\n병렬 크롤링 완료")
    print(f"처리: {len(tasks)}개")
    print(f"저장: {len(results)}개")
    print(f"중복제거: {duplicates_removed}개")
    print(f"성공: {success_count}개")
    print(f"저장위치: {output_file}")
    
    return results


def process_event_info_json(input_file: str, output_dir: str = "output", max_items: Optional[int] = None):
    """사건/인물정보 순차 처리"""
    print("\n사건정보 상세 크롤링 시작")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        events = json.load(f)
    
    if max_items:
        events = events[:max_items]
    
    print(f"항목: {len(events)}개\n")
    
    crawler = EventDetailCrawler()
    results = []
    
    for i, event in enumerate(events, 1):
        url = event.get('DCI_s', '')
        if not url:
            continue

        input_category_raw = event.get('category') or event.get('카테고리')
        if isinstance(input_category_raw, str):
            input_category_raw = input_category_raw.strip()
            
        if input_category_raw in ("사건정보", "사건", "event", "evnt"):
            preferred = "사건"
        elif input_category_raw in ("인간정보", "인물정보", "인물", "person"):
            preferred = "인물"
        else:
            preferred = None

        url_info = _extract_category_and_normalize_url(url, preferred_category=preferred)
        category = url_info.get("category", "사건")
        normalized_url = url_info.get("normalized_url", url)

        detail = crawler.get_event_detail(normalized_url)

        사건정보 = {}
        for item in detail.get('details', []):
            subtitle = item.get('subtitle', '')
            content = item.get('content', '')
            if subtitle:
                사건정보[subtitle] = _clean_text(content, subtitle)
        
        관련인물 = []
        for person in detail.get('related_people', []):
            cleaned_person = {}
            for key, value in person.items():
                cleaned_person[key] = _clean_text(value) if isinstance(value, str) else value
            관련인물.append(cleaned_person)
        
        관련사건 = []
        for event in detail.get('related_events', []):
            cleaned_event = {}
            for key, value in event.items():
                cleaned_event[key] = _clean_text(value) if isinstance(value, str) else value
            관련사건.append(cleaned_event)
        
        name_key = '사건명' if category == '사건' else '인물명'
        info_key = '사건정보' if category == '사건' else '인물정보'

        result_item = {
            'DCI_s': normalized_url,
            name_key: detail.get('event_name', ''),
            info_key: 사건정보,
            '관련인물': 관련인물,
            '관련사건': 관련사건,
            '카테고리': category
        }
        
        results.append(result_item)
        
        status = 'OK' if detail.get('success') else 'FAIL'
        print(f"{status} {i}/{len(events)} - {name_key}: {detail.get('event_name', '(없음)')[:30]}, 상세: {len(사건정보)}개, 관련인물: {len(관련인물)}명, 관련사건: {len(관련사건)}건")
    
    os.makedirs(output_dir, exist_ok=True)
    keyword = events[0].get('keyword', '사건정보') if events else '사건정보'
    output_file = os.path.join(output_dir, f'사건정보_상세_{keyword}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n완료")
    print(f"저장: {output_file}")
    
    return results


def collect_event_detail(
    input_file: str,
    output_dir: str = "output",
    max_items: Optional[int] = None,
    num_workers: int = 4,
    parallel: bool = True
):
    """사건/인물 상세 정보 수집"""
    if not os.path.exists(input_file):
        print(f"파일 없음: {input_file}")
        return []
    
    if parallel:
        return process_event_info_json_parallel(input_file, output_dir, max_items, num_workers)
    else:
        return process_event_info_json(input_file, output_dir, max_items)
