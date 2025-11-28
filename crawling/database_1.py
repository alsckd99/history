import requests
from bs4 import BeautifulSoup
import json
import time
from typing import Dict, Optional
import os
import multiprocessing as mp
import atexit
import re

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False


class ItkcContentCrawler:
    """ITKC 본문 크롤러"""
    
    def __init__(self):
        self.base_url = "http://db.itkc.or.kr/inLink?DCI="
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_content_by_dci(self, dci_s: str) -> Dict:
        """DCI로 본문 내용 가져오기"""
        url = f"{self.base_url}{dci_s}"
        
        try:
            if SELENIUM_AVAILABLE:
                try:
                    chrome_options = Options()
                    chrome_options.add_argument('--headless=new')
                    chrome_options.add_argument('--no-sandbox')
                    chrome_options.add_argument('--disable-dev-shm-usage')
                    chrome_options.add_argument('--disable-gpu')
                    chrome_options.add_argument('--window-size=1920,1080')
                    chrome_options.add_argument('--lang=ko-KR')
                    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
                    try:
                        driver.get(url)
                        WebDriverWait(driver, 10).until(
                            EC.any_of(
                                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.text_body')),
                                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.text_body2'))
                            )
                        )
                        rendered_html = driver.page_source
                        rsoup = BeautifulSoup(rendered_html, 'html.parser')
                        r_text_body = rsoup.select_one('div.text_body')
                        r_text_body2 = rsoup.select_one('div.text_body2')
                        main_content = r_text_body.get_text(separator='\n', strip=True) if r_text_body else ''
                        annotation_content = r_text_body2.get_text(separator='\n', strip=True) if r_text_body2 else ''
                        content_data = {
                            'dci_s': dci_s,
                            'url': url,
                            'main_content': main_content,
                            'annotation': annotation_content,
                            'content': main_content,
                        }
                    finally:
                        driver.quit()
                except Exception:
                    content_data = {
                        'dci_s': dci_s,
                        'url': url,
                        'main_content': '',
                        'annotation': '',
                        'content': '',
                    }
            else:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                t1 = soup.select_one('div.text_body')
                t2 = soup.select_one('div.text_body2')
                main_content = t1.get_text(separator='\n', strip=True) if t1 else ''
                annotation_content = t2.get_text(separator='\n', strip=True) if t2 else ''
                content_data = {
                    'dci_s': dci_s,
                    'url': url,
                    'main_content': main_content,
                    'annotation': annotation_content,
                    'content': main_content,
                }
            
            return content_data
        
        except requests.exceptions.RequestException as e:
            return {
                'dci_s': dci_s,
                'url': url,
                'success': False,
                'error': str(e)
            }
        except Exception as e:
            return {
                'dci_s': dci_s,
                'url': url,
                'success': False,
                'error': str(e)
            }


def _expand_bt_sj_items(item: Dict) -> list[Dict]:
    """BT_SJ 항목에서 하위 문서 DCI 추출"""
    if not SELENIUM_AVAILABLE:
        return []
    
    raw_data = item.get('raw_data', {})
    data_id = raw_data.get('자료ID')
    book_name = raw_data.get('서명', '')
    
    # 연려실기술인 경우 제12권~제18권만 수집
    is_yeolyeo = '연려실기술' in book_name
    yeolyeo_volumes = ['제12권', '제13권', '제14권', '제15권', '제16권', '제17권', '제18권']
    if is_yeolyeo:
        print(f"  [연려실기술] {', '.join(yeolyeo_volumes)} 만 수집")
    
    if not data_id:
        return []
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from webdriver_manager.chrome import ChromeDriverManager
        
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        try:
            url = f"https://db.itkc.or.kr/dir/item?itemId=BT#/dir/node?dataId={data_id}"
            print(f"BT_SJ 확장 중: {data_id}")
            
            driver.get(url)
            time.sleep(0.5)
            
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # td.third.gisa_tit.jusok_block 내의 a 태그들 추출
            td_elems = soup.select('td.third.gisa_tit.jusok_block')
            
            expanded_items = []
            
            # 1차 시도: 직접 추출
            if td_elems:
                print(f"  1차 시도: {len(td_elems)}개 발견")
                for td in td_elems:
                    a_tag = td.find('a')
                    if a_tag and a_tag.get('href'):
                        href = a_tag.get('href')
                        if 'dataId=' in href:
                            sub_data_id = href.split('dataId=')[1].split('&')[0]
                            # 제목에서 순서 번호 제거
                            title_text = a_tag.get_text(strip=True)
                            title_clean = re.sub(r'^\d+', '', title_text)
                            
                            expanded_items.append({
                                'dci_s': sub_data_id,
                                'title': title_clean,
                                'parent_data_id': data_id
                            })
                            print(f"  하위 문서: {title_clean[:30]}")
            
            # 2차 시도: jusok_block 다음의 second_jusok_block 링크들을 순회하며 클릭
            else:
                print("  1차 시도 실패, 2차 시도: jusok_block 내부 탐색")
                try:
                    # jusok_block 개수 확인
                    jusok_blocks_count = len(driver.find_elements(By.CSS_SELECTOR, 'th.jusok_block'))
                    print(f"  {jusok_blocks_count}개의 jusok_block 발견")
                    
                    # Stale element 방지: 매번 jusok_block을 다시 찾기
                    for block_idx in range(jusok_blocks_count):
                        try:
                            # 매번 jusok_block을 다시 찾기
                            jusok_blocks = driver.find_elements(By.CSS_SELECTOR, 'th.jusok_block')
                            if block_idx >= len(jusok_blocks):
                                break
                            
                            block = jusok_blocks[block_idx]
                            block_title_full = block.text.strip() if block.text else f"Block {block_idx + 1}"
                            block_title = block_title_full[:30]
                            
                            # 연려실기술인 경우 제12권~제18권만 처리
                            if is_yeolyeo:
                                if not any(vol in block_title_full for vol in yeolyeo_volumes):
                                    print(f"\n  [{block_idx + 1}/{jusok_blocks_count}] jusok_block: {block_title} - 건너뜀")
                                    continue
                            
                            print(f"\n  [{block_idx + 1}/{jusok_blocks_count}] jusok_block: {block_title}")
                            
                            # XPath를 사용하여 현재 th.jusok_block의 부모 tr 다음에 오는 td.second_jusok_block 찾기
                            # following-sibling::tr/td[@class='second_jusok_block']
                            parent_tr = block.find_element(By.XPATH, '..')
                            
                            # 링크 개수 확인
                            second_links_count = len(parent_tr.find_elements(By.XPATH, 
                                'following-sibling::tr/td[contains(@class, "second jusok_block")]/a'))
                            print(f"    → {second_links_count}개의 second jusok_block 링크 발견")
                            
                            # Stale element 방지: 매번 링크를 다시 찾아서 클릭
                            for idx in range(second_links_count):
                                try:
                                    # 매번 링크를 다시 찾기 (DOM 변경 대응)
                                    current_block = driver.find_elements(By.CSS_SELECTOR, 'th.jusok_block')[block_idx]
                                    current_parent_tr = current_block.find_element(By.XPATH, '..')
                                    current_links = current_parent_tr.find_elements(By.XPATH, 
                                        'following-sibling::tr/td[contains(@class, "second jusok_block")]/a')
                                    
                                    if idx >= len(current_links):
                                        break
                                    
                                    link = current_links[idx]
                                    link_text = link.text.strip()
                                    print(f"      [{idx+1}/{second_links_count}] 클릭 중: {link_text[:30]}")
                                    
                                    # 링크 클릭
                                    link.click()
                                    time.sleep(1.0)  # 페이지 로딩 대기 시간 증가
                                    
                                    # 페이지 로딩 완료 대기
                                    try:
                                        WebDriverWait(driver, 10).until(
                                            EC.presence_of_element_located((By.CSS_SELECTOR, 'td.third.gisa_tit.jusok_block'))
                                        )
                                    except:
                                        pass
                                    
                                    # 현재 페이지에서 td.third.gisa_tit.jusok_block 추출
                                    current_html = driver.page_source
                                    current_soup = BeautifulSoup(current_html, 'html.parser')
                                    third_elems = current_soup.select('td.third.gisa_tit.jusok_block')
                                    
                                    print(f"        → {len(third_elems)}개 하위 문서 발견")
                                    
                                    for td in third_elems:
                                        a_tag = td.find('a')
                                        if a_tag and a_tag.get('href'):
                                            href = a_tag.get('href')
                                            if 'dataId=' in href:
                                                sub_data_id = href.split('dataId=')[1].split('&')[0]
                                                # 제목에서 순서 번호 제거 (예: "7여름 6월" -> "여름 6월")
                                                title_text = a_tag.get_text(strip=True)
                                                title_clean = re.sub(r'^\d+', '', title_text)
                                                
                                                # 상위 문서 정보 포함 (예: "병오(丙午, 1606) - 봄 정월")
                                                full_title = f"{link_text} - {title_clean}"
                                                
                                                expanded_items.append({
                                                    'dci_s': sub_data_id,
                                                    'title': full_title,
                                                    'parent_data_id': data_id
                                                })
                                                print(f"          - {title_clean[:30]}")
                                    
                                    # 뒤로 가기 (원래 목록 페이지로 돌아가기)
                                    driver.back()
                                    time.sleep(1.0)  # 뒤로 가기 후 페이지 안정화 대기
                                    
                                    # 원래 페이지로 돌아왔는지 확인
                                    try:
                                        WebDriverWait(driver, 5).until(
                                            EC.presence_of_element_located((By.CSS_SELECTOR, 'th.jusok_block'))
                                        )
                                    except:
                                        pass
                                    
                                except Exception as link_err:
                                    print(f"        링크 클릭 실패: {link_err}")
                                    try:
                                        driver.back()  # 에러 발생 시에도 뒤로 가기 시도
                                        time.sleep(0.3)
                                    except:
                                        pass
                                    continue
                        
                        except Exception as block_err:
                            print(f"    jusok_block 처리 실패: {block_err}")
                            continue
                    
                except Exception as click_err:
                    print(f"  2차 시도 실패: {click_err}")
            
            print(f"  총 {len(expanded_items)}개 하위 문서 발견")
            return expanded_items
            
        finally:
            driver.quit()
            
    except Exception as e:
        print(f"BT_SJ 확장 오류 ({data_id}): {e}")
        return []


def _build_tasks_from_search_json(search_results, max_items: Optional[int] = None):
    """검색 결과에서 작업 목록 생성
    
    Args:
        search_results: 파일 경로(str) 또는 crawler 객체
    """
    # 파일 경로인 경우
    if isinstance(search_results, str):
        with open(search_results, 'r', encoding='utf-8') as f:
            search_data = json.load(f)
        keyword = search_data.get('search_info', {}).get('keyword', '')
        results_dict = search_data.get('results', {})
    # crawler 객체인 경우
    else:
        keyword = search_results.search_info.get('keyword', '')
        results_dict = search_results.results
    
    tasks = []

    def add_items(category_key: str, category_name: str):
        items = results_dict.get(category_key, [])
        for item in items:
            raw_data = item.get('raw_data', {})
            sec_id = item.get('sec_id', '')
            
            # BT_SJ인 경우 하위 문서들로 확장
            if sec_id == 'BT_SJ':
                print(f"\nBT_SJ 항목 처리: {raw_data.get('서명', '')}")
                expanded = _expand_bt_sj_items(item)
                
                for exp_item in expanded:
                    tasks.append({
                        'dci_s': exp_item['dci_s'],
                        'category': category_name,
                        'keyword': item.get('keyword', keyword),
                        'is_bt_sj': True,
                        'meta': {
                            '저자': raw_data.get('저자', item.get('저자', '')),
                            '저자몰년': raw_data.get('저자몰년', item.get('저자몰년', '')),
                            '권차명': raw_data.get('권차명', item.get('권차명', '')),
                            '주제분류': raw_data.get('주제분류', item.get('주제분류', '')),
                            '서명': raw_data.get('서명', item.get('서명', '')),
                            '기사명': exp_item['title'],
                            '부모자료ID': exp_item['parent_data_id'],
                        }
                    })
            else:
                # 일반 항목 처리
                dci_s = item.get('DCI_s') or raw_data.get('DCI_s')
                if not dci_s:
                    continue
                
                tasks.append({
                    'dci_s': dci_s,
                    'category': category_name,
                    'keyword': item.get('keyword', keyword),
                    'meta': {
                        '저자': raw_data.get('저자', item.get('저자', '')),
                        '저자몰년': raw_data.get('저자몰년', item.get('저자몰년', '')),
                        '권차명': raw_data.get('권차명', item.get('권차명', '')),
                        '주제분류': raw_data.get('주제분류', item.get('주제분류', '')),
                        '서명': raw_data.get('서명', item.get('서명', '')),
                        '기사명': raw_data.get('기사명', item.get('기사명', '')),
                    }
                })

    add_items('BT', '고전번역서')
    add_items('JT', '조선왕조실록')

    if max_items is not None:
        tasks = tasks[:max_items]
    return tasks, keyword


def _worker_process(task) -> Dict:
    """워커 프로세스: 본문 수집"""
    dci_s = task['dci_s']
    is_bt_sj = task.get('is_bt_sj', False)
    
    # BT_SJ인 경우 다른 URL 형식 사용
    if is_bt_sj:
        url = f"https://db.itkc.or.kr/dir/item?itemId=BT#/dir/node?dataId={dci_s}"
    else:
        url = f"http://db.itkc.or.kr/inLink?DCI={dci_s}"

    result: Dict = {
        'dci_s': dci_s,
        'url': url,
        'main_content': '',
        'annotation': '',
        'content': '',
        'is_bt_sj': is_bt_sj,
    }

    if not SELENIUM_AVAILABLE:
        result['error'] = 'Selenium not available'
    else:
        try:
            global g_driver
            
            # 이전 URL 저장
            previous_url = g_driver.current_url if g_driver else None
            
            # 새 페이지 로드
            g_driver.get(url)
            
            # 페이지가 실제로 변경될 때까지 대기
            if previous_url:
                try:
                    WebDriverWait(g_driver, 5).until(
                        lambda d: d.current_url != previous_url
                    )
                except Exception:
                    pass
            
            # 추가 대기: 페이지 로드 완료 확인
            time.sleep(0.5)
            
            # text_body 요소가 로드될 때까지 대기 (필수)
            try:
                WebDriverWait(g_driver, 10).until(
                    EC.any_of(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'div.text_body')),
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'div.text_body2'))
                    )
                )
                # 요소가 로드된 후 추가 안정화 대기
                time.sleep(0.3)
            except Exception as wait_err:
                # 대기 실패 시에도 일단 진행 (빈 페이지일 수 있음)
                pass

            html = g_driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            t1 = soup.select_one('div.text_body')
            t2 = soup.select_one('div.text_body2')
            main_content = t1.get_text(separator='\n', strip=True) if t1 else ''
            annotation = t2.get_text(separator='\n', strip=True) if t2 else ''

            result.update({
                'main_content': main_content,
                'annotation': annotation,
                'content': main_content,
            })
            try:
                cleaned, structured = tidy_annotations(html, main_content, annotation)
                result['annotation_clean_text'] = cleaned
                result['annotations_structured'] = structured
            except Exception:
                pass
        except Exception as e:
            result['error'] = str(e)

    # RAG용 간소화된 형태로 반환
    rag_result = {
        'url': result.get('url', ''),
        'content': result.get('main_content', ''),
        'metadata': {
            '저자': task['meta'].get('저자', ''),
            '서명': task['meta'].get('서명', ''),
            '제목': task['meta'].get('기사명', ''),
            '주제분류': task['meta'].get('주제분류', ''),
            '카테고리': task['category'],
            '키워드': task['keyword'],
        }
    }
    
    # 주석이 있으면 추가
    if result.get('annotation_clean_text'):
        rag_result['annotation'] = result['annotation_clean_text']
    
    # 필터링을 위해 임시로 필요한 정보 추가
    rag_result['_temp_main_content'] = result.get('main_content', '')
    rag_result['_temp_annotation'] = result.get('annotation', '')
    rag_result['_temp_is_bt_sj'] = result.get('is_bt_sj', False)
    
    return rag_result


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


def tidy_annotations(raw_html: str, main_content: str, annotation_text: str):
    """주석 정리"""
    label_map = {}
    soup = BeautifulSoup(raw_html or '', 'html.parser')
    for sp in soup.select('span.jusok[data-jusok-id]'):
        jid = sp.get('data-jusok-id')
        label_map[jid] = sp.get_text(strip=True)

    txt = (annotation_text or '').replace('\xa0', ' ')
    pattern = re.compile(r"\[주-(D\d{3})\]\s*(.+?)\s*[:：]\s*\n(.*?)(?=\n\[주-|\Z)", re.S)
    items = []
    for m in pattern.finditer(txt):
        jid, title, desc = m.group(1), m.group(2), m.group(3)
        title = ' '.join(title.split())
        desc = desc.strip()
        if ('…' in title or '...' in title) and jid in label_map:
            title = label_map[jid]
        items.append({'id': jid, 'label': title, 'text': desc})

    cleaned = '\n\n'.join([f"[{it['id']}] {it['label']}\n{it['text']}" for it in items])
    return cleaned, items


def collect_parallel(search_results, output_dir: str = "output", max_items: Optional[int] = None, num_workers: int = 2):
    """병렬로 본문 수집 (파일 또는 크롤러 객체)"""
    print(f"\nITKC 본문 병렬 수집 시작")
    
    tasks, keyword = _build_tasks_from_search_json(search_results, max_items=max_items)
    print(f"작업: {len(tasks)}개 | 워커: {num_workers}개")

    os.makedirs(output_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    results = []
    start = time.time()
    with ctx.Pool(processes=num_workers, initializer=init_worker) as pool:
        for i, content in enumerate(pool.imap_unordered(_worker_process, tasks, chunksize=1), 1):
            results.append(content)
            status = 'OK' if content.get('content') else 'FAIL'
            title = content.get('metadata', {}).get('제목', '')[:30]
            print(f"[{i}/{len(tasks)}] {status} {title}")

    elapsed = time.time() - start
    print(f"소요시간: {elapsed:.1f}s")

    def _has_kw(s: str) -> bool:
        return keyword in s if isinstance(s, str) else False

    before_cnt = len(results)
    # BT_SJ 항목은 키워드 필터링 건너뛰기
    filtered = [
        r for r in results
        if r.get('_temp_is_bt_sj', False) or
        (_has_kw(r.get('_temp_main_content', '')) and
         not _has_kw(r.get('_temp_annotation', '')))
    ]
    removed = before_cnt - len(filtered)
    print(f"키워드 필터: 본문포함 AND 주석미포함 -> 유지 {len(filtered)}개 / 제거 {removed}개")

    # 임시 필터링용 필드 제거
    for item in filtered:
        item.pop('_temp_main_content', None)
        item.pop('_temp_annotation', None)
        item.pop('_temp_is_bt_sj', None)

    json_filename = os.path.join(output_dir, f'문헌정보_상세_{keyword}.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"JSON 저장: {json_filename}")

    success_count = sum(1 for c in filtered if c.get('content'))
    print(f"\n병렬 수집 완료")
    print(f"성공: {success_count}개")
    print(f"실패: {len(filtered) - success_count}개")
    print(f"전체: {len(filtered)}개")


def collect_content_from_search(
    search_results,
    output_dir: str = "output",
    max_items: Optional[int] = None,
    num_workers: int = 4
):
    """검색 결과에서 본문 수집
    
    Args:
        search_results: 파일 경로(str) 또는 crawler 객체
    """
    # 파일 경로인 경우 존재 여부 확인
    if isinstance(search_results, str) and not os.path.exists(search_results):
        print(f"파일 없음: {search_results}")
        return []
    
    return collect_parallel(search_results, output_dir, max_items, num_workers)
