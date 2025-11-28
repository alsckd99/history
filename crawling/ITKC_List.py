import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import html
import re


class ItkcSearchCrawler:
    """한국고전종합DB(ITKC) OpenAPI 검색 크롤러"""
    
    # ITKC 카테고리 코드
    CATEGORY_CODES = {
        'BT': [  # 고전번역서 전체
            'BT_AA',
            'BT_KW',
            'BT_SJ'
        ],
        'JT': [  # 조선왕조실록
            'JT_AA'
        ],
        'PR': [  # 사건정보
            'PR_BB'
        ],
        'TR': [  # 사건정보
            'TR_AA'
        ]

    }
    
    def __init__(self):
        """초기화"""
        self.base_url = "http://db.itkc.or.kr/openapi/search"
        self.results = {
            'BT': [],  # 고전번역서
            'JT': [],  # 조선왕조실록
            'TR': []   # 용어정보
        }
        # 사건정보는 별도의 영역에 저장
        self.event_results: List[Dict] = []
        self.search_info = {
            'keyword': '',
            'total_count': {
                'BT': 0,
                'JT': 0,
                'PR': 0,
                'TR': 0
            },
            'search_time': ''
        }
    
    def search_category(
        self,
        keyword: str,
        sec_id: str,
        rows: int = 200,
        max_results: Optional[int] = None,
        parallel: bool = False,
        max_workers: int = 8,
    ) -> List[Dict]:
        """특정 카테고리에서 키워드 검색"""
        if sec_id.startswith("BT"):
            category_name = "고전번역서"
        elif sec_id.startswith("JT"):
            category_name = "조선왕조실록"
        elif sec_id.startswith("PR"):
            category_name = "사건정보"
        elif sec_id.startswith("TR"):
            category_name = "용어정보"
        else:
            category_name = "기타"
        
        print(f"\n[{sec_id}] 검색 시작: '{keyword}'")
        
        all_results = []
        start = 0  # API는 0부터 시작
        total_count = None
        
        # 병렬 수집만 사용
        try:
            if rows > 200:
                rows = 200
            params = {
                'keyword': keyword,
                'secId': sec_id,
                'start': start,
                'rows': rows
            }
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            xml_content = response.text
            root = ET.fromstring(xml_content)
            header = root.find('header')
            if header is not None:
                for field in header.findall('field'):
                    if field.get('name') == 'totalCount':
                        total_count = int(field.text)
                        print(f"[{sec_id}] totalCount: {total_count}")
                        break
            result_element = root.find('result')
            if result_element is None:
                print(f"[{category_name}] 결과 없음")
                print(f"[{category_name}] 검색 완료: 총 {len(all_results):,}개")
                return all_results
            first_docs = result_element.findall('doc')

            def add_docs_to_results(docs_list: List[ET.Element]) -> bool:
                for doc in docs_list:
                    doc_data = {}
                    for field in doc.findall('field'):
                        field_name = field.get('name')
                        field_value = field.text if field.text else ''
                        doc_data[field_name] = field_value
                    result = {
                        'rank': len(all_results) + 1,
                        'category': category_name,
                        'sec_id': sec_id,
                        'keyword': keyword,
                        'DCI_s': doc_data.get('DCI_s', ''),
                        'raw_data': doc_data,
                    }
                    all_results.append(result)
                    if max_results and len(all_results) >= max_results:
                        return True
                return False

            if first_docs:
                stop = add_docs_to_results(first_docs)
                if stop:
                    print(f"[{category_name}] 최대 결과 수 {max_results} 도달")
                    print(f"[{category_name}] 검색 완료: 총 {len(all_results):,}개")
                    return all_results

            # 남은 페이지 시작 지점 계산
            remaining_starts: List[int] = []
            if total_count is not None:
                remaining_starts = list(range(rows, total_count, rows))
            else:
                # totalCount가 없으면 첫 페이지 결과만 반환
                print(f"[{category_name}] totalCount 없음. 첫 페이지 결과만 사용")
                print(f"[{category_name}] 검색 완료: 총 {len(all_results):,}개")
                return all_results

            def fetch_page(page_start: int):
                p = {
                    'keyword': keyword,
                    'secId': sec_id,
                    'start': page_start,
                    'rows': rows
                }
                try:
                    r = requests.get(self.base_url, params=p, timeout=30)
                    r.raise_for_status()
                    r.encoding = 'utf-8'
                    xr = r.text
                    rt = ET.fromstring(xr)
                    re = rt.find('result')
                    if re is None:
                        return (page_start, [])
                    return (page_start, re.findall('doc'))
                except Exception:
                    return (page_start, [])

            page_results: Dict[int, List[ET.Element]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(fetch_page, s): s for s in remaining_starts}
                for future in as_completed(futures):
                    s, docs = future.result()
                    page_results[s] = docs

            # 시작 지점 기준으로 정렬하여 추가
            for s in sorted(page_results.keys()):
                docs = page_results[s]
                if not docs:
                    continue
                stop = add_docs_to_results(docs)
                if stop:
                    break

        except requests.exceptions.RequestException as e:
            print(f"[{category_name}] 요청 오류: {e}")
        except ET.ParseError as e:
            print(f"[{category_name}] XML 파싱 오류: {e}")
        except Exception as e:
            print(f"[{category_name}] 예상치 못한 오류: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n[{category_name}] 검색 완료: 총 {len(all_results):,}개")
        
        return all_results

    def _fetch_data_solr_id_by_dci(self, dci_s: str, sec_id: str = 'PR_BB') -> Optional[str]:
        """DCI_s로 data-solr-id 조회"""
        if not dci_s:
            return None
        try:
            url = 'https://db.itkc.or.kr/search/list'
            params = {'q': dci_s, 'secId': sec_id}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
            }
            print(f"[PR lookup] by DCI_s -> GET {url} q={dci_s} secId={sec_id}")
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'lxml')

            # dci_s와 일치하는 tr가 있으면 우선 사용
            exact = soup.find('tr', attrs={'data-solr-id': dci_s})
            if exact is not None:
                print(f"[PR lookup] exact match found: {dci_s}")
                return exact.get('data-solr-id')

            # 아니면 첫 번째 tr[data-solr-id] 사용
            first = soup.select_one('tr[data-solr-id]')
            if first is not None:
                print(f"[PR lookup] first result used: {first.get('data-solr-id')}")
                return first.get('data-solr-id')
            print("[PR lookup] no tr[data-solr-id] found")
        except Exception as e:
            print(f"[사건정보] data-solr-id 조회 실패 ({dci_s}): {e}")
        return None

    def _fetch_data_solr_id_by_query(self, query: str, sec_id: str = 'PR_BB') -> Optional[str]:
        """키워드로 data-solr-id 조회"""
        if not query:
            return None
        try:
            url = 'https://db.itkc.or.kr/search/list'
            params = {'q': query, 'secId': sec_id}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
            }
            print(f"[PR lookup] by query -> GET {url} q={query} secId={sec_id}")
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'lxml')
            first = soup.select_one('tr[data-solr-id]')
            if first is not None:
                print(f"[PR lookup] query first result: {first.get('data-solr-id')}")
                return first.get('data-solr-id')
            print("[PR lookup] query no tr[data-solr-id] found")
        except Exception as e:
            print(f"[사건정보] query data-solr-id 조회 실패 ({query}): {e}")
        return None

    def _fetch_solr_id_list_by_keyword(self, keyword: str, sec_id: str = 'PR_BB', expected: int = 0, page_unit: int = 20) -> List[str]:
        """키워드로 모든 data-solr-id 수집"""
        if not keyword:
            return []
        solr_ids: List[str] = []
        try:
            url = 'https://db.itkc.or.kr/search/list'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
            }

            # 1) 첫 페이지 요청 (pageIndex=1)
            params = {
                'q': f"query†{keyword}",
                'secId': sec_id,
                'pageIndex': 1,
                'pageUnit': page_unit,
            }
            print(f"[PR list] GET {url} q=query†{keyword} secId={sec_id} pageIndex=1 pageUnit={page_unit}")
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'lxml')

            # 수집 함수
            def collect_ids(s: BeautifulSoup):
                rows = s.select('tr[data-solr-id]')
                for row in rows:
                    sid = row.get('data-solr-id')
                    if sid:
                        solr_ids.append(sid)

            collect_ids(soup)

            # 총 페이지 수 파싱
            page_count = 1
            pager = soup.select_one('div.num')
            if pager:
                for a in pager.find_all('a'):
                    txt = (a.get_text() or '').strip()
                    try:
                        n = int(txt)
                        if n > page_count:
                            page_count = n
                    except Exception:
                        continue
            print(f"[PR list] collected {len(solr_ids)} (page 1/{page_count})")

            if expected and len(solr_ids) >= expected:
                return solr_ids[:expected]

            # 2) 나머지 페이지 순회
            for page_idx in range(2, page_count + 1):
                params['pageIndex'] = page_idx
                print(f"[PR list] GET {url} q=query†{keyword} secId={sec_id} pageIndex={page_idx} pageUnit={page_unit}")
                r = requests.get(url, params=params, headers=headers, timeout=30)
                r.raise_for_status()
                s2 = BeautifulSoup(r.text, 'lxml')
                collect_ids(s2)
                print(f"[PR list] collected {len(solr_ids)} (page {page_idx}/{page_count})")
                if expected and len(solr_ids) >= expected:
                    break

            return solr_ids if not expected else solr_ids[:expected]
        except Exception as e:
            print(f"[사건정보] data-solr-id 리스트 조회 실패: {e}")
            return solr_ids

    def _enrich_event_results_with_solr_id(self):
        """사건정보에 data-solr-id 추가"""
        if not self.event_results:
            print("[PR enrich] no event_results to enrich")
            return
        keyword = self.search_info.get('keyword', '')
        solr_ids = self._fetch_solr_id_list_by_keyword(keyword, 'PR_BB', expected=len(self.event_results), page_unit=20)
        print(f"[PR enrich] mapping event_results({len(self.event_results)}) with solr_ids({len(solr_ids)}) by order")
        n = min(len(self.event_results), len(solr_ids))
        base_url = "https://db.itkc.or.kr/people/item?gubun=evnt#/viewEvnt?gubun=evntcate&cate1=Z&cate2=&dataId="
        for idx in range(n):
            item = self.event_results[idx]
            sid = solr_ids[idx]
            # 요청: data_solr_id 필드는 제거, DCI_s에 최종 URL 저장
            if 'data_solr_id' in item:
                del item['data_solr_id']
            item['DCI_s'] = f"{base_url}{sid}"
            print(f"[PR enrich] #{idx+1} -> {sid}")
        # 나머지 항목은 비워둠
        for idx in range(n, len(self.event_results)):
            item = self.event_results[idx]
            if 'data_solr_id' in item:
                del item['data_solr_id']
            item['DCI_s'] = ''
            print(f"[PR enrich] #{idx+1} -> no id available")
    
    def search_all(
        self,
        keyword: str,
        categories: List[str] = ['BT', 'JT'],
        max_results_per_category: Optional[int] = None,
        parallel: bool = False,
        max_workers: int = 4,
    ):
        """지정된 카테고리에서 모두 검색"""
        print(f"\nITKC OpenAPI 검색 시작")
        print(f"키워드: {keyword}")
        print(f"카테고리: {', '.join(categories)}")
        if max_results_per_category:
            print(f"카테고리당 최대: {max_results_per_category:,}개")
        else:
            print("결과 수: 전체")
        
        self.search_info['keyword'] = keyword
        self.search_info['search_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 각 카테고리 검색
        for category in categories:
            if category not in self.CATEGORY_CODES:
                print(f"지원하지 않는 카테고리: {category}")
                continue
            
            category_names = {
                'BT': '고전번역서',
                'JT': '조선왕조실록'
            }
            
            print(f"\n{category_names.get(category, category)} 전체 검색 ({len(self.CATEGORY_CODES[category])}개 서브카테고리)")
            category_results = []
            
            for sec_id in self.CATEGORY_CODES[category]:
                results = self.search_category(keyword, sec_id, max_results=max_results_per_category, parallel=parallel, max_workers=max_workers)
                category_results.extend(results)
                if max_results_per_category and len(category_results) >= max_results_per_category:
                    category_results = category_results[:max_results_per_category]
                    break
            
            # 순위 재정렬
            for i, item in enumerate(category_results, 1):
                item['rank'] = i
            
            if category == 'PR':
                # 사건정보는 별도 보관
                self.event_results = category_results
                self.search_info['total_count']['PR'] = len(category_results)
            else:
                self.results[category] = category_results
                self.search_info['total_count'][category] = len(category_results)
        
        # 사건정보 보강: data-solr-id 채우기
        self._enrich_event_results_with_solr_id()

        # 결과 요약
        self.print_summary()
    
    def search_specific_codes(
        self,
        keyword: str,
        sec_ids: List[str],
        max_results: Optional[int] = None,
        parallel: bool = False,
        max_workers: int = 8,
    ):
        """특정 카테고리 코드만 검색"""
        print(f"\nITKC OpenAPI 특정 카테고리 검색")
        print(f"키워드: {keyword}")
        print(f"카테고리 코드: {', '.join(sec_ids)}")
        if max_results:
            print(f"최대 결과: {max_results:,}개")
        else:
            print("결과 수: 전체")
        
        self.search_info['keyword'] = keyword
        self.search_info['search_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        all_results = []
        
        for sec_id in sec_ids:
            results = self.search_category(keyword, sec_id, max_results=max_results, parallel=parallel, max_workers=max_workers)
            all_results.extend(results)
            
            if max_results and len(all_results) >= max_results:
                all_results = all_results[:max_results]
                break
        
        # 결과를 카테고리별로 분류
        for result in all_results:
            sec_id = result.get('sec_id', '')
            if sec_id.startswith('BT'):
                self.results['BT'].append(result)
            elif sec_id.startswith('JT'):
                self.results['JT'].append(result)
            elif sec_id.startswith('PR'):
                self.event_results.append(result)
            elif sec_id.startswith('TR'):
                self.results['TR'].append(result)
        
        # 카운트 업데이트
        self.search_info['total_count']['BT'] = len(self.results['BT'])
        self.search_info['total_count']['JT'] = len(self.results['JT'])
        self.search_info['total_count']['PR'] = len(self.event_results)
        self.search_info['total_count']['TR'] = len(self.results['TR'])
        
        # 순위 재정렬
        for category in ['BT', 'JT', 'TR']:
            for i, item in enumerate(self.results[category], 1):
                item['rank'] = i
        
        # 사건정보 보강: data-solr-id 채우기
        self._enrich_event_results_with_solr_id()

        # 결과 요약
        self.print_summary()
    
    def print_summary(self):
        """검색 결과 요약 출력"""
        print(f"\n검색 결과 요약")
        print(f"키워드: {self.search_info['keyword']}")
        print(f"검색 시간: {self.search_info['search_time']}")
        
        total = 0
        if self.results['BT']:
            print(f"고전번역서(BT): {len(self.results['BT']):,}개 결과")
            total += len(self.results['BT'])
        if self.results['JT']:
            print(f"조선왕조실록(JT): {len(self.results['JT']):,}개 결과")
            total += len(self.results['JT'])
        if self.results['TR']:
            print(f"용어정보(TR): {len(self.results['TR']):,}개 결과")
            total += len(self.results['TR'])
        if self.event_results:
            print(f"사건정보(PR): {len(self.event_results):,}개 결과 ")
        
        print(f"전체: {total:,}개 결과")
    
    def save_results(self, output_dir: str = "output", format: str = "json"):
        """검색 결과를 파일로 저장"""
        if not self.results['BT'] and not self.results['JT']:
            print("저장할 결과 없음")
            return False
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        keyword = self.search_info['keyword']
        
        try:
            # JSON 형식으로 저장 (BT/JT만 포함)
            if format in ['json', 'both']:
                json_filename = os.path.join(output_dir, f"itkc_search_{keyword}.json")
                json_data = {
                    'search_info': self.search_info,
                    'results': self.results,
                    'summary': {
                        'total_bt': len(self.results['BT']),
                        'total_jt': len(self.results['JT']),
                        'total': len(self.results['BT']) + len(self.results['JT'])
                    }
                }
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"JSON 저장: {json_filename}")
            
            return True
        
        except Exception as e:
            print(f"파일 저장 오류: {e}")
            return False
    
    def save_separate_files(self, output_dir: str = "output", format: str = "json"):
        """각 카테고리를 별도 파일로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        keyword = self.search_info['keyword']
        
        # 고전번역서 저장 (raw_data 포함)
        if self.results['BT']:
            if format in ['json', 'both']:
                bt_json = os.path.join(output_dir, f"고전번역서_{keyword}.json")
                with open(bt_json, 'w', encoding='utf-8') as f:
                    json.dump(self.results['BT'], f, ensure_ascii=False, indent=2)
                print(f"JSON 저장: {bt_json}")
        # 조선왕조실록 저장 (raw_data 포함)
        if self.results['JT']:
            if format in ['json', 'both']:
                jt_json = os.path.join(output_dir, f"조선왕조실록_{keyword}.json")
                with open(jt_json, 'w', encoding='utf-8') as f:
                    json.dump(self.results['JT'], f, ensure_ascii=False, indent=2)
                print(f"JSON 저장: {jt_json}")
        
        # 용어정보 저장 (raw_data 정리 후 저장)
        if self.results['TR']:
            if format in ['json', 'both']:
                # raw_data 정리: HTML entity 디코드 및 태그 제거
                cleaned_tr_results = []
                for item in self.results['TR']:
                    cleaned_item = item.copy()
                    if 'raw_data' in cleaned_item:
                        cleaned_raw = {}
                        for key, value in cleaned_item['raw_data'].items():
                            if isinstance(value, str):
                                # HTML entity 디코드 (&#x2F; → /, &quot; → " 등)
                                decoded = html.unescape(value)
                                # HTML 태그 제거 (<em class="hl1">텍스트</em> → 텍스트)
                                cleaned = re.sub(r'<[^>]+>', '', decoded)
                                cleaned_raw[key] = cleaned
                            else:
                                cleaned_raw[key] = value
                        cleaned_item['raw_data'] = cleaned_raw
                    cleaned_tr_results.append(cleaned_item)
                
                tr_json = os.path.join(output_dir, f"용어정보_{keyword}.json")
                with open(tr_json, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_tr_results, f, ensure_ascii=False, indent=2)
                print(f"JSON 저장: {tr_json}")
        
        # 사건정보 저장 (별도 파일)
        if self.event_results:
            if format in ['json', 'both']:
                pr_json = os.path.join(output_dir, f"사건정보_{keyword}.json")
                with open(pr_json, 'w', encoding='utf-8') as f:
                    json.dump(self.event_results, f, ensure_ascii=False, indent=2)
                print(f"JSON 저장: {pr_json}")
            

def search_and_save(
    keyword: str,
    sec_ids: List[str],
    output_dir: str = "output",
    parallel: bool = True,
    max_workers: int = 4,
    save_separate: bool = True,
    save_to_file: bool = True
):
    """검색 및 저장"""
    crawler = ItkcSearchCrawler()
    crawler.search_specific_codes(keyword, sec_ids, parallel=parallel, max_workers=max_workers)
    
    # 파일 저장 여부 확인
    if save_to_file:
        if save_separate:
            crawler.save_separate_files(output_dir=output_dir, format="json")
        else:
            crawler.save_results(output_dir=output_dir, format="json")
    
    return crawler

