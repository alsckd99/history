#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
분석 항목:
1. 정량적 데이터 분석 (키워드 빈도, 시계열, 주제 모델링)
2. 관계망 및 개체명 분석 (NER, 인물 관계망)
3. 지리공간 및 전략 분석
4. 감성 및 언어적 특징 분석
5. 사료 비판 및 교차 검증
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from langchain_community.llms import Ollama
except ImportError:
    from langchain.llms import Ollama

from graph_db import ArangoGraphDB

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("numpy 없음: pip install numpy")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib 없음: pip install matplotlib")


class AdvancedDocumentAnalyzer:
    """고급 문서 분석기"""
    
    def __init__(
        self,
        llm_model='gemma3:12b',
        arango_host='localhost',
        arango_port=8529
    ):
        """초기화"""
        print("\n고급 문서 분석 시스템 초기화 중...")
        
        # LLM 초기화
        try:
            self.llm = Ollama(model=llm_model, temperature=0.3)
            print(f"LLM 초기화 완료: {llm_model}")
        except Exception as e:
            print(f"LLM 초기화 실패: {e}")
            self.llm = None
        
        # 그래프 DB
        self.graph_db = ArangoGraphDB(
            host=arango_host,
            port=arango_port
        )
        
        # 문서 형식/주제 카테고리
        self.format_categories = [
            "일기", "사실 기록", "공문서", "편지", "보고서",
            "역사서", "개인 기록", "실록", "장계(狀啓)", "기타"
        ]
        
        self.topic_categories = [
            "전투/전쟁", "전술/전략", "정치/정책", "경제/재정", 
            "물자/보급", "인사/행정", "외교/국제", "사법/처벌",
            "개인/감정", "가족", "일상/생활", "종교/사상", "기타"
        ]
        # 그래프 기반 연구 주제 설정
        self.theme_category_map = {
            'event': '전투/전쟁',
            'organization': '정책/행정',
            'dynasty': '정치/제도',
            'person': '인물/리더십',
            'location': '전술/지리',
            'date': '시계열/연대',
            'entity': '일반/기타'
        }
        self.theme_descriptions = {
            '전투/전쟁': '전황 보고, 지형, 지휘 체계 등 군사 행위에 집중',
            '정책/행정': '명령·문서 체계, 행정 결정 과정, 인사 보고',
            '정치/제도': '왕조, 제도 개편, 조정 내 정치적 역학',
            '인물/리더십': '핵심 인물의 판단, 갈등, 협력 관계',
            '전술/지리': '지리·항로·거점 등 공간 전략 요소',
            '시계열/연대': '연대기적 기록, 날짜 중심 보고',
            '일반/기타': '상세 분류가 어려운 일반 기록'
        }
        self.default_theme_bank = self._build_default_theme_bank()
        self.graph_keyword_cache = {}
        
        # 문서 형식 감지를 위한 휴리스틱 신호
        self.format_signal_map = {
            "장계/공문": ["장계", "상소", "신이", "아뢰옵", "주상", "전하", "계본"],
            "일기/자기 기록": ["오늘", "이날", "기록하노라", "돌아보니", "내가", "느끼니"],
            "군령/명령서": ["명하여", "전교", "군령", "호령", "지시", "명령"],
            "상황 보고": ["보고", "전황", "첩보", "급보", "계상", "전달"]
        }
        
        # 감정 키워드 (한국어/한자)
        self.sentiment_positive = [
            "다행", "기쁘", "행", "평안", "승", "성공", "위로"
        ]
        self.sentiment_negative = [
            "통분", "분", "우", "비통", "애통", "근심", "걱정", 
            "패", "실패", "원통", "분하", "한탄"
        ]
    
    def analyze_document(self, document: Dict) -> Dict:
        """단일 문서 고급 분석"""
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        if not content:
            return {'error': '문서 내용이 비어있습니다'}
        
        print(f"\n{'='*70}")
        print(f"고급 문서 분석: {metadata.get('source', 'Unknown')}")
        print(f"{'='*70}")
        
        result = {
            'metadata': metadata,
            'content_length': len(content),
            'content_preview': content[:200] + '...' if len(content) > 200 else content
        }
        
        total_steps = 11
        
        # 1. 문서 개요
        print(f"\n[1/{total_steps}] 문서 개요 분석 중...")
        result['overview'] = self._analyze_overview(content, metadata)
        
        # 2. 형식 분류
        print(f"[2/{total_steps}] 문서 형식 분류 중...")
        result['format'] = self._classify_format(content)
        
        # 3. 연구 관점 분석 (주제/내용/형식 단서)
        print(f"[3/{total_steps}] 연구 관점 분석 중...")
        result['research_profile'] = self._analyze_research_profile(content, metadata)
        
        # 4. 정량적 분석 - 키워드 빈도
        print(f"[4/{total_steps}] 키워드 빈도 분석 중...")
        result['keyword_frequency'] = self._analyze_keyword_frequency(content)
        
        # 5. 정량적 분석 - 주제 모델링
        print(f"[5/{total_steps}] 주제 모델링 중...")
        result['topic_modeling'] = self._topic_modeling(content)
        
        # 6. 개체명 인식 (NER)
        print(f"[6/{total_steps}] 개체명 추출 중...")
        result['entities'] = self._extract_entities_ner(content)
        
        # 7. 인물 관계망 분석
        print(f"[7/{total_steps}] 인물 관계망 분석 중...")
        result['network'] = self._analyze_network(content, result.get('entities', {}))
        
        # 8. 감성 분석
        print(f"[8/{total_steps}] 감성 분석 중...")
        result['sentiment'] = self._analyze_sentiment(content)
        
        # 9. 문체 분석
        print(f"[9/{total_steps}] 문체 분석 중...")
        result['style'] = self._analyze_style(content)
        
        # 10. 요약 생성
        print(f"[10/{total_steps}] 문서 요약 생성 중...")
        result['summary'] = self._generate_summary(content)
        
        # 11. 사료 비판 (신뢰도, 편향성)
        print(f"[11/{total_steps}] 사료 비판 분석 중...")
        result['critical_analysis'] = self._critical_analysis(content, result)
        
        print(f"\n분석 완료!")
        return result
    
    def _analyze_overview(self, content: str, metadata: Dict) -> Dict:
        """문서 개요 분석"""
        overview = {
            'author': metadata.get('저자', '미상'),
            'title': metadata.get('서명', metadata.get('제목', '미상')),
            'source': metadata.get('source', ''),
            'category': metadata.get('카테고리', ''),
            'length_chars': len(content),
            'length_words': len(content.split())
        }
        
        # 시기/연대 추출 (일기라면)
        date_pattern = r'(\d{4})년|(\d{1,2})월|(\d{1,2})일'
        dates = re.findall(date_pattern, content[:500])
        if dates:
            overview['period'] = '연대 정보 있음'
        else:
            overview['period'] = '연대 정보 불명확'
        
        return overview
    
    def _classify_format(self, content: str) -> Dict:
        """문서 형식 분류"""
        if not self.llm:
            return {'category': '알 수 없음', 'confidence': 0.0}
        
        prompt = f"""다음 텍스트의 문서 형식을 정확히 분류하세요.

텍스트:
{content[:1500]}

가능한 형식: {', '.join(self.format_categories)}

다음 형식으로 답변하세요:
형식: [형식명]
신뢰도: [0-100]
특징: [문체적 특징 3가지]
사료적 가치: [1차/2차 사료 여부 및 가치]
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            category = '기타'
            confidence = 50.0
            features = []
            source_value = ''
            
            for line in response.split('\n'):
                line = line.strip()
                if '형식:' in line:
                    for cat in self.format_categories:
                        if cat in line:
                            category = cat
                            break
                elif '신뢰도:' in line:
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        confidence = float(numbers[0])
                elif '특징:' in line:
                    features.append(line.split(':', 1)[1].strip() if ':' in line else '')
                elif '사료' in line:
                    source_value = line.split(':', 1)[1].strip() if ':' in line else line
            
            return {
                'category': category,
                'confidence': confidence,
                'features': features[:3],
                'source_value': source_value
            }
        except Exception as e:
            print(f"  형식 분류 오류: {e}")
            return {'category': '알 수 없음', 'confidence': 0.0}
    
    def _build_default_theme_bank(self) -> Dict[str, Dict]:
        """그래프 미사용 시 활용할 기본 키워드 사전"""
        return {
            "전투/군사": {
                "keywords": [
                    "전투", "전장", "교전", "공격", "방어", "포격", "전열",
                    "군사", "병력", "수군", "적선", "장수", "기습", "퇴각"
                ],
                "description": "군사 행동, 전술, 전황 보고"
            },
            "정책/행정": {
                "keywords": [
                    "상소", "장계", "교지", "명령", "의논", "조정", "관찰사",
                    "윤허", "치계", "규정", "품의", "비답", "문서"
                ],
                "description": "정책 결정, 행정 보고, 명령 체계"
            },
            "경제/재정": {
                "keywords": [
                    "군량", "보급", "곡식", "양곡", "세금", "공납", "재정",
                    "물자", "쌀", "은", "총통", "돛", "선박", "화약"
                ],
                "description": "물자 조달, 재정 상황, 장비 확보"
            },
            "민생/생활": {
                "keywords": [
                    "백성", "민심", "민호", "기근", "생활", "피폐", "고통",
                    "민", "호소", "피난", "공포", "백성들", "사람들"
                ],
                "description": "백성의 삶, 민심, 사회상"
            },
            "외교/국제": {
                "keywords": [
                    "사신", "강화", "조약", "외교", "일본", "왜", "명나라",
                    "교섭", "회담", "공문", "통신사", "대마도"
                ],
                "description": "대외 교섭, 국제 관계, 외교 전략"
            },
            "감정/사상": {
                "keywords": [
                    "통분", "비통", "우국", "결의", "자책", "탄식", "염려",
                    "근심", "충절", "명분", "천명", "기도", "신념"
                ],
                "description": "저자의 감정, 사상, 정체성"
            }
        }
    
    def _split_sentences(self, content: str) -> List[str]:
        """간단한 문장 분할"""
        raw_sentences = re.split(r'(?<=[\.?!])\s+|[\n\r]+', content)
        sentences = [sent.strip() for sent in raw_sentences if len(sent.strip()) > 5]
        return sentences[:400]
    
    def _extract_evidence_sentences(
        self, 
        sentences: List[str], 
        keywords, 
        limit: int = 2
    ) -> List[str]:
        """키워드가 포함된 예시 문장 추출"""
        if not sentences or not keywords:
            return []
        
        keyword_list = list(keywords)
        evidences = []
        for sentence in sentences:
            if any(kw in sentence for kw in keyword_list):
                evidences.append(sentence[:300])
                if len(evidences) >= limit:
                    break
        return evidences
    
    def _collect_keyword_stats(self, content: str, keywords: List[str]) -> Tuple[Dict[str, int], int]:
        """키워드별 빈도 수집"""
        stats = {}
        total = 0
        for kw in keywords:
            count = content.count(kw)
            if count > 0:
                stats[kw] = count
                total += count
        return stats, total
    
    def _load_graph_keyword_bank(self, source: Optional[str] = None) -> Dict[str, List[str]]:
        """ArangoDB에서 키워드 사전을 로드"""
        if not self.graph_db or not getattr(self.graph_db, 'db', None):
            return {}
        try:
            keyword_map = self.graph_db.fetch_keywords_by_category(
                source=source,
                limit_per_category=80
            )
            return keyword_map or {}
        except Exception as e:
            print(f"  그래프 키워드 로드 실패: {e}")
            return {}
    
    def _load_graph_keywords_for_doc(self, metadata: Dict) -> Tuple[Dict[str, List[str]], str]:
        """문서 전용 그래프 키워드 → 없으면 글로벌/기본 사전"""
        source = metadata.get('source') or metadata.get('file_path')
        keyword_source = 'graph_document'
        keyword_map = {}
        
        if source:
            keyword_map = self._load_graph_keyword_bank(source)
        
        if not keyword_map:
            keyword_source = 'graph_global'
            if 'global' not in self.graph_keyword_cache:
                self.graph_keyword_cache['global'] = self._load_graph_keyword_bank()
            keyword_map = self.graph_keyword_cache.get('global') or {}
        
        if not keyword_map:
            keyword_source = 'fallback_default'
            keyword_map = {
                theme: data['keywords']
                for theme, data in self.default_theme_bank.items()
            }
        
        return keyword_map, keyword_source
    
    def _derive_theme_label(self, category: Optional[str]) -> str:
        if not category:
            return '일반/기타'
        return self.theme_category_map.get(category, category)
    
    def _theme_description(self, theme_label: str) -> str:
        return self.theme_descriptions.get(theme_label, '주요 맥락 파악용 그래프 기반 주제')
    
    def _load_relation_highlights(self, metadata: Dict, limit: int = 8) -> Tuple[List[Dict], str]:
        """그래프 관계 빈도 정보"""
        if not self.graph_db or not getattr(self.graph_db, 'db', None):
            return [], 'graph_unavailable'
        
        source = metadata.get('source') or metadata.get('file_path')
        origin = 'graph_document'
        relation_stats = []
        try:
            relation_stats = self.graph_db.fetch_relation_types(
                source=source,
                limit=limit
            ) if source else []
            if not relation_stats:
                origin = 'graph_global'
                relation_stats = self.graph_db.fetch_relation_types(limit=limit)
            if not relation_stats:
                origin = 'graph_empty'
            return relation_stats or [], origin
        except Exception as e:
            print(f"  관계 빈도 로드 실패: {e}")
            return [], 'graph_error'
    
    def _detect_format_signals(self, content: str, metadata: Dict) -> Dict:
        """휴리스틱 기반 문서 형식 신호"""
        head = content[:600]
        signals = []
        for label, cues in self.format_signal_map.items():
            hits = [cue for cue in cues if cue in head]
            if hits:
                signals.append({
                    'label': label,
                    'hit_count': len(hits),
                    'examples': hits[:3]
                })
        signals.sort(key=lambda x: x['hit_count'], reverse=True)
        
        date_markers = bool(re.search(r'(\d{4})년|\d{1,2}월|\d{1,2}일', head))
        
        return {
            'metadata_hint': metadata.get('카테고리') or metadata.get('형태') or metadata.get('source', ''),
            'candidates': signals[:3],
            'date_markers': date_markers
        }
    
    def _analyze_research_profile(self, content: str, metadata: Dict) -> Dict:
        """연구 관점(형식/내용/주제) 심층 분석"""
        sentences = self._split_sentences(content)
        
        keyword_map, keyword_source = self._load_graph_keywords_for_doc(metadata)
        
        theme_scores = []
        for category, keywords in keyword_map.items():
            if not keywords:
                continue
            stats, total_hits = self._collect_keyword_stats(content, keywords)
            if total_hits == 0:
                continue
            theme_label = self._derive_theme_label(category)
            density = round((total_hits / max(len(content), 1)) * 1000, 3)
            theme_scores.append({
                'category': category,
                'theme': theme_label,
                'description': self._theme_description(theme_label),
                'keyword_hits': dict(sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]),
                'total_hits': total_hits,
                'density_per_1k_chars': density,
                'keyword_pool_size': len(keywords),
                'score': round(total_hits + density + len(stats) * 0.3, 2),
                'evidence': self._extract_evidence_sentences(sentences, stats.keys())
            })
        
        theme_scores.sort(key=lambda x: x['score'], reverse=True)
        dominant_theme = theme_scores[0]['theme'] if theme_scores else '불명'
        
        relation_highlights, relation_source = self._load_relation_highlights(metadata)
        content_axes = []
        for rel in relation_highlights:
            axis_label = rel.get('type') or '관계'
            axis_hits = content.count(axis_label) if axis_label else 0
            content_axes.append({
                'axis': axis_label,
                'focus': '그래프 관계 기반',
                'keyword_hits': {axis_label: axis_hits} if axis_label else {},
                'relation_count': rel.get('count', 0),
                'evidence': self._extract_evidence_sentences(sentences, [axis_label]) if axis_label else []
            })
        
        format_signals = self._detect_format_signals(content, metadata)
        
        return {
            'dominant_theme': dominant_theme,
            'theme_scores': theme_scores,
            'content_axes': content_axes[:5],
            'format_signals': format_signals,
            'keyword_source': keyword_source,
            'keyword_bank_stats': {cat: len(words) for cat, words in keyword_map.items()},
            'relation_highlights': relation_highlights[:6],
            'relation_source': relation_source
        }
    
    def _analyze_keyword_frequency(self, content: str) -> Dict:
        """키워드 빈도 분석"""
        # 주요 역사 키워드 (확장 가능)
        important_keywords = {
            '전투': ['왜', '적', '적선', '전투', '싸움', '공격', '방어'],
            '인물': ['이순신', '원균', '권율', '선조', '진린', '어영담', '배설'],
            '장소': ['한산도', '부산', '명량', '고하도', '견내량', '노량'],
            '무기/장비': ['거북선', '판옥선', '대포', '화살', '총통'],
            '감정': ['통분', '분', '우', '다행', '평안', '걱정', '근심'],
            '가족': ['어머니', '자친', '아들', '형', '아우'],
            '물자': ['쌀', '군량', '화약', '돛', '밧줄']
        }
        
        frequency = {}
        for category, keywords in important_keywords.items():
            freq_dict = {}
            for kw in keywords:
                count = content.count(kw)
                if count > 0:
                    freq_dict[kw] = count
            if freq_dict:
                frequency[category] = freq_dict
        
        # 전체 빈도 Top 20
        all_keywords_flat = []
        for kws in important_keywords.values():
            all_keywords_flat.extend(kws)
        
        top_keywords = []
        for kw in all_keywords_flat:
            count = content.count(kw)
            if count > 0:
                top_keywords.append({'keyword': kw, 'count': count})
        
        top_keywords.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'by_category': frequency,
            'top_keywords': top_keywords[:20]
        }
    
    def _topic_modeling(self, content: str) -> List[Dict]:
        """주제 모델링"""
        if not self.llm:
            return []
        
        prompt = f"""다음 텍스트에서 주요 주제들을 추출하고 각 주제의 비중을 분석하세요.

텍스트:
{content[:2000]}

가능한 주제: {', '.join(self.topic_categories)}

다음 형식으로 답변하세요 (비중 합계 100%):
주제1: [주제명] - [비중]% - [해당 주제의 핵심 키워드 3개]
주제2: [주제명] - [비중]% - [해당 주제의 핵심 키워드 3개]
...
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            topics = []
            for line in response.split('\n'):
                line = line.strip()
                if '주제' in line and '%' in line:
                    # 주제명 추출
                    topic_name = None
                    for cat in self.topic_categories:
                        if cat in line:
                            topic_name = cat
                            break
                    
                    # 비중 추출
                    weight = 0
                    numbers = re.findall(r'(\d+)%', line)
                    if numbers:
                        weight = int(numbers[0])
                    
                    # 키워드 추출
                    keywords = []
                    if '-' in line:
                        parts = line.split('-')
                        if len(parts) >= 3:
                            keyword_part = parts[2].strip()
                            keywords = [k.strip() for k in keyword_part.split(',')][:3]
                    
                    if topic_name and weight > 0:
                        topics.append({
                            'topic': topic_name,
                            'weight': weight,
                            'keywords': keywords
                        })
            
            topics.sort(key=lambda x: x['weight'], reverse=True)
            return topics
            
        except Exception as e:
            print(f"  주제 모델링 오류: {e}")
            return []
    
    def _extract_entities_ner(self, content: str) -> Dict:
        """개체명 인식 (NER)"""
        if not self.llm:
            return {}
        
        prompt = f"""다음 텍스트에서 개체명을 추출하세요.

텍스트:
{content[:2000]}

다음 형식으로 답변하세요:
인물(PER): [이름1], [이름2], ...
장소(LOC): [장소1], [장소2], ...
기관(ORG): [기관1], [기관2], ...
날짜(DATE): [날짜1], [날짜2], ...
무기/장비(EQUIP): [무기1], [무기2], ...
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            entities = {
                'PER': [],   # 인물
                'LOC': [],   # 장소
                'ORG': [],   # 기관
                'DATE': [],  # 날짜
                'EQUIP': []  # 무기/장비
            }
            
            current_type = None
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # 타입 식별
                if '인물' in line or 'PER' in line:
                    current_type = 'PER'
                    line = line.split(':', 1)[1] if ':' in line else line
                elif '장소' in line or 'LOC' in line:
                    current_type = 'LOC'
                    line = line.split(':', 1)[1] if ':' in line else line
                elif '기관' in line or 'ORG' in line:
                    current_type = 'ORG'
                    line = line.split(':', 1)[1] if ':' in line else line
                elif '날짜' in line or 'DATE' in line:
                    current_type = 'DATE'
                    line = line.split(':', 1)[1] if ':' in line else line
                elif '무기' in line or 'EQUIP' in line:
                    current_type = 'EQUIP'
                    line = line.split(':', 1)[1] if ':' in line else line
                
                # 엔티티 추출
                if current_type:
                    items = [item.strip() for item in line.split(',') if item.strip()]
                    for item in items:
                        if item and len(item) > 1 and item not in entities[current_type]:
                            entities[current_type].append(item)
            
            # 빈 리스트 제거
            entities = {k: v for k, v in entities.items() if v}
            
            return entities
            
        except Exception as e:
            print(f"  NER 오류: {e}")
            return {}
    
    def _analyze_network(self, content: str, entities: Dict) -> Dict:
        """인물 관계망 분석"""
        if not self.llm or not entities.get('PER'):
            return {'relations': [], 'summary': '인물 정보 부족'}
        
        persons = entities.get('PER', [])[:10]  # 상위 10명
        
        prompt = f"""다음 텍스트에서 인물들 간의 관계를 분석하세요.

텍스트:
{content[:2000]}

주요 인물: {', '.join(persons)}

다음 형식으로 답변하세요:
[인물1] - [관계 유형] - [인물2] (감정: 긍정/부정/중립)
예: 이순신 - 협력 - 이억기 (감정: 긍정)

관계 유형: 협력, 갈등, 상하관계, 가족, 전우, 적대 등
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            relations = []
            for line in response.split('\n'):
                line = line.strip()
                if '-' in line and any(p in line for p in persons):
                    parts = [p.strip() for p in line.split('-')]
                    if len(parts) >= 3:
                        from_person = parts[0]
                        relation_type = parts[1]
                        to_info = parts[2]
                        
                        # 감정 추출
                        sentiment = '중립'
                        if '긍정' in to_info:
                            sentiment = '긍정'
                        elif '부정' in to_info:
                            sentiment = '부정'
                        
                        # 목표 인물 추출
                        to_person = to_info.split('(')[0].strip()
                        
                        relations.append({
                            'from': from_person,
                            'type': relation_type,
                            'to': to_person,
                            'sentiment': sentiment
                        })
            
            return {
                'relations': relations,
                'person_count': len(persons),
                'relation_count': len(relations)
            }
            
        except Exception as e:
            print(f"  관계망 분석 오류: {e}")
            return {'relations': [], 'summary': str(e)}
    
    def _analyze_sentiment(self, content: str) -> Dict:
        """감성 분석"""
        # 긍정/부정 키워드 빈도
        positive_count = sum(content.count(kw) for kw in self.sentiment_positive)
        negative_count = sum(content.count(kw) for kw in self.sentiment_negative)
        total = positive_count + negative_count
        
        if total > 0:
            positive_ratio = (positive_count / total) * 100
            negative_ratio = (negative_count / total) * 100
        else:
            positive_ratio = negative_ratio = 0
        
        # 전체 감성 점수 (-100 ~ +100)
        sentiment_score = positive_count - negative_count
        
        # 감성 분류
        if sentiment_score > 5:
            overall_sentiment = '긍정적'
        elif sentiment_score < -5:
            overall_sentiment = '부정적'
        else:
            overall_sentiment = '중립적'
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': round(positive_ratio, 1),
            'negative_ratio': round(negative_ratio, 1),
            'sentiment_score': sentiment_score,
            'overall': overall_sentiment
        }
    
    def _analyze_style(self, content: str) -> Dict:
        """문체 분석"""
        if not self.llm:
            return {}
        
        prompt = f"""다음 텍스트의 문체를 분석하세요.

텍스트:
{content[:1500]}

다음 항목을 분석하세요:
1. 문체 유형: 공식적/사적, 객관적/주관적, 건조한/감성적
2. 어조: 보고체/서술체/감탄체
3. 특징적 표현: 반복되는 문구나 독특한 표현
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            style_info = {
                'analysis': response[:500],
                'formal_vs_personal': '분석 중',
                'tone': '분석 중'
            }
            
            # 간단한 휴리스틱
            if '장계' in content[:200] or '狀啓' in content[:200]:
                style_info['formal_vs_personal'] = '공식적 (장계)'
            elif '일기' in content[:200] or '통분' in content[:500]:
                style_info['formal_vs_personal'] = '사적 (일기)'
            
            return style_info
            
        except Exception as e:
            print(f"  문체 분석 오류: {e}")
            return {}
    
    def _generate_summary(self, content: str) -> Dict:
        """문서 요약"""
        if not self.llm:
            return {'short': '', 'detailed': ''}
        
        # 짧은 요약
        short_prompt = f"""다음 텍스트를 1-2문장으로 요약하세요.

텍스트:
{content[:1000]}

요약:"""
        
        # 상세 요약
        detailed_prompt = f"""다음 텍스트를 5-10문장으로 상세히 요약하세요.

텍스트:
{content[:3000]}

주요 내용:
- 핵심 사건
- 주요 인물
- 중요한 결정이나 결과

상세 요약:"""
        
        try:
            short_summary = self.llm.invoke(short_prompt).strip()
            detailed_summary = self.llm.invoke(detailed_prompt).strip()
            
            return {
                'short': short_summary[:300],
                'detailed': detailed_summary[:1500]
            }
        except Exception as e:
            print(f"  요약 생성 오류: {e}")
            return {'short': '', 'detailed': ''}
    
    def _critical_analysis(self, content: str, analysis: Dict) -> Dict:
        """사료 비판 (편향성, 신뢰도, 한계)"""
        if not self.llm:
            return {}
        
        prompt = f"""다음 문서를 사료 비판 관점에서 분석하세요.

문서 정보:
- 형식: {analysis.get('format', {}).get('category', '알 수 없음')}
- 저자: {analysis.get('overview', {}).get('author', '미상')}
- 감성: {analysis.get('sentiment', {}).get('overall', '중립')}

텍스트:
{content[:1500]}

다음 항목을 분석하세요:
1. 신뢰도: 1차 사료인지, 얼마나 신뢰할 수 있는지
2. 편향성: 저자의 주관이나 편견이 강하게 드러나는 부분
3. 한계: 이 문서만으로는 파악하기 어려운 점
4. 교차 검증 필요성: 다른 사료와 비교가 필요한 부분
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            return {
                'reliability': '분석 중',
                'bias': '분석 중',
                'limitations': '분석 중',
                'full_analysis': response[:1000]
            }
            
        except Exception as e:
            print(f"  사료 비판 오류: {e}")
            return {}
    
    def save_analysis(self, analysis: Dict, output_dir='advanced_analysis'):
        """분석 결과 저장 (JSON + Markdown)"""
        os.makedirs(output_dir, exist_ok=True)
        
        source = analysis['metadata'].get('source', 'unknown')
        safe_filename = re.sub(r'[^\w\-_.]', '_', source)
        
        # JSON 저장
        json_path = os.path.join(output_dir, f"{safe_filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        # Markdown 저장
        md_path = os.path.join(output_dir, f"{safe_filename}.md")
        self._save_markdown(analysis, md_path)
        
        print(f"\n저장 완료:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")
    
    def _save_markdown(self, analysis: Dict, path: str):
        """Markdown 보고서 생성"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# 고급 문서 분석 보고서\n\n")
            
            # 1. 문서 개요
            f.write(f"## 1. 문서 개요\n\n")
            overview = analysis.get('overview', {})
            f.write(f"- 저자: {overview.get('author', '미상')}\n")
            f.write(f"- 제목: {overview.get('title', '미상')}\n")
            f.write(f"- 출처: {overview.get('source', '')}\n")
            f.write(f"- 시기: {overview.get('period', '불명')}\n")
            f.write(f"- 길이: {overview.get('length_chars', 0):,}자\n")
            
            # 사료적 가치
            fmt = analysis.get('format', {})
            if fmt:
                f.write(f"\n### 사료적 가치 및 특징\n\n")
                f.write(f"- 형식: {fmt.get('category', '알 수 없음')}\n")
                f.write(f"- 신뢰도: {fmt.get('confidence', 0):.1f}%\n")
                f.write(f"- 가치: {fmt.get('source_value', '분석 중')}\n")
                if fmt.get('features'):
                    f.write(f"- 특징:\n")
                    for feature in fmt['features']:
                        if feature:
                            f.write(f"  - {feature}\n")
            f.write(f"\n")
            
            research = analysis.get('research_profile', {})
            if research:
                f.write(f"### 연구 관점 분석\n\n")
                f.write(f"- 우세 주제: {research.get('dominant_theme', '불명')}\n")
                source_label = research.get('keyword_source')
                if source_label:
                    source_name = {
                        'graph_document': '지식 그래프 (문서 한정)',
                        'graph_global': '지식 그래프 (전체)',
                        'fallback_default': '기본 키워드 사전',
                        'graph_unavailable': '그래프 연결 없음',
                        'graph_empty': '그래프 데이터 부족',
                        'graph_error': '그래프 조회 오류'
                    }.get(source_label, source_label)
                    f.write(f"- 키워드 출처: {source_name}\n")
                
                theme_scores = research.get('theme_scores', [])
                if theme_scores:
                    f.write(f"\n**주요 주제 스코어**\n\n")
                    for theme in theme_scores[:3]:
                        keywords = ', '.join(list(theme.get('keyword_hits', {}).keys())[:3])
                        f.write(
                            f"- {theme.get('theme')}: 점수 {theme.get('score', 0)} "
                            f"(핵심 키워드: {keywords})\n"
                        )
                
                content_axes = research.get('content_axes', [])
                if content_axes:
                    f.write(f"\n**내용 축 근거 문장**\n\n")
                    for axis in content_axes[:3]:
                        focus = f" - {axis.get('focus')}" if axis.get('focus') else ''
                        f.write(f"- {axis.get('axis')}{focus}\n")
                        for evidence in axis.get('evidence', [])[:2]:
                            f.write(f"  - {evidence}\n")
                
                format_signals = research.get('format_signals', {})
                candidates = format_signals.get('candidates', [])
                if candidates:
                    f.write(f"\n**형식 단서**\n\n")
                    for candidate in candidates:
                        examples = ', '.join(candidate.get('examples', []))
                        f.write(
                            f"- {candidate.get('label')}: 발견된 표현 {examples} "
                            f"(총 {candidate.get('hit_count', 0)}회)\n"
                        )
                if format_signals.get('date_markers'):
                    f.write(f"- 날짜 표현 다수 발견 (연대 추적 가능)\n")
                
                relation_highlights = research.get('relation_highlights', [])
                if relation_highlights:
                    f.write(f"\n**그래프 관계 빈도**\n\n")
                    for rel in relation_highlights[:5]:
                        f.write(f"- {rel.get('type', '(미상)')}: {rel.get('count', 0)}회\n")
                    relation_source = research.get('relation_source')
                    if relation_source:
                        relation_source_label = {
                            'graph_document': '문서 기반',
                            'graph_global': '전체 그래프',
                            'graph_empty': '데이터 부족',
                            'graph_unavailable': '그래프 연결 없음',
                            'graph_error': '그래프 오류'
                        }.get(relation_source, relation_source)
                        f.write(f"- 관계 데이터 출처: {relation_source_label}\n")
                f.write(f"\n")
            
            # 2. 정량적 분석
            f.write(f"## 2. 정량적 분석\n\n")
            
            # 키워드 빈도
            kw_freq = analysis.get('keyword_frequency', {})
            if kw_freq:
                f.write(f"### 주요 키워드 빈도\n\n")
                
                by_cat = kw_freq.get('by_category', {})
                for category, keywords in by_cat.items():
                    f.write(f"**{category}**\n\n")
                    for kw, count in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]:
                        f.write(f"- {kw}: {count}회\n")
                    f.write(f"\n")
                
                top_kw = kw_freq.get('top_keywords', [])
                if top_kw:
                    f.write(f"### 최다 빈출 키워드 (Top 10)\n\n")
                    for item in top_kw[:10]:
                        f.write(f"- {item['keyword']}: {item['count']}회\n")
                    f.write(f"\n")
            
            # 주제 모델링
            topics = analysis.get('topic_modeling', [])
            if topics:
                f.write(f"### 주제 모델링\n\n")
                for topic in topics:
                    f.write(f"- {topic['topic']} ({topic['weight']}%)\n")
                    if topic.get('keywords'):
                        f.write(f"  - 핵심 키워드: {', '.join(topic['keywords'])}\n")
                f.write(f"\n")
            
            # 3. 개체 및 관계망 분석
            f.write(f"## 3. 개체 및 관계망 분석\n\n")
            
            entities = analysis.get('entities', {})
            if entities:
                f.write(f"### 주요 개체 (NER)\n\n")
                for entity_type, entity_list in entities.items():
                    type_name = {
                        'PER': '인물',
                        'LOC': '장소',
                        'ORG': '기관',
                        'DATE': '날짜',
                        'EQUIP': '무기/장비'
                    }.get(entity_type, entity_type)
                    
                    f.write(f"**{type_name}**\n\n")
                    for entity in entity_list[:10]:
                        f.write(f"- {entity}\n")
                    f.write(f"\n")
            
            # 인물 관계망
            network = analysis.get('network', {})
            if network.get('relations'):
                f.write(f"### 인물 관계망\n\n")
                for rel in network['relations']:
                    f.write(f"- {rel['from']} --[{rel['type']}]--> {rel['to']} (감정: {rel['sentiment']})\n")
                f.write(f"\n")
            
            # 4. 감성 및 문체 분석
            f.write(f"## 4. 감성 및 문체 분석\n\n")
            
            sentiment = analysis.get('sentiment', {})
            if sentiment:
                f.write(f"### 감성 분석\n\n")
                f.write(f"- 전체 감성: {sentiment.get('overall', '중립')}\n")
                f.write(f"- 감성 점수: {sentiment.get('sentiment_score', 0)}\n")
                f.write(f"- 긍정 키워드: {sentiment.get('positive_count', 0)}회 ({sentiment.get('positive_ratio', 0)}%)\n")
                f.write(f"- 부정 키워드: {sentiment.get('negative_count', 0)}회 ({sentiment.get('negative_ratio', 0)}%)\n")
                f.write(f"\n")
            
            style = analysis.get('style', {})
            if style:
                f.write(f"### 문체 분석\n\n")
                f.write(f"- 공식/사적: {style.get('formal_vs_personal', '분석 중')}\n")
                f.write(f"- 어조: {style.get('tone', '분석 중')}\n")
                if style.get('analysis'):
                    f.write(f"\n{style['analysis']}\n")
                f.write(f"\n")
            
            # 5. 요약
            f.write(f"## 5. 문서 요약\n\n")
            
            summary = analysis.get('summary', {})
            if summary.get('short'):
                f.write(f"### 간단 요약\n\n{summary['short']}\n\n")
            if summary.get('detailed'):
                f.write(f"### 상세 요약\n\n{summary['detailed']}\n\n")
            
            # 6. 사료 비판
            f.write(f"## 6. 사료 비판 및 교차 검증\n\n")
            
            critical = analysis.get('critical_analysis', {})
            if critical:
                f.write(f"### 신뢰도 및 한계\n\n")
                if critical.get('full_analysis'):
                    f.write(f"{critical['full_analysis']}\n\n")
            
            # 원문 미리보기
            f.write(f"## 원문 미리보기\n\n")
            f.write(f"```\n{analysis['content_preview']}\n```\n")


def analyze_documents_from_output(
    output_dir='output', 
    analysis_dir='advanced_analysis',
    use_existing_graph=True,
    graph_index_dir='graphrag_data'  # rag_graph.py와 동일한 경로
):
    """output 디렉토리의 모든 문서 고급 분석
    
    Args:
        output_dir: 문서 디렉토리
        analysis_dir: 분석 결과 저장 디렉토리
        use_existing_graph: 기존 지식 그래프 사용 여부
        graph_index_dir: 지식 그래프 인덱스 디렉토리
    """
    print(f"\n{'='*70}")
    print(f"고급 문서 분석 시작")
    print(f"{'='*70}")
    
    # 문서 로드
    from rag_graph import load_documents_from_output
    documents = load_documents_from_output(output_dir)
    
    if not documents:
        print("분석할 문서가 없습니다.")
        return
    
    print(f"\n총 {len(documents)}개 문서 발견")
    
    # 분석기 초기화
    analyzer = AdvancedDocumentAnalyzer()
    
    # 기존 지식 그래프 로드
    if use_existing_graph:
        graph_path = os.path.join(graph_index_dir, 'knowledge_graph.json')
        if os.path.exists(graph_path):
            print(f"\n기존 지식 그래프 로드 중: {graph_path}")
            analyzer.graph_db.import_graph(graph_path)
        else:
            print(f"\n⚠️  기존 지식 그래프가 없습니다: {graph_path}")
            print("  먼저 rag_graph.py를 실행하여 지식 그래프를 생성하세요.")
    
    # 각 문서 분석
    results = []
    for idx, doc in enumerate(documents, 1):
        print(f"\n[{idx}/{len(documents)}] 문서 분석 중...")
        
        try:
            analysis = analyzer.analyze_document(doc)
            
            # 지식 그래프에서 추가 정보 가져오기
            if analyzer.graph_db.db:
                analysis['graph_enrichment'] = _enrich_with_graph(
                    analysis, 
                    analyzer.graph_db
                )
            
            results.append(analysis)
            
            # 결과 저장
            analyzer.save_analysis(analysis, analysis_dir)
            
        except Exception as e:
            print(f"X 분석 실패: {e}")
            continue
    
    # 전체 요약
    print(f"\n{'='*70}")
    print(f"분석 완료!")
    print(f"{'='*70}")
    print(f"총 {len(results)}개 문서 분석 완료")
    print(f"결과 저장 위치: {analysis_dir}/")
    
    # 통계
    _print_statistics(results)


def _enrich_with_graph(analysis: Dict, graph_db: ArangoGraphDB) -> Dict:
    """지식 그래프로 분석 결과 보강"""
    enrichment = {
        'entity_details': [],
        'related_entities': []
    }
    
    try:
        # NER에서 추출한 인물들의 그래프 정보 가져오기
        entities = analysis.get('entities', {})
        persons = entities.get('PER', [])[:5]  # 상위 5명
        
        for person in persons:
            # 그래프에서 해당 인물 정보 조회
            entity_info = graph_db.query_entity(person)
            if entity_info:
                # 이웃 조회
                neighbors = graph_db.query_neighbors(person, depth=1)
                
                enrichment['entity_details'].append({
                    'name': person,
                    'sources': entity_info.get('sources', []),
                    'neighbor_count': len(neighbors.get('entities', []))
                })
                
                # 관련 엔티티 추가
                for neighbor in neighbors.get('entities', [])[:3]:
                    enrichment['related_entities'].append({
                        'from': person,
                        'to': neighbor.get('name', ''),
                        'type': neighbor.get('type', '')
                    })
        
        return enrichment
        
    except Exception as e:
        print(f"  그래프 보강 오류: {e}")
        return enrichment


def _print_statistics(results: List[Dict]):
    """전체 통계"""
    if not results:
        return
    
    print(f"\n전체 통계:")
    
    # 형식 통계
    formats = [r['format']['category'] for r in results if 'format' in r]
    if formats:
        print(f"\n문서 형식:")
        for fmt, count in Counter(formats).most_common(5):
            print(f"  - {fmt}: {count}개")
    
    # 주제 통계
    all_topics = []
    for r in results:
        if 'topic_modeling' in r:
            all_topics.extend([t['topic'] for t in r['topic_modeling']])
    if all_topics:
        print(f"\n주요 주제:")
        for topic, count in Counter(all_topics).most_common(5):
            print(f"  - {topic}: {count}회")
    
    # 감성 통계
    sentiments = [r['sentiment']['overall'] for r in results if 'sentiment' in r and 'overall' in r['sentiment']]
    if sentiments:
        print(f"\n감성 분포:")
        for sentiment, count in Counter(sentiments).most_common():
            print(f"  - {sentiment}: {count}개")
    
    research_themes = [
        r.get('research_profile', {}).get('dominant_theme')
        for r in results
        if r.get('research_profile', {}).get('dominant_theme') and r.get('research_profile', {}).get('dominant_theme') != '불명'
    ]
    if research_themes:
        print(f"\n우세 주제 통계:")
        for theme, count in Counter(research_themes).most_common(5):
            print(f"  - {theme}: {count}개")


if __name__ == "__main__":
    analyze_documents_from_output(
        output_dir='output',
        analysis_dir='advanced_analysis'
    )

