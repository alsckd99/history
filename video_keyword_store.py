import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Optional


@dataclass
class TimeSlice:
    start: float
    end: float
    keywords: List[Dict[str, Any]]  # 각 키워드는 children 필드를 가질 수 있음
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]  # 시간에 따른 키워드 연결 관계

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSlice":
        return cls(
            start=float(data.get("start", 0)),
            end=float(data.get("end", 0)),
            keywords=data.get("keywords", []),  # children 포함된 키워드
            entities=data.get("entities", []),
            relations=data.get("relations", []),  # relations 추가
        )


class VideoKeywordStore:
    """영상별 키워드-타임슬라이스 JSON을 로드/캐시"""

    def __init__(self, root_dir: str = "video_keywords"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
    
    def clear_cache(self):
        """캐시 클리어"""
        self.load_slices.cache_clear()

    def _build_path(self, video_id: str) -> str:
        safe_id = "".join(c for c in video_id if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.root_dir, f"{safe_id}.json")

    @lru_cache(maxsize=32)
    def load_slices(self, video_id: str, override_path: Optional[str] = None) -> List[TimeSlice]:
        path = override_path or self._build_path(video_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"키워드 JSON을 찾을 수 없습니다: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            slices = data.get("slices", [])
        else:
            slices = data
        return [TimeSlice.from_dict(item) for item in slices]

    def query(
        self,
        video_id: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
        top_k: int = 5,
        keyword_path: Optional[str] = None
    ) -> Dict[str, Any]:
        slices = self.load_slices(video_id, override_path=keyword_path)
        window = []
        print(f"[KeywordStore] 요청: start={start}, end={end}")
        for sl in slices:
            # 슬라이스의 시작 시간이 요청의 시작 시간보다 이후면 스킵
            # (아직 해당 시점에 도달하지 않음)
            if start is not None and sl.start > start:
                print(f"  [스킵] 슬라이스 {sl.start}-{sl.end}: 아직 도달 안함 (sl.start({sl.start}) > start({start}))")
                continue
            # 슬라이스 끝이 요청 시작보다 이전이거나 같으면 스킵
            if start is not None and sl.end <= start:
                print(f"  [스킵] 슬라이스 {sl.start}-{sl.end}: 이미 지남 (end({sl.end}) <= start({start}))")
                continue
            # 슬라이스 시작이 요청 끝보다 이후이거나 같으면 스킵
            if end is not None and sl.start >= end:
                print(f"  [스킵] 슬라이스 {sl.start}-{sl.end}: 범위 밖 (start({sl.start}) >= end({end}))")
                continue
            print(f"  [포함] 슬라이스 {sl.start}-{sl.end}")
            window.append(sl)
        aggregated = self._aggregate(window, top_k)
        # children 포함 여부 로그
        for kw in aggregated['keywords']:
            if 'children' in kw:
                print(f"[KeywordStore] '{kw['term']}'의 children: {[c.get('term') for c in kw['children']]}")
            else:
                print(f"[KeywordStore] '{kw['term']}'에 children 없음")
        print(f"[KeywordStore] 결과: {[k['term'] for k in aggregated['keywords']]}")
        
        # relations 수집 (모든 슬라이스에서)
        all_relations = []
        for sl in slices:
            if sl.relations:
                all_relations.extend(sl.relations)
        
        if all_relations:
            print(f"[KeywordStore] relations: {[(r.get('from'), r.get('to'), r.get('start')) for r in all_relations]}")
        
        return {
            "video_id": video_id,
            "start": start,
            "end": end,
            "keywords": aggregated["keywords"],
            "entities": aggregated["entities"],
            "relations": all_relations,  # relations 추가
            "slice_count": len(window)
        }

    @staticmethod
    def _aggregate(slices: List[TimeSlice], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
        keyword_scores: Dict[str, float] = {}
        keyword_data: Dict[str, Dict[str, Any]] = {}  # 전체 키워드 데이터 저장 (children 포함)
        entity_scores: Dict[str, float] = {}

        for sl in slices:
            for item in sl.keywords:
                term = item.get("term") or item.get("keyword")
                score = float(item.get("score", 1.0))
                if not term:
                    continue
                keyword_scores[term] = keyword_scores.get(term, 0.0) + score
                # 전체 키워드 데이터 저장 (children 포함)
                if term not in keyword_data:
                    keyword_data[term] = item

            for ent in sl.entities or []:
                name = ent.get("name") or ent.get("term")
                score = float(ent.get("score", 1.0))
                if not name:
                    continue
                entity_scores[name] = entity_scores.get(name, 0.0) + score

        def top_items(score_map: Dict[str, float]) -> List[Dict[str, Any]]:
            sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            return [
                {"term": term, "score": round(score, 4)}
                for term, score in sorted_items[:top_k]
            ]

        # 키워드에 children 필드 포함
        def top_keywords_with_children(score_map: Dict[str, float]) -> List[Dict[str, Any]]:
            sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            result = []
            for term, score in sorted_items[:top_k]:
                kw_item = {
                    "term": term,
                    "score": round(score, 4)
                }
                # children 필드가 있으면 포함
                if term in keyword_data and "children" in keyword_data[term]:
                    kw_item["children"] = keyword_data[term]["children"]
                result.append(kw_item)
            return result

        return {
            "keywords": top_keywords_with_children(keyword_scores),
            "entities": [
                {"name": term["term"], "score": term["score"]}
                for term in top_items(entity_scores)
            ]
        }

