import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Optional


@dataclass
class TimeSlice:
    start: float
    end: float
    keywords: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSlice":
        return cls(
            start=float(data.get("start", 0)),
            end=float(data.get("end", 0)),
            keywords=data.get("keywords", []),
            entities=data.get("entities", []),
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
            # 슬라이스 끝이 요청 시작보다 이전이거나 같으면 스킵
            if start is not None and sl.end <= start:
                print(f"  [스킵] 슬라이스 {sl.start}-{sl.end}: end({sl.end}) <= start({start})")
                continue
            # 슬라이스 시작이 요청 끝보다 이후이거나 같으면 스킵
            if end is not None and sl.start >= end:
                print(f"  [스킵] 슬라이스 {sl.start}-{sl.end}: start({sl.start}) >= end({end})")
                continue
            print(f"  [포함] 슬라이스 {sl.start}-{sl.end}")
            window.append(sl)
        aggregated = self._aggregate(window, top_k)
        print(f"[KeywordStore] 결과: {[k['term'] for k in aggregated['keywords']]}")
        return {
            "video_id": video_id,
            "start": start,
            "end": end,
            "keywords": aggregated["keywords"],
            "entities": aggregated["entities"],
            "slice_count": len(window)
        }

    @staticmethod
    def _aggregate(slices: List[TimeSlice], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
        keyword_scores: Dict[str, float] = {}
        entity_scores: Dict[str, float] = {}

        for sl in slices:
            for item in sl.keywords:
                term = item.get("term") or item.get("keyword")
                score = float(item.get("score", 1.0))
                if not term:
                    continue
                keyword_scores[term] = keyword_scores.get(term, 0.0) + score

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

        return {
            "keywords": top_items(keyword_scores),
            "entities": [
                {"name": term["term"], "score": term["score"]}
                for term in top_items(entity_scores)
            ]
        }

