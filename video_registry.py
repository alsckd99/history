import json
import os
from typing import Optional, Dict


class VideoRegistry:
    """영상 ID와 키워드 JSON 경로 매핑"""

    def __init__(self, registry_file: str = "video_keywords/registry.json"):
        self.registry_file = registry_file
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict[str, str]:
        if not os.path.exists(self.registry_file):
            return {}
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 상대 경로를 절대 경로로 변환
                normalized_data = {}
                for video_id, path in data.items():
                    if not os.path.isabs(path):
                        # 상대 경로인 경우 절대 경로로 변환
                        abs_path = os.path.abspath(path)
                        normalized_data[video_id] = abs_path
                    else:
                        normalized_data[video_id] = path
                return normalized_data
        except Exception:
            return {}

    def _save(self):
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def register(self, video_id: str, keyword_path: str):
        abs_path = os.path.abspath(keyword_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"키워드 JSON 경로를 찾을 수 없습니다: {abs_path}")
        self._data[video_id] = abs_path
        self._save()

    def resolve(self, video_id: str) -> Optional[str]:
        return self._data.get(video_id)

    def list_videos(self) -> Dict[str, str]:
        return dict(self._data)

