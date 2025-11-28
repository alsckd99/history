import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as hf_pipeline
)

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    import openai
except ImportError:
    openai = None


FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
DEFAULT_MODEL_ID = "openai/whisper-large-v3-turbo"
DEFAULT_LANGUAGE_MAP = {
    "korean": "ko",
    "english": "en",
    "japanese": "ja",
    "chinese": "zh"
}


@dataclass
class Segment:
    start: Optional[float]
    end: Optional[float]
    text: str
    confidence: Optional[float] = None


def run_ffmpeg_extract_audio(input_path: str, output_path: str) -> str:
    """비디오/오디오 파일에서 16kHz wav 추출"""
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def download_youtube_audio(youtube_url: str, tmp_dir: str) -> str:
    if not yt_dlp:
        raise RuntimeError("yt_dlp가 설치되어 있지 않습니다. pip install yt-dlp")
    temp_id = uuid.uuid4().hex
    outtmpl = os.path.join(tmp_dir, f"yt_{temp_id}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    audio_mp3 = f"{outtmpl.rsplit('.', 1)[0]}.mp3"
    if not os.path.exists(audio_mp3):
        raise FileNotFoundError("YouTube 오디오 다운로드 실패")
    return audio_mp3


def init_asr_pipeline(model_id: str, device: str):
    torch_dtype = torch.float16 if "cuda" in device else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )


def simple_cleanup(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    fillers = [
        "음", "어", "약간", "뭐랄까", "그냥", "그러니까", "아", "어머", "어휴",
        "저기", "그게", "말이야", "막", "이제", "뭐지", "그런데"
    ]
    for filler in fillers:
        text = re.sub(rf"\b{filler}\b", "", text)
    return text.strip()


def looks_historical(text: str) -> bool:
    history_keywords = [
        "전쟁", "난", "대첩", "전투", "왕", "장군", "황제", "조선", "고려", "신라",
        "백제", "발해", "임진왜란", "병자호란", "왜군", "수군", "조정", "사료",
        "기록", "연대", "년", "세기", "사건", "의병", "항쟁", "겸사", "관청"
    ]
    if any(keyword in text for keyword in history_keywords):
        return True
    if re.search(r"\d{3,4}\s*년", text):
        return True
    return False


class SegmentRefiner:
    def __init__(self, openai_model: Optional[str] = None):
        self.model = openai_model
        if self.model:
            if not openai:
                raise RuntimeError("openai 패키지가 필요합니다. pip install openai")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되어야 합니다.")
            openai.api_key = api_key

    def refine(self, text: str) -> str:
        base_clean = simple_cleanup(text)
        if not base_clean:
            return ""
        if not self.model:
            return base_clean if looks_historical(base_clean) else ""

        system_prompt = (
            "당신은 한국사 전문가입니다. 입력된 발화에서 역사·사료와 무관한 잡담, "
            "감탄사, 추임새, 일상 대화는 모두 제거하고 역사적 사실/사건/인물/지명/연대가 "
            "포함된 문장만 남겨 주세요. 결과가 없으면 빈 문자열을 반환하세요. "
            "출력은 JSON 형태로 {\"text\": \"...\"}만 포함해야 합니다."
        )
        user_prompt = f"전사 내용:\n{text}"
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
            )
            content = response["choices"][0]["message"]["content"].strip()
            data = json.loads(content)
            filtered = data.get("text", "").strip()
        except Exception:
            filtered = base_clean
        if not filtered:
            return ""
        return filtered


class AudioTextProcessor:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        language: str = "korean",
        openai_model: Optional[str] = None
    ):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.language_code = DEFAULT_LANGUAGE_MAP.get(language.lower(), "ko")
        self.asr = init_asr_pipeline(model_id, device)
        self.refiner = SegmentRefiner(openai_model)

    def _prepare_audio(self, source: str, tmp_dir: str, is_youtube: bool) -> str:
        if is_youtube:
            downloaded = download_youtube_audio(source, tmp_dir)
            wav_path = os.path.join(tmp_dir, "input.wav")
            return run_ffmpeg_extract_audio(downloaded, wav_path)
        if source.lower().startswith("http"):
            raise ValueError("URL 입력은 YouTube 옵션과 함께만 지원됩니다.")
        ext = os.path.splitext(source)[1].lower()
        wav_path = os.path.join(tmp_dir, "input.wav")
        if ext in [".wav", ".flac"] and self._is_16k_wav(source):
            return shutil.copy(source, wav_path)
        return run_ffmpeg_extract_audio(source, wav_path)

    @staticmethod
    def _is_16k_wav(path: str) -> bool:
        return False  # 간단화를 위해 항상 변환

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        result = self.asr(
            audio_path,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            generate_kwargs={"language": self.language_code}
        )
        segments = result.get("chunks") or []
        if not segments:
            segments = [{
                "text": result.get("text", ""),
                "timestamp": [0.0, 0.0]
            }]
        processed_segments: List[Segment] = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            timestamps = seg.get("timestamp") or seg.get("timestamps") or [None, None]
            start, end = timestamps if isinstance(timestamps, (list, tuple)) else (None, None)
            filtered_text = self.refiner.refine(text)
            if not filtered_text:
                continue
            processed_segments.append(
                Segment(
                    start=float(start) if start is not None else None,
                    end=float(end) if end is not None else None,
                    text=filtered_text,
                    confidence=seg.get("score")
                )
            )
        return {
            "segments": [asdict(seg) for seg in processed_segments],
            "language": self.language_code,
            "model_id": DEFAULT_MODEL_ID
        }

    def process(
        self,
        source: str,
        output_json: str,
        is_youtube: bool = False
    ):
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._prepare_audio(source, tmp_dir, is_youtube)
            transcription = self.transcribe(wav_path)
            transcription["source"] = source
            transcription["is_youtube"] = is_youtube
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
        return output_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="영상/오디오 → 텍스트 → 정제 모듈 (Whisper + 선택적 LLM)"
    )
    parser.add_argument("--input", required=True, help="로컬 파일 경로 또는 YouTube URL")
    parser.add_argument("--youtube", action="store_true", help="입력이 YouTube 링크인지 여부")
    parser.add_argument("--language", default="korean")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--openai-model", default=None, help="예: gpt-4o-mini, gpt-4.1-mini")
    parser.add_argument("--output", required=True, help="JSON 결과 저장 경로")
    return parser.parse_args()


def main():
    args = parse_args()
    processor = AudioTextProcessor(
        model_id=args.model_id,
        language=args.language,
        openai_model=args.openai_model
    )
    output_path = processor.process(
        source=args.input,
        output_json=args.output,
        is_youtube=args.youtube
    )
    print(f"[완료] 결과 저장: {output_path}")


if __name__ == "__main__":
    main()

