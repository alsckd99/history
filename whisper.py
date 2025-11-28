import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import yt_dlp


def download_youtube_audio(youtube_url, output_path="downloaded_audio"):
    """YouTube 영상에서 오디오 다운로드"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        audio_file = f"{output_path}.mp3"
        return audio_file
    
    except Exception as e:
        print(f"다운로드 오류: {str(e)}")
        return None


def transcribe_youtube_video(youtube_url, output_file="transcription.txt", language="korean"):
    """YouTube 영상 음성을 텍스트로 변환"""
    
    # 오디오 다운로드
    audio_file = download_youtube_audio(youtube_url)
    
    if audio_file is None or not os.path.exists(audio_file):
        print("오디오 파일 없음")
        return None
    
    # Whisper 모델 설정
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-large-v3-turbo"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    language_code = {
        "korean": "ko",
        "english": "en",
        "japanese": "ja",
        "chinese": "zh",
    }.get(language.lower(), "ko")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # 음성 변환
    result = pipe(
        audio_file,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        generate_kwargs={"language": language_code}
    )
    transcription = result["text"]
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    print(f"변환 완료: {output_file}")
    
    # 임시 파일 삭제
    try:
        os.remove(audio_file)
    except:
        pass
    
    return transcription


if __name__ == "__main__":
    print("=" * 60)
    print("YouTube 음성 -> 텍스트 변환기 (Whisper)")
    print("=" * 60)
    
    youtube_url = input("\nYouTube URL: ").strip()
    
    print("\n언어 선택:")
    print("1. 한국어 (기본)")
    print("2. 영어")
    print("3. 일본어")
    print("4. 중국어")
    lang_choice = input("선택 (1-4, Enter=1): ").strip()
    
    language_map = {
        "1": "korean",
        "2": "english",
        "3": "japanese",
        "4": "chinese",
        "": "korean"
    }
    
    language = language_map.get(lang_choice, "korean")
    
    print(f"\n언어: {language}")
    transcription = transcribe_youtube_video(youtube_url, "output/audio2text.txt", language)
    
    if transcription:
        print("\n" + "=" * 60)
        print("변환 텍스트 미리보기:")
        print("=" * 60)
        print(transcription[:500] + ("..." if len(transcription) > 500 else ""))
        print("\n완료")
