import json
import re
from pathlib import Path


def parse_annotation_text(annotation_text):
    """
    텍스트 형태의 주석을 구조화된 형식으로 파싱합니다.
    
    예시 입력:
    "[D001] 평성(平城)의 옛 계략이다\n한(漢) 나라 고제..."
    
    Returns:
        list: [{"id": "D001", "label": "...", "text": "..."}, ...]
    """
    if not annotation_text or not annotation_text.strip():
        return []
    
    annotations = []
    # [D001], [D002] 등의 패턴으로 분리
    pattern = r'\[D(\d+)\]\s*'
    parts = re.split(pattern, annotation_text)
    
    # parts는 ['', '001', '내용1', '002', '내용2', ...] 형태
    i = 1
    while i < len(parts) - 1:
        id_num = parts[i]
        content = parts[i + 1].strip()
        
        if content:
            # 첫 줄을 label로, 나머지를 text로
            lines = content.split('\n', 1)
            label = lines[0].strip()
            text = lines[1].strip() if len(lines) > 1 else label
            
            # label에서 콜론 이후 제거 (주-D001 형식 처리)
            if ':' in label and len(label.split(':')[0]) < 50:
                label_parts = label.split(':', 1)
                label = label_parts[0].strip()
                # text에 콜론 이후 내용 추가
                if len(label_parts) > 1:
                    text = label_parts[1].strip() + ('\n' + text if text != label else '')
            
            annotations.append({
                "id": f"D{id_num}",
                "label": label,
                "text": text if text else label
            })
        
        i += 2
    
    return annotations


def parse_annotation_with_prefix(annotation_text):
    """
    [주-D001] 형식의 주석을 파싱합니다.
    """
    if not annotation_text or not annotation_text.strip():
        return []
    
    annotations = []
    # [주-D001] 패턴으로 분리
    pattern = r'\[주-D(\d+)\]\s*'
    parts = re.split(pattern, annotation_text)
    
    i = 1
    while i < len(parts) - 1:
        id_num = parts[i]
        content = parts[i + 1].strip()
        
        if content:
            # 콜론으로 label과 text 분리
            if ':' in content or '：' in content:
                # 첫 번째 콜론으로 분리
                sep = ':' if ':' in content else '：'
                label_parts = content.split(sep, 1)
                label = label_parts[0].strip()
                text = label_parts[1].strip() if len(label_parts) > 1 else content
            else:
                lines = content.split('\n', 1)
                label = lines[0].strip()
                text = lines[1].strip() if len(lines) > 1 else label
            
            annotations.append({
                "id": f"D{id_num}",
                "label": label,
                "text": text if text else label
            })
        
        i += 2
    
    return annotations


def normalize_document(doc, source_type="auto"):
    """
    다양한 형식의 문서를 통일된 형식으로 정규화합니다.
    
    Args:
        doc: 원본 문서 딕셔너리
        source_type: 문서 소스 타입 ("itkc", "yeolyeo", "auto")
    
    Returns:
        dict: 정규화된 문서
    """
    # 소스 타입 자동 감지
    if source_type == "auto":
        if "dci_s" in doc:
            source_type = "itkc"  # 난중잡록, 기재사초 등
        elif "metadata" in doc:
            source_type = "yeolyeo"  # 연려실기술, 간양록 등
        else:
            source_type = "unknown"
    
    normalized = {
        "url": "",
        "content": "",
        "metadata": {
            "서명": "",
            "제목": "",
            "주제분류": "",
            "카테고리": "",
            "키워드": "",
            "날짜": "",
            "인물": [],
            "장소": []
        },
        "annotations": []
    }
    
    if source_type == "itkc":
        # 난중잡록, 기재사초 등 ITKC 형식
        normalized["url"] = doc.get("url", "")
        normalized["content"] = doc.get("content", "") or doc.get("main_content", "")
        
        normalized["metadata"]["서명"] = doc.get("서명", "")
        normalized["metadata"]["제목"] = doc.get("기사명", "")
        normalized["metadata"]["주제분류"] = doc.get("주제분류", "")
        normalized["metadata"]["카테고리"] = doc.get("category", "")
        normalized["metadata"]["키워드"] = doc.get("keyword", "")
        normalized["metadata"]["저자"] = doc.get("저자", "")
        
        # annotations 처리
        if doc.get("annotations_structured"):
            normalized["annotations"] = doc["annotations_structured"]
        elif doc.get("annotation_clean_text"):
            normalized["annotations"] = parse_annotation_text(
                doc["annotation_clean_text"]
            )
        elif doc.get("annotation"):
            # [주-D001] 형식 처리
            if "[주-D" in doc["annotation"]:
                normalized["annotations"] = parse_annotation_with_prefix(
                    doc["annotation"]
                )
            else:
                normalized["annotations"] = parse_annotation_text(
                    doc["annotation"]
                )
    
    elif source_type == "yeolyeo":
        # 연려실기술, 간양록 등 metadata 형식
        metadata = doc.get("metadata", {})
        
        normalized["url"] = doc.get("url", "")
        normalized["content"] = doc.get("content", "")
        
        normalized["metadata"]["서명"] = metadata.get("서명", "")
        normalized["metadata"]["제목"] = metadata.get("제목", "")
        normalized["metadata"]["주제분류"] = metadata.get("주제분류", "")
        normalized["metadata"]["카테고리"] = metadata.get("카테고리", "")
        normalized["metadata"]["키워드"] = metadata.get("키워드", "")
        normalized["metadata"]["저자"] = metadata.get("저자", "")
        normalized["metadata"]["날짜"] = metadata.get("날짜", "")
        normalized["metadata"]["인물"] = metadata.get("인물", [])
        normalized["metadata"]["장소"] = metadata.get("장소", [])
        
        # annotations 처리
        if doc.get("annotations"):
            normalized["annotations"] = doc["annotations"]
        elif doc.get("annotation"):
            normalized["annotations"] = parse_annotation_text(doc["annotation"])
    
    else:
        # 알 수 없는 형식 - 가능한 필드 복사
        normalized["url"] = doc.get("url", "")
        normalized["content"] = doc.get("content", "")
    
    return normalized


def normalize_json_file(input_path, output_path=None):
    """
    JSON 파일을 읽어서 정규화된 형식으로 저장합니다.
    
    Args:
        input_path: 입력 JSON 파일 경로
        output_path: 출력 JSON 파일 경로 (None이면 덮어쓰기)
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 단일 문서 또는 문서 리스트 처리
    if isinstance(data, list):
        documents = [normalize_document(doc) for doc in data]
    elif isinstance(data, dict):
        if "documents" in data:
            # 이미 documents 구조가 있는 경우
            documents = [normalize_document(doc) for doc in data["documents"]]
        else:
            # 단일 문서
            documents = [normalize_document(data)]
    else:
        print(f"알 수 없는 데이터 형식: {input_path}")
        return
    
    # 새로운 형식으로 저장
    result = {
        "document_count": len(documents),
        "documents": documents
    }
    
    output = output_path or input_path
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 정규화 완료: {output} ({len(documents)}개 문서)")


def normalize_directory(input_dir, output_dir=None):
    """
    디렉토리 내의 모든 JSON 파일을 정규화합니다.
    
    Args:
        input_dir: 입력 디렉토리 경로
        output_dir: 출력 디렉토리 경로 (None이면 덮어쓰기)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"디렉토리가 존재하지 않습니다: {input_dir}")
        return
    
    # 출력 디렉토리 생성
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    # 모든 JSON 파일 처리
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"JSON 파일이 없습니다: {input_dir}")
        return
    
    print(f"\n{input_dir} 디렉토리 정규화 시작 ({len(json_files)}개 파일)")
    print("-" * 60)
    
    for json_file in json_files:
        try:
            if output_dir:
                output_file = output_path / json_file.name
            else:
                output_file = json_file
            
            normalize_json_file(str(json_file), str(output_file))
        except Exception as e:
            print(f"✗ 오류 발생: {json_file.name} - {str(e)}")


def normalize_all_folders(base_dir, folders, output_base=None):
    """
    여러 폴더의 JSON 파일들을 일괄 정규화합니다.
    
    Args:
        base_dir: 기본 디렉토리
        folders: 정규화할 폴더 이름 리스트
        output_base: 출력 기본 디렉토리 (None이면 덮어쓰기)
    """
    base_path = Path(base_dir)
    
    for folder in folders:
        input_dir = base_path / folder
        
        if output_base:
            output_dir = Path(output_base) / folder
        else:
            output_dir = None
        
        if input_dir.exists():
            normalize_directory(str(input_dir), str(output_dir) if output_dir else None)
        else:
            print(f"폴더가 존재하지 않습니다: {input_dir}")


if __name__ == "__main__":
    # 정규화할 폴더 목록
    folders_to_normalize = [
        "난중잡록",
        "연려실기술",
        "기재사초",
        "간양록",
        "고대일록",
        "재조번방지"
    ]
    
    # output 디렉토리 기준으로 정규화
    base_directory = "output"
    
    print("=" * 60)
    print("문서 정규화 시작")
    print("=" * 60)
    
    normalize_all_folders(base_directory, folders_to_normalize)
    
    print("\n" + "=" * 60)
    print("모든 정규화 완료!")
    print("=" * 60)

