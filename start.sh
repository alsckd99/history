#!/bin/bash

# DGX Spark 호환 GraphRAG 시작 스크립트 (Linux)
# Docker로 Ollama + ArangoDB를 시작하고 Python GraphRAG를 실행합니다

set -e  # 에러 발생 시 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 옵션 파싱
SKIP_DOCKER=false
SKIP_MODEL=false
PYTHON_ONLY=false
VISUALIZE=false
VISUALIZE_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-docker)
      SKIP_DOCKER=true
      shift
      ;;
    --skip-model)
      SKIP_MODEL=true
      shift
      ;;
    --python-only)
      PYTHON_ONLY=true
      shift
      ;;
    --visualize)
      VISUALIZE=true
      shift
      ;;
    --visualize-only)
      VISUALIZE_ONLY=true
      shift
      ;;
    --help|-h)
      echo "Usage: ./start.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --skip-docker     Docker 서비스가 이미 실행 중이면 건너뛰기"
      echo "  --skip-model      Ollama 모델 다운로드 건너뛰기"
      echo "  --python-only     Python 스크립트만 실행 (Docker 시작 안 함)"
      echo "  --visualize       전체 실행 후 지식 그래프 시각화도 실행"
      echo "  --visualize-only  지식 그래프 시각화만 실행"
      echo "  --help, -h        이 도움말 표시"
      echo ""
      echo "Default: Docker 시작 → 모델 다운로드 → Python 실행"
      echo ""
      echo "Examples:"
      echo "  ./start.sh                    # 전체 실행 (권장)"
      echo "  ./start.sh --skip-docker      # Docker가 이미 실행 중일 때"
      echo "  ./start.sh --python-only      # Python만 실행"
      echo "  ./start.sh --visualize        # 전체 실행 + 시각화"
      echo "  ./start.sh --visualize-only   # 시각화만 실행"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Run './start.sh --help' for usage information"
      exit 1
      ;;
  esac
done


# 시각화만 실행 모드
if [ "$VISUALIZE_ONLY" = true ]; then
  echo -e "${BLUE}[시각화] 지식 그래프 시각화 중...${NC}"
  if command -v python3 &> /dev/null; then
    python3 visualize_graph.py
  elif command -v python &> /dev/null; then
    python visualize_graph.py
  else
    echo -e "${RED}✗ Python이 설치되지 않았습니다${NC}"
    exit 1
  fi
  exit 0
fi

# Python만 실행 모드
if [ "$PYTHON_ONLY" = true ]; then
  echo -e "${BLUE}[Python] rag_graph.py 실행 중...${NC}"
  if command -v python3 &> /dev/null; then
    python3 rag_graph.py
  elif command -v python &> /dev/null; then
    python rag_graph.py
  else
    echo -e "${RED}✗ Python이 설치되지 않았습니다${NC}"
    exit 1
  fi
  exit 0
fi

# 1. GPU 확인
echo -e "${BLUE}[1/6] GPU 확인 중...${NC}"
if command -v nvidia-smi &> /dev/null; then
  if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU 감지됨${NC}"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)
    echo "  GPU: $GPU_INFO"
  else
    echo -e "${YELLOW}⚠ GPU 접근 불가 (CPU 모드로 실행)${NC}"
  fi
else
  echo -e "${YELLOW}⚠ nvidia-smi 없음 (CPU 모드로 실행)${NC}"
fi

# 2. Docker Compose 확인
echo ""
echo -e "${BLUE}[2/6] Docker Compose 확인 중...${NC}"
DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null; then
  DOCKER_COMPOSE_CMD="docker compose"
  echo -e "${GREEN}✓ Docker Compose V2${NC}"
elif command -v docker-compose &> /dev/null; then
  DOCKER_COMPOSE_CMD="docker-compose"
  echo -e "${GREEN}✓ Docker Compose V1${NC}"
else
  echo -e "${RED}✗ Docker Compose가 설치되지 않았습니다${NC}"
  echo "설치: https://docs.docker.com/compose/install/"
  exit 1
fi

# 3. Docker 서비스 시작
if [ "$SKIP_DOCKER" = false ]; then
  echo ""
  echo -e "${BLUE}[3/6] Docker 서비스 시작 중...${NC}"
  $DOCKER_COMPOSE_CMD up -d
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Docker 서비스 시작 실패${NC}"
    exit 1
  fi
  echo -e "${GREEN}✓ Docker 서비스 시작 완료${NC}"
else
  echo ""
  echo -e "${YELLOW}[3/6] Docker 시작 건너뛰기 (--skip-docker)${NC}"
fi

# 4. 서비스 준비 대기
echo ""
echo -e "${BLUE}[4/6] 서비스 준비 대기 중...${NC}"
sleep 3

# Ollama 확인
echo "  Ollama 상태 확인 중..."
for i in {1..12}; do
  if docker exec history-ollama curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ Ollama 준비 완료${NC}"
    break
  fi
  if [ $i -eq 12 ]; then
    echo -e "  ${YELLOW}⚠ Ollama 응답 없음 (계속 진행)${NC}"
  else
    sleep 5
  fi
done

# ArangoDB 확인 (기존 8529)
echo "  ArangoDB (8529) 상태 확인 중..."
for i in {1..12}; do
  if curl -s http://localhost:8529/_api/version > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ ArangoDB (8529) 준비 완료${NC}"
    break
  fi
  if [ $i -eq 12 ]; then
    echo -e "  ${YELLOW}⚠ ArangoDB (8529) 응답 없음 (계속 진행)${NC}"
  else
    sleep 5
  fi
done

# ArangoDB 확인 (새로운 8530)
echo "  ArangoDB (8530) 상태 확인 중..."
for i in {1..12}; do
  if curl -s http://localhost:8530/_api/version > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ ArangoDB (8530) 준비 완료${NC}"
    break
  fi
  if [ $i -eq 12 ]; then
    echo -e "  ${YELLOW}⚠ ArangoDB (8530) 응답 없음 (계속 진행)${NC}"
  else
    sleep 5
  fi
done

# 5. Ollama 모델 다운로드
if [ "$SKIP_MODEL" = false ]; then
  echo ""
  echo -e "${BLUE}[5/6] Ollama 모델 다운로드 중...${NC}"
  echo "  모델: gemma3:12b"
  
  # 이미 모델이 있는지 확인
  if docker exec history-ollama ollama list | grep -q "gemma3:12b"; then
    echo -e "  ${GREEN}✓ 모델이 이미 설치되어 있습니다${NC}"
  else
    docker exec history-ollama ollama pull gemma3:12b
    if [ $? -eq 0 ]; then
      echo -e "  ${GREEN}✓ 모델 다운로드 완료${NC}"
    else
      echo -e "  ${YELLOW}⚠ 모델 다운로드 실패 (계속 진행)${NC}"
    fi
  fi
else
  echo ""
  echo -e "${YELLOW}[5/6] 모델 다운로드 건너뛰기 (--skip-model)${NC}"
fi

# 6. Python GraphRAG 실행
echo ""
echo -e "${BLUE}[6/6] Python GraphRAG 실행 중...${NC}"

if command -v python3 &> /dev/null; then
  python3 rag_graph.py
elif command -v python &> /dev/null; then
  python rag_graph.py
else
  echo -e "${RED}✗ Python이 설치되지 않았습니다${NC}"
  exit 1
fi

# 7. 시각화 실행 (--visualize 옵션 사용 시)
if [ "$VISUALIZE" = true ]; then
  echo ""
  echo -e "${BLUE}[추가] 지식 그래프 시각화 중...${NC}"
  if command -v python3 &> /dev/null; then
    python3 visualize_graph.py
  elif command -v python &> /dev/null; then
    python visualize_graph.py
  else
    echo -e "${RED}✗ Python이 설치되지 않았습니다${NC}"
    exit 1
  fi
  
  echo ""
  echo -e "${GREEN}✓ 시각화 완료!${NC}"
  echo "  생성된 파일:"
  echo "  • knowledge_graph.html (대화형 그래프)"
  echo "  • knowledge_graph_statistics.json (통계)"
  if [ -f "knowledge_graph.png" ]; then
    echo "  • knowledge_graph.png (정적 이미지)"
  fi
fi

# 완료 메시지
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}모든 작업 완료!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "${GREEN}서비스 접근:${NC}"
echo "  • ArangoDB (기존): http://localhost:8529"
echo "  • ArangoDB (새로운): http://localhost:8530"
echo "  • Ollama API: http://localhost:11434"
if [ "$VISUALIZE" = true ]; then
  echo "  • 지식 그래프: file://$(pwd)/knowledge_graph.html"
fi
echo ""
echo -e "${BLUE}관리 명령어:${NC}"
echo "  • 로그 보기: $DOCKER_COMPOSE_CMD logs -f"
echo "  • 서비스 중지: $DOCKER_COMPOSE_CMD down"
echo "  • 서비스 재시작: $DOCKER_COMPOSE_CMD restart"
echo ""
echo -e "${YELLOW}다시 실행:${NC}"
echo "  • 전체: ./start.sh"
echo "  • 전체 + 시각화: ./start.sh --visualize"
echo "  • Python만: ./start.sh --python-only"
echo "  • 시각화만: ./start.sh --visualize-only"
echo ""

