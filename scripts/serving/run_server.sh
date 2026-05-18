#!/bin/bash
"""
run_server.sh

Script chạy FastAPI server cho Toxic Comment Classification.

Cách dùng:
    # Chạy với model mặc định (models/production)
    bash scripts/serving/run_server.sh

    # Chạy với model cụ thể
    bash scripts/serving/run_server.sh --model-dir /path/to/model

    # Chạy ở port khác
    bash scripts/serving/run_server.sh --port 5000

Yêu cầu:
    - Python 3.10+
    - pip install -r requirements.txt
"""

# Mặc định
HOST="0.0.0.0"
PORT=8000
MODEL_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  Toxic Comment Classification API"
echo "========================================"
echo "  Host: $HOST"
echo "  Port: $PORT"
if [ -n "$MODEL_DIR" ]; then
    echo "  Model: $MODEL_DIR"
    export MODEL_DIR
fi
echo "========================================"

# Kích hoạt môi trường (nếu có)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Chạy server
python -m uvicorn src.serving.app:app --host "$HOST" --port "$PORT"
