#!/bin/bash
# deploy.sh — Tự động deploy model 'ban3_baseline_lr' vào registry
# Script được sinh tự động bởi ModelPackager

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."

cd "$PROJECT_ROOT" || {
    echo "❌ Không tìm thấy project root. Chạy script từ thư mục con của project."
    exit 1
}

echo "========================================"
echo "  🚀 Deploying model 'ban3_baseline_lr'..."
echo "========================================"

python scripts/serving/deploy_model.py ban3_baseline_lr --from-folder "$SCRIPT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "  ✅ Deploy thành công!"
    echo "  🚀 Start server: python scripts/serving/run_server.py"
    echo ""
else
    echo ""
    echo "  ❌ Deploy thất bại. Xem log ở trên."
    echo ""
fi
