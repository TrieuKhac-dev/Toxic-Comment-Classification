#!/usr/bin/env python
"""
run_server.py

Script chạy FastAPI server cho Toxic Comment Classification.
Hỗ trợ deploy model trực tiếp từ folder khi start server.

Usage:
    # Chạy server thông thường
    python scripts/serving/run_server.py

    # Deploy model + start server cùng lúc
    python scripts/serving/run_server.py --deploy ./ban10_fasttext_model/

    # Với port và host tùy chỉnh
    python scripts/serving/run_server.py --host 127.0.0.1 --port 8080

    # Với registry dir tùy chỉnh
    python scripts/serving/run_server.py --registry-dir ./my_models/registry
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toxic Comment Classification API Server (Multi-Model Multi-Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (mặc định: 8000)")
    parser.add_argument(
        "--registry-dir",
        default=None,
        help="Đường dẫn thư mục model registry",
    )
    parser.add_argument(
        "--deploy",
        default=None,
        help="Deploy model từ folder trước khi start server",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload khi code thay đổi (dùng cho development)",
    )

    args = parser.parse_args()

    # Thêm project root vào sys.path
    project_root = Path(__file__).resolve().parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Deploy model nếu có --deploy
    if args.deploy:
        deploy_folder = args.deploy
        if not os.path.isdir(deploy_folder):
            print(f"Error: Folder not found: {deploy_folder}")
            sys.exit(1)

        # Đọc model_name từ config.json trong folder
        config_path = os.path.join(deploy_folder, "config.json")
        if not os.path.exists(config_path):
            print(f"Error: config.json not found in {deploy_folder}")
            sys.exit(1)

        import json

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        model_name = config.get("model_name", "")
        if not model_name:
            print("Error: model_name not found in config.json")
            sys.exit(1)

        print(f"\n{'='*50}")
        print(f"  Deploying model '{model_name}' before starting server...")
        print(f"{'='*50}")

        from src.serving.model_packager import ModelPackager

        try:
            ModelPackager.deploy(
                model_name=model_name,
                from_folder=deploy_folder,
            )
        except Exception as e:
            print(f"  Deploy failed: {e}")
            print("  Starting server anyway...\n")

    # Set registry dir nếu có
    if args.registry_dir:
        os.environ["MODEL_REGISTRY_DIR"] = args.registry_dir

    # Start server
    print(f"\n{'='*50}")
    print(f"  Starting server at http://{args.host}:{args.port}")
    print(f"{'='*50}\n")

    import uvicorn

    uvicorn.run(
        "src.serving.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
