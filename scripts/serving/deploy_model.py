#!/usr/bin/env python
"""
deploy_model.py

CLI tool để deploy model từ folder (đã export từ Colab) vào Model Registry.

Usage:
    # Tự động detect từ folder
    python scripts/serving/deploy_model.py ban10_fasttext --from-folder ./ban10_fasttext_model/

    # Với version cụ thể
    python scripts/serving/deploy_model.py ban10_fasttext --from-folder ./ban10_fasttext_model/ --version v2

    # Force overwrite
    python scripts/serving/deploy_model.py ban10_fasttext --from-folder ./ban10_fasttext_model/ --force

    # Liệt kê models trong registry
    python scripts/serving/deploy_model.py --list

    # Liệt kê chi tiết
    python scripts/serving/deploy_model.py --list --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy model từ folder vào Model Registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scripts/serving/deploy_model.py ban10_fasttext --from-folder ./ban10_fasttext_model/
  python scripts/serving/deploy_model.py ban10_fasttext --from-folder ./ban10_fasttext_model/ --version v2
  python scripts/serving/deploy_model.py --list
        """,
    )

    # Mutually exclusive: deploy hoặc list
    parser.add_argument(
        "model_name",
        nargs="?",
        default=None,
        help="Tên model trong registry (vd: ban10_fasttext)",
    )
    parser.add_argument(
        "--from-folder",
        default=None,
        help="Đường dẫn folder chứa model files (đã export từ Colab)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version cụ thể (vd: v1, v2). Mặc định: tự động tăng",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè version cũ nếu đã tồn tại",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Liệt kê tất cả models trong registry",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Hiển thị chi tiết (dùng với --list)",
    )

    args = parser.parse_args()

    # --- List models ---
    if args.list:
        list_models(args.verbose)
        return

    # --- Deploy model ---
    if not args.model_name:
        parser.print_help()
        print("\n❌ Error: model_name is required when not using --list")
        sys.exit(1)

    if not args.from_folder:
        parser.print_help()
        print("\n❌ Error: --from-folder is required")
        sys.exit(1)

    deploy_model(args.model_name, args.from_folder, args.version, args.force)


def deploy_model(
    model_name: str,
    from_folder: str,
    version: str | None,
    force: bool,
) -> None:
    """Deploy model từ folder vào registry."""
    # Thêm project root vào sys.path
    _ensure_project_root()

    from src.serving.model_packager import ModelPackager

    try:
        dest = ModelPackager.deploy(
            model_name=model_name,
            from_folder=from_folder,
            version=version,
            force=force,
        )
        print(f"\n  📍 Deployed to: {dest}")
    except Exception as e:
        print(f"\n  ❌ Deploy failed: {e}")
        sys.exit(1)


def list_models(verbose: bool) -> None:
    """Liệt kê tất cả models trong registry."""
    _ensure_project_root()

    from src.serving.model_registry import ModelRegistry
    from src.serving.model_saver import get_registry_dir

    registry_dir = get_registry_dir()
    print(f"\n{'='*50}")
    print(f"  📦 Model Registry: {registry_dir}")
    print(f"{'='*50}")

    if not os.path.isdir(registry_dir):
        print(f"\n  ⚠️  Registry directory not found: {registry_dir}")
        print("  💡  Create models/registry/<model_name>/<version>/ with config.json")
        return

    models = ModelRegistry.list_available_models()
    if not models:
        print("\n  📭 No models found in registry")
        return

    print(f"\n  {'Model Name':<20} {'Version':<10} {'Framework':<15} {'Threshold':<10}")
    print(f"  {'-'*20} {'-'*10} {'-'*15} {'-'*10}")

    for m in models:
        print(
            f"  {m.model_name:<20} {m.version:<10} {m.framework:<15} {m.threshold:<10.3f}"
        )

    if verbose:
        print("\n  --- Detailed Info ---")
        for m in models:
            config_path = os.path.join(
                registry_dir, m.model_name, m.version, "config.json"
            )
            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
                print(f"\n  📍 {m.key}")
                print(f"     Files: {json.dumps(config.get('files', {}), indent=4)}")
                print(f"     Embedding: {config.get('embedding', {})}")
                print(f"     Preprocessing: {config.get('preprocessing', {})}")
                if "metrics" in config:
                    print(f"     Metrics: {config['metrics']}")

    print(f"\n  Total: {len(models)} model(s)")
    print()


def _ensure_project_root() -> None:
    """Thêm project root vào sys.path nếu chưa có."""
    project_root = Path(__file__).resolve().parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


if __name__ == "__main__":
    main()
