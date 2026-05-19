"""
model_packager.py

Model Packager — Công cụ đóng gói và deploy model một cách đơn giản.
Hỗ trợ 2 luồng chính:

1. export() — Trên Colab: đóng gói model + files thành 1 folder để download
2. deploy() — Trên Local: deploy folder đã download vào Model Registry

Usage (trên Colab):
    from src.serving.model_packager import ModelPackager

    ModelPackager.export(
        output_dir="ban10_fasttext_model",
        model_name="ban10_fasttext",
        model=model,
        framework="sklearn",
        threshold=0.45,
        extra_files={"tfidf.pkl": tfidf, "scaler_numeric.pkl": scaler_numeric},
        preprocessing_script="# custom preprocessing code",
        embedding_script="# custom embedding code",
        metrics={"f1": 0.92},
    )

Usage (trên Local):
    from src.serving.model_packager import ModelPackager

    ModelPackager.deploy("ban10_fasttext", from_folder="./ban10_fasttext_model")
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from src.serving.model_saver import (
    get_registry_dir,
    save_config,
    save_fasttext,
    save_joblib,
    save_keras,
    save_torch,
)

# === Constants ===

SUPPORTED_FRAMEWORKS = (
    "sklearn",
    "lightgbm",
    "xgboost",
    "pytorch",
    "tensorflow",
    "fasttext",
)
SUPPORTED_EXTENSIONS = {
    ".pkl": "joblib",
    ".joblib": "joblib",
    ".bin": "fasttext",  # FastText model
    ".h5": "keras",
    ".pt": "pytorch",
    ".pth": "pytorch",
}


class ModelPackagerError(Exception):
    """Exception cho ModelPackager."""

    pass


class ModelPackager:
    """
    Đóng gói và deploy model.

    Các phương thức static — không cần khởi tạo instance.
    """

    # ======================================================================
    #  EXPORT — Dùng trên Colab để đóng gói model thành folder
    # ======================================================================

    @staticmethod
    def export(
        output_dir: str,
        model_name: str,
        model: Any,
        framework: str = "sklearn",
        threshold: float = 0.5,
        extra_files: dict[str, Any] | None = None,
        preprocessing_script: str | None = None,
        embedding_script: str | None = None,
        metrics: dict[str, float] | None = None,
        description: str = "",
        model_filename: str | None = None,
    ) -> str:
        """
        Đóng gói model thành 1 folder để download từ Colab về local.

        Parameters
        ----------
        output_dir : str
            Tên folder đầu ra (sẽ được tạo).
        model_name : str
            Tên model (vd: 'ban10_fasttext').
        model : Any
            Object model đã train.
        framework : str
            Framework: 'sklearn', 'lightgbm', 'xgboost', 'pytorch', 'tensorflow', 'fasttext'.
        threshold : float
            Ngưỡng quyết định.
        extra_files : dict | None
            Các file bổ sung: {"ten_file": object_or_path, ...}.
        preprocessing_script : str | None
            Script preprocessing custom (nội dung string hoặc đường dẫn file).
        embedding_script : str | None
            Script embedding custom (nội dung string hoặc đường dẫn file).
        metrics : dict | None
            Metrics của model: {"f1": 0.92, "roc_auc": 0.95}.
        description : str
            Mô tả model.
        model_filename : str | None
            Tên file model. Mặc định: tự động dựa trên framework.

        Returns
        -------
        str
            Đường dẫn folder đã tạo.
        """
        # Validate
        if framework not in SUPPORTED_FRAMEWORKS:
            raise ModelPackagerError(
                f"Unsupported framework: '{framework}'. "
                f"Supported: {', '.join(SUPPORTED_FRAMEWORKS)}"
            )

        # Tạo folder output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"  📦 Exporting model '{model_name}'...")
        print(f"  📁 Output: {output_path.resolve()}")
        print(f"{'='*50}")

        # --- 1. Lưu model file ---
        if model_filename is None:
            model_filename = ModelPackager._default_model_filename(framework)

        ModelPackager._save_model_file(
            model, model_filename, str(output_path), framework
        )
        print(f"  ✅ Model saved: {model_filename}")

        # --- 2. Lưu extra files ---
        extra_files = extra_files or {}
        saved_extra: dict[str, str] = {}
        for filename, obj in extra_files.items():
            dest = output_path / filename
            if isinstance(obj, str) and os.path.isfile(obj):
                # Là đường dẫn file — copy
                shutil.copy2(obj, dest)
            else:
                # Là object — save bằng joblib
                save_joblib(obj, filename, str(output_path))
            saved_extra[filename] = filename
            print(f"  ✅ Extra file saved: {filename}")

        # --- 3. Lưu preprocessing script ---
        has_custom_preprocessing = False
        if preprocessing_script:
            ModelPackager._save_script(
                output_path, "preprocessing.py", preprocessing_script
            )
            has_custom_preprocessing = True
            print("  ✅ Preprocessing script saved: preprocessing.py")

        # --- 4. Lưu embedding script ---
        has_custom_embedding = False
        if embedding_script:
            ModelPackager._save_script(output_path, "embedding.py", embedding_script)
            has_custom_embedding = True
            print("  ✅ Embedding script saved: embedding.py")

        # --- 5. Tạo config.json ---
        config = ModelPackager._build_config(
            model_name=model_name,
            framework=framework,
            threshold=threshold,
            model_filename=model_filename,
            extra_files=saved_extra,
            has_custom_preprocessing=has_custom_preprocessing,
            has_custom_embedding=has_custom_embedding,
            metrics=metrics,
            description=description,
        )
        save_config(config, str(output_path))

        # --- 6. Tạo deploy script (cho local) ---
        ModelPackager._create_deploy_scripts(output_path, model_name)

        print("\n  ✅ Export hoàn tất!")
        print(f"  📁 Folder: {output_path.resolve()}")
        print("  📥 Download folder này về máy local")
        print(f"  🚀 Sau đó chạy: cd {output_dir} && bash deploy.sh (hoặc deploy.bat)")
        print(f"{'='*50}\n")

        return str(output_path.resolve())

    # ======================================================================
    #  DEPLOY — Dùng trên Local để deploy folder vào Model Registry
    # ======================================================================

    @staticmethod
    def deploy(
        model_name: str,
        from_folder: str | Path,
        version: str | None = None,
        force: bool = False,
    ) -> str:
        """
        Deploy model từ folder (đã export từ Colab) vào Model Registry.

        Parameters
        ----------
        model_name : str
            Tên model trong registry.
        from_folder : str | Path
            Đường dẫn folder chứa model files (đã download từ Colab).
        version : str | None
            Version cụ thể (vd: 'v1'). Mặc định: tự động tăng.
        force : bool
            Nếu True, ghi đè version cũ.

        Returns
        -------
        str
            Đường dẫn thư mục model version trong registry.
        """
        src_path = Path(from_folder)
        if not src_path.is_dir():
            raise ModelPackagerError(
                f"Folder not found: {src_path}\n"
                f"Hãy đảm bảo bạn đã download folder từ Colab về."
            )

        # Đọc config từ folder
        config_path = src_path / "config.json"
        if not config_path.exists():
            raise ModelPackagerError(
                f"config.json not found in {src_path}\n"
                f"Folder không hợp lệ — thiếu config.json."
            )

        with open(config_path, encoding="utf-8") as f:
            config: dict[str, Any] = json.load(f)

        # Xác định registry dir
        registry_dir = get_registry_dir()
        model_registry_path = Path(registry_dir) / model_name

        # Xác định version
        if version is None:
            version = ModelPackager._next_version(model_registry_path)

        # Tạo thư mục đích
        dest_dir = model_registry_path / version
        if dest_dir.exists():
            if force:
                shutil.rmtree(dest_dir)
                print(f"  ⚠️  Overwriting existing: {dest_dir}")
            else:
                raise ModelPackagerError(
                    f"Version '{version}' already exists for model '{model_name}'.\n"
                    f"  Use --force to overwrite, or let auto-version pick next version."
                )
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"  🚀 Deploying model '{model_name}' (version {version})...")
        print(f"  📁 From: {src_path.resolve()}")
        print(f"  📁 To:   {dest_dir.resolve()}")
        print(f"{'='*50}")

        # Copy tất cả files từ folder vào registry
        for item in src_path.iterdir():
            if item.name in ("deploy.sh", "deploy.bat"):
                continue  # Bỏ qua script deploy
            dest = dest_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)
                print(f"  ✅ Copied: {item.name}")
            elif item.is_dir():
                shutil.copytree(item, dest)
                print(f"  ✅ Copied directory: {item.name}")

        # Cập nhật config với model_name và version chính xác
        config["model_name"] = model_name
        config["version"] = version

        # Fix embedding type nếu không phù hợp
        # Nếu embedding type là "tfidf" nhưng không có file vectorizer trong files
        # → model là sklearn Pipeline đã tự xử lý feature extraction
        embedding_cfg = config.get("embedding", {})
        if embedding_cfg.get("type") == "tfidf":
            files = config.get("files", {})
            has_vectorizer = any(
                "vectorizer" in k.lower() or "tfidf" in k.lower() for k in files
            )
            if not has_vectorizer:
                print("  🔧 Auto-fix: embedding type changed from 'tfidf' to 'none'")
                print("    (sklearn Pipeline đã tự xử lý feature extraction)")
                config["embedding"]["type"] = "none"

        save_config(config, str(dest_dir))

        # Validate: thử load model
        try:
            ModelPackager._validate_model(str(dest_dir), config)
        except Exception as e:
            print(f"  ⚠️  Warning: Model validation failed: {e}")
            print("  💡 Model files are saved, but may not load correctly.")

        print("\n  ✅ Deploy hoàn tất!")
        print(f"  📍 Registry: {dest_dir.resolve()}")
        print("  🚀 Start server: python scripts/serving/run_server.py")
        print(f"{'='*50}\n")

        return str(dest_dir.resolve())

    # ======================================================================
    #  INTERNAL HELPERS
    # ======================================================================

    @staticmethod
    def _default_model_filename(framework: str) -> str:
        """Trả về tên file model mặc định dựa trên framework."""
        mapping = {
            "sklearn": "model.pkl",
            "lightgbm": "model.pkl",
            "xgboost": "model.pkl",
            "pytorch": "pytorch_model.bin",
            "tensorflow": "model.h5",
            "fasttext": "fasttext_model.bin",
        }
        return mapping.get(framework, "model.pkl")

    @staticmethod
    def _save_model_file(
        model: Any, filename: str, model_dir: str, framework: str
    ) -> str:
        """Lưu model file dựa trên framework."""
        if framework in ("sklearn", "lightgbm", "xgboost"):
            return save_joblib(model, filename, model_dir)
        elif framework == "fasttext":
            return save_fasttext(model, filename, model_dir)
        elif framework == "pytorch":
            return save_torch(model, filename, model_dir)
        elif framework == "tensorflow":
            return save_keras(model, filename, model_dir)
        else:
            return save_joblib(model, filename, model_dir)

    @staticmethod
    def _save_script(output_path: Path, filename: str, script: str) -> None:
        """
        Lưu script file.
        Nếu script là đường dẫn file, copy file.
        Nếu script là nội dung string, ghi file mới.
        """
        script_path = output_path / filename

        # Kiểm tra nếu là đường dẫn file
        if os.path.isfile(script):
            shutil.copy2(script, script_path)
            return

        # Nếu là nội dung string
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

    @staticmethod
    def _build_config(
        model_name: str,
        framework: str,
        threshold: float,
        model_filename: str,
        extra_files: dict[str, str],
        has_custom_preprocessing: bool,
        has_custom_embedding: bool,
        metrics: dict[str, float] | None,
        description: str,
    ) -> dict[str, Any]:
        """Xây dựng config.json tự động."""
        config: dict[str, Any] = {
            "model_name": model_name,
            "version": "__VERSION__",  # Sẽ được update khi deploy
            "model_framework": framework,
            "threshold": threshold,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "files": {
                "model": model_filename,
            },
        }

        # Thêm extra files vào config
        for fname in extra_files:
            config["files"][fname.replace(".", "_")] = fname

        # Custom preprocessing
        if has_custom_preprocessing:
            config["preprocessing"] = {
                "type": "custom",
                "script": "preprocessing.py",
            }
        else:
            # Preprocessing mặc định: normalize + lowercase + strip_spaces
            config["preprocessing"] = {
                "pipeline": [
                    {"name": "normalize_unicode", "enabled": True},
                    {"name": "lowercase", "enabled": True},
                    {"name": "strip_spaces", "enabled": True},
                ]
            }

        # Custom embedding
        if has_custom_embedding:
            config["embedding"] = {
                "type": "custom",
                "script": "embedding.py",
            }
        else:
            # Tự động detect embedding type từ extra_files
            embedding_type = ModelPackager._detect_embedding_type(
                extra_files, framework
            )
            if embedding_type == "tfidf":
                # Tìm tên file vectorizer trong extra_files
                vec_file = None
                for fname in extra_files:
                    if "vectorizer" in fname.lower() or "tfidf" in fname.lower():
                        vec_file = extra_files[fname]
                        break
                config["embedding"] = {
                    "type": "tfidf",
                }
                if vec_file:
                    config["files"]["vectorizer"] = vec_file
            elif embedding_type == "fasttext":
                config["embedding"] = {
                    "type": "fasttext",
                }
            elif embedding_type == "none":
                # Framework tự xử lý embedding (vd: FastText)
                config["embedding"] = {
                    "type": "none",
                }

        # Metrics
        if metrics:
            config["metrics"] = metrics

        return config

    @staticmethod
    def _detect_embedding_type(
        extra_files: dict[str, str],
        framework: str,
    ) -> str:
        """
        Tự động detect loại embedding dựa trên extra_files và framework.

        Quy tắc:
        - fasttext framework → "none" (FastText tự xử lý embedding)
        - sklearn/lightgbm/xgboost framework:
          - Nếu có extra_files chứa vectorizer/tfidf → "tfidf"
          - Nếu không có extra_files → "none" (model là Pipeline đã tự xử lý)
        - Còn lại → "tfidf" (mặc định)
        """
        # FastText framework tự xử lý embedding
        if framework == "fasttext":
            return "none"

        # Kiểm tra extra_files có vectorizer không
        for fname in extra_files:
            lower = fname.lower()
            if "vectorizer" in lower or "tfidf" in lower:
                return "tfidf"
            if "fasttext" in lower:
                return "fasttext"

        # Nếu framework là sklearn/lightgbm/xgboost và không có extra_files
        # → model là Pipeline đã tự xử lý feature extraction
        if framework in ("sklearn", "lightgbm", "xgboost") and not extra_files:
            return "none"

        # Mặc định: tfidf (phổ biến nhất)
        return "tfidf"

    @staticmethod
    def _next_version(model_registry_path: Path) -> str:
        """
        Tự động tìm version tiếp theo.
        Nếu chưa có: v1
        Nếu có v1, v2: v3
        """
        if not model_registry_path.is_dir():
            return "v1"

        existing = []
        for entry in model_registry_path.iterdir():
            if entry.is_dir() and entry.name.startswith("v"):
                try:
                    num = int(entry.name[1:])
                    existing.append(num)
                except ValueError:
                    continue

        if not existing:
            return "v1"

        next_num = max(existing) + 1
        return f"v{next_num}"

    @staticmethod
    def _create_deploy_scripts(output_path: Path, model_name: str) -> None:
        """Tạo deploy.sh và deploy.bat để chạy trên local."""
        # Linux/Mac script
        sh_script = f"""#!/bin/bash
# deploy.sh — Tự động deploy model '{model_name}' vào registry
# Script được sinh tự động bởi ModelPackager

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."

cd "$PROJECT_ROOT" || {{
    echo "❌ Không tìm thấy project root. Chạy script từ thư mục con của project."
    exit 1
}}

echo "========================================"
echo "  🚀 Deploying model '{model_name}'..."
echo "========================================"

python scripts/serving/deploy_model.py {model_name} --from-folder "$SCRIPT_DIR"

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
"""
        with open(output_path / "deploy.sh", "w", encoding="utf-8") as f:
            f.write(sh_script)

        # Windows script
        bat_script = f"""@echo off
REM deploy.bat — Tu dong deploy model '{model_name}' vao registry
REM Script duoc sinh tu dong boi ModelPackager

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

echo ========================================
echo   🚀 Deploying model '{model_name}'...
echo ========================================

python scripts/serving/deploy_model.py {model_name} --from-folder "%SCRIPT_DIR%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo   ✅ Deploy thanh cong!
    echo   🚀 Start server: python scripts/serving/run_server.py
    echo.
) else (
    echo.
    echo   ❌ Deploy that bai. Xem log o tren.
    echo.
)

pause
"""
        with open(output_path / "deploy.bat", "w", encoding="utf-8") as f:
            f.write(bat_script)

        # Make sh script executable (trên Linux/Mac)
        try:
            os.chmod(output_path / "deploy.sh", 0o755)
        except Exception:
            pass

    @staticmethod
    def _validate_model(model_dir: str, config: dict[str, Any]) -> None:
        """Thử load model để validate."""
        from src.serving.model_registry import ModelRegistry

        model_name = config.get("model_name", "test")
        version = config.get("version", "v1")

        # model_dir = models/registry/<model_name>/<version>
        # registry_dir = models/registry/
        registry_dir = str(Path(model_dir).parent.parent)

        # Tạm thời set registry dir để test
        old_registry = os.environ.get("MODEL_REGISTRY_DIR")
        os.environ["MODEL_REGISTRY_DIR"] = registry_dir

        try:
            predictor = ModelRegistry.get_predictor(model_name, version)
            print(f"  ✅ Model validation OK: {type(predictor).__name__}")
        finally:
            if old_registry:
                os.environ["MODEL_REGISTRY_DIR"] = old_registry
            else:
                os.environ.pop("MODEL_REGISTRY_DIR", None)
