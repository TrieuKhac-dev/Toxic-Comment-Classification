"""
dvc_integration.py

Tích hợp DVC với MLflow.
Cung cấp các hàm để:
- Lấy commit hash của DVC cho một dataset version.
- Log DVC version info vào MLflow run.
- Tạo DVC pipeline stages cho các bước xử lý.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from config.path_config import default_path_config


def get_dvc_hash(path: str) -> str | None:
    """
    Lấy DVC hash của một file/directory được DVC track.

    Parameters
    ----------
    path : str
        Đường dẫn tương đối từ project root (vd: 'datasets/custom_dataset/v1/raw').

    Returns
    -------
    str | None
        DVC hash (md5) hoặc None nếu không tìm thấy.
    """
    dvc_file = Path(default_path_config.project_root) / f"{path}.dvc"
    if not dvc_file.exists():
        return None

    with open(dvc_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data and "outs" in data and len(data["outs"]) > 0:
        md5: str | None = data["outs"][0].get("md5")
        return md5

    return None


def get_dvc_remote_url() -> str | None:
    """
    Lấy URL của DVC remote hiện tại.

    Returns
    -------
    str | None
        URL của default remote.
    """
    try:
        # Lấy tên remote mặc định từ config
        result = subprocess.run(
            ["dvc", "config", "core.remote"],
            capture_output=True,
            text=True,
            cwd=default_path_config.project_root,
        )
        if result.returncode != 0:
            return None
        remote_name = result.stdout.strip()
        if not remote_name:
            return None

        # Lấy URL của remote đó
        result = subprocess.run(
            ["dvc", "remote", "list"],
            capture_output=True,
            text=True,
            cwd=default_path_config.project_root,
        )
        for line in result.stdout.strip().split("\n"):
            if line.startswith(remote_name):
                return line.split("\t")[1] if "\t" in line else None
        return None
    except Exception:
        return None


def log_dvc_info_to_mlflow(
    tracker: Any, dataset_path: str = "datasets/custom_dataset/v1/raw"
) -> None:
    """
    Log DVC version info vào MLflow run hiện tại.

    Parameters
    ----------
    tracker : MLflowTracker
        Instance của MLflowTracker (đang trong context).
    dataset_path : str
        Đường dẫn dataset được DVC track.
    """
    dvc_hash = get_dvc_hash(dataset_path)
    if dvc_hash:
        tracker.log_params(
            {
                "dvc_dataset_path": dataset_path,
                "dvc_dataset_hash": dvc_hash,
            }
        )

    # Log git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=default_path_config.project_root,
        )
        if result.returncode == 0:
            tracker.set_tag("git_commit", result.stdout.strip())
    except Exception:
        pass


def create_dvc_stage(
    stage_name: str,
    cmd: str,
    deps: list[str] | None = None,
    outs: list[str] | None = None,
    params: list[str] | None = None,
    metrics: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Tạo một DVC stage (dvc.yaml) bằng CLI.

    Parameters
    ----------
    stage_name : str
        Tên stage.
    cmd : str
        Command để chạy.
    deps : list[str] | None
        Các dependencies.
    outs : list[str] | None
        Các output files/directories.
    params : list[str] | None
        Các params files (vd: config/*.py).
    metrics : list[str] | None
        Các metrics files.
    force : bool
        Overwrite nếu stage đã tồn tại.

    Returns
    -------
    dict
        Kết quả từ DVC CLI.
    """
    cmd_parts = ["dvc", "stage", "add", "-n", stage_name]

    if force:
        cmd_parts.append("--force")

    if deps:
        for dep in deps:
            cmd_parts.extend(["-d", dep])

    if outs:
        for out in outs:
            cmd_parts.extend(["-o", out])

    if params:
        for param in params:
            cmd_parts.extend(["-p", param])

    if metrics:
        for metric in metrics:
            cmd_parts.extend(["-m", metric])

    cmd_parts.append("--")
    cmd_parts.append(cmd)

    result = subprocess.run(
        cmd_parts,
        capture_output=True,
        text=True,
        cwd=default_path_config.project_root,
    )

    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def get_dvc_pipeline_status() -> dict[str, Any]:
    """
    Kiểm tra trạng thái DVC pipeline.

    Returns
    -------
    dict
        Trạng thái pipeline.
    """
    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
            cwd=default_path_config.project_root,
        )
        return {
            "success": result.returncode == 0,
            "status": result.stdout.strip()
            if result.returncode == 0
            else result.stderr.strip(),
        }
    except Exception as e:
        return {"success": False, "status": str(e)}
