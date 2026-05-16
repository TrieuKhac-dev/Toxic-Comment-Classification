"""
experiment_manager.py

Quản lý các experiment: tạo, so sánh, hiển thị kết quả.
Tích hợp MLflow để tracking và DVC để version data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from config.path_config import default_path_config


class ExperimentManager:
    """
    Quản lý experiment lifecycle.
    Cho phép tạo run, log kết quả, so sánh giữa các run.
    """

    def __init__(
        self,
        experiment_name: str = "toxic_comment_classification",
        tracking_uri: str | None = None,
    ):
        self.experiment_name = experiment_name
        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.tracking_uri: str = (
            tracking_uri
            or env_uri
            or str(Path(default_path_config.project_root) / "mlruns")
        )
        mlflow.set_tracking_uri(self.tracking_uri)

    def list_experiments(self) -> pd.DataFrame:
        """Liệt kê tất cả experiments."""
        experiments = mlflow.search_experiments()
        rows = []
        for exp in experiments:
            rows.append(
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "tags": exp.tags,
                }
            )
        return pd.DataFrame(rows)

    def list_runs(
        self,
        experiment_name: str | None = None,
        max_results: int = 100,
    ) -> pd.DataFrame:
        """
        Liệt kê tất cả runs của một experiment.

        Parameters
        ----------
        experiment_name : str | None
            Tên experiment. Mặc định: dùng self.experiment_name.
        max_results : int
            Số lượng run tối đa.

        Returns
        -------
        pd.DataFrame
            DataFrame chứa thông tin các runs.
        """
        exp_name = experiment_name or self.experiment_name
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
        )
        return runs

    def compare_runs(
        self,
        run_ids: list[str] | None = None,
        experiment_name: str | None = None,
        metrics: list[str] | None = None,
        params: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        So sánh nhiều runs.

        Parameters
        ----------
        run_ids : list[str] | None
            Danh sách run IDs. Nếu None, lấy tất cả runs của experiment.
        experiment_name : str | None
            Tên experiment (dùng khi run_ids=None).
        metrics : list[str] | None
            Các metrics cần so sánh. None = tất cả.
        params : list[str] | None
            Các params cần so sánh. None = tất cả.

        Returns
        -------
        pd.DataFrame
            Bảng so sánh.
        """
        if run_ids:
            runs_data = []
            for rid in run_ids:
                run = mlflow.get_run(rid)
                if not run:
                    continue
                row: dict[str, Any] = {
                    "run_id": rid,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                }
                row.update(run.data.params)
                row.update(run.data.metrics)
                runs_data.append(row)
            df = pd.DataFrame(runs_data)
        else:
            df = self.list_runs(experiment_name=experiment_name)

        # Lọc cột nếu cần
        if metrics or params:
            cols = ["run_id", "run_name"]
            if params:
                cols.extend(params)
            if metrics:
                cols.extend(metrics)
            cols = [c for c in cols if c in df.columns]
            df = df[cols]

        return df

    def get_best_run(
        self,
        metric: str = "f1",
        mode: str = "max",
        experiment_name: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Tìm run tốt nhất dựa trên một metric.

        Parameters
        ----------
        metric : str
            Tên metric.
        mode : str
            'max' hoặc 'min'.
        experiment_name : str | None
            Tên experiment.

        Returns
        -------
        dict | None
            Thông tin run tốt nhất.
        """
        df = self.list_runs(experiment_name=experiment_name)
        if df.empty or metric not in df.columns:
            return None

        ascending = mode == "min"
        df_sorted = df.sort_values(metric, ascending=ascending)
        best: dict[str, Any] = df_sorted.iloc[0].to_dict()
        return best

    def export_comparison_html(
        self,
        output_path: str | None = None,
        experiment_name: str | None = None,
    ) -> str:
        """
        Export bảng so sánh dưới dạng HTML.

        Parameters
        ----------
        output_path : str | None
            Đường dẫn file HTML output. Mặc định: lưu vào reports/.
        experiment_name : str | None
            Tên experiment.

        Returns
        -------
        str
            Đường dẫn file HTML.
        """
        df = self.compare_runs(experiment_name=experiment_name)

        if output_path is None:
            reports_dir = Path(default_path_config.project_root) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(reports_dir / "experiment_comparison.html")

        html = df.to_html(classes="table table-striped", index=False)
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Experiment Comparison - {experiment_name or self.experiment_name}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body>
    <div class="container mt-4">
        <h1>Experiment Comparison</h1>
        <p>Experiment: <strong>{experiment_name or self.experiment_name}</strong></p>
        {html}
    </div>
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        return output_path
