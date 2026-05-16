"""
base_pipeline.py

Base classes cho pipeline thông minh.
Cung cấp PipelineStep dataclass và BasePipeline.run_steps() static method.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineStep:
    """
    Một bước trong pipeline.

    Attributes
    ----------
    name : str
        Tên bước (dùng cho report).
    enabled_flag : str
        Tên attribute trong config để bật/tắt bước (vd: 'enable_html_removal').
    func : Callable
        Hàm xử lý. Nhận data + **kwargs, trả về (data, report_dict).
    kwargs : dict
        Tham số bổ sung cho func.
    """

    name: str
    enabled_flag: str
    func: Callable[..., tuple[Any, dict]]
    kwargs: dict[str, Any] = field(default_factory=dict)


class BasePipeline:
    """Base class cho các pipeline."""

    @staticmethod
    def run_steps(
        data: Any,
        config: Any,
        steps: list[PipelineStep],
    ) -> tuple[Any, dict]:
        """
        Chạy các bước pipeline theo thứ tự.
        Chỉ chạy bước được bật (enabled_flag = True trong config).

        Parameters
        ----------
        data : Any
            Dữ liệu đầu vào (vd: DataFrame, str).
        config : Any
            Đối tượng config chứa các enable_* flags.
        steps : list[PipelineStep]
            Danh sách các bước cần chạy.

        Returns
        -------
        tuple[Any, dict]
            (data đã xử lý, report tổng hợp).
        """
        report: dict[str, Any] = {}

        for step in steps:
            enabled = getattr(config, step.enabled_flag, False)
            if not enabled:
                report[step.name] = {
                    "applied": False,
                    "reason": f"{step.enabled_flag}=False",
                }
                continue

            data, step_report = step.func(data, **step.kwargs)
            step_report["applied"] = True
            report[step.name] = step_report

        return data, report
