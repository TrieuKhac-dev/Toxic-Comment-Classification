"""
visualize_results.py

Script hiển thị kết quả experiments dưới dạng HTML.
Chạy: python scripts/visualize_results.py [experiment_name]

Kết quả được lưu vào reports/experiment_comparison.html
"""

from __future__ import annotations

import os
import sys

# Thêm thư mục gốc project vào sys.path để import được src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.tracking.experiment_manager import ExperimentManager


def main() -> None:
    """Export comparison HTML cho tất cả experiments."""
    # Nhận experiment_name từ command line nếu có
    exp_name_arg = sys.argv[1] if len(sys.argv) > 1 else None

    manager = ExperimentManager()

    if exp_name_arg:
        # Export cho một experiment cụ thể
        print(f"Exporting comparison for experiment: {exp_name_arg}")
        html_path = manager.export_comparison_html(experiment_name=exp_name_arg)
        print(f"Exported: {html_path}")
    else:
        # Liệt kê experiments
        experiments_df = manager.list_experiments()
        print("Experiments:")
        print(experiments_df.to_string(index=False))
        print()

        # Export comparison cho từng experiment
        for _, row in experiments_df.iterrows():
            exp_name = row["name"]
            try:
                html_path = manager.export_comparison_html(experiment_name=exp_name)
                print(f"Exported: {html_path}")
            except Exception as e:
                print(f"Error exporting {exp_name}: {e}")

    print("\nDone! Open the HTML files in reports/ to view results.")


if __name__ == "__main__":
    main()
