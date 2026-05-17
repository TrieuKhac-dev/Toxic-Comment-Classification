"""
src/training/

Module đánh giá mô hình dùng chung cho mọi experiment.
Các hàm ở đây được trích xuất từ code lặp trong 15 notebooks training (Bản 3 → Bản 17).

Usage:
    from src.training.threshold import find_best_threshold, find_best_threshold_cost
    from src.training.metrics import evaluate_model, run_cross_validation
    from src.training.visualization import plot_roc_curve, plot_pr_curve, plot_calibration_curve
    from src.training.error_analysis import show_fp_fn_samples
"""
