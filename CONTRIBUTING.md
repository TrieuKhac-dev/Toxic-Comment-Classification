# Contributing

## Mục tiêu tài liệu

`CONTRIBUTING.md` là điểm vào chính cho contributor mới:

- setup môi trường nhanh,
- quy tắc đóng góp mức tổng quan,
- điều hướng sang các tài liệu chi tiết theo từng mục đích.

## Bản đồ tài liệu

- Quy trình làm việc chi tiết: [docs/workflows.md](docs/workflows.md)
- Hướng dẫn MLflow: [docs/mlflow_guide.md](docs/mlflow_guide.md)
- Quy tắc branch bắt buộc: [.github/branch_rules.md](.github/branch_rules.md)
- Mẫu Pull Request: [.github/pull_request_template.md](.github/pull_request_template.md)
- Hướng dẫn xử lý lỗi: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Setup môi trường (quick start)

Yêu cầu:

- Windows + PowerShell
- Miniconda hoặc Anaconda

Tạo môi trường lần đầu:

```powershell
conda env create -f .\environment.yml
conda activate toxic-cmt-clf
conda run -n toxic-cmt-clf pre-commit install   # Đảm bảo pre-commit hoạt động khi commit
```

Cập nhật môi trường:

```powershell
conda env update -n toxic-cmt-clf -f .\environment.yml --prune -v
conda run -n toxic-cmt-clf pre-commit install   # Luôn chạy lại sau khi update env
```

Luôn dùng lệnh trên thay vì:

```powershell
conda env update -f environment.yml --prune
```

### Thiết lập MLflow Tracking

MLflow được cấu hình mặc định lưu local vào thư mục `mlruns/`. Để xem UI:

```powershell
conda run -n toxic-cmt-clf mlflow ui
```

Mở trình duyệt tại `http://localhost:5000` để xem kết quả.

### Cấu trúc thư mục datasets

```text
datasets/
└── custom_dataset/
    └── v1/
        ├── raw/              # Dữ liệu gốc
        │   └── raw_dataset.csv
        ├── processed/        # Dữ liệu sau clean + preprocess
        │   └── processed_dataset.csv
        ├── split/            # Dữ liệu split (nếu có)
        └── meta/             # Tự động sinh bởi pipeline
            ├── cleaning_log.json
            ├── validation_report.json
            ├── preprocessing_params.json
            └── ...
```

### Cấu trúc thư mục scripts

```text
scripts/
├── dataset/              # Pipeline scripts
│   ├── validation.py     # Kiểm tra chất lượng dữ liệu
│   ├── cleaning.py       # Làm sạch dữ liệu
│   ├── preprocessing.py  # Tiền xử lý văn bản
│   └── download.py       # Tải dữ liệu từ Google Drive
└── mlflow/               # MLflow-related scripts
    ├── reproduce_experiment.py
    ├── track_dataset_meta.py
    └── visualize_results.py
```

### Tải dataset từ Google Drive

Dataset được lưu trên Google Drive. Liên hệ chủ project để lấy file_id, sau đó:

```powershell
conda run -n toxic-cmt-clf python scripts/dataset/download.py --file-id <FILE_ID> --dest datasets/custom_dataset/v1/raw/raw_dataset.csv
```

### Chạy pipeline xử lý dataset

Chạy các script theo thứ tự:

```powershell
# 1. Validation
conda run -n toxic-cmt-clf python scripts/dataset/validation.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv

# 2. Cleaning
conda run -n toxic-cmt-clf python scripts/dataset/cleaning.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv

# 3. Preprocessing
conda run -n toxic-cmt-clf python scripts/dataset/preprocessing.py --input datasets/custom_dataset/v1/processed/processed_dataset.csv

# 4. Track lên MLflow
conda run -n toxic-cmt-clf python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1
```

### Chạy thử nghiệm với Notebook

1. Mở `notebooks/training/experiment_template.ipynb`
2. Sửa các tham số trong cell "Experiment Config"
3. Chạy từng cell: Load Data → Validation → Cleaning → Preprocessing → Train + MLflow Tracking → So sánh kết quả

### So sánh kết quả experiments

```powershell
# Export HTML comparison
conda run -n toxic-cmt-clf python scripts/mlflow/visualize_results.py

# Reproduce best run
conda run -n toxic-cmt-clf python scripts/mlflow/reproduce_experiment.py --best f1

# Reproduce specific run
conda run -n toxic-cmt-clf python scripts/mlflow/reproduce_experiment.py --run-id <run_id>
```

## Quy tắc dependency (tóm tắt)

- Luôn sửa `environment.yml` trước, sau đó mới update environment.
- Dependency thuộc Conda: xóa khỏi `environment.yml` rồi chạy lệnh update chuẩn.
- Dependency thuộc PyPI: ngoài bước trên, cần chạy thêm `pip uninstall` cho gói đã xóa.

Chi tiết command và ví dụ nằm tại [docs/workflows.md](docs/workflows.md).

## Quy trình đóng góp (tóm tắt)

1. Tạo nhánh làm việc từ `main` theo chuẩn branch name trong [.github/branch_rules.md](.github/branch_rules.md).
2. Commit nhỏ, rõ ràng theo Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
3. Mở Pull Request và điền đầy đủ template tại [.github/pull_request_template.md](.github/pull_request_template.md).

## Báo lỗi và bảo mật

Khi tạo issue, vui lòng cung cấp bước tái hiện, kết quả mong đợi, kết quả thực tế và log liên quan.

Không chia sẻ token, API key hoặc thông tin nhạy cảm trong code, issue, PR.
