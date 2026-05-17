# Hướng dẫn sử dụng MLflow

## Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
- [2. MLflow (Machine Learning Lifecycle)](#2-mlflow-machine-learning-lifecycle)
- [3. Dataset trên Google Drive](#3-dataset-trên-google-drive)
- [4. Hướng dẫn thực tế cho người mới](#4-hướng-dẫn-thực-tế-cho-người-mới)
- [5. Troubleshooting](#5-troubleshooting)
- [6. Cheat Sheet](#6-cheat-sheet)

---

## 1. Giới thiệu

Dự án **Toxic Comment Classification** sử dụng **MLflow** để tracking thử nghiệm. Dataset được lưu trên Google Drive và tải về bằng cách mount Drive (trên Colab) hoặc dùng script `download.py` (trên local).

- **Tái lập kết quả (Reproducibility)**: Biết chính xác dataset, params, code đã dùng.
- **So sánh thử nghiệm**: Dễ dàng so sánh metrics giữa các lần chạy.
- **Cộng tác nhóm**: Chia sẻ dataset qua Google Drive, chia sẻ kết quả qua MLflow UI.

---

## 2. MLflow (Machine Learning Lifecycle)

### 2.1. Cài đặt & Khởi tạo

MLflow đã khai báo trong `environment.yml`:

```yaml
dependencies:
  - mlflow>=2.10,<3.0
```

**Cài đặt:**

```powershell
conda env update -n toxic-cmt-clf -f .\environment.yml --prune -v
mlflow ui    # Mở http://localhost:5000
```

### 2.2. Code integration: `MLflowTracker`

File `src/tracking/mlflow_tracker.py` - context manager tự động log params, metrics, artifacts.

```python
from src.tracking.mlflow_tracker import MLflowTracker

with MLflowTracker(
    experiment_name="toxic_comment_classification",
    run_name="lr_baseline",
    tags={"model_family": "logistic_regression", "dataset": "custom_dataset/v1"},
) as tracker:
    tracker.log_params({"model_type": "logistic_regression", "C": 1.0})
    tracker.log_metrics({"accuracy": 0.85, "f1": 0.82})
    tracker.log_artifact("path/to/file.csv")
    tracker.log_dict_as_yaml({"key": "value"}, "report.yaml")
    tracker.log_model(model, artifact_path="model")
    tracker.set_tag("git_commit", "abc123")
    print(f"Run ID: {tracker.run_id}")
```

### 2.3. Code integration: `ExperimentManager`

File `src/tracking/experiment_manager.py` - quản lý và so sánh experiments.

```python
from src.tracking.experiment_manager import ExperimentManager

manager = ExperimentManager(experiment_name="toxic_comment_classification")

# So sánh các runs
comparison = manager.compare_runs(
    metrics=["accuracy", "f1", "roc_auc"],
    params=["model_type", "C", "dataset_name"],
)

# Tìm run tốt nhất
best_run = manager.get_best_run(metric="f1", mode="max")

# Export HTML
html_path = manager.export_comparison_html()
```

---

## 3. Dataset trên Google Drive

### 3.1. Cấu trúc thư mục datasets

```text
datasets/
│
└── custom_dataset/
    └── v1/
        ├── raw/
        │   └── raw_dataset.csv
        ├── processed/
        │   └── processed_dataset.csv
        ├── split/
        └── meta/
            ├── validation_report.json
            ├── cleaning_log.json
            └── preprocessing_params.json
```

### 3.2. Tải dataset từ Google Drive

#### Trên local

Sử dụng script `scripts/dataset/download.py` để tải dataset từ Google Drive:

```powershell
python scripts/dataset/download.py --file-id <FILE_ID> --dest datasets/custom_dataset/v1/raw/raw_dataset.csv
```

Hoặc dùng hàm `download_local()` trong code:

```python
from src.dataset.loader import download_local

download_local(file_id="1ABCxyz123", dest="datasets/custom_dataset/v1/raw/raw_dataset.csv")
```

#### Trên Google Colab

Mount Google Drive và copy dữ liệu về thư mục datasets:

```python
from google.colab import drive
drive.mount("/content/drive")

import shutil, os
DRIVE_DATA_DIR = "/content/drive/MyDrive/CommentClassificationDataset"
LOCAL_RAW_DIR = "datasets/custom_dataset/v1/raw"
os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
shutil.copy(f"{DRIVE_DATA_DIR}/raw_dataset.csv", f"{LOCAL_RAW_DIR}/raw_dataset.csv")
```

> **Lưu ý:** File trên Google Drive cần được chia sẻ ở chế độ public (Anyone with the link can view).

### 3.3. Chạy pipeline xử lý dataset

Chạy các script theo thứ tự:

```powershell
# 1. Validation
conda run -n toxic-cmt-clf python scripts/dataset/validation.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv

# 2. Cleaning
conda run -n toxic-cmt-clf python scripts/dataset/cleaning.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv

# 3. Preprocessing
conda run -n toxic-cmt-clf python scripts/dataset/preprocessing.py --input datasets/custom_dataset/v1/processed/processed_dataset.csv
```

### 3.4. Track dataset meta lên MLflow

Sau khi chạy pipeline, track thông tin dataset lên MLflow:

```powershell
python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1
```

Script này sẽ:

1. Tính hash (md5) của toàn bộ thư mục dataset
2. Kiểm tra xem hash này đã được track chưa (tránh tạo run trùng)
3. Tạo MLflow run với tag `dataset_hash` và log các file trong `meta/`

### 3.5. Kiến trúc tổng thể

```text
Pipeline scripts (validation.py, cleaning.py, preprocessing.py)
    │
    ├── Xử lý dữ liệu và lưu vào datasets/<dataset>/<version>/
    ├── Tự động lưu báo cáo vào thư mục meta/
    │
    └── scripts/mlflow/track_dataset_meta.py → dataset_tracker.py → MLflow
        (1 run = 1 dataset hash)
```

**Nguyên tắc:**

- **Pipeline script** → chỉ làm pipeline, KHÔNG chứa MLflow code
- **MLflow script riêng** → đọc file trong meta/ và log vào MLflow
- **1 dataset hash = 1 MLflow run** (nếu hash không đổi thì không tạo run mới)

---

## 4. Hướng dẫn thực tế cho người mới

### 4.1. Clone project & chuẩn bị

```powershell
git clone https://github.com/TrieuKhac-dev/Toxic-Comment-Classification.git
cd Toxic-Comment-Classification

conda env create -f .\environment.yml
conda activate toxic-cmt-clf
pre-commit install
```

### 4.2. Tải dataset về

#### Trên local

Liên hệ chủ project để lấy file_id, sau đó:

```powershell
python scripts/dataset/download.py --file-id <FILE_ID> --dest datasets/custom_dataset/v1/raw/raw_dataset.csv
```

#### Trên Google Colab

Mount Google Drive và copy dữ liệu:

```python
from google.colab import drive
drive.mount("/content/drive")

import shutil, os
DRIVE_DATA_DIR = "/content/drive/MyDrive/CommentClassificationDataset"
LOCAL_RAW_DIR = "datasets/custom_dataset/v1/raw"
os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
shutil.copy(f"{DRIVE_DATA_DIR}/raw_dataset.csv", f"{LOCAL_RAW_DIR}/raw_dataset.csv")
```

### 4.3. Chạy pipeline dataset

```powershell
conda run -n toxic-cmt-clf python scripts/dataset/validation.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv
conda run -n toxic-cmt-clf python scripts/dataset/cleaning.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv
conda run -n toxic-cmt-clf python scripts/dataset/preprocessing.py --input datasets/custom_dataset/v1/processed/processed_dataset.csv
```

### 4.4. Track lên MLflow

```powershell
conda run -n toxic-cmt-clf python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1
```

### 4.5. Chạy thử nghiệm model

Mở `notebooks/training/experiment_template.ipynb`, sửa tham số trong cell **"Experiment Config"**, chạy từng cell.

### 4.6. So sánh kết quả

```powershell
# MLflow UI
conda run -n toxic-cmt-clf mlflow ui

# Export HTML
conda run -n toxic-cmt-clf python scripts/mlflow/visualize_results.py
```

---

## 5. Troubleshooting

### MLflow

| Vấn đề                                          | Giải pháp                                                |
| ----------------------------------------------- | -------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'mlflow'` | Chưa cài MLflow: `conda install mlflow`                  |
| `mlflow ui` không mở được                       | Kiểm tra port 5000: `mlflow ui --port 5001`              |
| Không thấy run mới trong UI                     | Refresh trang, kiểm tra đúng experiment name             |
| `mlruns/` quá lớn                               | Xóa các run cũ không cần thiết qua UI hoặc xóa `mlruns/` |

### Google Drive

| Vấn đề                                     | Giải pháp                                                             |
| ------------------------------------------ | --------------------------------------------------------------------- |
| `drive.mount()` không hoạt động trên Colab | Chạy cell mount, cấp quyền truy cập, kiểm tra đường dẫn Drive         |
| File không tìm thấy trong Drive            | Kiểm tra đường dẫn `DRIVE_DATA_DIR`, đảm bảo file tồn tại đúng vị trí |
| `download.py` không tải được file (local)  | Kiểm tra file_id, đảm bảo file đã được chia sẻ public                 |
| File tải về bị hỏng                        | Kiểm tra dung lượng file, thử tải bằng trình duyệt trước              |

---

## 6. Cheat Sheet

### MLflow Commands

```powershell
mlflow ui                         # Mở MLflow UI (http://localhost:5000)
mlflow ui --port 5001             # Mở ở port khác
mlflow experiments list           # Liệt kê experiments
mlflow runs list --experiment-id <id>  # Liệt kê runs
mlflow runs delete --run-id <id>  # Xóa run
```

### Workflow nhanh hàng ngày

#### Trên local (Windows)

```powershell
# 1. Tải dataset mới nhất từ Drive
python scripts/dataset/download.py --file-id <FILE_ID> --dest datasets/custom_dataset/v1/raw/raw_dataset.csv

# 2. Chạy pipeline
conda run -n toxic-cmt-clf python scripts/dataset/validation.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv
conda run -n toxic-cmt-clf python scripts/dataset/cleaning.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv
conda run -n toxic-cmt-clf python scripts/dataset/preprocessing.py --input datasets/custom_dataset/v1/processed/processed_dataset.csv

# 3. Track lên MLflow
conda run -n toxic-cmt-clf python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1

# 4. Chạy thử nghiệm (trong notebook hoặc script)

# 5. Xem kết quả
mlflow ui

# 6. Export comparison
python scripts/mlflow/visualize_results.py
```

#### Trên Google Colab

```python
# 1. Mount Drive và copy dataset
from google.colab import drive
drive.mount("/content/drive")
import shutil, os
DRIVE_DATA_DIR = "/content/drive/MyDrive/CommentClassificationDataset"
LOCAL_RAW_DIR = "datasets/custom_dataset/v1/raw"
os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
shutil.copy(f"{DRIVE_DATA_DIR}/raw_dataset.csv", f"{LOCAL_RAW_DIR}/raw_dataset.csv")

# 2. Chạy pipeline (dùng script hoặc import trực tiếp)
!python scripts/dataset/validation.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv
!python scripts/dataset/cleaning.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv
!python scripts/dataset/preprocessing.py --input datasets/custom_dataset/v1/processed/processed_dataset.csv

# 3. Track lên MLflow
!python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1

# 4. Chạy thử nghiệm trong notebook
```

---

> **Tài liệu tham khảo:**
>
> - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
> - [gdown Documentation](https://github.com/wkentaro/gdown)
