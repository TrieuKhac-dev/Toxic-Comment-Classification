# Hướng dẫn sử dụng DVC & MLflow

## Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
- [2. DVC (Data Version Control)](#2-dvc-data-version-control)
- [3. MLflow (Machine Learning Lifecycle)](#3-mlflow-machine-learning-lifecycle)
- [4. Kết hợp DVC + MLflow](#4-kết-hợp-dvc--mlflow)
- [5. Hướng dẫn thực tế cho người mới](#5-hướng-dẫn-thực-tế-cho-người-mới)
- [6. Troubleshooting](#6-troubleshooting)
- [7. Cheat Sheet](#7-cheat-sheet)

---

## 1. Giới thiệu

Dự án **Toxic Comment Classification** sử dụng **DVC** để quản lý phiên bản dữ liệu và **MLflow** để tracking thử nghiệm. Hai công cụ này kết hợp giúp:

- **Tái lập kết quả (Reproducibility)**: Biết chính xác dataset, params, code đã dùng.
- **So sánh thử nghiệm**: Dễ dàng so sánh metrics giữa các lần chạy.
- **Cộng tác nhóm**: Chia sẻ dataset qua Google Drive, chia sẻ kết quả qua MLflow UI.

---

## 2. DVC (Data Version Control)

### 2.1. Cài đặt & Khởi tạo

DVC đã khai báo trong `environment.yml`:

```yaml
dependencies:
  - dvc>=3.0,<4.0
  - pip:
      - dvc-gdrive>=3.0,<4.0
```

**Cài đặt:**

```powershell
conda env update -n toxic-cmt-clf -f .\environment.yml --prune -v
```

**Khởi tạo DVC (chạy 1 lần nếu clone mới):**

```powershell
dvc init
```

### 2.2. Cấu trúc DVC trong project

DVC track **theo version của dataset** (cả thư mục version), không track từng file riêng lẻ.

```
datasets/
│
├── custom_dataset/
│   └── v1/
│       ├── raw/
│       │   └── raw_dataset.csv
│       ├── processed/
│       │   └── processed_dataset.csv
│       ├── split/
│       └── meta/
│           ├── validation_report.json
│           ├── cleaning_log.json
│           └── preprocessing_params.json
│
└── v1.dvc                           # 1 file .dvc duy nhất cho cả version
```

**File `.dvc`** chứa hash md5 của toàn bộ thư mục version. Commit file này vào Git, dữ liệu thật được lưu ở remote.

### 2.3. Các lệnh DVC cơ bản

```powershell
# Track cả thư mục version
dvc add datasets/custom_dataset/v1/
git add datasets/custom_dataset/v1.dvc
git commit -m "feat: add custom_dataset v1"

# Push/Pull dữ liệu
dvc push
dvc pull

# Kiểm tra trạng thái
dvc status
dvc status --cloud

# Chuyển đổi phiên bản
git checkout <old-commit-hash>
dvc checkout
```

### 2.4. Làm việc với DVC remote (Google Drive)

```powershell
# Cấu hình remote (chạy 1 lần)
dvc remote add gdrive_remote gdrive://<FOLDER_ID>
dvc remote modify gdrive_remote gdrive_client_id <CLIENT_ID>
dvc remote modify gdrive_remote gdrive_client_secret <CLIENT_SECRET>
dvc config core.remote gdrive_remote

# Xem danh sách remote
dvc remote list
```

> Config được lưu trong `.dvc/config` (đã ignore trong `.gitignore`).

### 2.5. DVC Pipeline

DVC Pipeline định nghĩa các bước xử lý dữ liệu dưới dạng DAG trong `dvc.yaml`.

```powershell
# Chạy toàn bộ pipeline
dvc repro

# Chạy từ stage cụ thể
dvc repro clean_data

# Xem DAG
dvc dag
```

### 2.6. Code integration: `dvc_integration.py`

File `src/tracking/dvc_integration.py` cung cấp các hàm:

| Hàm                                             | Mô tả                                 |
| ----------------------------------------------- | ------------------------------------- |
| `get_dvc_hash(path)`                            | Lấy DVC hash (md5) của file/directory |
| `get_dvc_remote_url()`                          | Lấy URL của DVC remote hiện tại       |
| `log_dvc_info_to_mlflow(tracker, dataset_path)` | Log DVC version info vào MLflow run   |
| `create_dvc_stage(stage_name, cmd, ...)`        | Tạo DVC stage bằng CLI                |
| `get_dvc_pipeline_status()`                     | Kiểm tra trạng thái DVC pipeline      |

```python
from src.tracking.dvc_integration import get_dvc_hash, get_dvc_remote_url

dvc_hash = get_dvc_hash("datasets/custom_dataset/v1")
remote_url = get_dvc_remote_url()
```

### 2.7. BasePipeline framework (Validation, Cleaning, Preprocessing)

Project dùng **BasePipeline** (Python) để thực thi logic xử lý và **DVC Pipeline** (`dvc.yaml`) để orchestrate & cache.

```
src/pipeline/
├── base_pipeline.py              # BasePipeline + PipelineStep
├── validation_pipeline.py        # Kiểm tra chất lượng dữ liệu
├── cleaning_pipeline.py          # Làm sạch dữ liệu
└── preprocessing_pipeline.py     # Tiền xử lý văn bản
```

**Cách chạy các pipeline:**

```python
import pandas as pd
from src.pipeline.validation_pipeline import validate_dataset
from src.pipeline.cleaning_pipeline import clean_text_pipeline
from src.pipeline.preprocessing_pipeline import preprocess_text_pipeline

df = pd.read_csv("datasets/custom_dataset/v1/raw/raw_dataset.csv")

# 1. Validation
val_report = validate_dataset(df, dataset_name="custom_dataset", version="v1")

# 2. Cleaning
df_cleaned, cleaning_report = clean_text_pipeline(df, dataset_name="custom_dataset", version="v1")

# 3. Preprocessing
df_cleaned["processed_comment"] = df_cleaned["comment"].apply(
    lambda x: preprocess_text_pipeline(str(x), dataset_name="custom_dataset", version="v1")
)
```

Hoặc qua CLI:

```powershell
conda run -n toxic-cmt-clf python scripts/dataset/validation.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv
conda run -n toxic-cmt-clf python scripts/dataset/cleaning.py --input datasets/custom_dataset/v1/raw/raw_dataset.csv
conda run -n toxic-cmt-clf python scripts/dataset/preprocessing.py --input datasets/custom_dataset/v1/processed/processed_dataset.csv
```

---

## 3. MLflow (Machine Learning Lifecycle)

### 3.1. Cài đặt & Khởi tạo

```yaml
dependencies:
  - mlflow>=2.10,<3.0
```

```powershell
conda env update -n toxic-cmt-clf -f .\environment.yml --prune -v
mlflow ui    # Mở http://localhost:5000
```

### 3.2. Code integration: `MLflowTracker`

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

### 3.3. Code integration: `ExperimentManager`

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

## 4. Kết hợp DVC + MLflow

### 4.1. Kiến trúc tổng thể

```
DVC repro (có cache từng bước)
    │
    ├── validate → scripts/dataset/validation.py → meta/validation_report.json
    ├── clean    → scripts/dataset/cleaning.py    → processed/processed_dataset.csv + meta/cleaning_log.json
    ├── preprocess → scripts/dataset/preprocessing.py → processed/processed_dataset.csv + meta/preprocessing_params.json
    │
    ├── dvc add datasets/custom_dataset/v1/  (track cả version)
    │
    └── scripts/mlflow/track_dataset_meta.py → dataset_tracker.py → MLflow (1 run = 1 DVC hash)
```

**Nguyên tắc:**

- **Pipeline script** → chỉ làm pipeline, KHÔNG chứa MLflow code
- **MLflow script riêng** → đọc file trong meta/ và log vào MLflow
- **1 DVC hash = 1 MLflow run** (nếu hash không đổi thì không tạo run mới)

### 4.2. Luồng làm việc chi tiết

```bash
# === LẦN ĐẦU: Chạy full pipeline ===
dvc repro -p dataset=custom_dataset,version=v1
dvc add datasets/custom_dataset/v1/
python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1
git add .
git commit -m "dataset: v1 initial"
git push
dvc push

# === LẦN SAU: Chỉ sửa cleaning config ===
dvc repro -p dataset=custom_dataset,version=v1 --single-item clean
dvc add datasets/custom_dataset/v1/
python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1
git add .
git commit -m "dataset: v1 update cleaning"
git push
dvc push
```

### 4.3. Log DVC hash vào MLflow (cho training experiment)

```python
from src.tracking.mlflow_tracker import MLflowTracker
from src.tracking.dvc_integration import log_dvc_info_to_mlflow

with MLflowTracker(experiment_name="toxic_comment_classification", run_name="lr_baseline") as tracker:
    log_dvc_info_to_mlflow(tracker, "datasets/custom_dataset/v1")
    # ... training code ...
```

Hàm `log_dvc_info_to_mlflow()` log: `dvc_dataset_path`, `dvc_dataset_hash`, `git_commit`.

### 4.4. Reproduce experiment

```powershell
# Reproduce run cụ thể
conda run -n toxic-cmt-clf python scripts/mlflow/reproduce_experiment.py --run-id <run_id>

# Reproduce run tốt nhất dựa trên metric
conda run -n toxic-cmt-clf python scripts/mlflow/reproduce_experiment.py --experiment toxic_comment_classification --best f1
```

---

## 5. Hướng dẫn thực tế cho người mới

### 5.1. Clone project & lấy dữ liệu về

```powershell
git clone https://github.com/TrieuKhac-dev/Toxic-Comment-Classification.git
cd Toxic-Comment-Classification

conda env create -f .\environment.yml
conda activate toxic-cmt-clf
pre-commit install

dvc init
dvc remote add gdrive_remote gdrive://<FOLDER_ID>
dvc remote modify gdrive_remote gdrive_client_id <CLIENT_ID>
dvc remote modify gdrive_remote gdrive_client_secret <CLIENT_SECRET>
dvc config core.remote gdrive_remote

dvc pull
dvc status
```

### 5.2. Tạo dataset version mới

```powershell
# Bước 1: Đặt file dữ liệu gốc
mkdir -p datasets/my_dataset/v1/raw
copy D:\data\comments.csv datasets\my_dataset\v1\raw\raw_dataset.csv

# Bước 2: Chạy DVC pipeline
conda run -n toxic-cmt-clf dvc repro -p dataset=my_dataset,version=v1

# Bước 3: Track dataset bằng DVC
conda run -n toxic-cmt-clf dvc add datasets/my_dataset/v1/

# Bước 4: Log meta lên MLflow
conda run -n toxic-cmt-clf python scripts/mlflow/track_dataset_meta.py --dataset my_dataset --version v1

# Bước 5-6: Commit + Push
git add datasets/my_dataset/v1.dvc datasets/my_dataset/.gitignore
git commit -m "dataset: add my_dataset v1"
git push
conda run -n toxic-cmt-clf dvc push
```

### 5.3. Cập nhật dataset version hiện tại

```powershell
# Sửa config → chạy lại stage → track lại → MLflow → commit + push
conda run -n toxic-cmt-clf dvc repro -p dataset=custom_dataset,version=v1 --single-item clean
conda run -n toxic-cmt-clf dvc add datasets/custom_dataset/v1/
conda run -n toxic-cmt-clf python scripts/mlflow/track_dataset_meta.py --dataset custom_dataset --version v1
git add .
git commit -m "dataset: custom_dataset v1 update cleaning"
git push
conda run -n toxic-cmt-clf dvc push
```

### 5.4. Chạy thử nghiệm model mới

Mở `notebooks/training/experiment_template.ipynb`, sửa tham số trong cell **"Experiment Config"**, chạy từng cell.

### 5.5. So sánh kết quả

```powershell
# MLflow UI
conda run -n toxic-cmt-clf mlflow ui

# Export HTML
conda run -n toxic-cmt-clf python scripts/mlflow/visualize_results.py
```

### 5.6. Reproduce kết quả cũ

```powershell
git log --oneline
git checkout <commit-hash>
dvc checkout
conda run -n toxic-cmt-clf python scripts/mlflow/reproduce_experiment.py --run-id <run_id>
```

---

## 6. Troubleshooting

### DVC

| Vấn đề                                 | Giải pháp                                                          |
| -------------------------------------- | ------------------------------------------------------------------ |
| `dvc pull` yêu cầu xác thực Google     | Trình duyệt sẽ mở ra, đăng nhập Google và cấp quyền                |
| `dvc: command not found`               | Chưa cài DVC: `conda install dvc dvc-gdrive`                       |
| `ERROR: failed to push data to remote` | Kiểm tra kết nối mạng, kiểm tra quyền truy cập folder Google Drive |
| `WARNING: no cache`                    | Chạy `dvc checkout` hoặc `dvc pull` để tải dữ liệu về cache        |
| `dvc status` báo `changed`             | Dữ liệu đã thay đổi, cần chạy `dvc add` lại                        |

### MLflow

| Vấn đề                                          | Giải pháp                                                |
| ----------------------------------------------- | -------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'mlflow'` | Chưa cài MLflow: `conda install mlflow`                  |
| `mlflow ui` không mở được                       | Kiểm tra port 5000: `mlflow ui --port 5001`              |
| Không thấy run mới trong UI                     | Refresh trang, kiểm tra đúng experiment name             |
| `mlruns/` quá lớn                               | Xóa các run cũ không cần thiết qua UI hoặc xóa `mlruns/` |

---

## 7. Cheat Sheet

### DVC Commands

```powershell
dvc init                          # Khởi tạo DVC
dvc add <file/dir>                # Track file/directory
dvc status                        # Kiểm tra trạng thái
dvc checkout                      # Restore dữ liệu từ cache
dvc push                          # Push lên remote
dvc pull                          # Pull từ remote
dvc gc                            # Dọn dẹp cache
dvc remote add <name> <url>       # Thêm remote
dvc remote list                   # Liệt kê remotes
dvc repro                         # Chạy pipeline
dvc dag                           # Xem DAG
dvc diff                          # So sánh thay đổi
```

### MLflow Commands

```powershell
mlflow ui                         # Mở MLflow UI (http://localhost:5000)
mlflow ui --port 5001             # Mở ở port khác
mlflow experiments list           # Liệt kê experiments
mlflow runs list --experiment-id <id>  # Liệt kê runs
mlflow runs delete --run-id <id>  # Xóa run
```

### Workflow nhanh hàng ngày

```powershell
# 1. Pull dataset mới nhất
dvc pull

# 2. Chạy thử nghiệm (trong notebook hoặc script)

# 3. Xem kết quả
mlflow ui

# 4. Export comparison
python scripts/mlflow/visualize_results.py

# 5. Commit code + DVC metadata
git add .
git commit -m "feat: add experiment with logistic regression"
git push
dvc push
```

---

> **Tài liệu tham khảo:**
>
> - [DVC Documentation](https://dvc.org/doc)
> - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
> - [DVC + MLflow Integration Guide](https://dvc.org/doc/use-cases/ml-experiments)
