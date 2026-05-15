# Task Progress - Tích hợp DVC + MLflow + Pipeline Config

## 🎯 Mục tiêu

- **DVC**: Quản lý version dataset, remote là **Google Drive**
- **MLflow**: Tracking experiment, lưu kết quả, so sánh trực quan
- **Notebook (Colab)**: Môi trường chính để thử nghiệm cleaning/preprocessing/training methods
- **Local**: Chạy `mlflow ui` để xem kết quả, chạy DVC để tái tạo model ổn định

---

## 🏗️ Cấu trúc thư mục datasets

```
data/
└── custom_dataset/
    ├── README.md
    └── v1/
        ├── README.md
        ├── raw/                    # raw_dataset.csv (DVC track)
        ├── preprocessed/           # chỉ khi model ổn định
        ├── split/                  # chỉ khi model ổn định
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── meta/
            ├── cleaning_log.yaml           # Cleaning pipeline tự tạo
            ├── validation_report.yaml      # Validation pipeline tự tạo
            ├── preprocessing_params.yaml   # Preprocessing pipeline tự tạo
            ├── splits.yaml                # Split pipeline tự tạo (sau này)
            ├── statistics.json            # Pipeline tự tạo
            └── source_manifest.json       # Người dùng viết tay
```

**Lưu ý DVC:**

- Chỉ track `raw/` — `dvc add data/custom_dataset/v1/raw`
- `preprocessed/`, `split/`, `meta/` không track (có thể tái tạo lại)

---

## ✅ Checklist chi tiết

### Phase 1: Cập nhật Config files

- [ ] **`config/cleaning_config.py`** — Thêm 9 `enable_*` flags:
  - `enable_html_removal`, `enable_url_removal`, `enable_mention_removal`, `enable_emoji_removal`, `enable_special_chars_removal`, `enable_null_empty_removal`, `enable_non_text_removal`, `enable_duplicate_removal`, `enable_outlier_removal` (thay `outlier_enabled`)
- [ ] **`config/preprocessing_config.py`** — Thêm 3 `enable_*` flags:
  - `enable_normalize`, `enable_tokenize`, `enable_stopword_filter`
- [ ] **`config/path_config.py`** — Cập nhật:
  - Thêm `data_dir: str = "data"`
  - Helper methods dùng chung (nhận `dataset_name` làm tham số):
    - `get_dataset_dir(dataset_name)` → `data/<dataset_name>`
    - `get_drive_dataset_dir(dataset_name)` → `drive_data_dir/<dataset_name>`
    - `get_version_dir(dataset_name, version)` → `data/<dataset_name>/<version>`
    - `get_raw_dir(dataset_name, version)` → `data/<dataset_name>/<version>/raw`
    - `get_preprocessed_dir(dataset_name, version)` → `data/<dataset_name>/<version>/preprocessed`
    - `get_split_dir(dataset_name, version)` → `data/<dataset_name>/<version>/split`
    - `get_meta_dir(dataset_name, version)` → `data/<dataset_name>/<version>/meta`
  - Đọc `GDRIVE_FOLDER_ID` từ `.env` bằng `python-dotenv`

### Phase 2: Cập nhật Pipeline code (thiết kế thông minh + tự động lưu meta)

- [ ] **Tạo `src/pipeline/base_pipeline.py`**:
  - `PipelineStep` dataclass: `name`, `enabled_flag`, `func`, `kwargs`
  - `BasePipeline.run_steps()` — static method, duyệt steps, chỉ chạy step được bật
  - **Giữ đơn giản**: không auto-save meta trong BasePipeline. Để pipeline cụ thể tự xử lý.
- [ ] **Cập nhật `src/pipeline/cleaning_pipeline.py`**:
  - Đăng ký `CLEANING_STEPS` list
  - Dùng `BasePipeline.run_steps()`
  - Sau khi chạy, tự động lưu `cleaning_log.yaml` vào `meta/`
- [ ] **Cập nhật `src/pipeline/validation_pipeline.py`**:
  - Đăng ký `VALIDATION_STEPS` list
  - Dùng `BasePipeline.run_steps()`
  - Sau khi chạy, tự động lưu `validation_report.yaml` vào `meta/`
- [ ] **Cập nhật `src/pipeline/preprocessing_pipeline.py`**:
  - Đăng ký `PREPROCESSING_STEPS` list
  - Dùng `BasePipeline.run_steps()`
  - Sau khi chạy, tự động lưu `preprocessing_params.yaml` vào `meta/`

### Phase 3: Thiết lập DVC

- [ ] **Cập nhật `environment.yml`** — Thêm:
  ```yaml
  - pip:
      - dvc>=3.0,<4.0
      - dvc-gdrive>=3.0,<4.0
      - mlflow>=2.10,<3.0
      - pyyaml>=6.0,<7.0
      - python-dotenv>=1.0,<2.0
  ```
- [ ] **Tạo file `.env`** — Chứa `GDRIVE_FOLDER_ID=...` (không commit)
- [ ] **Tạo file `.env.example`** — Chứa `GDRIVE_FOLDER_ID=your-google-drive-folder-id` (commit được)
- [ ] **Cập nhật `.gitignore`** — Thêm `.env`, `.dvc/`, `mlruns/`
- [ ] **Cập nhật môi trường**:
  ```powershell
  conda env update -n toxic-cmt-clf -f .\environment.yml --prune -v
  ```
- [ ] **Khởi tạo DVC**: `dvc init`
- [ ] **Cấu hình remote Google Drive**: `dvc remote add -d gdrive_remote gdrive://<folder-id>`
- [ ] **Tạo cấu trúc thư mục**:
  ```
  data/custom_dataset/v1/{raw, preprocessed, split/train, split/val, split/test, meta}
  ```
- [ ] **Copy raw_dataset.csv vào** `data/custom_dataset/v1/raw/`
- [ ] **Track bằng DVC**: `dvc add data/custom_dataset/v1/raw`
- [ ] **Push lên Google Drive**: `dvc push`

### Phase 4: Thiết lập MLflow

- [ ] **Tạo `src/tracking/__init__.py`**
- [ ] **Tạo `src/tracking/mlflow_tracker.py`** — `MLflowTracker` class:
  - `start_run(run_name, tags)`
  - `log_params_from_config(config_obj, prefix)`
  - `log_metrics(metrics_dict)`
  - `log_artifact(file_path)`
  - `log_dict(dictionary, artifact_path)`
  - `log_dataset_info(dataset_name, version, dvc_hash)`
  - `end_run()`
  - `search_runs(experiment_name)`
  - Nhận `tracking_uri` từ config hoặc biến môi trường
- [ ] **Tạo `src/tracking/experiment_manager.py`** — `ExperimentManager` class:
  - `get_experiment_runs(experiment_name)`
  - `compare_runs(experiment_name, metrics)`
  - `plot_metric_comparison(experiment_name, metric)`
- [ ] **Tạo `src/tracking/dvc_integration.py`** — `DVCIntegration` class:
  - `get_dvc_hash(path)`
  - `checkout_dataset(path)`
  - `get_dataset_versions()`

### Phase 5: Notebook Training Experiments

- [ ] **Tạo `notebooks/training/experiment_template.ipynb`**:
  - Import MLflowTracker + Pipeline
  - Thử nghiệm với các config khác nhau
  - Log lên MLflow
  - So sánh kết quả bằng ExperimentManager
- [ ] **Cập nhật các notebook hiện có** (Bản_3, Bản_5, Bản_14, Bản_15):
  - Thêm cell MLflow tracking

### Phase 6: Scripts hỗ trợ

- [ ] **Tạo `scripts/reproduce_experiment.py`**:
  - Đọc MLflow run → lấy params
  - `dvc checkout` dataset
  - Chạy lại pipeline với đúng config
- [ ] **Tạo `scripts/visualize_results.py`**:
  - Query MLflow
  - Vẽ biểu đồ so sánh
  - Export báo cáo

---

## 🔄 Luồng làm việc tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                   COLAB (thử nghiệm hàng ngày)                   │
│                                                                  │
│  1. Mount Google Drive                                           │
│  2. Import MLflowTracker + Pipeline                              │
│  3. tracking_uri = '/content/drive/MyDrive/mlruns'               │
│  4. Thử nghiệm method A → log lên MLflow (Run 1)                │
│  5. Thử nghiệm method B → log lên MLflow (Run 2)                │
│  6. Dùng ExperimentManager để so sánh ngay trong notebook        │
│  7. Kết luận: method nào tốt nhất                                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LOCAL (xem kết quả chi tiết)                   │
│                                                                  │
│  1. Copy mlruns từ Google Drive về local                         │
│  2. Chạy: mlflow ui → localhost:5000                             │
│  3. Xem tất cả experiment, so sánh, download artifacts           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         KHI CÓ MODEL ỔN ĐỊNH (DVC + MLflow)                     │
│                                                                  │
│  1. dvc add data/custom_dataset/v1/raw                           │
│  2. dvc push → lên Google Drive                                  │
│  3. Ghi DVC hash vào MLflow run cuối                             │
│  4. Sau này: dvc checkout → tái tạo kết quả                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Rà soát lỗi và lỗ hổng

### 1. Lỗi hiện có trong codebase (cần sửa trước khi implement mới)

| #   | Vấn đề                                                                                                                      | File                                     | Mức độ        |
| --- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------- |
| 1   | `label_col` mặc định không đồng nhất: `dataset_config.py` dùng `"is_violation"`, các file khác dùng `"is_toxic"`            | `config/dataset_config.py`               | 🔴 Cao        |
| 2   | Import sai tên module: `feature_enginering` (thiếu 'e')                                                                     | `src/dataset/eda.py` dòng 19             | 🟡 Trung bình |
| 3   | Import `IsolationForest` thừa (đã import ở dòng 14 nhưng không dùng trực tiếp)                                              | `src/dataset/eda.py` dòng 14             | 🟢 Thấp       |
| 4   | Import alias không cần thiết: `get_stopwords as _get_stopwords`                                                             | `src/pipeline/preprocessing_pipeline.py` | 🟢 Thấp       |
| 5   | `scripts/dataset/validation.py` dùng `encoding=validation_config.encoding` nhưng `read_csv_with_columns` dùng `**pd_kwargs` | `scripts/dataset/validation.py` dòng 99  | 🟡 Trung bình |
| 6   | Thiếu module docstring cho `cleaning.py`, `validation.py`, `preprocessing.py`                                               | `src/dataset/`                           | 🟢 Thấp       |

### 2. Lỗ hổng thiết kế cần tránh

| #   | Lỗ hổng                                                                                                                                             | Giải pháp                                                                                                             |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 1   | **`base_pipeline.py` quá phức tạp** — Cố gắng làm BasePipeline quá thông minh (auto-save meta, hooks...) có thể gây khó hiểu                        | Giữ BasePipeline đơn giản: chỉ `run_steps()`. Việc lưu meta nên để pipeline cụ thể tự xử lý sau khi gọi `run_steps()` |
| 2   | **`cleaning_pipeline.py` phụ thuộc vào `feature_enginering.py`** — Outlier detection cần feature engineering tạm thời, tạo coupling không cần thiết | Giữ nguyên logic hiện tại (tạo feature tạm → detect outlier → xóa feature) nhưng đóng gói vào một wrapper function    |
| 3   | **MLflow tracking_uri không đồng nhất** — Local dùng `mlruns/`, Colab dùng Drive, dễ nhầm lẫn                                                       | `MLflowTracker` nên nhận `tracking_uri` từ config hoặc biến môi trường, có giá trị mặc định hợp lý                    |
| 4   | **DVC folder-id lưu trong `.env` dễ bị mất** — Nếu clone repo mới mà không có `.env`, DVC remote sẽ không hoạt động                                 | Thêm `.env.example` vào repo với hướng dẫn, và trong `CONTRIBUTING.md` ghi rõ bước này                                |
| 5   | **Notebook cũ (Bản_3, Bản_5...) có thể không tương thích** — Các notebook này dùng code cũ, import trực tiếp từ file, không qua pipeline            | Khi cập nhật, cần kiểm tra kỹ import và sửa cho phù hợp với cấu trúc mới                                              |
| 6   | **Thiếu `__init__.py` ở `src/tracking/`** — Module mới cần có `__init__.py` để import được                                                          | Nhớ tạo file này                                                                                                      |

### 3. Lời khuyên

#### Về thứ tự ưu tiên

1. **Sửa lỗi hiện có trước** (đặc biệt là `label_col` không đồng nhất) — nếu không, mọi thứ sẽ sai ngay từ đầu
2. **Làm Phase 1 + 2 trước** (Config + Pipeline) — vì đây là nền tảng
3. **Sau đó Phase 4** (MLflow) — có thể test ngay trên local
4. **Phase 3** (DVC) — chỉ cần thiết khi có model ổn định, nhưng nên setup sớm để track dataset
5. **Phase 5 + 6** — Làm cuối cùng

#### Về thiết kế

- **Giữ đơn giản**: Đừng làm BasePipeline quá thông minh. Một vòng lặp `for` với `if getattr()` là đủ.
- **Tách biệt concern**: Pipeline chỉ nên xử lý dữ liệu và trả về report. Việc lưu file hay log lên MLflow nên là trách nhiệm của caller (script/notebook).
- **Dùng `@dataclass` cho config**: Đã có `BaseConfig.override()`, rất tiện cho việc thử nghiệm trong notebook.

#### Về MLflow trên Colab

- Luôn set `tracking_uri` vào Google Drive để không mất dữ liệu khi runtime reset
- Dùng `ExperimentManager` trong notebook để xem kết quả, không cần MLflow UI
- Khi về local, copy `mlruns/` từ Drive về và chạy `mlflow ui`

#### Về DVC

- Chỉ track `raw/` — các thư mục khác (`preprocessed/`, `split/`) chỉ track khi có model ổn định
- `dvc.yaml` chưa cần tạo — chỉ tạo khi cần tái tạo pipeline hoàn chỉnh
- Nhớ push sau khi add: `dvc push`
