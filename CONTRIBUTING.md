# Contributing

## Mục tiêu tài liệu

`CONTRIBUTING.md` là điểm vào chính cho contributor mới:

- setup môi trường nhanh,
- quy tắc đóng góp mức tổng quan,
- điều hướng sang các tài liệu chi tiết theo từng mục đích.

## Bản đồ tài liệu

- Quy trình làm việc chi tiết: [docs/workflows.md](docs/workflows.md)
- Hướng dẫn DVC & MLflow: [docs/dvc_mlflow_guide.md](docs/dvc_mlflow_guide.md)
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

### Thiết lập DVC + Google Drive Remote

DVC dùng Google Drive làm remote storage. Cấu hình OAuth được lưu local (không commit lên Git).

> **Yêu cầu:** Bạn cần được chủ project share quyền truy cập vào folder Google Drive chứa dataset.
> Nếu chưa có, liên hệ chủ project để được thêm vào folder.

#### Hướng dẫn nhanh cho dev mới (clone lần đầu)

Sau khi clone project, chạy các lệnh sau:

```powershell
# 1. Khởi tạo DVC
dvc init

# 2. Thêm remote Google Drive (dùng chung folder của team, liên hệ chủ project để lấy FOLDER_ID)
dvc remote add gdrive_remote gdrive://<FOLDER_ID>


# 3. Cấu hình OAuth Client ID (lấy từ chủ project hoặc file scripts/dvc/client_secrets.json)
dvc remote modify gdrive_remote gdrive_client_id <CLIENT_ID>
dvc remote modify gdrive_remote gdrive_client_secret <CLIENT_SECRET>

# 4. Set remote mặc định
dvc config core.remote gdrive_remote

# 5. Pull dữ liệu từ Google Drive
conda run -n toxic-cmt-clf dvc pull
```

Khi chạy `dvc pull` lần đầu, trình duyệt sẽ mở ra yêu cầu đăng nhập Google và cấp quyền cho ứng dụng.

> **Lưu ý:** `<CLIENT_ID>` và `<CLIENT_SECRET>` lấy từ file `scripts/dvc/client_secrets.json` (liên hệ chủ project để nhận file này, hoặc tự tạo OAuth Client ID theo hướng dẫn bên dưới).

#### Nếu tự tạo OAuth Client ID (khi chưa có file client_secrets.json)

1. Vào <https://console.cloud.google.com/apis/credentials>
2. Nếu chưa có OAuth consent screen: Chọn **OAuth consent screen** → **External** → **Create**
   - App name: `Toxic Comment Classification`
   - User support email: email Google của bạn
   - Developer contact: email Google của bạn
   - **Save and Continue**
   - Mục **Scopes** → **Add or Remove Scopes** → thêm `.../auth/drive` → **Update** → **Save and Continue**
   - Mục **Test users** → **Add Users** → thêm email Google của bạn → **Save and Continue**
3. Quay lại **Credentials** → **+ Create Credentials** → **OAuth client ID**
   - Application type: **Desktop app**
   - Name: `DVC Desktop`
   - Nhấn **Create** → **Download JSON**
4. Đặt file vào `scripts/dvc/client_secrets.json`
5. Mở file lấy `client_id` và `client_secret` để dùng ở bước 3 ở trên

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
        ├── raw/              # DVC tracked
        │   └── raw_dataset.csv
        ├── processed/        # DVC tracked (output của clean + preprocess)
        │   └── processed_dataset.csv
        ├── split/            # DVC sẽ tự sinh (nếu có)
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
│   └── download.py       # Tải dữ liệu
├── dvc/                  # DVC-related scripts
│   ├── auth_gdrive.py
│   ├── client_secrets.json
│   └── settings.yaml
└── mlflow/               # MLflow-related scripts
    ├── reproduce_experiment.py
    ├── track_dataset_meta.py
    └── visualize_results.py
```

### Chạy DVC Pipeline

```powershell
# Chạy toàn bộ pipeline (validate → clean → preprocess → track_meta)
conda run -n toxic-cmt-clf dvc repro -p dataset=custom_dataset,version=v1

# Chạy từng stage riêng
conda run -n toxic-cmt-clf dvc repro -p dataset=custom_dataset,version=v1 --single-item validate
conda run -n toxic-cmt-clf dvc repro -p dataset=custom_dataset,version=v1 --single-item clean
conda run -n toxic-cmt-clf dvc repro -p dataset=custom_dataset,version=v1 --single-item preprocess
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
