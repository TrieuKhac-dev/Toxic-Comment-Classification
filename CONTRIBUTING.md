# Contributing

## Mục tiêu tài liệu

`CONTRIBUTING.md` là điểm vào chính cho contributor mới:

- setup môi trường nhanh,
- quy tắc đóng góp mức tổng quan,
- điều hướng sang các tài liệu chi tiết theo từng mục đích.

## Bản đồ tài liệu

- Quy trình làm việc chi tiết: [docs/workflows.md](docs/workflows.md)
- Quy tắc branch bắt buộc: [.github/BRANCH_RULES.md](.github/BRANCH_RULES.md)
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

### Khởi tạo cấu hình ClearML

1. Đăng ký tài khoản tại [ClearML](https://app.clear.ml/) (hoặc dùng server riêng nếu có).
2. Mở terminal tại thư mục dự án, chạy:

   ```powershell
   clearml-init
   ```

3. Truy cập [ClearML Workspace Configuration](https://app.clear.ml/settings/workspace-configuration), tạo credentials mới và copy đoạn cấu hình.

4. Quay lại terminal, dán đoạn cấu hình khi được hỏi.

5. File `clearml.conf` sẽ được tạo ở thư mục người dùng hoặc dự án.

## Quy tắc dependency (tóm tắt)

- Luôn sửa `environment.yml` trước, sau đó mới update environment.
- Dependency thuộc Conda: xóa khỏi `environment.yml` rồi chạy lệnh update chuẩn.
- Dependency thuộc PyPI: ngoài bước trên, cần chạy thêm `pip uninstall` cho gói đã xóa.

Chi tiết command và ví dụ nằm tại [docs/workflows.md](docs/workflows.md).

## Quy trình đóng góp (tóm tắt)

1. Tạo nhánh làm việc từ `main` theo chuẩn branch name trong [.github/BRANCH_RULES.md](.github/BRANCH_RULES.md).
2. Commit nhỏ, rõ ràng theo Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
3. Mở Pull Request và điền đầy đủ template tại [.github/pull_request_template.md](.github/pull_request_template.md).

## Báo lỗi và bảo mật

Khi tạo issue, vui lòng cung cấp bước tái hiện, kết quả mong đợi, kết quả thực tế và log liên quan.

Không chia sẻ token, API key hoặc thông tin nhạy cảm trong code, issue, PR.
