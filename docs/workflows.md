# Workflows

Tài liệu này mô tả quy trình thao tác chi tiết hằng ngày. Tổng quan contributor xem tại [../CONTRIBUTING.md](../CONTRIBUTING.md).

## 1) Workflow phát triển tính năng

1. Đồng bộ nhánh chính:

   ```powershell
   git checkout main
   git pull
   ```

2. Tạo nhánh mới theo quy tắc tên branch trong [../.github/BRANCH_RULES.md](../.github/BRANCH_RULES.md):

   ```powershell
   git checkout -b feature/<ten-ngan-gon>
   ```

3. Thực hiện thay đổi và commit theo Conventional Commits.
4. Push branch:

   ```powershell
   git push -u origin feature/<ten-ngan-gon>
   ```

5. Mở Pull Request và điền mẫu tại [../.github/pull_request_template.md](../.github/pull_request_template.md).

## 2) Workflow cập nhật dependency

Nguyên tắc: luôn sửa `environment.yml` trước khi chạy lệnh update.

### 2.1 Thêm dependency

1. Thêm package vào `environment.yml` (`dependencies` hoặc `pip:` tùy nguồn).
2. Cập nhật môi trường bằng lệnh chuẩn:

   ```powershell
   conda env update -n toxic-cmt-clf -f .\environment.yml --prune -v
   ```

### 2.2 Xóa dependency thuộc Conda

1. Xóa package khỏi `environment.yml`.
2. Chạy:

   ```powershell
   conda env update -n toxic-cmt-clf -f .\environment.yml --prune -v
   ```

### 2.3 Xóa dependency thuộc PyPI

1. Xóa package khỏi `environment.yml`.
2. Chạy:

   ```powershell
   pip uninstall -y <ten-goi-pypi>
   ```

### 2.4 Lưu ý khi ghi dependency trong `environment.yml`

- Viết đúng chính tả tên package theo registry chính thức (Conda channel hoặc PyPI).
- Hạn chế dùng ký tự mơ hồ như `*` và `~=` trong version spec.
- Ưu tiên ràng buộc phiên bản rõ ràng theo khoảng hoặc mốc cụ thể.
  - Ví dụ Conda: `numpy>=1.26,<2.0`
  - Ví dụ PyPI trong `pip:`: `requests>=2.32,<3.0`
- Với dependency cốt lõi (ví dụ `python`), pin version rõ ràng để tránh thay đổi lớn ngoài ý muốn.

## 3) Workflow xử lý hotfix

1. Tạo branch hotfix từ `main`:

   ```powershell
   git checkout main
   git pull
   git checkout -b hotfix/<ten-ngan-gon>
   ```

2. Sửa lỗi, commit, push và mở PR vào `main`.
3. Sau khi merge, xóa branch theo quy tắc tại [../.github/BRANCH_RULES.md](../.github/BRANCH_RULES.md).

## 4) Kiểm tra trước khi mở PR

- Chạy lại lệnh kiểm tra cục bộ cần thiết.
- Đảm bảo không còn file tạm/debug.
- Nếu đổi dependency, xác nhận `environment.yml` đã được cập nhật đúng.

Checklist chính thức nằm trong [../.github/pull_request_template.md](../.github/pull_request_template.md).

## 5) Conda command cheat sheet

Các lệnh Conda phổ biến:

### 5.1 Tạo, kích hoạt, cập nhật environment

- Tạo mới từ `environment.yml`:

  ```powershell
  conda env create -f .\environment.yml
  ```

- Kích hoạt environment:

  ```powershell
  conda activate toxic-cmt-clf
  ```

- Cập nhật environment theo file (khuyến nghị):

  ```powershell
  conda env update -n toxic-cmt-clf -f .\environment.yml --prune -v
  ```

### 5.2 Xem thông tin và package

- Liệt kê các environment hiện có:

  ```powershell
  conda env list
  ```

- Liệt kê package trong environment đang active:

  ```powershell
  conda list
  ```

- Xem thông tin Conda hiện tại:

  ```powershell
  conda info
  ```

### 5.3 Xóa environment

- Xóa environment theo tên:

  ```powershell
  conda remove -n toxic-cmt-clf --all
  ```
