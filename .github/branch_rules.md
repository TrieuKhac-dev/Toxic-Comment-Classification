# Branch Rules

Tài liệu này định nghĩa quy tắc bắt buộc cho branch strategy của dự án.

## 1) Nguyên tắc bắt buộc

1. `main` luôn phải ở trạng thái ổn định.
2. Không commit trực tiếp vào `main`.
3. Mọi thay đổi vào `main` đều đi qua Pull Request.
4. Branch làm việc là ngắn hạn và phải xóa sau khi merge.

## 2) Các branch được phép

- `main`: branch ổn định/release.
- `feature/*`: phát triển tính năng hoặc cải tiến thông thường.
- `hotfix/*`: sửa lỗi khẩn cấp cho production.

## 3) Quy tắc tạo branch

- `feature/*` tạo từ `main` mới nhất.
- `hotfix/*` tạo từ `main` mới nhất.
- Tên branch ngắn gọn, mô tả đúng mục đích.

Ví dụ:

- `feature/add-image-index-cache`
- `feature/update-search-ranking`
- `hotfix/fix-empty-query-crash`

## 4) Quy tắc merge

- `feature/*` merge vào `main` qua Pull Request.
- `hotfix/*` merge vào `main` qua Pull Request.
- Không tự merge khi chưa có review theo quy trình của team.

## 5) Xóa branch sau merge

- Bắt buộc xóa branch `feature/*` hoặc `hotfix/*` sau khi PR đã merge.
- Giữ repository gọn và tránh branch stale.

## 6) Tài liệu liên quan

- Tổng quan contributor: [../CONTRIBUTING.md](../CONTRIBUTING.md)
- Quy trình thao tác chi tiết: [../docs/workflows.md](../docs/workflows.md)
- Mẫu PR: [pull_request_template.md](pull_request_template.md)
