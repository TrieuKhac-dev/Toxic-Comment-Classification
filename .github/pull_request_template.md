# Pull Request Template

## Mục đích Pull Request

- [ ] Bug fix
- [ ] Tính năng mới
- [ ] Cải thiện tài liệu
- [ ] Refactor

Closes #ISSUE_NUMBER

## Mô tả thay đổi

Mô tả ngắn gọn:

- Đổi gì?
- Vì sao cần đổi?
- Ảnh/Screenshot (nếu có)

## Cách kiểm thử

Hướng dẫn reviewer chạy và kiểm tra thay đổi:

## Dependency impact

- [ ] Không thay đổi dependency
- [ ] Có thay đổi dependency trong `environment.yml`
- [ ] Đã chạy lệnh sync môi trường:

```powershell
conda env update -n toxic-cmt-clf -f .\environment.yml --prune -v
```

Nếu xóa package PyPI, xác nhận đã gỡ thêm bằng `pip uninstall`.

## Checklist trước khi request review

- [ ] Tuân thủ [.github/BRANCH_RULES.md](BRANCH_RULES.md)
- [ ] Đã self-review code
- [ ] Đã kiểm tra cục bộ các phần bị ảnh hưởng
- [ ] Đã cập nhật tài liệu liên quan (nếu cần)
