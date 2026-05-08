# Troubleshooting Guide – Hướng Dẫn Khắc Phục Lỗi Phổ Biến <!-- omit in toc -->

## Table of Contents (Mục Lục) <!-- omit in toc -->

- [Environment Errors (Lỗi Môi Trường)](#environment-errors-lỗi-môi-trường)
  - [Lỗi Conda Environment Không Nhận Đúng Trong PowerShell](#lỗi-conda-environment-không-nhận-đúng-trong-powershell)
    - [Ví Dụ Lỗi](#ví-dụ-lỗi)
    - [Nguyên Nhân](#nguyên-nhân)
    - [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)
    - [Cách Sửa](#cách-sửa)
    - [Lưu Ý Bảo Mật](#lưu-ý-bảo-mật)
    - [Nếu Vẫn Lỗi](#nếu-vẫn-lỗi)

## Environment Errors (Lỗi Môi Trường)

### Lỗi Conda Environment Không Nhận Đúng Trong PowerShell

Nội dung này hướng dẫn cách xử lý lỗi khi đã tạo môi trường conda và gọi lệnh activate đúng môi trường nhưng PowerShell không nhận đúng môi trường, biến PATH không được cập nhật, hoặc lệnh `echo $env:CONDA_PREFIX` trả về giá trị sai.

#### Ví Dụ Lỗi

Sau khi chạy:

```powershell
conda activate <env-name>
echo $env:CONDA_PREFIX
```

Kết quả trả về không phải đường dẫn môi trường conda bạn vừa activate, hoặc trả về rỗng.

#### Nguyên Nhân

- PowerShell chưa được hook đúng với conda (chưa init).
- Script activate của conda chưa được cấu hình cho PowerShell.
- PATH và các biến môi trường chưa được cập nhật do thiếu cấu hình shell.

#### Tài Liệu Tham Khảo

- [Conda Docs: conda init](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment)
- [PowerShell Profile](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_profiles)

#### Cách Sửa

##### Bước 1 – Kiểm tra lại activation\*\* <!-- omit in toc -->

Sau khi activate, chạy:

```powershell
echo $env:CONDA_PREFIX
```

Nếu đúng, kết quả phải là đường dẫn tới môi trường conda vừa activate (ví dụ: `C:\Users\<User>\anaconda3\envs\<env-name>`).

Nếu sai, thực hiện các bước sau:

##### Bước 2 – Init lại PowerShell cho conda\*\* <!-- omit in toc -->

Chạy lệnh sau trong PowerShell:

```powershell
conda init powershell
```

- Lệnh này sẽ cấu hình lại profile PowerShell để nhận hook của conda.
- Có thể cần quyền Administrator nếu profile nằm ở vị trí hệ thống.

##### Bước 3 – Đóng và mở lại PowerShell\*\* <!-- omit in toc -->

- Đóng cửa sổ PowerShell hiện tại.
- Mở lại một cửa sổ PowerShell mới.

##### Bước 4 – Thử lại activate\*\* <!-- omit in toc -->

Chạy lại:

```powershell
conda activate <env-name>
echo $env:CONDA_PREFIX
```

- Nếu kết quả trả về đúng đường dẫn môi trường, lỗi đã được khắc phục.
- Nếu vẫn lỗi, kiểm tra lại file profile PowerShell (`$PROFILE`) xem đã có dòng hook của conda chưa.

#### Lưu Ý Bảo Mật

- Việc init sẽ thêm script vào profile PowerShell, chỉ thực hiện khi tin tưởng nguồn cài đặt conda.
- Nếu dùng nhiều shell (cmd, bash, zsh), cần init cho từng shell tương ứng.

#### Nếu Vẫn Lỗi

- Kiểm tra lại biến môi trường bằng lệnh:

  ```powershell
  Get-ChildItem Env:
  ```

- Kiểm tra file `$PROFILE` có dòng liên quan đến conda không.
- Thử khởi động lại máy nếu profile đã đúng nhưng vẫn không nhận.
- Nếu dùng VS Code, đảm bảo terminal đang dùng PowerShell, không phải cmd hoặc bash.
