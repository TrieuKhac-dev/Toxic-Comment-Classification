"""
downloader.py

Các hàm tiện ích để tải dữ liệu từ Google Drive (file hoặc folder) bằng gdown.
Hỗ trợ cả Google Colab và local, không cần mount Drive.

Cách dùng:
    from src.utils.downloader import download

    # Tải 1 file từ folder
    download(
        folder_url="https://drive.google.com/drive/folders/...",
        dest_dir="datasets/custom_dataset/v1/raw",
        filename="raw_dataset.csv",
    )

    # Tải toàn bộ folder
    download(
        folder_url="https://drive.google.com/drive/folders/...",
        dest_dir="datasets/custom_dataset/v1/raw",
    )
"""

from __future__ import annotations

from pathlib import Path


def _find_file_recursive(directory: Path, filename: str) -> Path | None:
    """Tìm file trong directory (bao gồm subdirectory)."""
    for fpath in directory.rglob(filename):
        return fpath
    return None


def download(
    folder_url: str,
    dest_dir: str,
    filename: str | None = None,
) -> None:
    """
    Tải dữ liệu từ Google Drive folder public bằng gdown.

    Parameters
    ----------
    folder_url : str
        URL của folder Google Drive (ví dụ:
        'https://drive.google.com/drive/folders/1zayzzusvK9njgibfbqtUWv6N5aie2Cym').
    dest_dir : str
        Thư mục đích để lưu dữ liệu (ví dụ: 'datasets/custom_dataset/v1/raw').
    filename : str | None
        Tên file cần tải. Nếu None, tải toàn bộ folder.
    """
    try:
        import gdown
    except ImportError as err:
        raise ImportError(
            "Thiếu thư viện 'gdown'. Vui lòng cài bằng: pip install gdown"
        ) from err

    dst_path = Path(dest_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    if filename:
        # --- Chế độ 1: Tải 1 file cụ thể ---
        print("Đang tải file từ Google Drive...")
        print(f"  URL : {folder_url}")
        print(f"  File: {filename}")
        print(f"  Đích: {dst_path / filename}")

        gdown.download_folder(
            url=folder_url,
            output=str(dst_path),
            quiet=False,
            use_cookies=False,
        )

        # gdown.download_folder có thể tạo thư mục con bên trong dst_path,
        # nên cần tìm file theo cả subdirectory
        result_file = _find_file_recursive(dst_path, filename)
        if result_file is not None:
            # Nếu file nằm trong subdirectory, move lên dst_path
            if result_file.parent != dst_path:
                result_file.rename(dst_path / filename)
                print(f"✅ File đã được tải về: {dst_path / filename}")
            else:
                print(f"✅ File đã được tải về: {result_file}")
        else:
            print(f"⚠️  Không tìm thấy file {filename} trong {dst_path}")
            print("Kiểm tra lại folder_url hoặc tên file trong folder.")
    else:
        # --- Chế độ 2: Tải toàn bộ folder ---
        print("Đang tải folder từ Google Drive...")
        print(f"  URL : {folder_url}")
        print(f"  Đích: {dst_path}")

        gdown.download_folder(
            url=folder_url,
            output=str(dst_path),
            quiet=False,
            use_cookies=False,
        )

        # Kiểm tra folder đã có file chưa
        files = list(dst_path.iterdir())
        if files:
            print(f"✅ Folder đã được tải về: {dst_path}")
            print(f"   Số file: {len(files)}")
        else:
            print(f"⚠️  Folder {dst_path} trống sau khi tải.")
            print("Kiểm tra lại folder_url.")
