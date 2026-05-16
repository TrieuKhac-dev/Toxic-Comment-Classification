"""
Script xác thực Google Drive OAuth cho DVC.

Yêu cầu:
1. Tạo OAuth 2.0 Client ID (Desktop app) trên Google Cloud Console
2. Download file JSON, đặt tên là client_secrets.json trong thư mục scripts/
3. Chạy script này
"""

from pathlib import Path

from pydrive2.auth import GoogleAuth

# Dùng file client_secrets.json vừa tạo
secrets_path = Path(__file__).parent / "client_secrets.json"
if not secrets_path.exists():
    print(f"ERROR: Không tìm thấy {secrets_path}")
    print("Vui lòng tải OAuth Client ID JSON từ Google Cloud Console")
    print("và đặt vào scripts/client_secrets.json")
    exit(1)

gauth = GoogleAuth()
gauth.LoadClientConfigFile(str(secrets_path))
gauth.LocalWebserverAuth()
print("✅ Google Drive OAuth authentication successful!")
