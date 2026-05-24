import os
from huggingface_hub import snapshot_download
from pathlib import Path

# --- Configuration ---
HF_REPO_ID = "Phonsiri/astrology-raw-pdfs"
LOCAL_DIR = "input"
HF_TOKEN = os.environ.get("HF_TOKEN")

def download_raw_pdfs():
    if not HF_TOKEN:
        print("❌ Error: HF_TOKEN environment variable is not set.")
        return

    print(f"🚀 Starting download from {HF_REPO_ID} to '{LOCAL_DIR}'...")
    
    try:
        # ใช้ snapshot_download เพื่อโหลดไฟล์ทั้งหมดใน Repo
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            token=HF_TOKEN,
            local_dir_use_symlinks=False
        )
        print(f"✅ Download complete! Files are in '{LOCAL_DIR}'")
    except Exception as e:
        print(f"❌ Error during download: {e}")

if __name__ == "__main__":
    download_raw_pdfs()
