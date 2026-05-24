import os
from huggingface_hub import snapshot_download
from pathlib import Path

# --- Configuration ---
HF_REPO_ID = "Phonsiri/astrology-raw-pdfs"
LOCAL_DIR = "input"
CACHE_DIR = "hf_cache" # ใช้โฟลเดอร์สั้นๆ เพื่อเลี่ยง File name too long
HF_TOKEN = os.environ.get("HF_TOKEN")

def download_raw_pdfs():
    if not HF_TOKEN:
        print("❌ Error: HF_TOKEN environment variable is not set.")
        return

    # เปิดใช้งาน hf_transfer เพื่อความเร็ว (ถ้าติดตั้งไว้)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print(f"🚀 Starting download from {HF_REPO_ID} to '{LOCAL_DIR}'...")
    
    try:
        # ใช้ snapshot_download พร้อมกำหนด cache_dir ให้สั้นลง
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            token=HF_TOKEN,
            local_dir_use_symlinks=False,
            cache_dir=CACHE_DIR
        )
        print(f"✅ Download complete! Files are in '{LOCAL_DIR}'")
    except Exception as e:
        print(f"❌ Error during download: {e}")

if __name__ == "__main__":
    download_raw_pdfs()
