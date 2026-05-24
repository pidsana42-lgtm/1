import os
from pathlib import Path
from huggingface_hub import HfApi

# --- Configuration ---
LOCAL_HF_DIR = "hf"
HF_REPO_ID = "Phonsiri/astrology-raw-pdfs" # Repository สำหรับเก็บไฟล์ PDF ต้นฉบับ
HF_TOKEN = os.environ.get("HF_TOKEN")

def push_raw_pdfs():
    if not HF_TOKEN:
        print("❌ Please set HF_TOKEN environment variable.")
        return

    api = HfApi()
    
    # สร้าง Repo หากยังไม่มี (เป็น Dataset Repo เพื่อเก็บไฟล์)
    print(f"🚀 Creating/Checking Repository: {HF_REPO_ID}")
    api.create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, repo_type="dataset", exist_ok=True)

    print(f"📦 Uploading PDFs from '{LOCAL_HF_DIR}' to Hugging Face...")
    
    # อัปโหลดทั้งโฟลเดอร์ hf ไปที่ root ของ repo
    api.upload_folder(
        folder_path=LOCAL_HF_DIR,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    
    print(f"✅ Successfully uploaded all PDFs to https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    push_raw_pdfs()
