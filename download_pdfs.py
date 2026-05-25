import os
import re
import requests
from huggingface_hub import HfApi
from pathlib import Path

# --- Configuration ---
HF_REPO_ID = "Phonsiri/astrology-raw-pdfs"
LOCAL_DIR = "input"
HF_TOKEN = os.environ.get("HF_TOKEN")

def clean_filename(filename):
    # ดึงเฉพาะชื่อไฟล์
    base = os.path.basename(filename)
    
    # อนุญาตเฉพาะภาษาไทย, อังกฤษ, ตัวเลข, จุด, ขีดกลาง, และ Underscore
    # ช่วงภาษาไทยคือ \u0e00-\u0e7f
    cleaned = re.sub(r'[^\w\s\.\-\u0e00-\u0e7f]', '', base)
    
    # เปลี่ยนช่องว่างเป็น Underscore
    cleaned = re.sub(r'\s+', '_', cleaned)
    
    # จำกัดความยาวไม่เกิน 60 ตัวอักษรก่อนนามสกุลไฟล์เพื่อไม่ให้ชื่อยาวเกินไป
    name, ext = os.path.splitext(cleaned)
    if len(name) > 60:
        name = name[:60]
        
    return name + ext

def download_raw_pdfs():
    if not HF_TOKEN:
        print("❌ Error: HF_TOKEN environment variable is not set.")
        return

    Path(LOCAL_DIR).mkdir(exist_ok=True)
    api = HfApi()
    
    print(f"📦 Fetching list of files from HF Repo: {HF_REPO_ID}...")
    try:
        all_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
    except Exception as e:
        print(f"❌ Failed to fetch file list: {e}")
        return

    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("ℹ️ No PDF files found in the repository.")
        return

    print(f"🚀 Found {len(pdf_files)} PDF files. Starting download...")
    
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {HF_TOKEN}"})

    success_count = 0
    for idx, filename in enumerate(pdf_files, 1):
        safe_name = clean_filename(filename)
        dest_path = Path(LOCAL_DIR) / safe_name
        
        # เลี่ยงการดาวน์โหลดซ้ำถ้าไฟล์มีอยู่แล้ว
        if dest_path.exists():
            print(f"  [{idx}/{len(pdf_files)}] Skip: {safe_name} (already exists)")
            success_count += 1
            continue

        url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{filename}"
        print(f"  [{idx}/{len(pdf_files)}] Downloading: {safe_name}...", end="", flush=True)
        
        try:
            response = session.get(url, stream=True)
            response.raise_for_status()
            
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(" Done! ✅")
            success_count += 1
        except Exception as e:
            print(f" Failed! ❌ ({e})")

    print(f"\n🎉 Completed! Successfully downloaded {success_count}/{len(pdf_files)} files to '{LOCAL_DIR}'")

if __name__ == "__main__":
    download_raw_pdfs()
