from datasets import Dataset, Image, Features, Value
import os
import json
from pathlib import Path

# --- Hugging Face Configuration ---
DATA_DIR = "output_data"
HF_REPO_ID = "Phonsiri/astrology-dataset" # ตั้งเป้าหมายไปที่ User: Phonsiri
# แนะนำให้ใช้ Token ผ่าน Environment Variable: export HF_TOKEN="your_token"
HF_TOKEN = os.environ.get("HF_TOKEN") 

def push_to_huggingface():
    all_data = []
    data_path = Path(DATA_DIR)
    
    # รวบรวมข้อมูลจาก JSONL ทั้งหมด
    print("📦 Gathering data from JSONL files...")
    for jsonl_file in data_path.rglob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    all_data.append({
                        "source": entry["source"],
                        "page": entry["page"],
                        "image": entry.get("image_path"),
                        "text": entry.get("text", ""),
                        "caption": entry.get("caption", "")
                    })
                except: continue

    if not all_data:
        print("❌ No data found to push.")
        return

    # กำหนดโครงสร้าง Dataset (5 คอลัมน์)
    features = Features({
        "source": Value("string"),
        "page": Value("int32"),
        "image": Image(),
        "text": Value("string"),
        "caption": Value("string")
    })

    print(f"🚀 Creating Dataset with {len(all_data)} samples...")
    dataset = Dataset.from_list(all_data, features=features)
    
    # Push ขึ้น Hugging Face (ทำเป็น Batch อัตโนมัติโดยไลบรารี datasets)
    print(f"📤 Pushing to Hugging Face: {HF_REPO_ID}...")
    dataset.push_to_hub(HF_REPO_ID, token=HF_TOKEN, private=True) # ตั้งเป็น Private ไว้ก่อนเพื่อความปลอดภัย
    
    print("✅ Successfully pushed to Hugging Face!")

if __name__ == "__main__":
    if not HF_TOKEN:
        print("❌ Please set HF_TOKEN environment variable first.")
    else:
        push_to_huggingface()
