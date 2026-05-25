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

    # Auto-upload/update dataset card (README.md) on HF
    try:
        import io
        from huggingface_hub import HfApi
        api = HfApi()
        card_content = f"""---
pretty_name: Astrology PDF OCR & Captions Dataset
task_categories:
- image-to-text
- text-generation
language:
- th
- en
tags:
- astrology
- ocr
- gemma3
- pdf-parsing
license: apache-2.0
---

# Astrology PDF OCR & Captions Dataset

This dataset contains OCR text and image descriptions (captions) extracted from Thai Astrology PDF documents. The extraction is performed using the **Gemma 3 27B** multimodal model.

## Dataset Structure

The dataset contains the following columns:

| Column | Type | Description |
|---|---|---|
| `source` | string | The name of the source PDF document. |
| `page` | int32 | The page number in the source document. |
| `image` | image | The original page image (rendered from PDF). |
| `text` | string | Full OCR text extracted from the page (formatted in Markdown). |
| `caption` | string | Detailed description of any charts, tables, or astrological diagrams in the page. |

## Creation Process

1. **PDF Rendering:** Source PDFs are rendered to JPEG images at 150 DPI.
2. **Double-Inference Pipeline:**
   - **OCR:** Gemma 3 27B is prompted to extract all text exactly as shown.
   - **Captioning:** Gemma 3 27B is prompted to describe charts/diagrams/tables in detail.
3. **Execution Engine:** Powered by `vLLM` offline inference.

*Developed for Advanced Astrology AI Dataset Construction.*
"""
        api.upload_file(
            path_or_fileobj=io.BytesIO(card_content.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print("✅ Dataset card updated on Hugging Face!")
    except Exception as e:
        print(f"⚠️ Warning: Failed to update dataset card on HF: {e}")

if __name__ == "__main__":
    if not HF_TOKEN:
        print("❌ Please set HF_TOKEN environment variable first.")
    else:
        push_to_huggingface()
