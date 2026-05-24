import os
import json
import base64
import time
from pathlib import Path
from pdf2image import convert_from_path
from openai import OpenAI
from datasets import Dataset, Image, Features, Value

# --- Cloud Configuration ---
MODEL_PATH = "google/gemma-4-31b-it" 
INPUT_DIR = "input"
OUTPUT_DIR = "output_data"
TEMP_IMAGE_DIR = "temp_pages"

# --- Hugging Face Configuration ---
HF_REPO_ID = "Phonsiri/astrology-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN")

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def push_current_data_to_hf():
    """รวบรวมข้อมูลทั้งหมดที่มีตอนนี้แล้วพุชขึ้น HF (เป็นแบต)"""
    if not HF_TOKEN:
        print("  ⚠️ HF_TOKEN not set, skipping auto-push.")
        return

    all_data = []
    data_path = Path(OUTPUT_DIR)
    
    print(f"  📤 Preparing to sync all processed data to HF...")
    for jsonl_file in data_path.rglob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    all_data.append({
                        "image": entry["image_path"],
                        "text": entry["content"],
                        "source": entry["source"],
                        "page": entry["page"]
                    })
                except: continue

    if not all_data: return

    features = Features({
        "image": Image(),
        "text": Value("string"),
        "source": Value("string"),
        "page": Value("int32")
    })

    dataset = Dataset.from_list(all_data, features=features)
    dataset.push_to_hub(HF_REPO_ID, token=HF_TOKEN, private=True)
    print(f"  ✅ Auto-synced {len(all_data)} samples to Hugging Face!")

def process_page_vllm(image_path, page_num):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""ถอดข้อมูลจากภาพหน้าโหราศาสตร์หน้าที {page_num}:
    1. ข้อความภาษาไทยทั้งหมด (Markdown)
    2. อธิบายภาพดวงชาตา/ตารางดาว อย่างละเอียด
    3. ตอบเฉพาะเนื้อหาเท่านั้น"""

    try:
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  ❌ Error on page {page_num}: {e}")
        return None

def main():
    Path(INPUT_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(TEMP_IMAGE_DIR).mkdir(exist_ok=True)

    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))

    for pdf_path in pdf_files:
        clean_name = pdf_path.stem.replace(' ', '_')
        pdf_output_dir = Path(OUTPUT_DIR) / clean_name
        images_dir = pdf_output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        jsonl_path = pdf_output_dir / f"{clean_name}.jsonl"
        
        last_page = 0
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        last_page = max(last_page, entry.get("page", 0))
                    except: pass

        print(f"\n--- Processing: {pdf_path.name} ---")
        pages = convert_from_path(pdf_path, dpi=150)
        
        new_pages_processed = 0
        with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for i, page in enumerate(pages):
                page_num = i + 1
                if page_num <= last_page: continue
                
                img_name = f"page_{page_num:03d}.jpg"
                img_path = images_dir / img_name
                page.save(img_path, "JPEG", quality=85)
                
                content = process_page_vllm(img_path, page_num)
                
                if content:
                    data_row = {
                        "source": pdf_path.name,
                        "page": page_num,
                        "image_path": str(img_path),
                        "content": content
                    }
                    f_jsonl.write(json.dumps(data_row, ensure_ascii=False) + "\n")
                    f_jsonl.flush()
                    new_pages_processed += 1
                    print(f"  ✅ Page {page_num} Saved Local.")

        # จบ 1 ไฟล์ PDF (1 แบต) ทำการพุชขึ้น HF ทันที
        if new_pages_processed > 0:
            push_current_data_to_hf()

if __name__ == "__main__":
    main()
