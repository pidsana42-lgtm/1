import os
import json
import base64
import time
from pathlib import Path
from pdf2image import convert_from_path
from openai import OpenAI

# --- Cloud Configuration ---
# แนะนำให้ตั้ง MODEL_PATH เป็นชื่อรุ่นใน HF หรือ Path ที่โหลดโมเดลมาไว้ใน GPU
MODEL_PATH = "google/gemma-4-31b-it" 
INPUT_DIR = "input"
OUTPUT_DIR = "output_data"
TEMP_IMAGE_DIR = "temp_pages"

# เชื่อมต่อกับ vLLM Local Server (H100)
# รัน vllm serve รอไว้ก่อนรันสคริปต์นี้
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def process_page_vllm(image_path, page_num):
    """ส่งภาพหน้าหนังสือให้ vLLM (Gemma-4) ประมวลผล"""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""คุณคือผู้เชี่ยวชาญด้านโหราศาสตร์ไทย 
    นี่คือหน้าหนังสือโหราศาสตร์หน้าที {page_num}
    งานของคุณคือ:
    1. ถอดข้อความภาษาไทยในภาพนี้ออกมาให้แม่นยำที่สุดในรูปแบบ Markdown
    2. หากเจอ "ภาพดวงชาตา" ให้เขียนคำบรรยายรายละเอียดดาวในภพต่างๆ ลงมาด้วย
    3. ไม่ต้องมีคำเกริ่นนำ ให้ส่งกลับมาเฉพาะเนื้อหา Markdown เท่านั้น"""

    try:
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.1, # ใช้ Low temp เพื่อความแม่นยำของเนื้อหา
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
        
        # Resume Logic: เช็คว่าทำค้างไว้หน้าไหน
        last_page = 0
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        last_page = max(last_page, entry.get("page", 0))
                    except: pass

        print(f"\n--- Processing: {pdf_path.name} (Start from Page {last_page + 1}) ---")
        
        # แปลง PDF เป็นภาพ (150 DPI เหมาะกับ H100)
        pages = convert_from_path(pdf_path, dpi=150)
        
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
                        "content": content,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    f_jsonl.write(json.dumps(data_row, ensure_ascii=False) + "\n")
                    f_jsonl.flush() # เซฟเก็บแบตทีละหน้า
                    print(f"  ✅ Page {page_num} Processed and Saved.")

if __name__ == "__main__":
    main()
