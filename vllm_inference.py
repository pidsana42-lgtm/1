import os
import json
import torch
import time
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from datasets import Dataset, Features, Value

# --- Configuration ---
MODEL_ID = "google/gemma-4-31b-it"
INPUT_DIR = "input"
OUTPUT_DIR = "output_data"
TEMP_IMAGE_DIR = "temp_pages"

# --- Hugging Face Configuration ---
HF_REPO_ID = "Phonsiri/astrology-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"🔄 Loading model {MODEL_ID}...")
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN
)
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
print("✅ Model loaded!")

def process_page(image_path, page_num):
    image = Image.open(image_path).convert("RGB")

    prompt = f"""ถอดข้อมูลจากภาพหน้าโหราศาสตร์หน้าที่ {page_num}:
1. ข้อความภาษาไทยทั้งหมด (Markdown)
2. อธิบายภาพดวงชาตา/ตารางดาว อย่างละเอียด
3. ตอบเฉพาะเนื้อหาเท่านั้น"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )

        response = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()

    except Exception as e:
        print(f"  ❌ Error on page {page_num}: {e}")
        return None

def push_to_hf():
    if not HF_TOKEN:
        print("  ⚠️ HF_TOKEN not set, skipping push.")
        return

    all_data = []
    data_path = Path(OUTPUT_DIR)

    for jsonl_file in data_path.rglob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    all_data.append({
                        "text": entry["content"],
                        "source": entry["source"],
                        "page": entry["page"],
                    })
                except:
                    continue

    if not all_data:
        print("  ⚠️ No data to push.")
        return

    features = Features({
        "text": Value("string"),
        "source": Value("string"),
        "page": Value("int32"),
    })

    dataset = Dataset.from_list(all_data, features=features)
    dataset.push_to_hub(HF_REPO_ID, token=HF_TOKEN, private=True)
    print(f"  ✅ Pushed {len(all_data)} samples to {HF_REPO_ID}")

def main():
    Path(INPUT_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(TEMP_IMAGE_DIR).mkdir(exist_ok=True)

    pdf_files = sorted(Path(INPUT_DIR).glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in input/")
        return

    for pdf_path in pdf_files:
        clean_name = pdf_path.stem.replace(" ", "_")
        pdf_output_dir = Path(OUTPUT_DIR) / clean_name
        images_dir = pdf_output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = pdf_output_dir / f"{clean_name}.jsonl"

        # Resume: หาหน้าสุดท้ายที่ทำไปแล้ว
        last_page = 0
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        last_page = max(last_page, entry.get("page", 0))
                    except:
                        pass
            if last_page > 0:
                print(f"  ▶️  Resuming from page {last_page + 1}")

        print(f"\n--- Processing: {pdf_path.name} ---")
        pages = convert_from_path(pdf_path, dpi=150)
        total = len(pages)
        new_pages = 0

        with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for i, page in enumerate(pages):
                page_num = i + 1
                if page_num <= last_page:
                    continue

                img_path = images_dir / f"page_{page_num:03d}.jpg"
                page.save(img_path, "JPEG", quality=85)

                print(f"  🔍 Page {page_num}/{total}...", end=" ", flush=True)
                content = process_page(img_path, page_num)

                if content:
                    row = {
                        "source": pdf_path.name,
                        "page": page_num,
                        "image_path": str(img_path),
                        "content": content,
                    }
                    f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f_jsonl.flush()
                    new_pages += 1
                    print(f"✅ Saved")
                else:
                    print(f"❌ Skipped")

        if new_pages > 0:
            print(f"\n📤 Pushing {new_pages} new pages to HF...")
            push_to_hf()

    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
