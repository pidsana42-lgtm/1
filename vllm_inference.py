import os
import json
import torch
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoProcessor
from datasets import Dataset, Features, Value, Image as HFImage, load_dataset
from vllm import LLM, SamplingParams


# --- Configuration ---
MODEL_ID = "google/gemma-4-31b-it"
INPUT_DIR = "input"
OUTPUT_DIR = "output_data"
TEMP_IMAGE_DIR = "temp_pages"
BATCH_SIZE = 6  # จำนวนหน้าที่ประมวลผลพร้อมกัน (ปรับตาม VRAM)

# --- Hugging Face Configuration ---
HF_REPO_ID = "Phonsiri/astrology-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"🔄 Loading model {MODEL_ID} via vLLM...")
llm = LLM(
    model=MODEL_ID,
    max_model_len=8192,
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
    hf_overrides={
        "vision_config": {"default_output_length": 560},
        "vision_soft_tokens_per_image": 560
    },
    mm_processor_kwargs={"max_soft_tokens": 560}
)
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
print("✅ Model loaded via vLLM!")


def make_prompt(task, page_num):
    if task == "ocr":
        return (
            f"ถอดข้อความทั้งหมดจากภาพหน้าโหราศาสตร์หน้าที่ {page_num} "
            f"ให้ครบถ้วนทุกตัวอักษร ทั้งภาษาไทยและภาษาอื่นๆ "
            f"จัดรูปแบบเป็น Markdown ตอบเฉพาะข้อความเท่านั้น ห้ามอธิบายเพิ่ม"
        )
    else:  # caption
        return (
            f"บรรยายภาพในหน้าโหราศาสตร์หน้าที่ {page_num} อย่างละเอียด "
            f"เช่น ดวงชาตา ตารางดาว สัญลักษณ์ทางโหราศาสตร์ ตาราง หรือภาพประกอบใดๆ "
            f"อธิบายตำแหน่ง สี รูปร่าง และความหมายที่มองเห็น ตอบเป็นภาษาไทย"
        )


def run_batch(image_prompt_pairs):
    """
    รัน inference พร้อมกัน BATCH_SIZE หน้าโดยใช้ vLLM Offline Mode
    image_prompt_pairs: list of (PIL.Image, prompt_str)
    คืนค่า list of str
    """
    vllm_inputs = []

    for image, prompt in image_prompt_pairs:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        vllm_inputs.append({
            "prompt": text,
            "multi_modal_data": {"image": image}
        })

    # ตั้งค่า sampling parameters สำหรับ OCR/Caption
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048
    )

    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

    results = []
    for output in outputs:
        results.append(output.outputs[0].text.strip())

    return results


def process_batch_pages(batch_info):
    """
    batch_info: list of (image_path, page_num)
    คืนค่า list of {"text": ..., "caption": ...}
    """
    images = [Image.open(p).convert("RGB") for p, _ in batch_info]
    page_nums = [n for _, n in batch_info]

    # --- Batch OCR ---
    ocr_pairs = [(img, make_prompt("ocr", num)) for img, num in zip(images, page_nums)]
    ocr_results = run_batch(ocr_pairs)

    # --- Batch Caption ---
    cap_pairs = [(img, make_prompt("caption", num)) for img, num in zip(images, page_nums)]
    cap_results = run_batch(cap_pairs)

    return [
        {"text": text, "caption": cap}
        for text, cap in zip(ocr_results, cap_results)
    ]


def push_to_hf():
    if not HF_TOKEN:
        print("  ⚠️ HF_TOKEN not set, skipping push.")
        return

    all_data = []
    for jsonl_file in Path(OUTPUT_DIR).rglob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    img_path = entry.get("image_path")
                    img_obj = None
                    if img_path and os.path.exists(img_path):
                        try:
                            img_obj = Image.open(img_path).convert("RGB")
                        except Exception as e:
                            print(f"    ⚠️ Warning: Could not open image {img_path}: {e}")
                    all_data.append({
                        "source": entry["source"],
                        "page": entry["page"],
                        "image": img_obj,
                        "text": entry.get("text", ""),
                        "caption": entry.get("caption", ""),
                    })
                except:
                    continue

    if not all_data:
        print("  ⚠️ No data to push.")
        return

    features = Features({
        "source": Value("string"),
        "page": Value("int32"),
        "image": HFImage(),
        "text": Value("string"),
        "caption": Value("string"),
    })

    dataset = Dataset.from_list(all_data, features=features)
    dataset.push_to_hub(HF_REPO_ID, token=HF_TOKEN, private=True)
    print(f"  ✅ Pushed {len(all_data)} samples to {HF_REPO_ID}")


def sync_from_hf():
    if not HF_TOKEN:
        print("  ℹ️ HF_TOKEN not set, skipping HF sync.")
        return

    try:
        print(f"📥 Checking Hugging Face Dataset '{HF_REPO_ID}' to sync progress...")
        dataset = load_dataset(HF_REPO_ID, split="train", token=HF_TOKEN)
        print(f"✅ Found {len(dataset)} existing records on Hugging Face.")
        
        synced_count = 0
        for row in dataset:
            source = row["source"]
            page = int(row["page"])
            clean_name = Path(source).stem.replace(" ", "_")
            
            pdf_output_dir = Path(OUTPUT_DIR) / clean_name
            images_dir = pdf_output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = pdf_output_dir / f"{clean_name}.jsonl"
            
            existing_pages = set()
            if jsonl_path.exists():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            existing_pages.add(json.loads(line).get("page", 0))
                        except:
                            pass
            
            if page not in existing_pages:
                img_path = images_dir / f"page_{page:03d}.jpg"
                if row.get("image") is not None:
                    try:
                        row["image"].save(img_path, "JPEG", quality=85)
                    except Exception as e:
                        print(f"    ⚠️ Warning: Could not save image for page {page}: {e}")
                
                local_row = {
                    "source": source,
                    "page": page,
                    "image_path": str(img_path),
                    "text": row.get("text", ""),
                    "caption": row.get("caption", ""),
                }
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(local_row, ensure_ascii=False) + "\n")
                synced_count += 1
                
        if synced_count > 0:
            print(f"🎉 Synced {synced_count} records and images from Hugging Face successfully!")
        else:
            print("👍 Local data is already fully synced with Hugging Face.")
            
    except Exception as e:
        print(f"ℹ️ Hugging Face dataset sync skipped or not found (It's okay if this is the first run): {e}")


def main():
    Path(INPUT_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(TEMP_IMAGE_DIR).mkdir(exist_ok=True)

    # Sync กับ HF ก่อนเพื่อทำต่อแบบไร้รอยต่อ
    sync_from_hf()

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

        # Resume
        last_page = 0
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        last_page = max(last_page, json.loads(line).get("page", 0))
                    except:
                        pass
            if last_page > 0:
                print(f"  ▶️  Resuming from page {last_page + 1}")

        print(f"\n--- Processing: {pdf_path.name} ---")
        pages = convert_from_path(pdf_path, dpi=150)
        total = len(pages)
        new_pages = 0

        # บันทึกภาพทุกหน้าก่อน
        pending = []
        for i, page in enumerate(pages):
            page_num = i + 1
            if page_num <= last_page:
                continue
            img_path = images_dir / f"page_{page_num:03d}.jpg"
            page.save(img_path, "JPEG", quality=85)
            pending.append((img_path, page_num))

        # ประมวลผลเป็น batch
        with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for b_start in range(0, len(pending), BATCH_SIZE):
                batch = pending[b_start:b_start + BATCH_SIZE]
                page_range = f"{batch[0][1]}–{batch[-1][1]}"
                print(f"  🔍 Batch pages {page_range}/{total} (size={len(batch)})...", flush=True)

                try:
                    results = process_batch_pages(batch)
                    for (img_path, page_num), result in zip(batch, results):
                        row = {
                            "source": pdf_path.name,
                            "page": page_num,
                            "image_path": str(img_path),
                            "text": result["text"],
                            "caption": result["caption"],
                        }
                        f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f_jsonl.flush()
                        new_pages += 1
                    print(f"  ✅ Saved pages {page_range}")
                except Exception as e:
                    print(f"  ❌ Batch error: {e}")
                    # fallback ทีละหน้า
                    for img_path, page_num in batch:
                        try:
                            r = process_batch_pages([(img_path, page_num)])
                            result = r[0]
                            row = {
                                "source": pdf_path.name,
                                "page": page_num,
                                "image_path": str(img_path),
                                "text": result["text"],
                                "caption": result["caption"],
                            }
                            f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                            f_jsonl.flush()
                            new_pages += 1
                            print(f"    ✅ Page {page_num} (fallback)")
                        except Exception as e2:
                            print(f"    ❌ Page {page_num} failed: {e2}")

                # Push ขึ้น HF ทุก batch
                print(f"  📤 Pushing batch to HF...")
                push_to_hf()

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
