import os
import re
import json
import sys
import argparse
from pathlib import Path
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Image as HFImage, load_dataset

# ตัวแปรโกลบอลสำหรับเก็บโมเดลเมื่อมีการเปิดใช้โหมด LLM
llm = None
processor = None

def clean_ocr_text(text):
    """
    ทำความสะอาดข้อความดิบจาก OCR โดยใช้ Regex กรอง Noise พื้นฐาน
    """
    if not text:
        return ""
    
    # 1. จัดระเบียบการเว้นวรรคและบรรทัดใหม่
    text = text.replace("\r\n", "\n")
    
    # 2. แก้ปัญหาช่องว่างแทรกระหว่างสระ/วรรณยุกต์ไทย (เช่น 'ท ี่' -> 'ที่', 'ก า ร' -> 'การ')
    text = re.sub(r"\s([ะาิีึืุูั็่้๊๋ํ์ๅ])", r"\1", text)
    
    # 3. ลบเครื่องหมายประหลาดขยะรอบๆ ข้อความ (ยกเว้น markdown formatting เช่น * หรือ #)
    text = re.sub(r"(?<![#*])([|\\\/_~])(?![#*])", " ", text)
    
    # 4. ลบช่องว่างส่วนเกิน
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    return text.strip()

def clean_conversational_noise(text):
    """
    ลบคำพูดทักทายโต้ตอบของ AI (เช่น "ภาพที่คุณส่งมาคือ...", "ครับ", "ค่ะ")
    """
    if not text:
        return ""
    
    patterns_to_remove = [
        r"^ภาพที่ปรากฏคือ[^:\n]+โดยมีรายละเอียดดังนี้(ครับ|ค่ะ)?:?\s*",
        r"^ภาพที่คุณส่งมา[^:\n]+โดยมีรายละเอียดดังนี้(ครับ|ค่ะ)?:?\s*",
        r"^รูปภาพที่คุณส่งมา[^:\n]+โดยมีรายละเอียดดังนี้(ครับ|ค่ะ)?:?\s*",
        r"^ภาพนี้แสดงถึง[^:\n]+โดยมีรายละเอียดดังนี้(ครับ|ค่ะ)?:?\s*",
        r"^รายละเอียดของภาพที่ปรากฏมีดังนี้(ครับ|ค่ะ)?:?\s*",
        r"^ในภาพนี้ไม่มี[^:\n]+มีเพียงรายละเอียดดังนี้(ครับ|ค่ะ)?:?\s*",
        r"^รูปภาพนี้คือ[^:\n]+ซึ่งมีลักษณะดังนี้(ครับ|ค่ะ)?:?\s*",
        r"^รายละเอียดของภาพ:\s*",
        r"^โดยมีรายละเอียดดังนี้(ครับ|ค่ะ)?:?\s*",
        r"^มีรายละเอียดดังนี้(ครับ|ค่ะ)?:?\s*"
    ]
    
    cleaned = text
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
    cleaned = re.sub(r"\s*นะครับ\s*$", "", cleaned)
    cleaned = re.sub(r"\s*นะค่ะ\s*$", "", cleaned)
    cleaned = re.sub(r"\s*ค่ะ\s*$", "", cleaned)
    cleaned = re.sub(r"\s*ครับ\s*$", "", cleaned)
    cleaned = re.sub(r"\s*ครับผม\s*$", "", cleaned)
    cleaned = re.sub(r"ตรงกึ่งกลางหน้าครับ", "ตรงกึ่งกลางหน้า", cleaned)
    cleaned = re.sub(r"ปรากฏอยู่กลางหน้าครับ", "ปรากฏอยู่กลางหน้า", cleaned)
    cleaned = re.sub(r"ปรากฏอยู่กลางหน้าค่ะ", "ปรากฏอยู่กลางหน้า", cleaned)
    cleaned = re.sub(r"รายละเอียดดังนี้ครับ", "รายละเอียดดังนี้", cleaned)
    
    return cleaned.strip()

def should_skip_page(text, caption):
    """
    ตรวจสอบหน้าขยะ หน้าว่าง ตราประทับเปล่าๆ หรือผลปฏิเสธจาก AI
    """
    if not text and not caption:
        return True
        
    clean_txt = text.replace(" ", "").replace("\n", "")
    if clean_txt in [
        "หอสมุดแห่งชาติจังหวัดสุพรรณบุรีเฉลิมพระเกียรติ",
        "หอสมุดแห่งชาติ",
        "จังหวัดสุพรรณบุรีเฉลิมพระเกียรติ",
        "หอสมุดแห่งชาติจังหวัดสุพรรณบุรีเฉลิมพระเกียรติพิมพ์ที่โรงพิมพ์ศิวาสรม"
    ]:
        return True
        
    refusal_keywords = [
        "ไม่พบข้อความใดๆ",
        "ไม่ใช่หน้าโหราศาสตร์",
        "ภาพนี้เป็นเพียง",
        "ไม่มีเนื้อหาใดๆ",
        "รบกวนช่วยถ่ายภาพ",
        "รบกวนส่งรูปภาพ",
        "ไม่ปรากฏดวงชะตา",
        "หน้ากระดาษเปล่า",
        "ไม่มีตัวอักษรปรากฏอยู่",
        "พื้นผิวสีดำ",
        "เป็นเพียงภาพพื้นผิว"
    ]
    
    for kw in refusal_keywords:
        if kw in text or kw in caption:
            return True
            
    if len(text.strip()) < 30:
        keywords = ["ดาว", "ดวง", "ราศี", "ลัคนา", "โหร", "พยากรณ์", "ทาย", "ชะตา", "นพเคราะห์", "ฤกษ์"]
        if not any(k in text for k in keywords) and not any(k in caption for k in keywords):
            return True
            
    return False

def init_llm(model_id, gpu_memory_utilization=0.80):
    """
    โหลดโมเดลผ่าน Transformers เพื่อใช้ประมวลผลข้อความภาษาไทย
    """
    global llm, processor
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ Error: ไม่พบไลบรารี torch หรือ transformers ในสภาพแวดล้อมนี้ กรุณาติดตั้งก่อนใช้งาน")
        sys.exit(1)
        
    print(f"🔄 กำลังโหลดโมเดล {model_id} ผ่าน Transformers สำหรับแก้ภาษา...")
    processor = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ โหลดโมเดล LLM สำหรับเรียงข้อความเสร็จสิ้น!")

def clean_text_with_llm(text_list):
    """
    ใช้ LLM ขัดเกลาคำผิด เรียงไวยากรณ์ ตัดขึ้นบรรทัดใหม่ และจำแนกหมวดหมู่สำหรับประมวลผลต่อ (แบบ Transformers)
    """
    if not llm or not text_list:
        return [{"text": t, "category": "อื่นๆ"} for t in text_list]
        
    import torch
    import re
    
    print(f"  ⚡ กำลังประมวลผลข้อความผ่านโมเดล LLM จำนวน {len(text_list)} หน้า...")
    
    cleaned_results = []
    for idx, text in enumerate(text_list):
        print(f"    📄 กำลังประมวลผลหน้า {idx+1}/{len(text_list)}...")
        messages = [
            {"role": "system", "content": (
                "คุณคือผู้เชี่ยวชาญการจัดชำระคัมภีร์และตำราโหราศาสตร์ไทยโบราณ\n"
                "หน้าที่ของคุณคือ:\n"
                "1. จัดจำแนกหมวดหมู่ของข้อความในหน้านี้ โดยเลือกจากรายการต่อไปนี้เท่านั้น:\n"
                "   - 'โหราศาสตร์ไทย/ดวงดาว' (ตารางดาว, แผนภาพจักรราศี, คำทำนายดาวจร/ดาวสถิต, ลัคนา)\n"
                "   - 'ทำนายฝัน' (การตีความฝันต่าง ๆ)\n"
                "   - 'ไพ่ยิปซี/ทาโรต์' (ความหมายและการวางไพ่ทาโรต์)\n"
                "   - 'ลายมือ' (เส้นลายมือและการพยากรณ์)\n"
                "   - 'ความรัก/เนื้อคู่' (การสมพงศ์ดวงชะตาคู่ครอง, ความรัก)\n"
                "   - 'ฤกษ์ยาม/นพเคราะห์' (การหาเวลาอันเป็นมงคล, พิธีกรรม)\n"
                "   - 'ความเชื่อ/ไสยศาสตร์' (คาถาอาคม, การบูชาสิ่งศักดิ์สิทธิ์)\n"
                "   - 'อื่นๆ' (หากไม่ตรงกับหมวดหมู่ใด ๆ)\n"
                "2. ทำความสะอาด แก้ไขคำสะกดผิด และจัดเรียงข้อความดิบจาก OCR ใหม่ตามลำดับการอ่านธรรมชาติและตามหลักไวยากรณ์ไทย\n"
                "   - เรียงลำดับเนื้อความใหม่กรณีคอลัมน์สลับหรือข้อความเรียงตัวอักษรผิดเพี้ยน\n"
                "   - ห้ามแต่งเติมข้อมูลใหม่ ห้ามดัดแปลงความหมายเดิม และห้ามเขียนข้อความอธิบายใดๆ เพิ่มเติม\n\n"
                "กรุณาตอบกลับในรูปแบบที่กำหนดไว้ด้านล่างนี้อย่างเคร่งครัด ห้ามพิมพ์คำอธิบายอื่นนอกเหนือจากแท็กเหล่านี้:\n"
                "[หมวดหมู่]: <ชื่อหมวดหมู่ที่เลือก>\n"
                "[ข้อความสะอาด]:\n"
                "<ข้อความที่ปรับปรุงเสร็จเรียบร้อยแล้ว>"
            )},
            {"role": "user", "content": f"ข้อความ OCR ดิบ:\n{text}"}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, return_tensors="pt").to(llm.device)
        
        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=False
            )
            
        # Extract response
        input_len = inputs.input_ids.shape[1]
        resp_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        
        # ดึงหมวดหมู่
        category_match = re.search(r"\[หมวดหมู่\]:\s*(.*?)(?:\n|$)", resp_text)
        category = category_match.group(1).strip() if category_match else "อื่นๆ"
        category = re.sub(r"['\"`<>\[\]]", "", category)
        
        # ดึงข้อความสะอาด
        text_match = re.search(r"\[ข้อความสะอาด\]:\s*(.*)", resp_text, re.DOTALL)
        cleaned_text = text_match.group(1).strip() if text_match else resp_text
        cleaned_text = re.sub(r"^\[ข้อความสะอาด\]:\s*", "", cleaned_text)
        
        cleaned_results.append({
            "text": cleaned_text,
            "category": category
        })
        
    return cleaned_results

def recursive_chunk_text(text, chunk_size=1500, chunk_overlap=200):
    """
    สไลซ์ข้อความขนาดยาวออกเป็นท่อนๆ (Chunks) สำหรับ RAG
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        if end < text_len:
            paragraph_boundary = text.rfind("\n\n", start, end)
            if paragraph_boundary != -1 and paragraph_boundary > start + (chunk_size // 2):
                end = paragraph_boundary + 2
            else:
                line_boundary = text.rfind("\n", start, end)
                if line_boundary != -1 and line_boundary > start + (chunk_size // 2):
                    end = line_boundary + 1
                else:
                    word_boundary = text.rfind(" ", start, end)
                    if word_boundary != -1 and word_boundary > start + (chunk_size // 2):
                        end = word_boundary + 1
                        
        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append(chunk_content)
            
        start = end - chunk_overlap if end < text_len else end
        if start >= text_len or end == text_len:
            break
            
    return chunks

def push_progress_to_hf(cleaned_rows, repo_id, hf_token):
    """
    พุชข้อมูลที่คลีนเสร็จแล้วขึ้น Hugging Face Dataset (Private) เพื่อเซฟประวัติความคืบหน้า
    """
    try:
        print(f"📤 กำลังอัปเดตความคืบหน้าขึ้น Hugging Face: '{repo_id}'...")
        
        all_data = []
        for row in cleaned_rows:
            all_data.append({
                "source": row["source"],
                "page": row["page"],
                "image": row["image"],
                "text": row["text"],
                "caption": row["caption"],
                "category": row.get("category", "อื่นๆ")
            })
            
        features = Features({
            "source": Value("string"),
            "page": Value("int32"),
            "image": HFImage(),
            "text": Value("string"),
            "caption": Value("string"),
            "category": Value("string")
        })
        
        dataset = Dataset.from_list(all_data, features=features)
        dataset.push_to_hub(repo_id, token=hf_token, private=True)
        print(f"✅ อัปเดตความคืบหน้าขึ้น Hugging Face สำเร็จ! (รวม {len(all_data)} หน้า)")
    except Exception as e:
        print(f"⚠️ ไม่สามารถพุชขึ้น Hugging Face ได้ชั่วคราว: {e}")

def process_dataset(input_dir, output_dir, chunk_size, chunk_overlap, use_llm, llm_model, 
                    hf_input_repo, hf_output_repo, batch_size, gpu_memory_utilization):
    """
    ฟังก์ชันแกนหลักของการรวบรวม ทำความสะอาด และบันทึกผลลัพธ์
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cpt_jsonl_file = output_path / "cpt_output.jsonl"
    cpt_txt_file = output_path / "cpt_corpus.txt"
    rag_jsonl_file = output_path / "rag_output.jsonl"
    sft_json_file = output_path / "sft_multimodal.json"
    
    hf_token = os.environ.get("HF_TOKEN")
    
    # 1. โหลดข้อมูลดิบ (ดึงจาก HF ก่อน ถ้าไม่ได้ให้ใช้ข้อมูลโลคัล)
    raw_pages = []
    if hf_input_repo and hf_token:
        try:
            print(f"📥 กำลังโหลด Dataset ดิบจาก Hugging Face: '{hf_input_repo}'...")
            dataset = load_dataset(hf_input_repo, split="train", token=hf_token)
            print(f"✅ โหลดสำเร็จ! พบข้อมูลทั้งหมด {len(dataset)} รายการ")
            for row in dataset:
                raw_pages.append({
                    "source": row.get("source", ""),
                    "page": int(row.get("page", 0)),
                    "image": row.get("image"),
                    "text": row.get("text", "") or "",
                    "caption": row.get("caption", "") or ""
                })
        except Exception as e:
            print(f"⚠️ ไม่สามารถดึงจาก Hugging Face: {e} จะสลับไปใช้ไฟล์ในเครื่องแทน...")
            
    if not raw_pages:
        # Fallback to local files
        jsonl_files = sorted(input_path.rglob("*.jsonl"))
        if not jsonl_files:
            print(f"❌ ไม่พบข้อมูลดิบทั้งบน Hugging Face และโฟลเดอร์ '{input_dir}'")
            return
            
        print(f"📦 สแกนข้อมูลดิบจากไฟล์โลคัลทั้งหมด {len(jsonl_files)} ไฟล์...")
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        raw_text = entry.get("text", "") or ""
                        raw_caption = entry.get("caption", "") or ""
                        
                        img_path = entry.get("image_path", "")
                        img_obj = None
                        if img_path and os.path.exists(img_path):
                            img_obj = PILImage.open(img_path).convert("RGB")
                            
                        raw_pages.append({
                            "source": entry.get("source", ""),
                            "page": int(entry.get("page", 0)),
                            "image": img_obj,
                            "text": raw_text,
                            "caption": raw_caption
                        })
                    except Exception as e:
                        continue

    if not raw_pages:
        print("❌ ไม่มีหน้าข้อมูลใดๆ ที่สามารถดึงมาประมวลผลได้")
        return

    # 2. โหลดประวัติความคืบหน้าที่ถูกทำความสะอาดแล้วจาก HF ปลายทาง (เพื่อทำ Auto-Resume)
    cleaned_dict = {}
    if hf_output_repo and hf_token:
        try:
            print(f"📥 กำลังตรวจสอบความคืบหน้าที่สะอาดแล้วจาก Hugging Face: '{hf_output_repo}'...")
            cleaned_dataset = load_dataset(hf_output_repo, split="train", token=hf_token)
            print(f"✅ ดึงประวัติความสำเร็จ! พบหน้าที่สะอาดแล้ว {len(cleaned_dataset)} รายการ")
            for row in cleaned_dataset:
                key = (row.get("source", ""), int(row.get("page", 0)))
                cleaned_dict[key] = {
                    "source": row.get("source", ""),
                    "page": int(row.get("page", 0)),
                    "image": row.get("image"),
                    "text": row.get("text", "") or "",
                    "caption": row.get("caption", "") or "",
                    "category": row.get("category", "อื่นๆ")
                }
        except Exception as e:
            print(f"ℹ️ ยังไม่มี Dataset สะอาดในปลายทาง หรือสร้างครั้งแรก: {e}")

    # 3. คัดกรองหน้าขยะและหน้าที่ประมวลผลเสร็จแล้วออก
    pages_to_clean = []
    skipped_trash = 0
    
    for p in raw_pages:
        source = p["source"]
        page = p["page"]
        
        # กรองขยะเบื้องต้น
        if should_skip_page(p["text"], p["caption"]):
            skipped_trash += 1
            continue
            
        # ถ้าเคยทำความสะอาดแล้ว ให้ข้ามไปใช้ของเก่าเลย
        if (source, page) in cleaned_dict:
            continue
            
        pages_to_clean.append(p)

    # จัดเรียงหน้าที่จะคลีนตามแหล่งที่มาและหน้าจริง เพื่อรักษาลำดับการอ่าน
    pages_to_clean.sort(key=lambda x: (x["source"], x["page"]))
    
    total_to_clean = len(pages_to_clean)
    print(f"\n📊 สรุปรายการประมวลผล:")
    print(f"  - คัดขยะออก: {skipped_trash} หน้า")
    print(f"  - ข้ามหน้าที่เสร็จไปแล้ว: {len(cleaned_dict)} หน้า")
    print(f"  - หน้าที่ต้องคลีนเพิ่ม: {total_to_clean} หน้า")

    # 4. ประมวลผลทีละแบทช์และบันทึกขึ้น HF
    if total_to_clean > 0:
        if use_llm:
            init_llm(llm_model, gpu_memory_utilization)
            
        for i in range(0, total_to_clean, batch_size):
            batch = pages_to_clean[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_to_clean + batch_size - 1) // batch_size
            print(f"\n🔍 กำลังประมวลผลแบทช์ {batch_num}/{total_batches} (หน้าที่ {i+1}–{min(i + batch_size, total_to_clean)})...")
            
            # ล้างข้อความเนื้อหา OCR และจัดแยกหมวดหมู่
            if use_llm:
                raw_texts = [p["text"] for p in batch]
                cleaned_data = clean_text_with_llm(raw_texts)
                for p, item in zip(batch, cleaned_data):
                    p["text"] = item["text"]
                    p["category"] = item["category"]
            else:
                for p in batch:
                    p["text"] = clean_ocr_text(p["text"])
                    p["category"] = "อื่นๆ"
            
            # ล้างคำบรรยายภาพและคำเกริ่นนำของบอท
            for p in batch:
                p["caption"] = clean_conversational_noise(clean_ocr_text(p["caption"]))
                
            # เพิ่มผลลัพธ์เข้าดิกชันนารีเก็บความสำเร็จ
            for p in batch:
                key = (p["source"], p["page"])
                cleaned_dict[key] = p
                
            # พุชประวัติขึ้น Hugging Face หลังจบแบทช์
            if hf_output_repo and hf_token:
                push_progress_to_hf(cleaned_dict.values(), hf_output_repo, hf_token)

    # 5. สรุปและเขียนบันทึกไฟล์โลคัล (จากข้อมูลสะอาดครบทุกหน้าเรียงลำดับเสร็จสมบูรณ์)
    final_pages = sorted(cleaned_dict.values(), key=lambda x: (x["source"], x["page"]))
    
    cpt_entries = []
    rag_entries = []
    sft_entries = []
    full_corpus_text = []

    for page_data in final_pages:
        source = page_data["source"]
        page = page_data["page"]
        cleaned_text = page_data["text"]
        cleaned_caption = page_data["caption"]
        
        # ปรับ path ชั่วคราวสำหรับอ้างอิงรูปโลคัล
        clean_name = Path(source).stem.replace(" ", "_")
        image_path = f"output_data/{clean_name}/images/page_{page:03d}.jpg"
        
        # 1. เขียนบันทึกสำหรับ CPT
        cpt_page_text = f"เอกสาร: {source}\nหน้าที่: {page}\nหมวดหมู่: {page_data.get('category', 'อื่นๆ')}\n\nเนื้อหา:\n{cleaned_text}\n"
        if cleaned_caption:
            cpt_page_text += f"\nรายละเอียดรูปภาพและดวงชะตา:\n{cleaned_caption}\n"
        cpt_page_text += "\n" + "="*40 + "\n\n"
        
        cpt_entries.append({"text": cpt_page_text.strip()})
        full_corpus_text.append(cpt_page_text)
        
        # 2. เขียนบันทึกสำหรับ RAG (ผสาน Text + Caption เป็นก้อนเดียวกัน)
        rag_combined_content = f"# แหล่งที่มา: {source} (หน้าที่ {page})\n## หมวดหมู่: {page_data.get('category', 'อื่นๆ')}\n\n"
        rag_combined_content += f"## เนื้อหาข้อความ:\n{cleaned_text}\n\n"
        if cleaned_caption:
            rag_combined_content += f"## ดวงชะตาและแผนภาพประกอบ:\n{cleaned_caption}\n"
            
        sub_chunks = recursive_chunk_text(rag_combined_content, chunk_size, chunk_overlap)
        
        for idx, chunk_text in enumerate(sub_chunks):
            chunk_id = f"{Path(source).stem}_p{page:03d}_c{idx:02d}"
            rag_row = {
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "category": page_data.get("category", "อื่นๆ"),
                "image_path": image_path,
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "page": page,
                    "category": page_data.get("category", "อื่นๆ"),
                    "chunk_idx": idx,
                    "total_chunks": len(sub_chunks),
                    "image_path": image_path
                }
            }
            rag_entries.append(rag_row)

        # 3. จัดรูปแบบสำหรับ Multimodal SFT (LLaMA-Factory VQA format)
        sft_row = {
            "images": [image_path],
            "conversations": [
                {
                    "from": "user",
                    "value": "<image>\nกรุณาถอดข้อความภาษาไทยและวิเคราะห์ดวงชะตา ตารางดาว หรือภาพประกอบจากหน้านี้อย่างละเอียด"
                },
                {
                    "from": "assistant",
                    "value": f"## หมวดหมู่:\n{page_data.get('category', 'อื่นๆ')}\n\n## เนื้อหาข้อความ:\n{cleaned_text}\n\n## ดวงชะตาและแผนภาพประกอบ:\n{cleaned_caption if cleaned_caption else 'ไม่มีภาพประกอบหรือตารางดาวในหน้านี้'}"
                }
            ]
        }
        sft_entries.append(sft_row)

    # บันทึกไฟล์โลคัลทั้งหมดลงดิสก์
    with open(cpt_jsonl_file, "w", encoding="utf-8") as f:
        for entry in cpt_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(cpt_txt_file, "w", encoding="utf-8") as f:
        f.writelines(full_corpus_text)
        
    with open(rag_jsonl_file, "w", encoding="utf-8") as f:
        for entry in rag_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(sft_json_file, "w", encoding="utf-8") as f:
        json.dump(sft_entries, f, ensure_ascii=False, indent=2)
            
    print("\n🎉 จัดระเบียบและทำความสะอาดข้อมูลเสร็จสมบูรณ์!")
    print(f"📊 สรุปผลลัพธ์:")
    print(f"  - คงเหลือหน้าคุณภาพดี: {len(cpt_entries)} หน้า")
    print(f"💾 CPT Output (JSONL) -> {cpt_jsonl_file}")
    print(f"💾 CPT Output (TXT)   -> {cpt_txt_file}")
    print(f"💾 RAG Output (JSONL) -> {rag_jsonl_file} ({len(rag_entries)} Chunks)")
    print(f"💾 Multimodal SFT (JSON) -> {sft_json_file}")

def main():
    parser = argparse.ArgumentParser(description="สคริปต์ขั้นสูงสำหรับการขัดเกลาคำผิดด้วย LLM และ Regex สำหรับทำ CPT/RAG")
    parser.add_argument("--input-dir", type=str, default="output_data", help="โฟลเดอร์ข้อมูลดิบ .jsonl (ใช้ในกรณีออฟไลน์)")
    parser.add_argument("--output-dir", type=str, default="clean", help="โฟลเดอร์เซฟผลลัพธ์โลคัล")
    parser.add_argument("--chunk-size", type=int, default=1500, help="ขนาดตัวอักษรสูงสุดต่อ 1 Chunk")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="ความกว้างของข้อความทับซ้อนกันระหว่าง Chunk")
    parser.add_argument("--use-llm", action="store_true", help="เปิดใช้งานโหมดใช้ LLM (vLLM) ช่วยเรียงประโยคและคำสะกดผิด")
    parser.add_argument("--llm-model", type=str, default="google/gemma-4-E4B-it", help="โมเดลที่จะใช้แก้อักษรไทย (ค่าเริ่มต้น gemma-4-E4B-it เพื่อประหยัด VRAM)")
    parser.add_argument("--hf-input-repo", type=str, default="Phonsiri/astrology-dataset", help="ชื่อ repository ข้อมูลดิบบน Hugging Face")
    parser.add_argument("--hf-output-repo", type=str, default="Phonsiri/astrology-dataset-clean", help="ชื่อ repository ผลลัพธ์สะอาดบน Hugging Face")
    parser.add_argument("--batch-size", type=int, default=8, help="จำนวนหน้าที่ประมวลผลและพุชขึ้น HF ต่อ 1 แบทช์")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="สัดส่วนการจอง VRAM ของ GPU สำหรับ vLLM (เช่น 0.80 สำหรับใช้ 80%)")
    
    args = parser.parse_args()
    process_dataset(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        chunk_size=args.chunk_size, 
        chunk_overlap=args.chunk_overlap, 
        use_llm=args.use_llm, 
        llm_model=args.llm_model,
        hf_input_repo=args.hf_input_repo,
        hf_output_repo=args.hf_output_repo,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

if __name__ == "__main__":
    main()
