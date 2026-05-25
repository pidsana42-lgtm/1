import os
import re
import json
import sys
import argparse
from pathlib import Path

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

def init_llm(model_id):
    """
    โหลดโมเดลผ่าน vLLM เพื่อใช้ประมวลผลข้อความภาษาไทย
    """
    global llm, processor
    try:
        from vllm import LLM as VLLM_LLM
        from transformers import AutoProcessor
    except ImportError:
        print("❌ Error: ไม่พบไลบรารี vllm หรือ transformers ในสภาพแวดล้อมนี้ กรุณาติดตั้งก่อนใช้งานโหมด LLM")
        sys.exit(1)
        
    print(f"🔄 กำลังโหลดโมเดล {model_id} ผ่าน vLLM สำหรับแก้ภาษา...")
    llm = VLLM_LLM(
        model=model_id,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.30,  # ใช้หน่วยความจำจำกัดเพื่อแชร์พื้นที่กับการทำ inference ทั่วไป
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("✅ โหลดโมเดล LLM สำหรับเรียงข้อความเสร็จสิ้น!")

def clean_text_with_llm(text_list):
    """
    ใช้ LLM ขัดเกลาคำผิด เรียงไวยากรณ์ และตัดขึ้นบรรทัดใหม่ให้สมบูรณ์สำหรับประมวลผลต่อ
    """
    if not llm or not text_list:
        return text_list
        
    from vllm import SamplingParams
    
    prompts = []
    for text in text_list:
        messages = [
            {"role": "system", "content": (
                "คุณคือผู้เชี่ยวชาญการจัดชำระคัมภีร์และตำราโหราศาสตร์ไทยโบราณ\n"
                "หน้าที่ของคุณคือ ล้างข้อมูล แก้ไขคำสะกดผิด และเรียงลำดับเนื้อความที่ได้จากระบบ OCR ให้ถูกต้องตามลำดับการอ่านธรรมชาติและตามหลักไวยากรณ์ไทย\n"
                "1. เรียงลำดับเนื้อหาใหม่ให้ถูกต้องตามลำดับการอ่านจริง (เช่น กรณีมีคอลัมน์สลับกัน หรือมีเลขข้อที่สลับกันอันเนื่องมาจากข้อบกพร่องของ OCR เช่น ข้อ ๘, ๙, ๑๐ แล้วเป็น ข้อ ๒๑ ให้ใช้บริบทปรับแก้อัตโนมัติเป็น ข้อ ๑๑ เป็นต้น)\n"
                "2. แก้ไขคำสะกดผิดเพี้ยนอันเนื่องมาจากความผิดพลาดในการสแกน OCR (เช่น ปรับปรุงสระ/วรรณยุกต์ลอย/สลับตำแหน่ง หรือข้อความแยกตัวอักษรผิดเพี้ยน เช่น 'คลงไคล่' ให้เป็น 'คลั่งไคล้')\n"
                "3. รักษาโครงสร้างเนื้อหา ลบเลขหัวข้อซ้ำซ้อนหรือหน้าสแกนสอดแทรกที่ไม่จำเป็น แต่ห้ามแต่งเติมข้อมูลใหม่ ห้ามดัดแปลงความหมายเดิม และห้ามเขียนข้อความอธิบายใดๆ เพิ่มเติมทั้งสิ้น\n"
                "4. ส่งกลับเฉพาะเนื้อความคัมภีร์ที่จัดเรียงและปรับแก้ไขแล้วเท่านั้น ห้ามตอบนอกเหนือจากข้อความคัมภีร์โดยเด็ดขาด"
            )},
            {"role": "user", "content": f"ข้อความ OCR ดิบ:\n{text}"}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048
    )
    
    print(f"  ⚡ กำลังประมวลผลข้อความผ่านโมเดล LLM จำนวน {len(text_list)} หน้า...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    
    return [output.outputs[0].text.strip() for output in outputs]

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

def process_dataset(input_dir, output_dir, chunk_size, chunk_overlap, use_llm, llm_model):
    """
    ฟังก์ชันแกนหลักของการรวบรวม ทำความสะอาด และบันทึกผลลัพธ์
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cpt_jsonl_file = output_path / "cpt_output.jsonl"
    cpt_txt_file = output_path / "cpt_corpus.txt"
    rag_jsonl_file = output_path / "rag_output.jsonl"
    
    jsonl_files = sorted(input_path.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"⚠️ ไม่พบไฟล์ .jsonl ในโฟลเดอร์ '{input_dir}'")
        return
        
    print(f"📦 สแกนข้อมูลดิบทั้งหมด {len(jsonl_files)} ไฟล์...")
    
    valid_pages = []
    skipped_pages = 0
    
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    raw_text = entry.get("text", "")
                    raw_caption = entry.get("caption", "")
                    
                    # คัดกรองหน้าขยะออกไปก่อน
                    if should_skip_page(raw_text, raw_caption):
                        skipped_pages += 1
                        continue
                        
                    valid_pages.append({
                        "source": entry.get("source", ""),
                        "page": entry.get("page", 0),
                        "image_path": entry.get("image_path", ""),
                        "text": raw_text,
                        "caption": raw_caption
                    })
                except Exception as e:
                    continue

    if not valid_pages:
        print("❌ ไม่มีหน้าข้อมูลที่มีสาระผ่านเกณฑ์เลยสักหน้า")
        return

    # เรียงลำดับหน้าเอกสารตามแหล่งที่มาและเลขหน้า เพื่อให้ได้บริบทที่ปะติดปะต่อกัน
    valid_pages.sort(key=lambda x: (x["source"], x["page"]))

    # ล้างข้อความ OCR (เลือกโหมดประมวลผลด้วย LLM หรือ Regex ปกติ)
    if use_llm:
        # เริ่มการโหลดและทำความสะอาดข้อความด้วย LLM
        init_llm(llm_model)
        raw_texts = [p["text"] for p in valid_pages]
        cleaned_texts = clean_text_with_llm(raw_texts)
        for page_data, cleaned in zip(valid_pages, cleaned_texts):
            page_data["text"] = cleaned
    else:
        print("⚙️ รันระบบกรองขยะ Regex พื้นฐานสำหรับเนื้อหา OCR...")
        for page_data in valid_pages:
            page_data["text"] = clean_ocr_text(page_data["text"])

    cpt_entries = []
    rag_entries = []
    full_corpus_text = []

    # ดำเนินการสรุปผลและแปลงฟอร์แมต
    for page_data in valid_pages:
        source = page_data["source"]
        page = page_data["page"]
        image_path = page_data["image_path"]
        cleaned_text = page_data["text"]
        # คำบรรยายภาพที่ได้จาก Gemma4 สะอาดอยู่แล้ว ให้ล้างแค่คำเกริ่นนำ AI
        cleaned_caption = clean_conversational_noise(clean_ocr_text(page_data["caption"]))
        
        # 1. เขียนบันทึกสำหรับ CPT
        cpt_page_text = f"เอกสาร: {source}\nหน้าที่: {page}\n\nเนื้อหา:\n{cleaned_text}\n"
        if cleaned_caption:
            cpt_page_text += f"\nรายละเอียดรูปภาพและดวงชะตา:\n{cleaned_caption}\n"
        cpt_page_text += "\n" + "="*40 + "\n\n"
        
        cpt_entries.append({"text": cpt_page_text.strip()})
        full_corpus_text.append(cpt_page_text)
        
        # 2. เขียนบันทึกสำหรับ RAG (ผสาน Text + Caption เป็นก้อนเดียวกัน)
        rag_combined_content = f"# แหล่งที่มา: {source} (หน้าที่ {page})\n\n"
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
                "image_path": image_path,
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "page": page,
                    "chunk_idx": idx,
                    "total_chunks": len(sub_chunks),
                    "image_path": image_path
                }
            }
            rag_entries.append(rag_row)

    # บันทึกไฟล์ทั้งหมดลงดิสก์
    with open(cpt_jsonl_file, "w", encoding="utf-8") as f:
        for entry in cpt_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(cpt_txt_file, "w", encoding="utf-8") as f:
        f.writelines(full_corpus_text)
        
    with open(rag_jsonl_file, "w", encoding="utf-8") as f:
        for entry in rag_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print("\n🎉 จัดระเบียบและทำความสะอาดข้อมูลเสร็จสมบูรณ์!")
    print(f"📊 สรุปผลลัพธ์:")
    print(f"  - คัดกรองหน้าขยะออก: {skipped_pages} หน้า")
    print(f"  - คงเหลือหน้าคุณภาพดี: {len(cpt_entries)} หน้า")
    print(f"💾 CPT Output (JSONL) -> {cpt_jsonl_file}")
    print(f"💾 CPT Output (TXT)   -> {cpt_txt_file}")
    print(f"💾 RAG Output (JSONL) -> {rag_jsonl_file} ({len(rag_entries)} Chunks)")

def main():
    parser = argparse.ArgumentParser(description="สคริปต์ขั้นสูงสำหรับการขัดเกลาคำผิดด้วย LLM และ Regex สำหรับทำ CPT/RAG")
    parser.add_argument("--input-dir", type=str, default="output_data", help="โฟลเดอร์ข้อมูลดิบ .jsonl")
    parser.add_argument("--output-dir", type=str, default="clean", help="โฟลเดอร์เซฟผลลัพธ์")
    parser.add_argument("--chunk-size", type=int, default=1500, help="ขนาดตัวอักษรสูงสุดต่อ 1 Chunk")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="ความกว้างของข้อความทับซ้อนกันระหว่าง Chunk")
    parser.add_argument("--use-llm", action="store_true", help="เปิดใช้งานโหมดใช้ LLM (vLLM) ช่วยเรียงประโยคและคำสะกดผิด")
    parser.add_argument("--llm-model", type=str, default="google/gemma-4-E4B-it", help="โมเดลที่จะใช้แก้อักษรไทย (ค่าเริ่มต้น gemma-4-E4B-it เพื่อประหยัด VRAM)")
    
    args = parser.parse_args()
    process_dataset(
        args.input_dir, 
        args.output_dir, 
        args.chunk_size, 
        args.chunk_overlap, 
        args.use_llm, 
        args.llm_model
    )

if __name__ == "__main__":
    main()
