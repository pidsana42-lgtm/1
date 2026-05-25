import os
import re
import json
import sys
import argparse
from pathlib import Path
from datasets import Dataset, Features, Value, load_dataset

# ตัวแปรโกลบอลสำหรับโมเดล vLLM
llm = None
processor = None

def init_llm(model_id, gpu_memory_utilization=0.80):
    """
    โหลดโมเดลผ่าน vLLM สำหรับใช้สร้างคำถาม-คำตอบเชิงลึกแบบมี <think>
    """
    global llm, processor
    try:
        from vllm import LLM as VLLM_LLM
        from transformers import AutoProcessor
    except ImportError:
        print("❌ Error: ไม่พบไลบรารี vllm หรือ transformers ในสภาพแวดล้อมนี้ กรุณาติดตั้งก่อนใช้งาน")
        sys.exit(1)
        
    print(f"🔄 กำลังโหลดโมเดล {model_id} ผ่าน vLLM สำหรับสร้างคู่คำถาม...")
    llm = VLLM_LLM(
        model=model_id,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("✅ โหลดโมเดล LLM สำหรับสร้าง QA เสร็จสมบูรณ์!")

def generate_qa_with_llm(chunk_list):
    """
    ป้อนเนื้อหา RAG Chunks ให้โมเดลประมวลผลสร้างบทสนทนา 2 รอบ (Multi-turn) 
    รอบที่ 1: ป้อน Context + คำถามแรก -> กระบวนการคิด <think> + คำตอบแรก
    รอบที่ 2: คำถามที่สอง (ถามต่อยอด/เจาะลึก) -> กระบวนการคิด <think> + คำตอบที่สอง
    """
    if not llm or not chunk_list:
        return []
        
    from vllm import SamplingParams
    
    prompts = []
    for chunk in chunk_list:
        text_content = chunk.get("text", "")
        messages = [
            {"role": "system", "content": (
                "คุณคือผู้เชี่ยวชาญการสร้างชุดข้อมูลสำหรับเทรน AI โหราศาสตร์ไทยในฐานะหมอดู (Astrologer)\n"
                "หน้าที่ของคุณคือการอ่านเนื้อหาโหราศาสตร์ไทยโบราณที่กำหนดให้ แล้วสร้างบทสนทนาโต้ตอบกัน 2 รอบ (Multi-turn - 2 turns) ระหว่าง 'ผู้ใช้งานที่ต้องการดูดวงชะตาชีวิต/ทายดวง (human)' และ 'หมอดูอัจฉริยะ (gpt)' ที่ทำนายตามกฎเกณฑ์คัมภีร์โบราณ\n\n"
                "กติกาสำคัญในการตั้งคำถามของลูกค้า (Human):\n"
                "1. ต้องเป็นคำถามแบบ 'เปิดกว้าง' ไม่ถามแค่ให้ตอบว่า 'ใช่' หรือ 'ไม่ใช่' เท่านั้น แต่ต้องเปิดโอกาสให้เห็นแนวทางชีวิต โอกาส หรืออุปสรรคที่จะเกิดขึ้น\n"
                "2. ต้องไม่คลุมเครือ มีความชัดเจน สั้นกระชับ เป็นภาษาพูดที่คนทั่วไปใช้สนทนากับหมอดูจริงๆ\n"
                "3. มีการระบุ 'ข้อมูลพื้นฐาน/สถานการณ์สั้นๆ (Background Scenario)' เพื่อปูพื้นหลังให้สมจริง เช่น 'เพิ่งย้ายงานใหม่', 'เพิ่งเลิกกับแฟน', 'สัมภาษณ์งานไปแล้วกำลังรอผล', 'ช่วงนี้หมดไฟมาก', หรือบอกเล่าความฝันของตนเอง แต่ยังไม่ได้ระบุข้อมูลทางโหราศาสตร์ที่จำเป็น\n"
                "4. มีการกำหนด 'กรอบเวลาที่ชัดเจน' เสนอด้วย เช่น 'ภายใน 3-6 เดือนนี้', 'ภายในครึ่งปีหลังนี้', 'ปีนี้' ฯลฯ\n"
                "5. ต้องเลือกตั้งคำถามให้สอดคล้องกับเนื้อหาในคัมภีร์ที่กำหนดให้ โดยอ้างอิงและปรับประยุกต์รูปแบบคำถามยอดฮิตตามหมวดหมู่ต่อไปนี้:\n"
                "   - 💼 หมวดการงาน & การเรียน: เกณฑ์ย้ายงาน/ย้ายแผนก, การประสบความสำเร็จ/อุปสรรคของโปรเจกต์, ผลตอบรับจากการร่วมหุ้นทำธุรกิจ, ทางเลือกทางรอดเมื่อหมดไฟ, ความเหมาะของคณะ/สาขาวิชาที่เลือก\n"
                "   - 💰 หมวดการเงิน & โชคลาภ: สภาพคล่องครึ่งปีหลังเทียบกับครึ่งปีแรก, เกณฑ์มีโชคใหญ่หรือได้คืนหนี้สิน, ความเสี่ยงในการลงทุน (หุ้น/กองทุน/อสังหาฯ), การหาลู่ทางเพิ่มรายได้ใหม่\n"
                "   - ❤️ หมวดความรัก & ความสัมพันธ์:\n"
                "       * สำหรับคนโสด: เกณฑ์สละโสดและลักษณะคนที่จะเข้ามา, ความจริงใจของคนคุยใหม่, สิ่งในดวงชะตาที่ควรแก้เพื่อสมหวัง\n"
                "       * สำหรับคนมีคู่/สถานะไม่ชัดเจน: แนวโน้มความคลี่คลายของความรักที่ตึงเครียด, การพัฒนาสถานะจากคนคุยเป็นแฟน, เรื่องที่คู่ส่งเสริมกัน หรือข้อควรระวังเพื่อเลี่ยงการเลิกรา\n"
                "   - 🏥 หมวดสุขภาพ & ภาพรวมชีวิต: เกณฑ์อุบัติเหตุ/ปัญหาสุขภาพร้ายแรง, วิธีเสริมดวง/ทำบุญแก้เคล็ดช่วงดวงตก, ภาพรวมเรื่องที่เด่นที่สุดและควรห่วงที่สุดในปีหน้า\n"
                "6. ห้ามใส่ศัพท์เทคนิคของคัมภีร์, ข้อมูล RAG, หรือตัวเลขดาวโหราศาสตร์ปนลงไปในข้อความคำถามของลูกค้า (Human) เด็ดขาด ปล่อยให้ลูกค้าถามด้วยภาษาธรรมชาติธรรมดาที่สุด\n\n"
                "กติกาสำคัญสำหรับบอทหมอดู (GPT) และรูปแบบบทสนทนารวม:\n"
                "1. **ระบบความช่างซักถามเพื่อขอข้อมูลเพิ่ม (Clarification & Interactive Loop):**\n"
                "   - ในรอบแรก (Turn 1) ลูกค้ามักให้ข้อมูลไม่ครบถ้วน เช่น เล่าแค่เรื่องความฝัน หรือเล่าเรื่องปัญหาหัวใจแต่ยังไม่บอกข้อมูลส่วนตัวที่ต้องใช้ตามเงื่อนไขของคัมภีร์ (เช่น วันเกิด วันเดือนปีเกิด เวลาตกฟาก หรือตำแหน่งดาวในภพเฉพาะ)\n"
                "   - **Turn 1 Response (response1):** ภายใต้แท็ก `<think>` ให้วิเคราะห์สาระจากคัมภีร์และวิเคราะห์ว่าในการทำนายระดับลึกจำเป็นต้องใช้ข้อมูลใดเพิ่ม จากนั้นในคำตอบจริงของหมอดู ให้ทำนายภาพรวมกว้างๆ เท่าที่ข้อมูลมีอย่างเป็นมิตร และ **'แสดงความช่างถาม' โดยขอข้อมูลที่ยังขาดอย่างเฉพาะเจาะจง** (เช่น สอบถามวันเดือนปีเกิด เวลาเกิด หรือตำแหน่งดาวเฉพาะดวงเพื่อนำมาประกอบการคำนวณตามเกณฑ์ในคัมภีร์)\n"
                "   - **Turn 2 Question (question2):** ลูกค้าตอบรับอย่างเป็นมิตรและยินดีให้ข้อมูลเพิ่มตามที่หมอดูขอ (เช่น 'หนูเกิดวันอังคารที่ 5 มีนาคม เวลา 8 โมงเช้าค่ะ', 'ในใบเกิดระบุว่าเกิดเวลาตกฟาก 22.30 น. ค่ะ', 'ในใบดวงของหนูมีดาว ๔ อยู่ร่วมกับดาว ๑ ค่ะหมอ')\n"
                "   - **Turn 2 Response (response2):** ภายใต้แท็ก `<think>` ให้นำข้อมูลเฉพาะที่ลูกค้าเพิ่งให้เข้ามาประมวลผลร่วมกับหลักเกณฑ์คัมภีร์เพื่อวิเคราะห์ชะตาขั้นลึกทีละขั้นตอน จากนั้นตอบพยากรณ์อย่างเจาะลึก แม่นยำ และชี้แนะแนวทางเสริมดวงชะตาหรือวิธีปรับดวงให้ดีขึ้นอย่างเป็นมิตรและเป็นรูปธรรม\n"
                "2. ให้คำทำนายตามกฎเกณฑ์คัมภีร์อย่างอบอุ่น สุภาพ ไพเราะ และให้ข้อคิดที่ดี\n\n"
                "ส่งกลับเป็นโครงสร้าง JSON ดังนี้เท่านั้น ห้ามพิมพ์อธิบายขยายนอก JSON:\n"
                "{\n"
                "  \"question1\": \"[คำถามดูดวงรอบแรกของคนธรรมดา ระบุบริบทสั้นๆ และกรอบเวลา โดยข้อมูลส่วนตัว/โหราศาสตร์ยังไม่ครบถ้วน]\",\n"
                "  \"response1\": \"<think>\\n[วิเคราะห์ข้อมูลเบื้องต้นและระบุสิ่งที่ยังขาดเพื่อนำไปทำนายตามคัมภีร์โบราณ...]\\n</think>\\n[คำทำนายภาพกว้างอย่างอบอุ่น และซักถามวันเกิด/วันเดือนปีเกิด/เวลาตกฟาก หรือตำแหน่งดาวเพิ่มเติมอย่างสุภาพและเฉพาะเจาะจงเพื่อนำไปคำนวณ...]\",\n"
                "  \"question2\": \"[คำตอบของลูกค้าที่ให้ข้อมูลส่วนตัวเพิ่มเติมตามที่หมอดูสอบถามไปรอบแรก]\",\n"
                "  \"response2\": \"<think>\\n[นำข้อมูลส่วนตัวใหม่ของลูกค้ามาจับคู่คำนวณตามกฎเกณฑ์ในคัมภีร์อย่างละเอียดทีละขั้นตอน...]\\n</think>\\n[คำทำนายขั้นลึกที่แม่นยำตามเกณฑ์คัมภีร์ พร้อมแนะแนวทางแก้ไขหรือเสริมดวงชะตา...]\"\n"
                "}"
            )},
            {"role": "user", "content": f"เนื้อหาคัมภีร์ตำราโหราศาสตร์โบราณ สำหรับวิเคราะห์ดวงชะตา:\n{text_content}"}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=3072  # ขยายขนาดเพื่อรองรับบทสนทนาโต้ตอบ 2 รอบที่มีแท็กคิดวิเคราะห์สองชุด
    )
    
    print(f"  ⚡ กำลังสร้างบทสนทนา Multi-turn QA ผ่านโมเดลจำนวน {len(chunk_list)} รายการ...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    
    results = []
    for chunk, output in zip(chunk_list, outputs):
        raw_text = output.outputs[0].text.strip()
        
        # ตัดบล็อก markdown json ออกหากปนมา
        cleaned_output = raw_text
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        elif cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
        cleaned_output = cleaned_output.strip()
        
        try:
            parsed = json.loads(cleaned_output)
            q1 = parsed.get("question1", "").strip()
            r1 = parsed.get("response1", "").strip()
            q2 = parsed.get("question2", "").strip()
            r2 = parsed.get("response2", "").strip()
            
            if q1 and r1 and q2 and r2:
                results.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "source": chunk.get("source", ""),
                    "page": chunk.get("page", 0),
                    "question1": q1,
                    "response1": r1,
                    "question2": q2,
                    "response2": r2
                })
            else:
                raise ValueError("Missing JSON keys for multi-turn")
        except Exception as e:
            print(f"  ⚠️ ไม่สามารถพาร์สผลลัพธ์ของ Chunk {chunk.get('chunk_id')} เป็น JSON ได้ ({e}), กำลังทำ Fallback...")
            # Fallback หาก JSON โครงสร้างผิดเพี้ยน
            fallback_q1 = "ช่วยทำนายดวงชะตาตามหลักเกณฑ์ในคัมภีร์หน้านี้ให้ทีค่ะหมอ"
            fallback_r1 = f"<think>\nวิเคราะห์สาระสำคัญตำราโหราศาสตร์หน้า {chunk.get('page')}\n</think>\n{raw_text[:len(raw_text)//2]}"
            fallback_q2 = "มีข้อควรระวังหรือเกณฑ์อะไรที่ต้องใส่ใจเป็นพิเศษเพิ่มเติมอีกไหมคะ?"
            fallback_r2 = f"<think>\nค้นหารายละเอียดเชิงลึกเพิ่มเติมในบริบท\n</think>\n{raw_text[len(raw_text)//2:]}"
            
            results.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "source": chunk.get("source", ""),
                "page": chunk.get("page", 0),
                "question1": fallback_q1,
                "response1": fallback_r1,
                "question2": fallback_q2,
                "response2": fallback_r2
            })
            
    return results

def push_sft_progress_to_hf(qa_rows, repo_id, hf_token):
    """
    พุชชุดข้อมูล SFT QA ที่ทำเสร็จสะสมเก็บไว้บน Hugging Face ป้องกันข้อมูลสูญหาย
    """
    try:
        print(f"📤 กำลังอัปเดตความคืบหน้า SFT ขึ้น Hugging Face: '{repo_id}'...")
        all_data = []
        for row in qa_rows:
            all_data.append({
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "page": int(row["page"]),
                "question1": row["question1"],
                "response1": row["response1"],
                "question2": row["question2"],
                "response2": row["response2"]
            })
            
        features = Features({
            "chunk_id": Value("string"),
            "source": Value("string"),
            "page": Value("int32"),
            "question1": Value("string"),
            "response1": Value("string"),
            "question2": Value("string"),
            "response2": Value("string")
        })
        
        dataset = Dataset.from_list(all_data, features=features)
        dataset.push_to_hub(repo_id, token=hf_token, private=True)
        print(f"✅ อัปเดตขึ้น Hugging Face สำเร็จ! (สะสม {len(all_data)} รายการ)")
    except Exception as e:
        print(f"⚠️ ไม่สามารถพุชขึ้น Hugging Face ได้ชั่วคราว: {e}")

def process_qa_generation(input_file, output_dir, use_llm, llm_model, 
                          hf_output_repo, batch_size, gpu_memory_utilization):
    """
    ฟังก์ชันแกนหลักของการสร้างชุดข้อมูล SFT
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sft_local_file = output_path / "sft_qa_thinking.json"
    
    # 1. โหลดข้อมูล RAG chunks ที่สะสางแล้ว
    chunks = []
    if not input_path.exists():
        print(f"❌ ไม่พบไฟล์ขยะที่คลีนแล้วที่ '{input_file}'! กรุณารันเตรียมข้อมูลด้วย prepare_data.py ก่อน")
        return
        
    print(f"📖 กำลังโหลด RAG Chunks จาก '{input_file}'...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except:
                continue
                
    print(f"✅ โหลดสำเร็จ! พบข้อมูลทั้งหมด {len(chunks)} Chunks")
    
    # 2. ตรวจเช็กความคืบหน้าที่เคยสร้าง SFT ไว้แล้วจาก HF เพื่อทำ Auto-Resume
    hf_token = os.environ.get("HF_TOKEN")
    generated_dict = {}
    
    if hf_output_repo and hf_token:
        try:
            print(f"📥 ตรวจเช็กประวัติ QA บน Hugging Face: '{hf_output_repo}'...")
            existing_dataset = load_dataset(hf_output_repo, split="train", token=hf_token)
            print(f"✅ พบข้อมูล SFT เดิมที่สร้างเสร็จแล้ว {len(existing_dataset)} คู่")
            for row in existing_dataset:
                chunk_id = row.get("chunk_id", "")
                if chunk_id:
                    generated_dict[chunk_id] = {
                        "chunk_id": chunk_id,
                        "source": row.get("source", ""),
                        "page": int(row.get("page", 0)),
                        "question1": row.get("question1", "") or "",
                        "response1": row.get("response1", "") or "",
                        "question2": row.get("question2", "") or "",
                        "response2": row.get("response2", "") or ""
                    }
        except Exception as e:
            print(f"ℹ️ ยังไม่มีประวัติ Dataset SFT เดิม หรือเป็นรอบรันครั้งแรก: {e}")
            
    # 3. กรอง Chunks ที่ต้องการประมวลผลเพิ่ม
    chunks_to_process = [c for c in chunks if c.get("chunk_id") not in generated_dict]
    
    # เรียงลำดับเพื่อให้เนื้อความเรียงปะติดปะต่อกัน
    chunks_to_process.sort(key=lambda x: (x.get("source", ""), x.get("page", 0), x.get("chunk_id", "")))
    
    total_to_process = len(chunks_to_process)
    print(f"\n📊 สรุปรายการสร้าง SFT:")
    print(f"  - จำนวน Chunk ทั้งหมด: {len(chunks)} รายการ")
    print(f"  - สร้างเสร็จไปแล้วก่อนหน้า: {len(generated_dict)} รายการ")
    print(f"  - ต้องสร้างเพิ่ม: {total_to_process} รายการ")
    
    # 4. เริ่มประมวลผลเป็น Batch
    if total_to_process > 0:
        init_llm(llm_model, gpu_memory_utilization)
        
        for i in range(0, total_to_process, batch_size):
            batch = chunks_to_process[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_to_process + batch_size - 1) // batch_size
            print(f"\n🔍 กำลังสร้าง SFT แบทช์ {batch_num}/{total_batches} (Chunck ที่ {i+1}–{min(i + batch_size, total_to_process)})...")
            
            # รันผ่าน LLM
            qa_results = generate_qa_with_llm(batch)
            
            # อัปเดตความสำเร็จลง Dict
            for result in qa_results:
                cid = result["chunk_id"]
                generated_dict[cid] = result
                
            # พุชชุดข้อมูล SFT ทุกแบทช์
            if hf_output_repo and hf_token:
                push_sft_progress_to_hf(generated_dict.values(), hf_output_repo, hf_token)
                
    # 5. สรุปผลลัพธ์แปลงเป็น ShareGPT format เพื่อนำไปใช้เทรนได้โดยตรง (Multi-turn)
    sharegpt_entries = []
    final_rows = sorted(generated_dict.values(), key=lambda x: (x.get("source", ""), x.get("page", 0), x.get("chunk_id", "")))
    
    # แมปข้อความคัมภีร์ RAG ดั้งเดิมเพื่อใช้เป็น System Prompt
    chunk_text_map = {c["chunk_id"]: c["text"] for c in chunks}
    
    for row in final_rows:
        cid = row["chunk_id"]
        context_text = chunk_text_map.get(cid, "")
        
        sharegpt_entry = {
            "system": f"ข้อมูลประกอบคำทำนายจากคัมภีร์ตำราโหราศาสตร์โบราณ:\n---\n{context_text}\n---\nคุณคือหมอดูอัจฉริยะ ทำนายดวงชะตาชีวิตตามกฎเกณฑ์คัมภีร์ด้านบนนี้อย่างสุภาพ อบอุ่น และแม่นยำที่สุด",
            "conversations": [
                {
                    "from": "human",
                    "value": row["question1"]
                },
                {
                    "from": "gpt",
                    "value": row["response1"]
                },
                {
                    "from": "human",
                    "value": row["question2"]
                },
                {
                    "from": "gpt",
                    "value": row["response2"]
                }
            ]
        }
        sharegpt_entries.append(sharegpt_entry)
        
    # บันทึกลงดิสก์เครื่องคลาวในรูปแบบ ShareGPT JSON
    with open(sft_local_file, "w", encoding="utf-8") as f:
        json.dump(sharegpt_entries, f, ensure_ascii=False, indent=2)
        
    print("\n🎉 สร้างชุดข้อมูล SFT QA สำเร็จสมบูรณ์!")
    print(f"💾 SFT ShareGPT (JSON) -> {sft_local_file} (รวม {len(sharegpt_entries)} รายการ)")

def main():
    parser = argparse.ArgumentParser(description="สคริปต์สังเคราะห์คู่คำถาม-คำตอบเชิงลึกแบบมีคิดวิเคราะห์ <think> เพื่อทำ SFT")
    parser.add_argument("--input-file", type=str, default="clean/rag_output.jsonl", help="ตำแหน่งไฟล์ RAG chunks ที่ล้างแล้ว")
    parser.add_argument("--output-dir", type=str, default="clean", help="ตำแหน่งไดเรกทอรีเก็บเอาต์พุต SFT")
    parser.add_argument("--llm-model", type=str, default="google/gemma-4-26B-A4B-it", help="โมเดลที่จะใช้แกะเนื้อหาเพื่อถามตอบ")
    parser.add_argument("--hf-output-repo", type=str, default="Phonsiri/astrology-sft-thinking", help="ชื่อ repository ผลลัพธ์ SFT บน Hugging Face")
    parser.add_argument("--batch-size", type=int, default=8, help="จำนวนชิ้นงานประมวลผลต่อ 1 แบทช์")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="สัดส่วนการจอง VRAM ของ GPU สำหรับโมเดล vLLM (0.0-1.0)")
    
    args = parser.parse_args()
    process_qa_generation(
        input_file=args.input_file,
        output_dir=args.output_dir,
        use_llm=True,
        llm_model=args.llm_model,
        hf_output_repo=args.hf_output_repo,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

if __name__ == "__main__":
    main()
