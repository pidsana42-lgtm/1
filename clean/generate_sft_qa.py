import os
import re
import json
import sys
import argparse
from pathlib import Path
from datasets import Dataset, Features, Value, load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

llm = None
processor = None

def init_llm(model_id):
    """
    โหลดโมเดลภาษาผ่าน Transformers สำหรับใช้ประมวลผลบนเครื่อง
    """
    global llm, processor
    
    print(f"🔄 กำลังโหลดโมเดล {model_id} ผ่าน Transformers...")
    try:
        processor = AutoTokenizer.from_pretrained(model_id)
        llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ โหลดโมเดล LLM ผ่าน Transformers สำเร็จ!")
    except Exception as e:
        print(f"❌ Error: ไม่สามารถโหลดโมเดลได้: {e}")
        sys.exit(1)


def generate_qa_with_llm(chunk_list):
    """
    ป้อนเนื้อหา RAG Chunks ให้โมเดลประมวลผลสร้างบทสนทนา 2 รอบ (Multi-turn) ที่มีการเรียกใช้เครื่องมือค้นหา (Tool Calling)
    โดยใช้ Hugging Face Transformers
    """
    if not llm or not chunk_list:
        return []
        
    prompts = []
    for idx, chunk in enumerate(chunk_list):
        text_content = chunk.get("text", "")
        
        # ปรับปรุง System Prompt ให้มีกติกาและสอนการเรียกใช้เครื่องมือ search_astrology_manual
        messages = [
            {"role": "system", "content": (
                "คุณคือผู้เชี่ยวชาญการสร้างชุดข้อมูลสำหรับเทรน AI โหราศาสตร์ไทยในฐานะหมอดูอัจฉริยะ (Friendly Astrologer)\n"
                "หน้าที่ของคุณคือการอ่านเนื้อหาโหราศาสตร์ไทยโบราณที่กำหนดให้ แล้วสร้างบทสนทนาโต้ตอบกัน 2 รอบ (Multi-turn - 2 turns) ระหว่าง 'ผู้ใช้งานที่ต้องการดูดวง (human)' และ 'หมอดู (gpt)'\n\n"
                "จุดประสงค์สำคัญ: จำลองคู่สนทนาที่มีการเรียกใช้เครื่องมือสืบค้นคัมภีร์ (Tool Calling) เพื่อฝึกให้โมเดลสามารถสืบค้นคีย์เวิร์ดที่เกี่ยวข้อง และนำเนื้อหาจากตำราโบราณมาวิเคราะห์ทำนายดวงชะตาได้อย่างถูกต้องและสมจริง\n\n"
                "รายละเอียดเครื่องมือที่คุณสามารถเรียกใช้งานได้:\n"
                "{\n"
                "  \"name\": \"search_astrology_manual\",\n"
                "  \"description\": \"ค้นหาข้อมูลและกฎเกณฑ์ในคัมภีร์ตำราโหราศาสตร์โบราณ โดยใช้คำสำคัญ (Keywords) ที่เกี่ยวข้องกับคำถามของผู้ใช้ เพื่อนำมาประกอบคำพยากรณ์\",\n"
                "  \"parameters\": {\n"
                "    \"type\": \"object\",\n"
                "    \"properties\": {\n"
                "      \"keywords\": {\n"
                "        \"type\": \"array\",\n"
                "        \"items\": {\"type\": \"string\"},\n"
                "        \"description\": \"คำสำคัญ (Keywords) ภาษาไทยที่เกี่ยวข้องกับประเด็นหรือคำถามที่ต้องการค้นหาคำทำนายในตำรา\"\n"
                "      }\n"
                "    },\n"
                "    \"required\": [\"keywords\"]\n"
                "  }\n"
                "}\n\n"
                "กติกาสำคัญในการตั้งคำถามของลูกค้า (Human):\n"
                "1. คำถามต้องดูสมจริงเหมือนลูกค้ามาคุยกับหมอดูทั่วไป มีระบุสถานการณ์สั้นๆ (Background Scenario) เช่น ย้ายงานใหม่ อกหัก หรืออยากหาฤกษ์มงคล และกรอบเวลาที่ชัดเจน (ภายใน 3 เดือน, ปีนี้ ฯลฯ)\n"
                "2. ลูกค้าต้องไม่ถามศัพท์เทคนิคโหรหรือระบุข้อมูลคัมภีร์ตรงๆ แต่ให้หมอดูเป็นผู้สืบค้นและถอดความเอง\n\n"
                "กติกาสำคัญสำหรับหมอดู (GPT) และรูปแบบของคู่สนทนา:\n"
                "1. **รอบแรก (Turn 1)**:\n"
                "   - **Human**: ถามคำถามทั่วไปเกี่ยวกับเรื่องรัก/งาน/สุขภาพ/โชคลาภ\n"
                "   - **GPT (Assistant)**:\n"
                "     * `thought_before_tool`: วิเคราะห์เจตนาของลูกค้า หาประเด็นและตัดสินใจว่าจะเรียกเครื่องมือด้วยคีย์เวิร์ดภาษาไทยอะไร เช่น `[\"เกณฑ์การงาน\", \"ดาวพฤหัสบดี\"]` หรือ `[\"วิธีแก้เคล็ด\", \"ดวงตก\"]` เพื่อสืบค้นคัมภีร์\n"
                "     * `tool_call`: เรียกใช้เครื่องมือ `search_astrology_manual` พร้อมระบุอาร์กิวเมนต์เป็นคีย์เวิร์ด\n"
                "     * `tool_call`: เรียกใช้เครื่องมือ `search_astrology_manual` พร้อมระบุอาร์กิวเมนต์เป็นคีย์เวิร์ด\n"
                "     * `tool_response`: ผลลัพธ์จากการรันเครื่องมือ (ให้คัดลอกส่วนสำคัญหรือสรุปเนื้อหาจากข้อความคัมภีร์ที่กำหนดให้ด้านล่างนี้ มาจำลองเป็นผลลัพธ์ของเครื่องมือ)\n"
                "     * `thought_after_tool`: วิเคราะห์ข้อความในคัมภีร์ที่ได้รับจากเครื่องมือเทียบกับสถานการณ์ของลูกค้า ภายใต้กรอบคิดวิเคราะห์ <think>...</think>\n"
                "     * `response_text`: พยากรณ์ผลลัพธ์ภาพรวมอย่างอบอุ่น ชวนคุยโต้ตอบเป็นกันเอง และเอ่ยปากถามเพื่อสร้างปฏิสัมพันธ์โต้ตอบในรอบที่ 2\n"
                "2. **รอบสอง (Turn 2)**:\n"
                "   - **Human**: ถามคำถามถามต่อที่เจาะลึกหรือเชื่อมโยงกับคำตอบรอบแรก\n"
                "   - **GPT (Assistant)**:\n"
                "     - กรณีต้องการค้นหาเพิ่มเติมในตำรา: ให้ระบุ `thought_before_tool` / `tool_call` / `tool_response` และ `thought_after_tool` / `response_text` ในลักษณะเดียวกับรอบแรก\n"
                "     - กรณีตอบได้ทันทีจากเนื้อความเดิม: ให้ระบุ `thought_before_tool` และ `tool_call`, `tool_response` เป็น null ทั้งหมด และระบุ `thought_after_tool` พร้อมทำนายคำตอบสุดท้ายใน `response_text`\n\n"
                "ส่งกลับเป็นโครงสร้าง JSON ดังนี้เท่านั้น ห้ามพิมพ์อธิบายขยายนอก JSON:\n"
                "{\n"
                "  \"turns\": [\n"
                "    {\n"
                "      \"human_question\": \"[คำถามรอบแรกของลูกค้า]\",\n"
                "      \"thought_before_tool\": \"[วิเคราะห์เจตนาและหาคำสืบค้นในคัมภีร์]\",\n"
                "      \"tool_call\": {\n"
                "        \"name\": \"search_astrology_manual\",\n"
                "        \"arguments\": {\n"
                "          \"keywords\": [\"คีย์เวิร์ด1\", \"คีย์เวิร์ด2\"]\n"
                "        }\n"
                "      },\n"
                "      \"tool_response\": \"[ข้อความคัมภีร์ตำราโบราณที่ดึงมาประกอบ จากเนื้อหาที่กำหนดให้]\",\n"
                "      \"thought_after_tool\": \"<think>\\n[วิเคราะห์ผลลัพธ์คัมภีร์เปรียบเทียบกับดวงชะตาลูกค้า...]\\n</think>\",\n"
                "      \"response_text\": \"[คำทำนายเบื้องต้นและซักถามกระตุ้นความอยากรู้ต่อ]\"\n"
                "    },\n"
                "    {\n"
                "      \"human_question\": \"[คำถามรอบสองของลูกค้า]\",\n"
                "      \"thought_before_tool\": \"[วิเคราะห์คำถามรอบสอง หรือเป็น null]\",\n"
                "      \"tool_call\": null,\n"
                "      \"tool_response\": null,\n"
                "      \"thought_after_tool\": \"<think>\\n[ผูกดวงวิเคราะห์เชิงลึกหาทางออกหรือการทำนายเสริมสิริมงคล...]\\n</think>\",\n"
                "      \"response_text\": \"[คำทำนายเจาะลึกชี้แนะแนวทางที่ชัดเจนและจบรอบสนทนาอย่างน่าประทับใจ]\"\n"
                "    }\n"
                "  ]\n"
                "}"
            )},
            {"role": "user", "content": f"เนื้อหาคัมภีร์ตำราโหราศาสตร์โบราณ สำหรับวิเคราะห์ดวงชะตา:\n{text_content}"}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        
    print(f"  ⚡ กำลังสร้างบทสนทนาที่มี Tool Calling ผ่านโมเดลจำนวน {len(chunk_list)} รายการ...")
    
    # ประมวลผลข้อความผ่าน Transformers
    raw_texts = []
    for p_idx, prompt in enumerate(prompts):
        print(f"    - กำลังรันลำดับย่อยที่ {p_idx+1}/{len(prompts)} ผ่าน Transformers...")
        inputs = processor(prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=3584,
                temperature=0.3,
                do_sample=True
            )
        input_len = inputs.input_ids.shape[1]
        raw_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        raw_texts.append(raw_text)

    results = []
    for chunk, raw_text in zip(chunk_list, raw_texts):
        # ล้างและจัดการโค้ดบล็อก Markdown
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
            turns = parsed.get("turns", [])
            
            # ตรวจสอบว่ามีข้อมูลรอบสนทนาอย่างน้อย 1-2 รอบหรือไม่
            if len(turns) >= 2:
                results.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "source": chunk.get("source", ""),
                    "page": chunk.get("page", 0),
                    "turns": turns
                })
            else:
                raise ValueError("Turns count is less than 2")
                
        except Exception as e:
            print(f"  ⚠️ ไม่สามารถพาร์สผลลัพธ์ของ Chunk {chunk.get('chunk_id')} เป็น JSON ได้ ({e}), กำลังทำ Fallback...")
            
            # Fallback หาก JSON โครงสร้างผิดเพี้ยนเพื่อไม่ให้งานหลุด
            text_content = chunk.get("text", "")
            fallback_q1 = "ช่วยทำนายดวงชะตาตามหลักเกณฑ์ในคัมภีร์หน้านี้ให้ทีค่ะหมอ"
            fallback_r1_thought_before = "วิเคราะห์คำถามดูดวงทั่วไปของลูกค้า เพื่อหาคำสืบค้นในตำราโหราศาสตร์"
            fallback_r1_tool_call = {
                "name": "search_astrology_manual",
                "arguments": {
                    "keywords": ["ทำนายดวง", "วิเคราะห์ดวง"]
                }
            }
            fallback_r1_tool_response = text_content[:600] if len(text_content) > 600 else text_content
            fallback_r1_thought_after = f"<think>\nวิเคราะห์เนื้อหาคัมภีร์หน้า {chunk.get('page', 0)} เพื่อตอบลูกค้า\n</think>"
            fallback_r1_content = f"สวัสดีครับ จากการเปิดค้นตำราหมอพบข้อมูลว่า... (โปรดสอบถามวันเดือนปีเกิดเพิ่มเติมเพื่อวิเคราะห์ดวงเจาะลึกครับ)"
            
            fallback_q2 = "มีข้อควรระวังหรือเกณฑ์อะไรที่ต้องใส่ใจเป็นพิเศษเพิ่มเติมอีกไหมคะ?"
            fallback_r2_thought_before = "วิเคราะห์เจตนาลูกค้าที่ต้องการหาข้อควรระวังเพิ่มเติม"
            fallback_r2_tool_call = {
                "name": "search_astrology_manual",
                "arguments": {
                    "keywords": ["ข้อควรระวัง", "ข้อห้าม"]
                }
            }
            fallback_r2_tool_response = text_content[-600:] if len(text_content) > 600 else text_content
            fallback_r2_thought_after = "<think>\nวิเคราะห์เกณฑ์ข้อห้ามและอุปสรรคตามกฎเกณฑ์ตอนท้ายบทตำรา\n</think>"
            fallback_r2_content = "สำหรับเรื่องที่ควรระวังนั้น ตามคัมภีร์เตือนว่าให้หลีกเลี่ยงการลงมือในช่วงเวลาอับโชค..."
            
            results.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "source": chunk.get("source", ""),
                "page": chunk.get("page", 0),
                "turns": [
                    {
                        "human_question": fallback_q1,
                        "thought_before_tool": fallback_r1_thought_before,
                        "tool_call": fallback_r1_tool_call,
                        "tool_response": fallback_r1_tool_response,
                        "thought_after_tool": fallback_r1_thought_after,
                        "response_text": fallback_r1_content
                    },
                    {
                        "human_question": fallback_q2,
                        "thought_before_tool": fallback_r2_thought_before,
                        "tool_call": fallback_r2_tool_call,
                        "tool_response": fallback_r2_tool_response,
                        "thought_after_tool": fallback_r2_thought_after,
                        "response_text": fallback_r2_content
                    }
                ]
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
                "turns_json": json.dumps(row["turns"], ensure_ascii=False)
            })
            
        features = Features({
            "chunk_id": Value("string"),
            "source": Value("string"),
            "page": Value("int32"),
            "turns_json": Value("string")
        })
        
        dataset = Dataset.from_list(all_data, features=features)
        dataset.push_to_hub(repo_id, token=hf_token, private=True)
        print(f"✅ อัปเดตขึ้น Hugging Face สำเร็จ! (สะสม {len(all_data)} รายการ)")
    except Exception as e:
        print(f"⚠️ ไม่สามารถพุชขึ้น Hugging Face ได้ชั่วคราว: {e}")

def process_qa_generation(input_file, input_repo, output_dir, use_llm, llm_model, 
                          hf_output_repo, batch_size, gpu_memory_utilization):
    """
    ฟังก์ชันแกนหลักของการสร้างชุดข้อมูล SFT
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sft_local_file = output_path / "sft_qa_thinking.json"
    
    # 1. โหลดข้อมูล RAG chunks (จากไฟล์ หรือดาวน์โหลดตรงจาก Hugging Face)
    chunks = []
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        # ลองค้นหา token จาก cache ของ Hugging Face ในเครื่อง
        for p in [Path.home() / ".cache" / "huggingface" / "token", Path.home() / ".huggingface" / "token"]:
            if p.exists():
                try:
                    hf_token = p.read_text().strip()
                    break
                except:
                    pass
    
    if not input_path.exists() and input_repo:
        if not hf_token:
            print(f"❌ Error: ไม่พบไฟล์ในเครื่อง '{input_file}' และไม่ได้ระบุ HF_TOKEN เพื่อดึงข้อมูลจาก Hugging Face")
            return
        try:
            print(f"📥 ไม่พบไฟล์ในเครื่อง กำลังดึงข้อมูลจาก Hugging Face Repo: '{input_repo}'...")
            hf_dataset = load_dataset(input_repo, split="train", token=hf_token)
            
            # ตรวจสอบว่าใน Dataset มีคอลัมน์ chunk_id หรือไม่
            has_chunk_id = "chunk_id" in hf_dataset.column_names
            
            if has_chunk_id:
                print("✅ พบคอลัมน์ chunk_id ใน Dataset บน Hugging Face")
                for row in hf_dataset:
                    chunks.append({
                        "chunk_id": row.get("chunk_id", ""),
                        "source": row.get("source", ""),
                        "page": int(row.get("page", 0)),
                        "text": row.get("text", "")
                    })
            else:
                print("ℹ️ ไม่พบคอลัมน์ chunk_id ใน Dataset ต้นทาง, กำลังทำ Dynamic Chunking ในหน่วยความจำ...")
                # ฟังก์ชันหั่น Chunk เหมือนใน prepare_data.py
                def recursive_chunk_text(text, chunk_size=1500, chunk_overlap=200):
                    if len(text) <= chunk_size:
                        return [text]
                    sub_chunks = []
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
                            sub_chunks.append(chunk_content)
                        start = end - chunk_overlap if end < text_len else end
                        if start >= text_len or end == text_len:
                            break
                    return sub_chunks

                for row in hf_dataset:
                    source = row.get("source", "")
                    page = int(row.get("page", 0))
                    cleaned_text = row.get("text", "") or ""
                    cleaned_caption = row.get("caption", "") or ""
                    category = row.get("category", "อื่นๆ")
                    
                    # ผสานเนื้อหาเพื่อทำ RAG Chunk
                    rag_combined_content = f"# แหล่งที่มา: {source} (หน้าที่ {page})\n## หมวดหมู่: {category}\n\n"
                    rag_combined_content += f"## เนื้อหาข้อความ:\n{cleaned_text}\n\n"
                    if cleaned_caption:
                        rag_combined_content += f"## ดวงชะตาและแผนภาพประกอบ:\n{cleaned_caption}\n"
                    
                    # หั่น Chunk ขนาด 1500 อักษร ทับซ้อน 200 อักษร
                    sub_chunks = recursive_chunk_text(rag_combined_content, chunk_size=1500, chunk_overlap=200)
                    for idx, chunk_text in enumerate(sub_chunks):
                        chunk_id = f"{Path(source).stem}_p{page:03d}_c{idx:02d}"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "source": source,
                            "page": page,
                            "text": chunk_text
                        })
            print(f"✅ ประมวลผลเป็น RAG Chunks สำเร็จ! พบทั้งหมด {len(chunks)} Chunks")
        except Exception as e:
            print(f"❌ ไม่สามารถโหลดข้อมูลจาก Hugging Face ได้: {e}")
            return
    else:
        print(f"📖 กำลังโหลด RAG Chunks จากไฟล์ในเครื่อง '{input_file}'...")
        with open(input_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    # เพิ่มความปลอดภัยกรณีไม่มี chunk_id ในไฟล์โลคัล
                    if "chunk_id" not in data or not data["chunk_id"]:
                        source = data.get("source", "unknown")
                        page = int(data.get("page", 0))
                        data["chunk_id"] = f"{Path(source).stem}_p{page:03d}_local_{line_idx:03d}"
                    chunks.append(data)
                except:
                    continue
        print(f"✅ โหลดสำเร็จ! พบข้อมูลทั้งหมด {len(chunks)} Chunks")
    
    # 2. ตรวจเช็กความคืบหน้าที่เคยสร้าง SFT ไว้แล้วจาก HF เพื่อทำ Auto-Resume
    generated_dict = {}
    
    if hf_output_repo and hf_token:
        try:
            print(f"📥 ตรวจเช็กประวัติ QA บน Hugging Face: '{hf_output_repo}'...")
            existing_dataset = load_dataset(hf_output_repo, split="train", token=hf_token)
            print(f"✅ พบข้อมูล SFT เดิมที่สร้างเสร็จแล้ว {len(existing_dataset)} คู่")
            for row in existing_dataset:
                chunk_id = row.get("chunk_id", "")
                if chunk_id:
                    # ตรวจทานความเข้ากันได้ย้อนหลัง (Backward Compatibility)
                    turns = []
                    if "turns_json" in row:
                        try:
                            turns = json.loads(row["turns_json"])
                        except:
                            turns = []
                    elif "question1" in row:
                        # อพยพโครงสร้างข้อมูลเดิม (Migration)
                        turns = [
                            {
                                "human_question": row.get("question1", ""),
                                "thought_before_tool": "สืบค้นข้อมูลในตำราประกอบคำพยากรณ์รอบที่หนึ่ง",
                                "tool_call": {
                                    "name": "search_astrology_manual",
                                    "arguments": {
                                        "keywords": ["ดวงชะตา", "ทำนายดวง"]
                                    }
                                },
                                "tool_response": "ข้อมูลคัมภีร์ดั้งเดิม",
                                "thought_after_tool": "<think>\nวิเคราะห์เกณฑ์ดวงชะตารอบแรกตามตำราโบราณ\n</think>",
                                "response_text": row.get("response1", "")
                            },
                            {
                                "human_question": row.get("question2", ""),
                                "thought_before_tool": "สืบค้นเพิ่มเติมสำหรับหัวข้อถามตอบรอบที่สอง",
                                "tool_call": None,
                                "tool_response": None,
                                "thought_after_tool": "<think>\nวิเคราะห์เชิงลึกสำหรับอุปสรรคและโอกาสเพิ่มเติม\n</think>",
                                "response_text": row.get("response2", "")
                            }
                        ]
                        
                    generated_dict[chunk_id] = {
                        "chunk_id": chunk_id,
                        "source": row.get("source", ""),
                        "page": int(row.get("page", 0)),
                        "turns": turns
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
        init_llm(llm_model)
        
        for i in range(0, total_to_process, batch_size):
            batch = chunks_to_process[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_to_process + batch_size - 1) // batch_size
            print(f"\n🔍 กำลังสร้าง SFT แบทช์ {batch_num}/{total_batches} (Chunk ที่ {i+1}–{min(i + batch_size, total_to_process)})...")
            
            # รันผ่าน LLM
            qa_results = generate_qa_with_llm(batch)
            
            # อัปเดตความสำเร็จลง Dict
            for result in qa_results:
                cid = result["chunk_id"]
                generated_dict[cid] = result
                
            # พุชชุดข้อมูล SFT ทุกแบทช์
            if hf_output_repo and hf_token:
                push_sft_progress_to_hf(generated_dict.values(), hf_output_repo, hf_token)
                
    # 5. สรุปผลลัพธ์แปลงเป็น ShareGPT format เพื่อนำไปใช้เทรนได้โดยตรง (Multi-turn + Tool Calling)
    sharegpt_entries = []
    final_rows = sorted(generated_dict.values(), key=lambda x: (x.get("source", ""), x.get("page", 0), x.get("chunk_id", "")))
    
    # ดึงคัมภีร์ดั้งเดิมมาอัปเดตและใช้สำหรับการเก็บประวัติใน RAG
    chunk_text_map = {c["chunk_id"]: c["text"] for c in chunks}
    
    # โครงสร้างคำจำกัดความเครื่องมือสำหรับส่งสอน SFT
    astrology_tools = [
        {
            "name": "search_astrology_manual",
            "description": "ค้นหาข้อมูลและกฎเกณฑ์ในคัมภีร์ตำราโหราศาสตร์โบราณ โดยใช้คำสำคัญ (Keywords) ที่เกี่ยวข้องกับคำถามของผู้ใช้ เพื่อนำมาประกอบคำพยากรณ์",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "คำสำคัญ (Keywords) ภาษาไทยที่ต้องการใช้ค้นหาคำทำนายในคัมภีร์ เช่น ['ดวงตก', 'แก้เคล็ด', 'การงาน']"
                    }
                },
                "required": ["keywords"]
            }
        }
    ]
    
    for row in final_rows:
        cid = row["chunk_id"]
        context_text = chunk_text_map.get(cid, "")
        
        conversations = []
        for turn in row["turns"]:
            # 1. ข้อความจากผู้ใช้
            conversations.append({
                "from": "human",
                "value": turn["human_question"]
            })
            
            # 2. จำลองการเรียกใช้เครื่องมือในบทสนทนา
            if turn.get("tool_call"):
                # คิดวิเคราะห์ก่อนเรียกเครื่องมือ
                gpt_thought_val = f"<think>\n{turn.get('thought_before_tool', '').strip()}\n</think>"
                conversations.append({
                    "from": "gpt",
                    "value": gpt_thought_val,
                    "tool_calls": [turn["tool_call"]]
                })
                
                # ผลลัพธ์จำลองที่ส่งกลับจากเครื่องมือ
                tool_res = turn.get("tool_response")
                if isinstance(tool_res, (dict, list)):
                    tool_res_str = json.dumps(tool_res, ensure_ascii=False)
                else:
                    tool_res_str = str(tool_res)
                
                # สอดรับผลลัพธ์เป็น observation role
                conversations.append({
                    "from": "observation",
                    "value": tool_res_str
                })
                
                # คิดวิเคราะห์หลังได้ผลลัพธ์ + พยากรณ์จริง
                gpt_final_val = f"<think>\n{turn.get('thought_after_tool', '').strip()}\n</think>\n{turn['response_text']}"
                conversations.append({
                    "from": "gpt",
                    "value": gpt_final_val
                })
            else:
                # กรณีข้ามการเรียกเครื่องมือในรอบย่อย
                gpt_direct_val = f"<think>\n{turn.get('thought_after_tool', '').strip() or turn.get('thought_before_tool', '').strip()}\n</think>\n{turn['response_text']}"
                conversations.append({
                    "from": "gpt",
                    "value": gpt_direct_val
                })
        
        sharegpt_entry = {
            "tools": astrology_tools,
            "system": (
                "คุณคือหมอดูอัจฉริยะ ทำนายดวงชะตาชีวิตตามคัมภีร์ตำราโหราศาสตร์โบราณอย่างสุภาพ อบอุ่น และแม่นยำที่สุด "
                "โดยสามารถใช้เครื่องมือ search_astrology_manual เพื่อค้นหาคำทำนายและแนวทางแก้เคล็ดจากตำราตามคีย์เวิร์ดที่ลูกค้าถามได้"
            ),
            "conversations": conversations
        }
        sharegpt_entry_final = sharegpt_entry
        sharegpt_entries.append(sharegpt_entry_final)
        
    # บันทึกลงดิสก์เครื่องคลาวในรูปแบบ ShareGPT JSON
    with open(sft_local_file, "w", encoding="utf-8") as f:
        json.dump(sharegpt_entries, f, ensure_ascii=False, indent=2)
        
    print("\n🎉 สร้างชุดข้อมูล SFT QA สำเร็จสมบูรณ์!")
    print(f"💾 SFT ShareGPT (JSON) -> {sft_local_file} (รวม {len(sharegpt_entries)} รายการ)")

def main():
    parser = argparse.ArgumentParser(description="สคริปต์สังเคราะห์คู่คำถาม-คำตอบเชิงลึกแบบมีคิดวิเคราะห์ <think> เพื่อทำ SFT")
    parser.add_argument("--input-file", type=str, default="clean/rag_output.jsonl", help="ตำแหน่งไฟล์ RAG chunks ที่ล้างแล้วในเครื่อง")
    parser.add_argument("--input-repo", type=str, default="Phonsiri/astrology-dataset-clean", help="ชื่อ repository ข้อมูล RAG chunks สะอาดบน Hugging Face สำหรับดึงเมื่อไม่มีไฟล์ในเครื่อง")
    parser.add_argument("--output-dir", type=str, default="clean", help="ตำแหน่งไดเรกทอรีเก็บเอาต์พุต SFT")
    parser.add_argument("--llm-model", type=str, default="google/gemma-4-26B-A4B-it", help="โมเดลที่จะใช้แกะเนื้อหาเพื่อถามตอบ")
    parser.add_argument("--hf-output-repo", type=str, default="Phonsiri/astrology-sft-thinking", help="ชื่อ repository ผลลัพธ์ SFT บน Hugging Face")
    parser.add_argument("--batch-size", type=int, default=8, help="จำนวนชิ้นงานประมวลผลต่อ 1 แบทช์")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80, help="สัดส่วนการจอง VRAM ของ GPU สำหรับโมเดล (ไม่มีผลเนื่องจากปิด vLLM ไปแล้ว)")
    
    args = parser.parse_args()
    process_qa_generation(
        input_file=args.input_file,
        input_repo=args.input_repo,
        output_dir=args.output_dir,
        use_llm=True,
        llm_model=args.llm_model,
        hf_output_repo=args.hf_output_repo,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

if __name__ == "__main__":
    main()
