# Astrology PDF OCR & Dataset Pipeline (H100 + Transformers)

โปรเจกต์สำหรับดึงข้อมูลข้อความและภาพดวงชาตาจากไฟล์ PDF โหราศาสตร์ เพื่อสร้าง Dataset สำหรับเทรน AI (CPT/Fine-tuning) บนระบบ Cloud (H100) โดยใช้ **Gemma 4 31B** ผ่าน Hugging Face Transformers โดยตรง

---

## 🔄 Data Pipeline Flow

1. **Cloud (H100)**: โหลดโมเดล Gemma 4 31B → ประมวลผล PDF ทีละหน้า
2. **Hugging Face**: ส่ง Dataset ที่แกะเสร็จแล้วขึ้น HF (User: Phonsiri)

---

## 📊 โครงสร้าง Dataset

แต่ละ row มี **5 คอลัมน์** แยกชัดเจน:

| คอลัมน์ | ประเภท | คำอธิบาย |
|---|---|---|
| `source` | string | ชื่อไฟล์ PDF ต้นฉบับ |
| `page` | int | หมายเลขหน้า |
| `image` | image | ภาพของหน้า PDF ต้นฉบับ |
| `text` | string | ข้อความ OCR ทั้งหมดจากหน้านั้น (Markdown) |
| `caption` | string | คำบรรยายภาพ/ดวงชาตา/ตารางดาวในหน้านั้น |

> แต่ละหน้า PDF จะ inference **2 ครั้ง**: ครั้งที่ 1 สำหรับ OCR, ครั้งที่ 2 สำหรับบรรยายภาพ

---

## 🛠️ สคริปต์ที่สำคัญ (Key Scripts)

| ไฟล์ | ทำอะไร |
|---|---|
| `vllm_inference.py` | โหลด Gemma 4 แล้วแกะข้อมูลจาก PDF → JSONL (OCR + Caption) |
| `setup_cloud.sh` | ติดตั้ง dependencies (รันครั้งแรกครั้งเดียว) |
| `push_to_hf.py` | รวบรวมข้อมูลและพุชขึ้น Hugging Face Dataset |
| `download_pdfs.py` | ดึง PDF ต้นฉบับจาก Hugging Face ลงมาที่ `input/` |

---

## 🚀 ขั้นตอนการรันบน Cloud ใหม่ (H100 / A100)

### 1. โคลนและติดตั้งครั้งแรก
เมื่อเริ่มรันบนเครื่อง Cloud ใหม่ ให้รันคำสั่งกลุ่มนี้เพื่อดาวน์โหลดสคริปต์, ตั้งค่าสิทธิ์, และดาวน์โหลดโมเดล/ความคุ้มกัน CUDA:
```bash
# โคลนโปรเจกต์
git clone https://github.com/pidsana42-lgtm/1.git
cd 1

# ตั้งค่า Token เป็น Global (สำคัญมาก: ห้ามลืม export เพื่อให้สคริปต์ลูกเรียกใช้ได้)
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# ติดตั้ง System Dependencies (poppler สำหรับแปลง PDF)
sudo apt-get install -y poppler-utils

# รันตัวติดตั้ง (สคริปต์จะเช็กเวอร์ชัน CUDA ในเครื่อง แล้วติดตั้ง vLLM + ตัวกู้คืนไลบรารีที่เหมาะสมให้เอง)
bash setup_cloud.sh
```

### 2. วาง PDF ลงโฟลเดอร์ input/
```bash
# หากมีไฟล์ PDF อยู่ในเครื่อง Cloud แล้ว ให้ย้ายเข้า input/
mv /path/to/your/pdfs/*.pdf input/

# หรือถ้าต้องการดึงไฟล์ PDF ต้นฉบับจาก Hugging Face ให้รัน:
python3 download_pdfs.py
```

### 3. เริ่มรันประมวลผล (รันต่ออัตโนมัติ)
ในการรันทุกครั้ง ให้รันผ่านสคริปต์ครอบ **`run.sh`** เพื่อเชื่อมต่อไลบรารี CUDA เข้ากับโค้ดโดยอัตโนมัติ:
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
bash run.sh
```

ระบบจะ:
- **Auto-Sync:** ดึงประวัติที่ทำเสร็จแล้วบน Hugging Face ลงมากู้คืนโฟลเดอร์ผลลัพธ์บนเครื่อง Cloud ใหม่นี้
- **Auto-Resume:** ข้ามหน้าเก่าที่ทำเสร็จแล้ว และรันหน้าถัดไปต่อให้อัตโนมัติทันที
- **vLLM Offline Mode:** ประมวลผลแบบ Batch ด้วยความเร็วสูงโดยไม่ต้องสตาร์ต Server แยก
- **Auto-Push:** อัปเดตข้อมูลขึ้น Hugging Face ใหม่ทุก ๆ 1 Batch

### 4. การล้างข้อมูลและจัดเรียงด้วย LLM (Post-Processing)
เมื่อประมวลผลดึงข้อความจาก PDF ทั้งหมดเรียบร้อยแล้ว สามารถรันระบบล้างและจัดเรียงคลังข้อมูลด้วย LLM เพื่อปั้นชุดข้อมูลคุณภาพสูงสำหรับ CPT หรือ RAG ได้โดยตรง:
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
bash run_clean.sh
```
ระบบจะช่วยดำเนินการ:
- **Auto-Sorting**: จัดเรียงลำดับหน้าเอกสารตาม PDF และเลขหน้าให้อัตโนมัติ ป้องกันข้อมูลสลับหน้า
- **LLM-assisted Text Reordering**: ใช้ LLM (Gemma 4) จัดลำดับเนื้อความ ค้นหาและแก้ปัญหากล่องข้อความ/คอลัมน์สลับกัน หรือลำดับหัวข้อผิดเพี้ยนจากการดึงข้อมูล OCR (เช่น การอ่านเลขข้อข้าม `ข้อ ๘`, `ข้อ ๙`, `ข้อ ๑๐` แล้วสแกนเป็น `ข้อ ๒๑` ระบบจะใช้บริบทแก้กลับเป็น `ข้อ ๑๑` ให้โดยอัตโนมัติ)
- **Thai Grammatical Correction**: สะกดคำผิด ตรวจสอบวรรณยุกต์ลอย หรือสระสลับตำแหน่ง เช่น เปลี่ยนตัวสะกดที่เพี้ยนอย่าง "คลงไคล่" เป็น "คลั่งไคล้" ให้อ่านราบรื่น
- **Structured Output**: แบ่งเนื้อหาออกมาในรูปแบบ JSONL และไฟล์คลังคำดิบพร้อมเอาไปเทรนต่อในโฟลเดอร์ `clean/`

### 5. การดึงโค้ดเวอร์ชันล่าสุดในครั้งถัดไป
หากมีการแก้ไขสคริปต์เพิ่มเติม ให้รันคำสั่งเหล่านี้เพื่ออัปเดตโค้ด:
```bash
git pull origin main
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
bash run.sh
```

---

## 🌟 ฟีเจอร์เด่น
- **ไม่ต้องรัน Server** — ใช้ Transformers โดยตรง ง่ายกว่า vLLM
- **2 Tasks ต่อหน้า** — OCR แยก + Caption แยก เก็บคนละคอลัมน์
- **Resume Ability** — รันต่อจากหน้าที่ค้างไว้ได้ทันที ไม่ต้องเริ่มใหม่
- **Auto Sync** — เชื่อมต่อ Hugging Face Hub อัตโนมัติ

---

## 📁 โครงสร้างโฟลเดอร์
- `input/` — ไฟล์ PDF ต้นฉบับ
- `output_data/` — ผลลัพธ์ JSONL + ภาพ แยกตามชื่อไฟล์ PDF
- `temp_pages/` — ไฟล์ภาพชั่วคราวระหว่างประมวลผล

---

## ⚠️ หมายเหตุ
- ต้องติดตั้ง `poppler-utils` ก่อนรัน (`sudo apt-get install -y poppler-utils`)
- `HF_TOKEN` ต้องมีสิทธิ์ write เพื่อ push dataset ขึ้น HF
- RAM GPU ต้องมีพอสำหรับ Gemma 4 31B (bfloat16 ~62GB) — แนะนำ H100 80GB

---
*Developed for Advanced Astrology AI Dataset Construction.*
