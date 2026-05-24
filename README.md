# Astrology PDF OCR & Dataset Pipeline (H100 + vLLM)

โปรเจกต์สำหรับดึงข้อมูลข้อความและภาพดวงชาตาจากไฟล์ PDF โหราศาสตร์ เพื่อสร้าง Dataset สำหรับเทรน AI (CPT/Fine-tuning) บนระบบ Cloud (H100) โดยใช้ **Gemma 4 31B**

---

## 🔄 Data Pipeline Flow

1. **Cloud (H100)**: รัน vLLM Server + ประมวลผล PDF ที่มีอยู่บนเครื่อง
2. **Hugging Face**: ส่ง Dataset ที่แกะเสร็จแล้ว (Text + Image) ขึ้น HF (User: Phonsiri)

---

## 🛠️ สคริปต์ที่สำคัญ (Key Scripts)

- `start_vllm.sh`: รัน vLLM Server (Gemma 4 31B) บน GPU
- `setup_cloud.sh`: ติดตั้ง dependencies ทั้งหมด (รันครั้งแรกครั้งเดียว)
- `vllm_inference.py`: แกะข้อมูลจาก PDF ทีละหน้าผ่าน vLLM → เซฟเป็น JSONL + Images
- `push_to_hf.py`: รวบรวมข้อมูลทั้งหมดพุชขึ้น Hugging Face Dataset
- `download_pdfs.py`: ดึง PDF ต้นฉบับจาก Hugging Face ลงมาที่ `input/`

---

## 🚀 ขั้นตอนการรันบน Cloud (H100)

### 1. ติดตั้งครั้งแรก
```bash
git clone https://github.com/pidsana42-lgtm/1.git
cd 1
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# ติดตั้ง system deps (poppler จำเป็นสำหรับ pdf2image)
sudo apt-get install -y poppler-utils

# ติดตั้ง Python libs
bash setup_cloud.sh
```

### 2. วาง PDF ลง input/
```bash
# ถ้า PDF อยู่บน cloud แล้ว ย้ายเข้า input/
mv /path/to/your/pdfs/*.pdf input/

# หรือถ้าอยู่บน Hugging Face ให้รัน
python3 download_pdfs.py
```

### 3. รัน vLLM Server (Terminal 1)
```bash
bash start_vllm.sh
```
รอจน server พร้อม (เห็น `Uvicorn running on...`)

### 4. รันประมวลผล (Terminal 2)
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# แกะข้อมูลจาก PDF (มีระบบ Resume รันต่อจากเดิมอัตโนมัติ)
python3 vllm_inference.py

# พุช Dataset ขึ้น Hugging Face
python3 push_to_hf.py
```

### 5. ครั้งถัดไป (ไม่ต้อง setup ใหม่)
```bash
cd ~/1
git pull origin main
git restore start_vllm.sh  # ถ้าไฟล์หาย
```

---

## 🌟 ฟีเจอร์เด่น
- **Auto Backend**: vLLM เลือก FLASH_ATTN อัตโนมัติสำหรับ H100
- **Resume Ability**: รันต่อจากหน้าที่ค้างไว้ได้ทันที ไม่ต้องเริ่มใหม่
- **Multimodal Dataset**: เก็บทั้งข้อความ Markdown และภาพต้นฉบับ
- **Auto Sync**: เชื่อมต่อกับ Hugging Face Hub ทั้งขาเข้าและขาออก

---

## 📁 โครงสร้างโฟลเดอร์
- `input/`: ไฟล์ PDF ต้นฉบับ
- `output_data/`: ผลลัพธ์ที่แกะเสร็จแล้ว แยกตามชื่อไฟล์ PDF
- `temp_pages/`: ไฟล์ภาพชั่วคราวระหว่างประมวลผล

---

## ⚠️ หมายเหตุ
- `start_vllm.sh` ถ้าหายให้รัน `git restore start_vllm.sh` (อย่าใช้ `rm` ลบ)
- `poppler-utils` ต้องติดตั้งก่อนรัน `vllm_inference.py` ทุกครั้งที่ใช้ cloud ใหม่

---
*Developed for Advanced Astrology AI Dataset Construction.*
