# Astrology PDF OCR & Dataset Pipeline (H100 + vLLM Optimized)

โปรเจกต์สำหรับดึงข้อมูลข้อความและภาพดวงชาตาจากไฟล์ PDF โหราศาสตร์ เพื่อสร้าง Dataset สำหรับเทรน AI (CPT/Fine-tuning) บนระบบ Cloud (H100) โดยใช้ **Gemma 4 31B**

---

## 🔄 Data Pipeline Flow

1.  **Local**: ส่งไฟล์ PDF จากเครื่องขึ้นไปพักที่ Hugging Face (Raw Storage)
2.  **Cloud (H100)**: ดึงไฟล์ลงมาประมวลผลผ่าน vLLM Server (High Speed)
3.  **Hugging Face**: ส่ง Dataset ที่แกะเสร็จแล้ว (Text + Image) กลับขึ้นไปที่ HF (User: Phonsiri)

---

## 🛠️ สคริปต์ที่สำคัญ (Key Scripts)

- `push_pdfs_to_hf.py`: (รันที่เครื่องเรา) ส่งไฟล์ PDF ในโฟลเดอร์ `hf/` ขึ้น Hugging Face
- `setup_cloud.sh`: (รันบน Cloud) ติดตั้ง Library และดึง PDF ต้นฉบับลงมาเครื่อง GPU
- `vllm_inference.py`: (รันบน Cloud) เชื่อมต่อ vLLM เพื่อแกะข้อมูลทีละหน้า และเซฟเป็น JSONL + Images
- `push_to_hf.py`: (รันบน Cloud) รวบรวมข้อมูลทั้งหมดพุชขึ้น Hugging Face Dataset

---

## 🚀 ขั้นตอนการติดตั้งและรันบน Cloud (H100)

### 1. เตรียมเครื่องและข้อมูล
```bash
# Clone โค้ด
git clone https://github.com/pidsana42-lgtm/1.git
cd 1

# ตั้งค่า Token
export HF_TOKEN="your_huggingface_token"

# รัน Setup (ติดตั้ง Libs + ดึง PDF ลงมา)
bash setup_cloud.sh
```

### 2. รัน vLLM Server (Terminal 1)
ปรับแต่งมาเพื่อ H100 โดยใช้ **SDPA Backend** เพื่อความเสถียรสูงสุด:
```bash
export VLLM_ATTENTION_BACKEND=SDPA
vllm serve google/gemma-4-31b-it \
  --limit-mm-per-prompt image=1 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --device cuda
```

### 3. เริ่มประมวลผล (Terminal 2)
```bash
# เริ่มแกะข้อมูล (มีระบบ Resume รันต่อจากเดิมอัตโนมัติ)
python3 vllm_inference.py

# ส่ง Dataset ขึ้น Hugging Face (User: Phonsiri)
python3 push_to_hf.py
```

---

## 🌟 ฟีเจอร์เด่น
- **SDPA Optimized**: ใช้ Scaled Dot Product Attention เพื่อประสิทธิภาพสูงสุดบน H100
- **Resume Ability**: รันต่อจากหน้าที่ค้างไว้ได้ทันที ไม่ต้องเริ่มใหม่
- **Multimodal Dataset**: เก็บทั้งข้อความ Markdown และภาพต้นฉบับ (Base64/Image Type)
- **Automatic Sync**: เชื่อมต่อกับ Hugging Face Hub ทั้งขาเข้าและขาออก

---

## 📁 โครงสร้างโฟลเดอร์
- `input/`: ไฟล์ PDF ต้นฉบับที่ดึงมาจาก HF
- `output_data/`: ผลลัพธ์ที่แกะเสร็จแล้ว แยกตามชื่อไฟล์ PDF
- `temp_pages/`: ไฟล์ภาพชั่วคราวระหว่างประมวลผล

---
*Developed for Advanced Astrology AI Dataset Construction.*
