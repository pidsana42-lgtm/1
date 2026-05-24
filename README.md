# Astrology PDF to Markdown & JSONL Extractor (Gemma-4 Vision)

เครื่องมืออัตโนมัติสำหรับสกัดข้อความภาษาไทยและ "ภาพดวงชาตา" จากไฟล์ PDF หนังสือโหราศาสตร์ ให้ออกมาเป็นรูปแบบ **Markdown** และ **JSONL** เพื่อเตรียมข้อมูลสำหรับกระบวนการ Continuous Pre-Training (CPT) หรือ Fine-tuning AI แบบ Multimodal

ขับเคลื่อนขุมพลังโดย **Gemma 4 31B (Multimodal)** ผ่าน Google GenAI API

---

## 🌟 จุดเด่นของระบบ (Features)

1. **Multimodal Extraction**: AI สามารถ "มองเห็น" และอธิบายรายละเอียดของภาพดวงชาตา หรือตารางดวงดาว ออกมาเป็นตัวอักษรได้
2. **Dual Output Format**:
   - **Markdown (.md)**: สำหรับอ่านและตรวจสอบความถูกต้อง มีภาพประกอบคู่กับข้อความ
   - **JSONL (.jsonl)**: สำหรับเทรน AI มีข้อมูลโครงสร้าง (`source`, `page`, `content`) และ **Base64 Image Data**
3. **Resume Capability (รันต่อจากเดิม)**: ระบบจะเช็คไฟล์เดิมอัตโนมัติ หากรันค้างไว้จะข้ามหน้าที่เสร็จแล้วและเริ่มงานต่อจากจุดล่าสุดทันที
4. **Incremental Saving (เซฟเก็บแบต)**: บันทึกข้อมูลลงดิสก์ทันทีที่ประมวลผลเสร็จในแต่ละหน้า ทั้ง MD และ JSONL
5. **Auto-Recovery**: ระบบต้านทาน Server Error (500) และ Quota Limit (429) แบบ Exponential Backoff
6. **Rate Limiting**: ควบคุมความเร็วที่ **15 RPM** (หน่วงเวลา 5 วินาทีต่อหน้า) เพื่อความเสถียรสูงสุดตามโควตา API

---

## 📋 สิ่งที่ต้องเตรียม (Prerequisites)

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Poppler](https://poppler.freedesktop.org/)
- Google AI Studio API Key (Gemma-4 Access)

---

## ⚙️ การติดตั้ง (Installation)

1. **ติดตั้ง Dependencies ระดับ OS:**
   ```bash
   brew install tesseract poppler
   ```

2. **ติดตั้ง Python Libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **ตั้งค่า API Key:**
   ใส่คีย์ในไฟล์ `gemini_ocr.py`:
   ```python
   API_KEY = "YOUR_API_KEY_HERE"
   ```

---

## 🚀 วิธีใช้งาน (Usage)

1. วางไฟล์ PDF ในโฟลเดอร์ **`input/`**
2. รันสคริปต์:
   ```bash
   python3 gemini_ocr.py
   ```
3. หากโปรแกรมหยุดทำงาน สามารถรันคำสั่งเดิมซ้ำเพื่อ **ทำงานต่อ (Resume)** ได้ทันที
4. ตรวจสอบผลลัพธ์ในโฟลเดอร์ **`output_data/[ชื่อไฟล์]/`**

---

## 📁 โครงสร้างข้อมูล (Data Structure)

ในโฟลเดอร์ผลลัพธ์จะประกอบด้วย:
- `images/`: ภาพหน้าหนังสือแยกแต่ละหน้า (.jpg)
- `[ชื่อไฟล์].md`: เนื้อหา Markdown พร้อมภาพประกอบ
- `[ชื่อไฟล์].jsonl`: ข้อมูลสำหรับเทรน AI (ประกอบด้วยข้อความและภาพ Base64)

---

## ⚠️ การแก้ไขปัญหา (Troubleshooting)

- **Error 500:** ระบบจะรอและลองใหม่โดยอัตโนมัติ ไม่ต้องปิดโปรแกรม
- **Error 429:** ระบบจะหยุดพัก 30 วินาทีแล้วทำงานต่อเอง
- **ต้องการเริ่มใหม่ทั้งหมด:** ให้ลบโฟลเดอร์ใน `output_data/` ของไฟล์นั้นทิ้ง ระบบจะเริ่มนับ 1 ใหม่ให้เอง
# 1
