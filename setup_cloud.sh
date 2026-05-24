#!/bash/bin
# สคริปต์เตรียมเครื่องบน Cloud สำหรับ H100 (Using SDPA Backend)

echo "🚀 Installing stable dependencies for H100..."
pip install --upgrade pip
pip install vllm openai pdf2image datasets huggingface_hub

echo "📂 Creating directories..."
mkdir -p input output_data temp_pages

echo "📥 Downloading raw PDFs from Hugging Face..."
# ดึงไฟล์ PDF ทั้งหมดจาก Repo มาไว้ที่ input/
if [ -n "$HF_TOKEN" ]; then
    python3 download_pdfs.py
else
    echo "⚠️ HF_TOKEN not set, skipping PDF download."
fi

echo "🖥️  To start vLLM server on H100 with SDPA, run:"
echo "export VLLM_ATTENTION_BACKEND=SDPA"
echo "vllm serve google/gemma-4-31b-it --limit-mm-per-prompt image=1 --max-model-len 4096 --dtype bfloat16"

echo "✅ Ready! (H100 + SDPA Optimized)"
