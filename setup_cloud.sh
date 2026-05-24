#!/bin/bash
# สคริปต์เตรียมเครื่องบน Cloud สำหรับ H100

echo "🚀 Installing system dependencies..."
sudo apt-get install -y poppler-utils 2>/dev/null || conda install -c conda-forge poppler -y

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install "numpy<2.0.0" vllm openai pdf2image datasets huggingface_hub hf_transfer

echo "📂 Creating directories..."
mkdir -p input output_data temp_pages

echo "📥 Downloading raw PDFs from Hugging Face..."
if [ -n "$HF_TOKEN" ]; then
    python3 download_pdfs.py
else
    echo "⚠️ HF_TOKEN not set, skipping PDF download."
fi

echo "✅ Ready! Run:"
echo "  Terminal 1: bash start_vllm.sh"
echo "  Terminal 2: python3 vllm_inference.py"
