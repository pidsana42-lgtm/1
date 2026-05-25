#!/bin/bash
# สคริปต์เตรียมเครื่องบน Cloud สำหรับ H100

echo "🚀 Installing system dependencies..."
sudo apt-get install -y poppler-utils 2>/dev/null || conda install -c conda-forge poppler -y

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install "numpy<2.0.0" transformers accelerate pdf2image datasets huggingface_hub hf_transfer Pillow torchvision
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu124 --extra-index-url https://wheels.vllm.ai/nightly/cu129 --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match || pip install vllm --pre

echo "📂 Creating directories..."
mkdir -p input output_data temp_pages

echo "📥 Downloading raw PDFs from Hugging Face..."
if [ -n "$HF_TOKEN" ]; then
    python3 download_pdfs.py
else
    echo "⚠️ HF_TOKEN not set, skipping PDF download."
fi

echo ""
echo "✅ Ready! Run:"
echo "  export HF_TOKEN=hf_xxxx"
echo "  python3 vllm_inference.py"
