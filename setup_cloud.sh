#!/bin/bash
# สคริปต์เตรียมเครื่องบน Cloud สำหรับ H100

echo "🚀 Installing system dependencies..."
sudo apt-get install -y poppler-utils 2>/dev/null || conda install -c conda-forge poppler -y

echo "📦 Installing Python dependencies..."
python3 -m pip install --upgrade pip

# ตรวจสอบเวอร์ชัน CUDA ของ PyTorch ในเครื่อง
CUDA_VERSION=$(python3 -c "
try:
    import torch
    v = torch.version.cuda
    if v:
        print(v)
    else:
        print('12.4')
except Exception as e:
    print('12.4')
" 2>/dev/null)

echo "🔍 Detected CUDA version from PyTorch: $CUDA_VERSION"

# ติดตั้งแพ็กเกจพื้นฐาน
python3 -m pip install "numpy<2.0.0" transformers accelerate pdf2image datasets huggingface_hub hf_transfer Pillow torchvision

# ติดตั้ง vLLM Nightly ให้ตรงกับเวอร์ชัน CUDA
if [[ "$CUDA_VERSION" == 13* ]]; then
    echo "⚡ Installing vLLM for CUDA 13.0..."
    python3 -m pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu130 --extra-index-url https://download.pytorch.org/whl/cu130 || python3 -m pip install -U vllm --pre
else
    echo "⚡ Installing vLLM for CUDA 12.9..."
    python3 -m pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu129 --extra-index-url https://download.pytorch.org/whl/cu129 || python3 -m pip install -U vllm --pre
fi

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
echo "  bash run.sh"
