#!/bin/bash
# สคริปต์เตรียมเครื่องบน Cloud สำหรับ H100

echo "🚀 Installing system dependencies..."
sudo apt-get install -y poppler-utils 2>/dev/null || conda install -c conda-forge poppler -y

echo "📦 Installing Python dependencies..."
python3 -m pip install --upgrade pip

# ตรวจสอบว่ามี ROCm/HIP (AMD Instinct GPU) หรือไม่
IS_ROCM=$(python3 -c "
try:
    import torch
    print('True' if torch.version.hip else 'False')
except Exception:
    print('False')
" 2>/dev/null)

if [ "$IS_ROCM" = "True" ]; then
    echo "🔍 Detected AMD Instinct GPU (ROCm/HIP environment)"
    # ติดตั้งแพ็กเกจพื้นฐาน โดยไม่ติดตั้ง nvidia-cuda-runtime-cu12 เพื่อป้องกันปัญหาไลบรารีสับสน
    python3 -m pip install "numpy<2.0.0" transformers accelerate pdf2image datasets huggingface_hub hf_transfer Pillow torchvision
    echo "⚡ Installing vLLM for AMD ROCm (MI300X/MI250)..."
    python3 -m pip install -U vllm --extra-index-url https://download.pytorch.org/whl/rocm6.1 || python3 -m pip install -U vllm
else
    # กรณีเครื่องเป็น NVIDIA (CUDA)
    # ล้าง vllm / torch ตัวเดิมออกก่อนเสมอเพื่อแก้ปัญหา Driver และป้องกันการข้ามติดตั้ง (satisfied) ของ pip
    echo "🧹 Cleaning up existing PyTorch and vLLM packages to ensure stable installation..."
    pip uninstall -y vllm torch torchvision torchaudio flashinfer-python flashinfer-cubin apache-tvm-ffi tilelang 2>/dev/null || true
    conda remove -y vllm torch torchvision torchaudio 2>/dev/null || true

    # ติดตั้ง PyTorch + CUDA 12.1 ใหม่เพื่อแก้ปัญหา Driver
    echo "⚡ Installing stable PyTorch and vLLM for CUDA 12.1..."
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir vllm
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
