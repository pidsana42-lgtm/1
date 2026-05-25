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
    # ตรวจสอบเวอร์ชัน CUDA ของ PyTorch ในเครื่อง
    CUDA_VERSION=$(python3 -c "
    try:
        import torch
        v = torch.version.cuda
        if v:
            print(v)
        else:
            print('12.1')
    except Exception as e:
        print('12.1')
    " 2>/dev/null)

    echo "🔍 Detected CUDA version from PyTorch: $CUDA_VERSION"

    # ติดตั้งแพ็กเกจพื้นฐาน (รวมถึง nvidia-cuda-runtime-cu12 เพื่อให้ไลบรารี libcudart.so.12 พร้อมใช้งานเสมอ)
    python3 -m pip install "numpy<2.0.0" transformers accelerate pdf2image datasets huggingface_hub hf_transfer Pillow torchvision nvidia-cuda-runtime-cu12

    # พยายามติดตั้ง vLLM เวอร์ชันปกติที่เข้ากันได้กับระบบก่อนเพื่อหลีกเลี่ยงปัญหา Driver เก่า
    echo "⚡ Installing stable vLLM..."
    if python3 -m pip install vllm; then
        echo "✅ Installed stable vLLM successfully!"
    else
        echo "⚠️ Stable vLLM install failed, trying version-specific fallback..."
        if [[ "$CUDA_VERSION" == 12.4* ]]; then
            echo "⚡ Installing vLLM for CUDA 12.4..."
            python3 -m pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124
        elif [[ "$CUDA_VERSION" == 12.1* ]]; then
            echo "⚡ Installing vLLM for CUDA 12.1..."
            python3 -m pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
        else
            echo "⚡ Installing default vLLM..."
            python3 -m pip install vllm
        fi
    fi
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
