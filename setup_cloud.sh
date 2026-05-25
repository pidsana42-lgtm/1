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
    # ล้าง vllm / torch / transformers และแพ็กเกจ CUDA ย่อยทั้งหมดที่เป็นของ cu129/cu130/cu13 ออกให้เกลี้ยงเพื่อป้องกันปัญหา Mismatch
    echo "🧹 Cleaning up existing PyTorch, vLLM, transformers, and old nvidia-cuda packages..."
    python3 -m pip uninstall -y vllm torch torchvision torchaudio flashinfer-python flashinfer-cubin apache-tvm-ffi tilelang transformers 2>/dev/null || true
    python3 -m pip uninstall -y nvidia-nccl-cu12 nvidia-nccl-cu13 nvidia-cudnn-cu12 nvidia-cudnn-cu13 nvidia-cublas-cu12 nvidia-cublas-cu13 nvidia-cuda-runtime-cu12 nvidia-cuda-runtime-cu13 nvidia-cuda-cupti-cu12 nvidia-cuda-cupti-cu13 nvidia-curand-cu12 nvidia-curand-cu13 nvidia-cusolver-cu12 nvidia-cusolver-cu13 nvidia-cusparse-cu12 nvidia-cusparse-cu13 nvidia-nvtx-cu12 nvidia-nvtx-cu13 nvidia-nvjitlink-cu12 nvidia-nvjitlink-cu13 nvidia-cuda-nvrtc-cu12 nvidia-cuda-nvrtc-cu13 2>/dev/null || true
    conda remove -y vllm torch torchvision torchaudio transformers 2>/dev/null || true

    # ล้าง Cache เพื่อคืนพื้นที่ดิสก์ (แก้ปัญหา No space left on device)
    echo "🧹 Purging package manager caches to free up disk space..."
    python3 -m pip cache purge 2>/dev/null || true
    conda clean -ay 2>/dev/null || true

    # ติดตั้ง vLLM Nightly Wheels (CUDA 12.9) และไลบรารีที่จำเป็นอื่นๆ
    echo "⚡ Installing vLLM Nightly Wheels (CUDA 12.9) and dependencies..."
    python3 -m pip install -U vllm --pre \
      --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
      --extra-index-url https://download.pytorch.org/whl/cu129
    python3 -m pip install --no-cache-dir "transformers>=5.9.0" accelerate pdf2image datasets huggingface_hub hf_transfer Pillow
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
