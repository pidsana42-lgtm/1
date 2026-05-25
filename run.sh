#!/bin/bash
# สคริปต์รัน vllm_inference.py โดยโหลดไลบรารี CUDA จาก python packages อัตโนมัติ

echo "⚙️ Setting up CUDA library paths from python packages..."
# ค้นหาไดเรกทอรี nvidia/*/lib ทั้งหมดใน environment site-packages
NVIDIA_LIBS=$(python3 -c "
import glob, sys
try:
    sp_path = next(p for p in sys.path if 'site-packages' in p)
    paths = glob.glob(sp_path + '/nvidia/*/lib')
    print(':'.join(paths))
except Exception as e:
    pass
" 2>/dev/null)

if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
    echo "✅ CUDA libraries linked successfully!"
else
    echo "⚠️ Warning: Could not auto-detect nvidia libraries in site-packages."
fi

# ปิดใช้งาน DeepGEMM (บั๊กของ vLLM 0.21.x ที่พยายามเช็ก FP8 kernel บนโมเดล bf16)
export VLLM_USE_DEEP_GEMM=0

# ตรวจสอบตัวแปร HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN is not set. Please export HF_TOKEN=\"your_token\" before running."
    exit 1
fi

echo "🚀 Starting inference pipeline..."
python3 vllm_inference.py
