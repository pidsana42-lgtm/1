#!/bin/bash
# สคริปต์รัน clean/prepare_data.py พร้อมโหมด LLM โดยโหลดไลบรารี CUDA อัตโนมัติ

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

# ปิดใช้งาน DeepGEMM (ป้องกันบั๊ก vLLM 0.21.x ในการโหลดโมเดล BF16)
export VLLM_USE_DEEP_GEMM=0

# ตรวจสอบตัวแปร HF_TOKEN (จำเป็นต้องใช้เนื่องจากโมเดลตระกูล Gemma เป็น Gated Model)
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN is not set. Please export HF_TOKEN=\"your_token\" before running."
    exit 1
fi

echo "🚀 Starting data cleaning pipeline with LLM-assisted reordering..."
python3 clean/prepare_data.py --use-llm
