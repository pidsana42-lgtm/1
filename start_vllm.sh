export VLLM_ATTENTION_BACKEND=TORCH_SDPA
python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31b-it \
  --limit-mm-per-prompt '{"image": 1}' \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --trust-remote-code
