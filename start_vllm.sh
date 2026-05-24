vllm serve google/gemma-4-31b-it \
  --limit-mm-per-prompt '{"image": 1}' \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --trust-remote-code \
  --attention-config '{"backend": "FLASH_ATTN"}'
