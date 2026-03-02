#!/usr/bin/env python3
"""vLLM V100 サーバ起動スクリプト

FlashAttention SM70 バックエンドで vLLM の OpenAI 互換 API サーバを起動する。
起動後、curl や OpenAI Python SDK でリクエストを送信できる。

使い方:
  # デフォルト (Llama-2-7B):
  docker run --gpus '"device=0"' -p 8000:8000 --ipc=host \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      flash_attn_v100 python examples/serve_vllm.py

  # カスタムモデル:
  docker run --gpus '"device=0"' -p 8000:8000 --ipc=host \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      flash_attn_v100 python examples/serve_vllm.py \
          --model meta-llama/Llama-2-7b-hf \
          --max-model-len 2048

リクエスト例:
  # Completions API:
  curl -s http://localhost:8000/v1/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "NousResearch/Llama-2-7b-hf", "prompt": "Hello!", "max_tokens": 64}'

  # Chat API:
  curl -s http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "NousResearch/Llama-2-7b-hf",
           "messages": [{"role": "user", "content": "What is FlashAttention?"}],
           "max_tokens": 128}'

  # OpenAI Python SDK:
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
  response = client.chat.completions.create(
      model="NousResearch/Llama-2-7b-hf",
      messages=[{"role": "user", "content": "Hello!"}],
      max_tokens=64,
  )
  print(response.choices[0].message.content)
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="vLLM V100 server with FlashAttention SM70")
    parser.add_argument("--model", default="NousResearch/Llama-2-7b-hf",
                        help="HuggingFace model name or path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=64,
                        help="KV cache block size (SM70: 64 recommended)")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", "half",                         # V100: FP16 only
        "--trust-remote-code",
        "--enforce-eager",                         # torch.compile off for stability
        "--disable-frontend-multiprocessing",
        "--block-size", str(args.block_size),      # SM70 paged KV: 64
    ]

    print("=" * 60)
    print("vLLM V100 Server (FlashAttention SM70)")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Port:       {args.port}")
    print(f"  Max SeqLen: {args.max_model_len}")
    print(f"  Block Size: {args.block_size}")
    print(f"  dtype:      float16 (V100 FP16 only)")
    print()
    print(f"  Command: {' '.join(cmd)}")
    print()
    print(f"  API endpoint: http://0.0.0.0:{args.port}/v1")
    print("=" * 60)
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
