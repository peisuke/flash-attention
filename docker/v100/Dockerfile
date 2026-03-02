# =============================================================================
# FlashAttention V100 (SM70) — flash_attn 推論 + vLLM サーバ
# =============================================================================
#
# ビルド:
#   docker build -t flash_attn_v100 -f docker/v100/Dockerfile .
#
# flash_attn 直接推論サンプル:
#   docker run --gpus '"device=0"' --rm flash_attn_v100 python examples/inference_flash_attn.py
#
# vLLM サーバ起動:
#   docker run --gpus '"device=0"' -p 8000:8000 --ipc=host \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       flash_attn_v100 python examples/serve_vllm.py
#
# vLLM サーバ起動 (カスタムモデル):
#   docker run --gpus '"device=0"' -p 8000:8000 --ipc=host \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       flash_attn_v100 python examples/serve_vllm.py --model meta-llama/Llama-2-7b-hf
#
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.11-py3

# ---------- Stage 1: flash-attn SM70 ビルド (~20分) ----------

WORKDIR /workspace/flash-attention
COPY setup.py README.md ./
COPY csrc/ csrc/
COPY flash_attn/ flash_attn/

RUN FLASH_ATTN_CUDA_ARCHS="70" pip install -e . && \
    python -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} OK')"

# ---------- Stage 2: vLLM v0.6.5 ソースビルド ----------

# vLLM の依存パッケージ (torch, flash-attn, triton は既存を使用)
RUN pip install --no-cache-dir \
    psutil sentencepiece "requests>=2.26.0" tqdm py-cpuinfo \
    "transformers>=4.45.0,<4.48.0" "tokenizers>=0.19.1,<0.22" protobuf \
    "fastapi>=0.107.0" aiohttp "openai>=1.0" "uvicorn[standard]" \
    "pydantic>=2.9" pillow \
    "prometheus_client>=0.18.0" "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" "typing_extensions>=4.10" "filelock>=3.16.1" \
    partial-json-parser pyzmq msgspec importlib_metadata \
    "mistral_common>=1.5.0" pyyaml einops \
    cloudpickle blake3 interegular jsonschema \
    accelerate safetensors huggingface_hub \
    "outlines>=0.0.43,<0.1" "lm-format-enforcer>=0.10.9,<0.11" \
    "compressed-tensors==0.8.0" "depyf==0.18.0" "gguf==0.10.0" \
    lark ray

RUN pip install --no-cache-dir setuptools-scm cmake ninja

# vLLM ソースからビルド (pip wheel は NGC PyTorch と ABI 不一致のため)
RUN git clone --depth 1 --branch v0.6.5 \
        https://github.com/vllm-project/vllm.git /tmp/vllm && \
    cd /tmp/vllm && \
    TORCH_CUDA_ARCH_LIST="7.0" \
    VLLM_TARGET_DEVICE=cuda \
    MAX_JOBS=4 \
    pip install --no-deps --no-build-isolation -e . && \
    rm -rf /tmp/vllm/.git

# ---------- Stage 3: vLLM SM70 パッチ ----------

# パッチ 1/3: SM80 チェックを SM70 に変更
RUN sed -i 's/has_device_capability(80)/has_device_capability(70)/' \
    /tmp/vllm/vllm/attention/selector.py && \
    grep -q 'has_device_capability(70)' /tmp/vllm/vllm/attention/selector.py && \
    echo "Patch 1/3: selector SM70 check OK"

# パッチ 2/3: vllm_flash_attn を SM70 対応 flash_attn のシムに置換
COPY docker/v100/vllm_flash_attn_shim.py /tmp/vllm/vllm/vllm_flash_attn/__init__.py
RUN python -c "\
from vllm.vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache; \
print('Patch 2/3: vllm_flash_attn shim OK')"

# パッチ 3/3: block_size=64 を許可 (SM70 paged KV cache 用)
RUN sed -i "s/choices=\[8, 16, 32, 256\]/choices=[8, 16, 32, 64, 256]/" \
    /tmp/vllm/vllm/engine/arg_utils.py && \
    grep -q '64' /tmp/vllm/vllm/engine/arg_utils.py && \
    echo "Patch 3/3: block_size=64 allowed"

# ---------- Stage 4: サンプルスクリプト配置 ----------

COPY docker/v100/examples/ /workspace/examples/

# 環境検証
RUN python -c "\
import torch, flash_attn, vllm; \
print(f'PyTorch:    {torch.__version__}'); \
print(f'flash_attn: {flash_attn.__version__}'); \
print(f'vLLM:       {vllm.__version__}'); \
print(f'CUDA:       {torch.version.cuda}'); \
print('Environment OK')"

WORKDIR /workspace
EXPOSE 8000
