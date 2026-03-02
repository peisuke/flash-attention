#!/usr/bin/env python3
"""FlashAttention V100 (SM70) 推論サンプル

flash_attn API を直接使った推論の例。以下を実演:
  1. flash_attn_func — 基本的な attention 計算
  2. flash_attn_with_kvcache — KV cache を使ったインクリメンタルデコード
  3. HuggingFace モデルでの利用 — GPT-2 で実際にテキスト生成

使い方:
  docker run --gpus '"device=0"' --rm flash_attn_v100 python examples/inference_flash_attn.py
"""

import torch
import time


def example_flash_attn_func():
    """1. flash_attn_func: 基本的な attention 計算"""
    from flash_attn import flash_attn_func

    print("=" * 60)
    print("Example 1: flash_attn_func (basic attention)")
    print("=" * 60)

    batch_size = 2
    seqlen = 256
    nheads = 12
    headdim = 64

    # Q, K, V を生成 (V100 は FP16 のみ)
    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16, device="cuda")

    # non-causal attention
    out = flash_attn_func(q, k, v, causal=False)
    print(f"  Non-causal: input={q.shape}, output={out.shape}")

    # causal attention
    out_causal = flash_attn_func(q, k, v, causal=True)
    print(f"  Causal:     input={q.shape}, output={out_causal.shape}")

    # GQA (Grouped Query Attention): Q=12 heads, KV=4 heads
    nheads_kv = 4
    k_gqa = torch.randn(batch_size, seqlen, nheads_kv, headdim, dtype=torch.float16, device="cuda")
    v_gqa = torch.randn(batch_size, seqlen, nheads_kv, headdim, dtype=torch.float16, device="cuda")
    out_gqa = flash_attn_func(q, k_gqa, v_gqa, causal=True)
    print(f"  GQA:        Q heads={nheads}, KV heads={nheads_kv}, output={out_gqa.shape}")

    # PyTorch reference と比較
    scale = headdim ** -0.5
    attn_ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
    ).transpose(1, 2)
    max_diff = (out.float() - attn_ref.float()).abs().max().item()
    print(f"  Max diff vs PyTorch SDPA: {max_diff:.6f}")
    print()


def example_kvcache():
    """2. flash_attn_with_kvcache: インクリメンタルデコード"""
    from flash_attn import flash_attn_with_kvcache

    print("=" * 60)
    print("Example 2: flash_attn_with_kvcache (incremental decode)")
    print("=" * 60)

    batch_size = 1
    max_seqlen = 512
    nheads = 12
    nheads_kv = 12
    headdim = 64

    # KV cache を事前に確保
    k_cache = torch.zeros(batch_size, max_seqlen, nheads_kv, headdim, dtype=torch.float16, device="cuda")
    v_cache = torch.zeros(batch_size, max_seqlen, nheads_kv, headdim, dtype=torch.float16, device="cuda")

    # Prefill: 最初の 32 トークンを処理
    prefill_len = 32
    q_prefill = torch.randn(batch_size, prefill_len, nheads, headdim, dtype=torch.float16, device="cuda")
    k_new = torch.randn(batch_size, prefill_len, nheads_kv, headdim, dtype=torch.float16, device="cuda")
    v_new = torch.randn(batch_size, prefill_len, nheads_kv, headdim, dtype=torch.float16, device="cuda")
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

    out_prefill = flash_attn_with_kvcache(
        q_prefill, k_cache, v_cache,
        k=k_new, v=v_new,
        cache_seqlens=cache_seqlens,
        causal=True,
    )
    cache_seqlens += prefill_len
    print(f"  Prefill: {prefill_len} tokens -> output={out_prefill.shape}")

    # Decode: 1 トークンずつ生成
    num_decode_steps = 5
    for step in range(num_decode_steps):
        q_decode = torch.randn(batch_size, 1, nheads, headdim, dtype=torch.float16, device="cuda")
        k_step = torch.randn(batch_size, 1, nheads_kv, headdim, dtype=torch.float16, device="cuda")
        v_step = torch.randn(batch_size, 1, nheads_kv, headdim, dtype=torch.float16, device="cuda")

        out_decode = flash_attn_with_kvcache(
            q_decode, k_cache, v_cache,
            k=k_step, v=v_step,
            cache_seqlens=cache_seqlens,
            causal=True,
        )
        cache_seqlens += 1

    print(f"  Decode:  {num_decode_steps} steps -> cache length={cache_seqlens.item()}")
    print()


def example_huggingface():
    """3. HuggingFace モデルでの推論 (GPT-2)"""
    print("=" * 60)
    print("Example 3: HuggingFace GPT-2 text generation")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  SKIP: transformers not installed")
        return

    model_name = "gpt2"
    print(f"  Loading {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # V100 は FP16 のみ
        attn_implementation="flash_attention_2",  # flash_attn を使用
    ).cuda()

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"  Prompt: {prompt!r}")

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

    print(f"  Generated ({num_tokens} tokens, {elapsed:.2f}s):")
    print(f"  {generated}")
    print()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    import flash_attn
    print(f"flash_attn: {flash_attn.__version__}")
    print()

    example_flash_attn_func()
    example_kvcache()
    example_huggingface()

    print("All examples completed.")


if __name__ == "__main__":
    main()
