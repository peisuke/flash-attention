# Add V100 (SM70/Volta) support for FlashAttention v2

PR作成URL: https://github.com/peisuke/flash-attention/compare/main...v100-sm70-support

## Summary

- Add V100 (SM70, Volta) GPU support for FlashAttention v2 forward and backward passes (FP16 only)
- Support head dimensions 32, 64, 96, 128, 192, 256 with SM70-optimized block size configurations
- Use SM70-compatible `SM70_8x8x4` MMA atom with standard `make_tiled_copy_B/C` (avoid `warpcontiguousN` OOB on SM70)
- Relax compute capability checks from SM80+ to SM70+ with FP16-only constraint
- Add comprehensive benchmarks: training (GPT fwd/bwd) and serving (vLLM Llama-2-7B)

## Key Technical Changes

| File | Change |
|------|--------|
| `kernel_traits.h` | SM70 MMA atom (`SM70_8x8x4_F32F16F16F32_TN`), SmemCopyAtom fallbacks, PdS layout config |
| `flash_bwd_kernel.h` | Use `make_tiled_copy_B/C` for SM70 (avoid warpcontiguousN OOB) |
| `flash_fwd_kernel.h` | SM70 guards for cp.async, LDSM, predicated gmem copies |
| `flash_fwd_launch_template.h` | V100 block size configs per head dimension |
| `flash_bwd_launch_template.h` | SM70 arch guard |
| `flash_api.cpp` | Runtime checks: SM70+ allowed, FP16-only on Volta, causal hdim>192 blocked, no dropout/ALiBi on SM70 bwd |
| `utils.h` | SM70 gemm K-loop fix (4:1 K_mma/K_copy ratio), register layout conversions |
| `mask.h` / `softmax.h` / `alibi.h` / `dropout.h` | SM70 compatibility guards |
| `setup.py` | Add SM70 to default CUDA arch list |
| `flash_attn_interface.py` | V100 block size heuristics |

## Benchmarks

### Training (GPT fwd/bwd on V100)

3 backends are near-identical in speed on V100 (within 1%). flash_attn is slightly fastest overall (backward 3-5ms faster).

| Model | seqlen | flash_attn Total(ms) | PyTorch SDPA | xFormers |
|-------|--------|---------------------|-------------|---------|
| GPT-small (124M) | 512 | **160.9** | 161.3 | 161.9 |
| GPT-small | 2048 | **197.4** | 199.3 | 199.3 |
| GPT-medium (350M) | 1024 | **207.7** | 208.1 | 208.6 |

### Serving (vLLM + Llama-2-7B on V100)

flash_attn with SM70 splitkv (kBlockN=64, page_block_size=64) is near-competitive with xFormers (page_block_size=16).

| Metric | FLASH_ATTN (bs=64) | XFORMERS (bs=16) | Ratio |
|--------|-----------|----------|-------|
| Avg TTFT | 976ms | 876ms | 1.10x |
| Avg TPOT | 37.8ms | 33.1ms | 1.14x |
| Avg Throughput | 133.2 tok/s | 154.8 tok/s | 1.16x |

At low concurrency (1-2 requests), flash_attn throughput matches xFormers. The ~16% gap at high concurrency comes from xFormers' finer KV cache granularity (block_size 16 vs 64).

Previous results with block_size=256 showed a 1.71x throughput gap — the splitkv kBlockN=64 fix reduced this to 1.16x.

> Note: On V100, PyTorch SDPA and xFormers dispatch to the same CUTLASS memory-efficient attention kernel (flash backend requires SM80+).

Full report: [`benchmarks/V100_BENCHMARK_REPORT.md`](benchmarks/V100_BENCHMARK_REPORT.md)

## Limitations on V100

- **FP16 only** (BF16 not supported on Volta hardware)
- **No dropout** in forward or backward (blocked at runtime)
- **No ALiBi** in backward (blocked at runtime)
- **hdim=256 causal** not supported (exceeds 96KB shared memory limit)
- **SplitKV** uses kBlockM=32/kNWarps=2 (64 threads) to avoid register spilling on SM70
- **SM75 (Turing)** explicitly blocked (64KB smem limit insufficient)

## Test Plan

- [x] Clean build: `FLASH_ATTN_CUDA_ARCHS="70" pip install -e .`
- [x] Forward numerical check: hdim 32-256 x seqlen 16-256 x causal/non-causal (max error < 0.001)
- [x] Backward numerical check: dQ/dK/dV against reference (max error < 0.003)
- [x] E2E NN comparison: 2-layer Transformer + CrossEntropyLoss, flash_attn vs PyTorch SDPA (18 configs)
- [x] PR review fixes: splitkv SM70, varlen dropout/causal guards, ALiBi bwd block, P-buffer smem fix, SM75 block, kvcache causal guard
- [x] Training benchmark: GPT-small/medium x 3 backends x 3 seqlens
- [x] Serving benchmark: vLLM Llama-2-7B x 2 backends x 4 workloads x 5 concurrency levels
