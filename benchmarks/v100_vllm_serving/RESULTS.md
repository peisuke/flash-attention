# vLLM Serving Benchmark Results — V100 (SM70)

## Setup
- **GPU**: NVIDIA V100 32GB
- **Model**: NousResearch/Llama-2-7b-hf (FP16)
- **Framework**: vLLM v0.6.5
- **PyTorch**: 2.6.0a0+df5bbc0 (NGC 24.11)
- **CUDA**: 12.6
- **Settings**: `--max-model-len 4096`, `--gpu-memory-utilization 0.9`, `--enforce-eager`
- **Flash Attention**: SM70-compatible build with `--block-size 64` (SM70 splitkv kBlockN=64)
- **xFormers**: v0.0.29.post2 (CUTLASS backend, block_size=16)
- **Requests per scenario**: 32

## Summary

| Metric | FLASH_ATTN (bs=64) | XFORMERS (bs=16) | XFORMERS Speedup |
|--------|-------------------|----------|-------------------|
| Avg TTFT | 976ms | 876ms | 1.10x |
| Avg TPOT | 37.8ms | 33.1ms | 1.14x |
| Avg Throughput | 133.2 tok/s | 154.8 tok/s | 1.16x |

Previous results with `block_size=256`: TPOT 1.74x gap, Throughput 1.71x gap.
With `block_size=64`: TPOT 1.14x gap, Throughput 1.16x gap — **major improvement**.

## Detailed Results

### FLASH_ATTN (SM70, block_size=64)

| Workload | Conc | OK | TTFT avg(ms) | TTFT p99(ms) | TPOT(ms) | Tput(tok/s) | Req/s |
|----------|------|----|--------------|--------------|----------|-------------|-------|
| short (128/128) | 1 | 32/32 | 61 | 76 | 26.8 | 37.0 | 0.29 |
| short (128/128) | 2 | 32/32 | 91 | 122 | 28.1 | 69.8 | 0.55 |
| short (128/128) | 4 | 32/32 | 175 | 581 | 28.5 | 134.8 | 1.05 |
| short (128/128) | 8 | 32/32 | 756 | 2690 | 30.5 | 221.3 | 1.73 |
| short (128/128) | 16 | 32/32 | 856 | 1412 | 34.7 | 388.9 | 3.04 |
| medium (512/256) | 1 | 32/32 | 113 | 115 | 26.2 | 37.7 | 0.15 |
| medium (512/256) | 2 | 32/32 | 171 | 228 | 28.5 | 68.8 | 0.27 |
| medium (512/256) | 4 | 32/32 | 334 | 409 | 30.4 | 126.4 | 0.49 |
| medium (512/256) | 8 | 32/32 | 699 | 787 | 35.4 | 210.5 | 0.82 |
| medium (512/256) | 16 | 32/32 | 1086 | 1550 | 47.6 | 310.0 | 1.21 |
| long (1024/512) | 1 | 32/32 | 219 | 221 | 26.7 | 36.9 | 0.07 |
| long (1024/512) | 2 | 32/32 | 330 | 443 | 29.4 | 66.8 | 0.13 |
| long (1024/512) | 4 | 32/32 | 679 | 834 | 33.7 | 114.3 | 0.22 |
| long (1024/512) | 8 | 32/32 | 1084 | 1659 | 44.3 | 172.8 | 0.34 |
| long (1024/512) | 16 | 32/32 | 1949 | 3288 | 62.7 | 241.1 | 0.47 |
| very_long (2048/256) | 1 | 32/32 | 458 | 462 | 26.6 | 35.3 | 0.14 |
| very_long (2048/256) | 2 | 32/32 | 686 | 924 | 32.4 | 57.2 | 0.22 |
| very_long (2048/256) | 4 | 32/32 | 1143 | 1832 | 41.0 | 88.4 | 0.35 |
| very_long (2048/256) | 8 | 32/32 | 2057 | 3653 | 60.5 | 117.1 | 0.46 |
| very_long (2048/256) | 16 | 32/32 | 6575 | 26505 | 82.0 | 128.5 | 0.50 |

### XFORMERS (CUTLASS, block_size=16)

| Workload | Conc | OK | TTFT avg(ms) | TTFT p99(ms) | TPOT(ms) | Tput(tok/s) | Req/s |
|----------|------|----|--------------|--------------|----------|-------------|-------|
| short (128/128) | 1 | 32/32 | 68 | 341 | 26.6 | 37.1 | 0.29 |
| short (128/128) | 2 | 32/32 | 89 | 118 | 27.8 | 70.6 | 0.55 |
| short (128/128) | 4 | 32/32 | 210 | 979 | 28.9 | 131.9 | 1.03 |
| short (128/128) | 8 | 32/32 | 219 | 248 | 29.7 | 256.4 | 2.00 |
| short (128/128) | 16 | 32/32 | 406 | 455 | 31.8 | 460.6 | 3.60 |
| medium (512/256) | 1 | 32/32 | 110 | 112 | 28.1 | 35.2 | 0.14 |
| medium (512/256) | 2 | 32/32 | 166 | 221 | 29.4 | 66.8 | 0.26 |
| medium (512/256) | 4 | 32/32 | 327 | 402 | 29.9 | 128.8 | 0.50 |
| medium (512/256) | 8 | 32/32 | 1298 | 3116 | 31.4 | 220.0 | 0.86 |
| medium (512/256) | 16 | 32/32 | 1065 | 1530 | 37.8 | 382.7 | 1.49 |
| long (1024/512) | 1 | 32/32 | 208 | 211 | 28.1 | 35.1 | 0.07 |
| long (1024/512) | 2 | 32/32 | 314 | 420 | 28.6 | 68.6 | 0.13 |
| long (1024/512) | 4 | 32/32 | 645 | 795 | 29.7 | 129.3 | 0.25 |
| long (1024/512) | 8 | 32/32 | 1038 | 1587 | 35.0 | 216.7 | 0.42 |
| long (1024/512) | 16 | 32/32 | 1850 | 3136 | 45.1 | 329.2 | 0.64 |
| very_long (2048/256) | 1 | 32/32 | 416 | 420 | 28.6 | 33.2 | 0.13 |
| very_long (2048/256) | 2 | 32/32 | 626 | 837 | 30.5 | 61.0 | 0.24 |
| very_long (2048/256) | 4 | 32/32 | 1043 | 1668 | 34.1 | 105.1 | 0.41 |
| very_long (2048/256) | 8 | 32/32 | 1876 | 3333 | 44.2 | 155.6 | 0.61 |
| very_long (2048/256) | 16 | 32/32 | 5548 | 19667 | 57.7 | 172.3 | 0.67 |

## Analysis

### block_size=64 vs block_size=256 improvement

The SM70 splitkv kernel with kBlockN=64 allows `page_block_size=64` (previously required 256).
This reduced the throughput gap from **1.71x to 1.16x** (averaged across all scenarios).

| Concurrency | FLASH_ATTN tok/s (bs=64) | XFORMERS tok/s (bs=16) | Ratio |
|-------------|-------------------------|----------------------|-------|
| 1 (avg) | 36.7 | 35.2 | **0.96x (FA faster)** |
| 2 (avg) | 65.7 | 66.8 | 1.02x |
| 4 (avg) | 116.0 | 123.8 | 1.07x |
| 8 (avg) | 180.4 | 212.2 | 1.18x |
| 16 (avg) | 267.1 | 336.2 | 1.26x |

### Remaining gap at high concurrency

At concurrency 16, xFormers is still ~1.2-1.4x faster due to block_size 16 vs 64 (4x finer
KV cache granularity). This means xFormers wastes less memory on partially-filled cache blocks,
allowing more concurrent sequences to fit in GPU memory.

### Key Takeaway

With the splitkv kBlockN=64 fix, flash_attn SM70 is now practical for vLLM serving:
- **Low concurrency (1-2)**: flash_attn matches or slightly beats xFormers
- **High concurrency (8-16)**: xFormers has ~20% advantage from finer KV cache granularity
- **Training**: flash_attn remains the best choice (no paged KV, fastest backward)
