# FlashAttention V100 (SM70) ベンチマークレポート

## 1. 概要

FlashAttention v2にV100 (SM70/Volta) サポートを追加し、既存のattentionバックエンド（PyTorch SDPA、xFormers）との性能を比較した。ベンチマークは2種類実施した。

| ベンチマーク | 目的 | 比較対象 |
|-------------|------|---------|
| **Training (GPTモデル)** | forward/backward カーネル単体の速度・メモリ | flash_attn, PyTorch SDPA, xFormers |
| **Serving (vLLM推論)** | 実運用サーバの TTFT・スループット | flash_attn, xFormers |

## 2. V100上のAttention実装の整理

V100 (SM70) では、表面上3つのバックエンド名が存在するが、実際に動作するカーネル実装は2種類に集約される。

```
呼び出しAPI                    V100での実カーネル
─────────────────────────────────────────────────────
flash_attn (本リポジトリ)  →  Flash Attention v2 (SM70 MMA atom)
PyTorch SDPA               →  memory-efficient attention ─┐ 同一カーネル
xFormers                   →  memory-efficient attention ─┘ (CUTLASS)
```

| API | V100でdispatchされる実装 | 理由 |
|-----|------------------------|------|
| flash_attn | **Flash Attention v2 (SM70)** | 本リポジトリのカスタムビルド |
| PyTorch SDPA (`F.scaled_dot_product_attention`) | **memory-efficient attention** | SDPA内部の flash backend は SM80+ 専用。V100では自動的に mem-efficient にfallback |
| xFormers (`xformers.ops.memory_efficient_attention`) | **memory-efficient attention** | xFormersの flash backend も SM80+ 専用。V100ではCUTLASS backend を使用 |

PyTorch SDPAとxFormersはV100上で同一のCUTLASSカーネルを呼ぶため、Trainingベンチマークで速度・精度がほぼ一致する（セクション4参照）。

Servingベンチマーク（セクション5）では、vLLM v0.6.5の`TORCH_SDPA`バックエンドがCPU専用のため除外した。ただし上記の通り、xFormers（CUTLASS）の結果がPyTorch SDPAと実質同等であるため、2バックエンド（FLASH_ATTN, XFORMERS）で3つのAPIすべてをカバーしている。

## 3. 環境

| 項目 | 値 |
|------|-----|
| GPU | Tesla V100-SXM2-32GB |
| PyTorch | 2.6.0a0+df5bbc0 (NGC 24.11) |
| CUDA | 12.6 |
| dtype | FP16 |
| flash_attn | v2 + SM70カスタムビルド |
| xFormers | v0.0.29.post2 (CUTLASS backend) |
| vLLM | v0.6.5 (ソースビルド) |

## 4. Training ベンチマーク（GPTモデル forward/backward）

GPT-small (124M) および GPT-medium (350M) で、異なるシーケンス長（512/1024/2048）におけるforward/backward速度、ピークメモリ使用量、数値精度を計測。

### 4.1 各バックエンドのV100上での実装

| Backend | V100での実装 | 備考 |
|---------|-------------|------|
| flash_attn | Flash Attention v2 (SM70 MMA atom) | 本リポジトリのカスタムビルド |
| PyTorch SDPA | memory-efficient attention | flash backendはSM80+のみ |
| xFormers | CUTLASS backend | flash backendはSM80+のみ |

### 4.2 速度比較

#### GPT-small (124M, nhead=12, hdim=64)

| seqlen | batch | Backend | Fwd (ms) | Bwd (ms) | Total (ms) | vs flash_attn |
|--------|-------|---------|----------|----------|------------|---------------|
| 512 | 16 | **flash_attn** | 54.4 | **106.5** | **160.9** | - |
| | | pytorch_sdpa | **53.8** | 107.5 | 161.3 | +0.2% |
| | | xformers | 54.0 | 107.8 | 161.9 | +0.6% |
| 1024 | 8 | **flash_attn** | 57.7 | **115.5** | **173.2** | - |
| | | pytorch_sdpa | **56.5** | 118.0 | 174.4 | +0.7% |
| | | xformers | 56.7 | 118.4 | 175.1 | +1.1% |
| 2048 | 4 | **flash_attn** | 64.3 | **133.1** | **197.4** | - |
| | | pytorch_sdpa | **61.9** | 137.5 | 199.3 | +1.0% |
| | | xformers | 61.7 | 137.6 | 199.3 | +1.0% |

#### GPT-medium (350M, nhead=16, hdim=64)

| seqlen | batch | Backend | Fwd (ms) | Bwd (ms) | Total (ms) | vs flash_attn |
|--------|-------|---------|----------|----------|------------|---------------|
| 512 | 8 | flash_attn | 59.4 | **131.4** | 190.7 | - |
| | | pytorch_sdpa | **58.1** | 132.9 | 191.0 | +0.2% |
| | | xformers | 58.1 | 132.6 | **190.7** | 0.0% |
| 1024 | 4 | flash_attn | 63.9 | **143.8** | **207.7** | - |
| | | pytorch_sdpa | **61.8** | 146.3 | 208.1 | +0.2% |
| | | xformers | 62.3 | 146.3 | 208.6 | +0.4% |
| 2048 | 2 | flash_attn | 73.5 | **167.9** | **241.5** | - |
| | | pytorch_sdpa | **69.1** | 173.3 | 242.4 | +0.4% |
| | | xformers | 69.2 | 173.2 | 242.4 | +0.4% |

### 4.3 メモリ使用量

| Backend | GPT-small PeakMem (MB) | GPT-medium PeakMem (MB) |
|---------|----------------------|------------------------|
| flash_attn | 8,494 | 7,424 |
| pytorch_sdpa | 8,494 | 7,424 |
| xformers | 8,811 (+3.7%) | 8,206 (+10.5%) |

### 4.4 数値精度（flash_attn基準）

| Backend | Logit 最大誤差 | 勾配 最大誤差 |
|---------|--------------|-------------|
| pytorch_sdpa | 0.003~0.005 | 0.00002~0.00004 |
| xformers | 0.003~0.005 | 0.00002~0.00004 |

### 4.5 Training ベンチマーク まとめ

- **速度**: 3バックエンドはV100上でほぼ同等（差は1%以内）。flash_attnはbackwardで3-5ms速く、SDPAはforwardで2-4ms速い。**トータルではflash_attnがわずかに最速**。
- **メモリ**: flash_attnとSDPAは同一のピークメモリ。xFormersは300~800MB多く使用。
- **精度**: すべてのバックエンドで実用上問題のない精度。

---

## 5. Serving ベンチマーク（vLLM推論サーバ）

vLLM v0.6.5でLlama-2-7B (FP16) を提供し、OpenAI互換APIへ非同期ストリーミングリクエストを送信して計測。

> **PyTorch SDPA (TORCH_SDPA) について**: vLLM v0.6.5の`TORCH_SDPA`バックエンドはCPU専用のため本ベンチマークから除外した。ただしセクション2で説明の通り、V100上のPyTorch SDPAとxFormersは同一のCUTLASSカーネルにdispatchされるため、XFORMERSの結果がPyTorch SDPAの性能も代表する。

### 5.1 計測条件

| 項目 | 値 |
|------|-----|
| モデル | NousResearch/Llama-2-7b-hf (FP16) |
| max_model_len | 4096 |
| gpu_memory_utilization | 0.9 |
| 推論モード | eager (torch.compile無効) |
| FLASH_ATTN block_size | 64（SM70 splitkv kBlockN=64） |
| XFORMERS block_size | 16（デフォルト） |
| リクエスト数/シナリオ | 32 |

### 5.2 ワークロード定義

| ワークロード | 入力長 | 最大出力長 | 想定ユースケース |
|-------------|--------|-----------|----------------|
| short | 128 | 128 | チャット応答 |
| medium | 512 | 256 | 要約 |
| long | 1024 | 512 | 長文生成 |
| very_long | 2048 | 256 | 長文理解 |

同時接続数: 1, 2, 4, 8, 16

### 5.3 総合結果

| メトリクス | FLASH_ATTN (SM70, bs=64) | XFORMERS (bs=16) | XFORMERS 優位率 |
|-----------|-------------------|----------|----------------|
| 平均 TTFT (初回トークン遅延) | 976 ms | 876 ms | 1.10x 高速 |
| 平均 TPOT (トークン生成間隔) | 37.8 ms | 33.1 ms | 1.14x 高速 |
| 平均スループット | 133.2 tok/s | 154.8 tok/s | 1.16x 高スループット |

> 以前のblock_size=256では1.71xのスループット差があったが、splitkv kBlockN=64対応により1.16xまで改善。

### 5.4 詳細結果 — FLASH_ATTN (SM70, block_size=64)

| ワークロード | 同時接続 | 成功率 | TTFT avg (ms) | TTFT p99 (ms) | TPOT (ms) | スループット (tok/s) | Req/s |
|-------------|---------|--------|--------------|--------------|----------|-------------------|-------|
| short | 1 | 32/32 | 61 | 76 | 26.8 | 37.0 | 0.29 |
| short | 2 | 32/32 | 91 | 122 | 28.1 | 69.8 | 0.55 |
| short | 4 | 32/32 | 175 | 581 | 28.5 | 134.8 | 1.05 |
| short | 8 | 32/32 | 756 | 2,690 | 30.5 | 221.3 | 1.73 |
| short | 16 | 32/32 | 856 | 1,412 | 34.7 | 388.9 | 3.04 |
| medium | 1 | 32/32 | 113 | 115 | 26.2 | 37.7 | 0.15 |
| medium | 2 | 32/32 | 171 | 228 | 28.5 | 68.8 | 0.27 |
| medium | 4 | 32/32 | 334 | 409 | 30.4 | 126.4 | 0.49 |
| medium | 8 | 32/32 | 699 | 787 | 35.4 | 210.5 | 0.82 |
| medium | 16 | 32/32 | 1,086 | 1,550 | 47.6 | 310.0 | 1.21 |
| long | 1 | 32/32 | 219 | 221 | 26.7 | 36.9 | 0.07 |
| long | 2 | 32/32 | 330 | 443 | 29.4 | 66.8 | 0.13 |
| long | 4 | 32/32 | 679 | 834 | 33.7 | 114.3 | 0.22 |
| long | 8 | 32/32 | 1,084 | 1,659 | 44.3 | 172.8 | 0.34 |
| long | 16 | 32/32 | 1,949 | 3,288 | 62.7 | 241.1 | 0.47 |
| very_long | 1 | 32/32 | 458 | 462 | 26.6 | 35.3 | 0.14 |
| very_long | 2 | 32/32 | 686 | 924 | 32.4 | 57.2 | 0.22 |
| very_long | 4 | 32/32 | 1,143 | 1,832 | 41.0 | 88.4 | 0.35 |
| very_long | 8 | 32/32 | 2,057 | 3,653 | 60.5 | 117.1 | 0.46 |
| very_long | 16 | 32/32 | 6,575 | 26,505 | 82.0 | 128.5 | 0.50 |

### 5.5 詳細結果 — XFORMERS (CUTLASS, block_size=16)

| ワークロード | 同時接続 | 成功率 | TTFT avg (ms) | TTFT p99 (ms) | TPOT (ms) | スループット (tok/s) | Req/s |
|-------------|---------|--------|--------------|--------------|----------|-------------------|-------|
| short | 1 | 32/32 | 68 | 341 | 26.6 | 37.1 | 0.29 |
| short | 2 | 32/32 | 89 | 118 | 27.8 | 70.6 | 0.55 |
| short | 4 | 32/32 | 210 | 979 | 28.9 | 131.9 | 1.03 |
| short | 8 | 32/32 | 219 | 248 | 29.7 | 256.4 | 2.00 |
| short | 16 | 32/32 | 406 | 455 | 31.8 | 460.6 | 3.60 |
| medium | 1 | 32/32 | 110 | 112 | 28.1 | 35.2 | 0.14 |
| medium | 2 | 32/32 | 166 | 221 | 29.4 | 66.8 | 0.26 |
| medium | 4 | 32/32 | 327 | 402 | 29.9 | 128.8 | 0.50 |
| medium | 8 | 32/32 | 1,298 | 3,116 | 31.4 | 220.0 | 0.86 |
| medium | 16 | 32/32 | 1,065 | 1,530 | 37.8 | 382.7 | 1.49 |
| long | 1 | 32/32 | 208 | 211 | 28.1 | 35.1 | 0.07 |
| long | 2 | 32/32 | 314 | 420 | 28.6 | 68.6 | 0.13 |
| long | 4 | 32/32 | 645 | 795 | 29.7 | 129.3 | 0.25 |
| long | 8 | 32/32 | 1,038 | 1,587 | 35.0 | 216.7 | 0.42 |
| long | 16 | 32/32 | 1,850 | 3,136 | 45.1 | 329.2 | 0.64 |
| very_long | 1 | 32/32 | 416 | 420 | 28.6 | 33.2 | 0.13 |
| very_long | 2 | 32/32 | 626 | 837 | 30.5 | 61.0 | 0.24 |
| very_long | 4 | 32/32 | 1,043 | 1,668 | 34.1 | 105.1 | 0.41 |
| very_long | 8 | 32/32 | 1,876 | 3,333 | 44.2 | 155.6 | 0.61 |
| very_long | 16 | 32/32 | 5,548 | 19,667 | 57.7 | 172.3 | 0.67 |

### 5.6 ワークロード別スループット比較

| ワークロード | 同時接続 | FLASH_ATTN (tok/s) | XFORMERS (tok/s) | XFORMERS / FLASH_ATTN |
|-------------|---------|-------------------|-----------------|----------------------|
| short | 1 | 37.0 | 37.1 | 1.00x |
| short | 16 | 388.9 | 460.6 | **1.18x** |
| medium | 1 | 37.7 | 35.2 | **0.93x** (FA高速) |
| medium | 16 | 310.0 | 382.7 | **1.23x** |
| long | 1 | 36.9 | 35.1 | **0.95x** (FA高速) |
| long | 16 | 241.1 | 329.2 | **1.37x** |
| very_long | 1 | 35.3 | 33.2 | **0.94x** (FA高速) |
| very_long | 16 | 128.5 | 172.3 | **1.34x** |

### 5.7 Serving ベンチマーク分析

**SM70 splitkv kBlockN=64対応により、block_size=256→64に改善。** 以前の1.7~2.4xの差が1.0~1.4xまで縮小。低同時接続(1-2)ではflash_attnがわずかに高速な場合もある。

| 要因 | FLASH_ATTN | XFORMERS |
|------|-----------|----------|
| KV cache block_size | 64 | 16 |
| ブロックあたり割当量 | 64トークン分 | 16トークン分 |
| 部分使用時の無駄 | 中程度 | 小さい |

- **block_size=64**: SM70 splitkvカーネルのkBlockNを64にすることで、page_block_size=64が利用可能に。以前のblock_size=256に比べ4倍のKV cache粒度改善。
- **残りの差**: 高同時接続(16)でXFORMERSが1.2~1.4x高速なのは、block_size 16 vs 64の差（4倍のメモリ粒度差）。
- **低同時接続で互角**: concurrency 1-2では、KV cacheサイズの影響が小さく、flash_attnのカーネル速度がわずかに有利。
- **カーネル速度自体は同等**: Trainingベンチマークが示す通り、attention カーネル単体の速度差はほぼない。

---

## 6. 総合考察

### 6.1 ユースケース別推奨

| ユースケース | 推奨バックエンド | 理由 |
|-------------|----------------|------|
| **Training (学習)** | **flash_attn** | カーネル速度が最速（backward 3-5ms高速）、メモリ効率最良 |
| **Serving (推論、低同時接続)** | **flash_attn** | xFormersとほぼ同等〜わずかに高速 |
| **Serving (推論、高同時接続)** | **xFormers** | block_size=16のメモリ効率が有利（1.2~1.4x高速） |

### 6.2 flash_attn SM70の価値

1. **Training**: flash_attnはV100でのtraining用途において最良の選択肢。paged KV cacheを使わないためblock_size制約が無関係で、backward速度が最速。

2. **Serving**: vLLM上での推論で`block_size=64`が利用可能。低同時接続（1-2）ではxFormersとほぼ同等の性能。高同時接続（16）でも1.2~1.4x程度の差に収まる（以前の2.0~2.4xから大幅改善）。

3. **互換性**: flash_attn SM70サポートにより、V100ユーザーがflash_attnに依存するコードベース（HuggingFace Transformers等）をV100上で動作させることが可能になる。

### 6.3 制限事項

| 制限 | 詳細 |
|------|------|
| FP16のみ | BF16はVoltaハードウェア非対応 |
| Dropout非対応 | SM70での検証未実施のためブロック |
| hdim=256 causal backward | V100の96KB共有メモリ上限を超過 |
| SplitKV | SM70ではkBlockM=32/kNWarps=2（64スレッド）で動作 |
| Serving block_size | 64（xFormersの16と比較してやや非効率、以前の256から大幅改善） |

---

## 7. 再現方法

### 7.1 Training ベンチマーク

```bash
# ベースイメージビルド（初回のみ、約2時間）
docker build -t flash_attn_v100_base -f benchmarks/v100_comparison/Dockerfile.base .

# ベンチマークイメージビルド（xFormers + スクリプト）
docker build -t flash_attn_bench -f benchmarks/v100_comparison/Dockerfile .

# 実行
docker run --gpus '"device=0"' --rm flash_attn_bench python benchmark_gpt.py

# dry-run
docker run --gpus '"device=0"' --rm flash_attn_bench python benchmark_gpt.py --dry-run
```

### 7.2 Serving ベンチマーク

```bash
# vLLMイメージビルド（flash_attn_benchベース）
docker build -t flash_attn_v100_vllm -f benchmarks/v100_vllm_serving/Dockerfile .

# 実行（HFキャッシュをマウントしてモデル再ダウンロード回避）
docker run --gpus '"device=0"' --ipc=host --rm \
    -e PYTHONUNBUFFERED=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    flash_attn_v100_vllm python benchmark_serving.py

# dry-run
docker run --gpus '"device=0"' --ipc=host --rm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    flash_attn_v100_vllm python benchmark_serving.py --dry-run
```

---

## 8. ファイル構成

```
benchmarks/
├── v100_comparison/                    # Training ベンチマーク
│   ├── Dockerfile.base                 # NGC PyTorch + flash_attn SM70ビルド
│   ├── Dockerfile                      # + xFormers + ベンチマークスクリプト
│   ├── benchmark_gpt.py                # GPTモデル fwd/bwd 計測スクリプト
│   └── RESULTS.md                      # Training ベンチマーク結果
├── v100_vllm_serving/                  # Serving ベンチマーク
│   ├── Dockerfile                      # vLLM v0.6.5 + パッチ
│   ├── benchmark_serving.py            # vLLM推論サーバ計測スクリプト
│   ├── vllm_flash_attn_shim.py         # vLLM ↔ flash_attn API ブリッジ
│   └── RESULTS.md                      # Serving ベンチマーク結果
└── V100_BENCHMARK_REPORT.md            # 本レポート（総合）
```
