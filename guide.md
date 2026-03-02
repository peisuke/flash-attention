# FlashAttention V100 (SM70) ビルド & vLLM 利用ガイド

FlashAttention v2 を V100 (SM70/Volta) GPU でビルドし、vLLM で推論サーバとして利用するための手順書。

## 前提条件

- NVIDIA V100 GPU (compute capability 7.0)
- CUDA 12.x
- PyTorch 2.x (CUDA 12 対応ビルド)
- Python 3.10+

## V100 での制限事項

| 制限 | 詳細 |
|------|------|
| FP16 のみ | BF16 は Volta ハードウェア非対応 |
| Dropout 非対応 | forward/backward ともにブロック |
| ALiBi backward 非対応 | forward のみ利用可 |
| hdim=256 causal 非対応 | 96KB 共有メモリ上限を超過 |
| paged KV cache block_size | 64 の倍数が必要（xFormers は 16） |

---

## 方法 1: Docker を使う（推奨）

Docker を使うと環境の再現性が高く、ホスト環境を汚さない。

### Step 1: flash-attention リポジトリの取得

```bash
git clone https://github.com/peisuke/flash-attention.git
cd flash-attention
git checkout v100-sm70-support
```

### Step 2: ベースイメージのビルド（flash-attn SM70）

NGC PyTorch コンテナをベースに、flash-attn を SM70 向けにビルドする。
初回ビルドには約 2 時間かかる。

```bash
docker build -t flash_attn_v100_base \
    -f benchmarks/v100_comparison/Dockerfile.base .
```

**Dockerfile.base の内容:**
```dockerfile
FROM nvcr.io/nvidia/pytorch:24.11-py3
WORKDIR /workspace/flash-attention
COPY setup.py README.md ./
COPY csrc/ csrc/
COPY flash_attn/ flash_attn/
RUN FLASH_ATTN_CUDA_ARCHS="70" pip install -e .
```

ビルド完了後、動作確認:
```bash
docker run --gpus '"device=0"' --rm flash_attn_v100_base \
    python -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} OK')"
```

### Step 3: vLLM イメージのビルド

flash-attn ベースイメージの上に vLLM v0.6.5 をインストールし、SM70 互換パッチを当てる。

まず、xFormers を追加したベンチマーク用イメージをビルド:
```bash
docker build -t flash_attn_bench \
    -f benchmarks/v100_comparison/Dockerfile .
```

次に、vLLM イメージをビルド:
```bash
docker build -t flash_attn_v100_vllm \
    -f benchmarks/v100_vllm_serving/Dockerfile .
```

このイメージには以下の vLLM パッチが含まれる:
1. **SM70 セレクタ**: `has_device_capability(80)` → `has_device_capability(70)` に変更し、V100 で flash_attn バックエンドを選択可能にする
2. **API シム**: vLLM の `vllm_flash_attn` モジュールを、本リポジトリの flash_attn をラップするシムに置換（`out` パラメータの互換性対応）
3. **block_size=64**: vLLM の `--block-size` 引数に 64 を許可

### Step 4: vLLM サーバの起動

```bash
# HuggingFace キャッシュをマウントしてモデル再ダウンロードを回避
docker run --gpus '"device=0"' -p 8000:8000 --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    flash_attn_v100_vllm \
    python -m vllm.entrypoints.openai.api_server \
        --model NousResearch/Llama-2-7b-hf \
        --port 8000 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        --dtype half \
        --trust-remote-code \
        --enforce-eager \
        --block-size 64
```

**重要なオプション:**
- `--dtype half`: V100 は FP16 のみ対応
- `--block-size 64`: SM70 の paged KV cache 用（デフォルトの 16 は使えない）
- `--enforce-eager`: torch.compile を無効化（V100 での安定性向上）

### Step 5: リクエストの送信

```bash
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "NousResearch/Llama-2-7b-hf",
        "prompt": "Hello, world!",
        "max_tokens": 64
    }' | python -m json.tool
```

---

## 方法 2: ホストに直接インストール

Docker を使わず、ホスト環境に直接インストールする方法。

### Step 1: 前提パッケージの確認

```bash
# CUDA バージョン確認（12.x 推奨）
nvcc --version

# PyTorch が CUDA 対応であること
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# V100 であること
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# → Tesla V100-SXM2-32GB, 7.0
```

### Step 2: flash-attention のビルド

```bash
git clone https://github.com/peisuke/flash-attention.git
cd flash-attention
git checkout v100-sm70-support

# SM70 のみビルド（ビルド時間を短縮、約 20 分）
FLASH_ATTN_CUDA_ARCHS="70" pip install -e .
```

**ビルドオプション:**
- `FLASH_ATTN_CUDA_ARCHS="70"`: SM70 のみビルド。ビルド時間を大幅短縮し、`FLASH_ATTN_SM70_ONLY` マクロが自動有効化される
- `FLASH_ATTN_CUDA_ARCHS="70;80"`: SM70 と SM80 の両方をビルド（A100 と V100 両方で使う場合）。ビルド時間が長くなる
- `NVCC_THREADS=4`: 並列コンパイル数（デフォルト 4）

ビルド完了後の確認:
```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

### Step 3: vLLM のインストール

```bash
# vLLM v0.6.5 をソースからインストール
pip install setuptools-scm cmake ninja

git clone --depth 1 --branch v0.6.5 \
    https://github.com/vllm-project/vllm.git /tmp/vllm

cd /tmp/vllm
TORCH_CUDA_ARCH_LIST="7.0" \
VLLM_TARGET_DEVICE=cuda \
MAX_JOBS=4 \
pip install --no-deps --no-build-isolation -e .
```

> **注意**: vLLM の pip wheel は NGC PyTorch 等のカスタムビルドと ABI 互換性がない場合がある。ソースビルドを推奨。

### Step 4: vLLM への SM70 パッチ適用

3 つのパッチを手動で適用する。

**パッチ 1: SM70 セレクタ**

vLLM のアテンションセレクタが V100 で flash_attn を選択できるようにする:

```bash
# vLLM のインストールパスを確認
VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")

# SM80 チェックを SM70 に変更
sed -i 's/has_device_capability(80)/has_device_capability(70)/' \
    "$VLLM_PATH/attention/selector.py"

# 確認
grep 'has_device_capability(70)' "$VLLM_PATH/attention/selector.py" && \
    echo "Patch 1/3 OK"
```

**パッチ 2: flash_attn API シム**

vLLM の内蔵 `vllm_flash_attn` を本リポジトリの flash_attn で置換する:

```bash
# シムファイルをコピー
cp /path/to/flash-attention/benchmarks/v100_vllm_serving/vllm_flash_attn_shim.py \
    "$VLLM_PATH/vllm_flash_attn/__init__.py"

# 確認
python -c "from vllm.vllm_flash_attn import flash_attn_varlen_func; print('Patch 2/3 OK')"
```

**パッチ 3: block_size=64 許可**

```bash
# block_size の選択肢に 64 を追加
sed -i "s/choices=\[8, 16, 32, 256\]/choices=[8, 16, 32, 64, 256]/" \
    "$VLLM_PATH/engine/arg_utils.py"

# 確認
grep '64' "$VLLM_PATH/engine/arg_utils.py" && echo "Patch 3/3 OK"
```

### Step 5: vLLM サーバの起動

```bash
python -m vllm.entrypoints.openai.api_server \
    --model NousResearch/Llama-2-7b-hf \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --dtype half \
    --trust-remote-code \
    --enforce-eager \
    --block-size 64
```

---

## トラブルシューティング

### ビルドが非常に遅い（数時間以上）

`FLASH_ATTN_CUDA_ARCHS` に SM80+ を含めると、splitkv カーネルの SM70/SM80+ 両方のテンプレートがインスタンス化され、ビルド時間が倍増する。

```bash
# 悪い例（SM70+SM80 で 12 時間以上）
FLASH_ATTN_CUDA_ARCHS="70;80" pip install -e .

# 良い例（SM70 のみで 20 分）
FLASH_ATTN_CUDA_ARCHS="70" pip install -e .
```

SM70 と SM80 の両方が必要な場合は、ビルド時間の増加を許容するか、別々の環境でビルドする。

### BF16 エラー

```
RuntimeError: FlashAttention on Volta (V100) only supports FP16, not BF16.
```

V100 は BF16 を物理的にサポートしていない。`--dtype half` を使用する。
HuggingFace モデルのデフォルトが BF16 の場合は明示的に FP16 を指定:

```python
model = AutoModelForCausalLM.from_pretrained("model_name", torch_dtype=torch.float16)
```

### vLLM で `--block-size 64` が拒否される

パッチ 3 が正しく適用されていない。`arg_utils.py` 内の `choices` に `64` が含まれているか確認:

```bash
grep "block.size" "$VLLM_PATH/engine/arg_utils.py"
```

### causal mode で hdim=256 がエラー

```
RuntimeError: FlashAttention on Volta (V100) does not support causal mode with head_dim > 192.
```

V100 の共有メモリ上限 (96KB) により hdim=256 の causal mode はサポートされていない。non-causal mode か hdim<=192 のモデルを使用する。

### vLLM で FLASH_ATTN バックエンドが選択されない

パッチ 1 が正しく適用されていない。確認:

```bash
grep 'has_device_capability' "$VLLM_PATH/attention/selector.py"
```

`has_device_capability(70)` になっていることを確認。

また、環境変数で明示的にバックエンドを指定することもできる:

```bash
VLLM_ATTENTION_BACKEND=FLASH_ATTN python -m vllm.entrypoints.openai.api_server ...
```

### pip install で metadata-generation-failed

git submodule の初期化で失敗する場合がある:

```bash
# git の safe.directory 設定（Docker 内でよく発生）
git config --global --add safe.directory /path/to/flash-attention

# または setup.py develop を直接使用
python setup.py develop
```

---

## パフォーマンス参考値

vLLM + Llama-2-7B (FP16) on V100-32GB でのベンチマーク結果:

| メトリクス | FLASH_ATTN (bs=64) | XFORMERS (bs=16) | 比率 |
|-----------|-------------------|-----------------|------|
| 平均 TTFT | 976ms | 876ms | 1.10x |
| 平均 TPOT | 37.8ms | 33.1ms | 1.14x |
| 平均スループット | 133.2 tok/s | 154.8 tok/s | 1.16x |

- 低同時接続 (1-2): flash_attn は xFormers とほぼ同等
- 高同時接続 (16): xFormers が 1.2-1.4x 高速（KV cache block_size 16 vs 64 の差）
- Training 用途では flash_attn が最速（backward 3-5ms 高速）
