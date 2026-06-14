#!/usr/bin/env bash
# SFT trace-gen node (Direction 2, step 1): drive the REAL chroma_search env with a
# strong OpenRouter teacher (Kimi K2), keep finished high-recall trajectories, format
# as `messages`, push to HF. No GPU training here — network/CPU bound; cheap GPU box.
# Command is still `bash run.sh` (only the file content differs from the parent RL node).
set -euxo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

MODEL_TOK="Qwen/Qwen3-1.7B"   # tokenizer for the env's budget accounting

# ---------- environment setup (bare-pod fix ladder, same as parent) ----------
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
apt-get install -y -qq libnuma1 2>/dev/null || true

uv sync --extra fsdp
uv pip install --python .venv/bin/python -q rank-bm25 pandas pyarrow aiohttp huggingface_hub datasets

# ---------- dataset (same deterministic build as 4c-twin; seed 42) ----------
.venv/bin/python chroma/build_dataset.py --out data --n-train 1000 --n-val 200 --pool-extra 990 --eval-md data_EVAL.md

# ---------- teacher trace generation -> HF dataset ----------
set +e
.venv/bin/python chroma/teacher_gen.py \
  --train data/train.parquet \
  --corpus data/corpus.jsonl \
  --tokenizer "$MODEL_TOK" \
  --model moonshotai/kimi-k2 \
  --token-budget 4096 \
  --search-topk 8 \
  --max-tasks 800 \
  --target 400 \
  --recall-threshold 0.8 \
  --concurrency 24 \
  --max-turns 8 \
  --hf-repo-name chroma-sft-traces \
  2>&1 | tee gen.log
GEN_EXIT=${PIPESTATUS[0]}
set -e

# ---------- EVAL.md ----------
{
  echo "# SFT trace-gen — OpenRouter teacher -> HF (exit $GEN_EXIT)"
  echo
  echo "## Generation stats"
  echo '```json'
  cat .openresearch/artifacts/sft_gen_stats.json 2>/dev/null || echo "{}"
  echo '```'
  echo
  echo "## HF repo / secret resolution"
  grep -E "\[secrets\]|\[hf\]|FATAL|WARNING" gen.log | tail -10 || true
  echo
  echo "## Last gen progress lines"
  echo '```'
  grep -E "\[gen\]" gen.log | tail -6 || true
  echo '```'
} > EVAL.md
cat EVAL.md
exit "$GEN_EXIT"
