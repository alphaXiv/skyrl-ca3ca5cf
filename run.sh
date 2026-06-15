#!/usr/bin/env bash
# SFT cold-start node (Direction 2, step 2): LoRA SFT of Qwen3-1.7B on the teacher
# demos (alphaXiv/chroma-sft-traces), export the merged HF model, push it to HF so
# the RL-from-SFT node can load it. Command is still `bash run.sh`.
set -euxo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

MODEL="Qwen/Qwen3-1.7B"
SFT_DATASET="alphaXiv/chroma-sft-traces"
HF_OUT_NAME="chroma-sft-qwen3-1.7b"
RUN_NAME="chroma-sft-1.7b"
NUM_GPUS=$(nvidia-smi -L | wc -l)

# ---------- environment setup (bare-pod fix ladder) ----------
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
apt-get install -y -qq libnuma1 2>/dev/null || true

uv sync --extra fsdp
uv pip install --python .venv/bin/python -q huggingface_hub datasets

EXPORT="$HOME/sft_export/$RUN_NAME"

# ---------- LoRA SFT ----------
# hf_save_interval>0 (and > total steps) guarantees exactly ONE final merged HF export.
set +e
.venv/bin/python -m skyrl.train.main_sft \
  strategy=fsdp \
  model.path="$MODEL" \
  model.lora.rank=32 \
  model.lora.alpha=64 \
  model.lora.target_modules=all-linear \
  dataset_name="$SFT_DATASET" \
  dataset_split=train \
  messages_key=messages \
  train_on_what=last_assistant_message \
  max_length=8192 \
  num_epochs=3 \
  batch_size=16 \
  micro_train_batch_size_per_gpu=1 \
  remove_microbatch_padding=true \
  seed=42 \
  optimizer_config.lr=1e-4 \
  optimizer_config.weight_decay=0.0 \
  optimizer_config.max_grad_norm=1.0 \
  optimizer_config.num_warmup_steps=5 \
  optimizer_config.scheduler=constant_with_warmup \
  placement.num_nodes=1 \
  placement.num_gpus_per_node="$NUM_GPUS" \
  fsdp_config.cpu_offload=false \
  fsdp_config.reshard_after_forward=true \
  logger=wandb \
  project_name=chroma-skyrl \
  run_name="$RUN_NAME" \
  ckpt_path="$HOME/ckpts/$RUN_NAME" \
  export_path="$EXPORT" \
  hf_save_interval=1000 \
  2>&1 | tee sft.log
SFT_EXIT=${PIPESTATUS[0]}
set -e

# ---------- locate merged export + push to HF ----------
POLICY_DIR=$(ls -d "$EXPORT"/global_step_*/policy 2>/dev/null | sort | tail -1 || true)
PUSH_NOTE="no export found"
if [ -n "$POLICY_DIR" ] && [ -f "$POLICY_DIR/config.json" ]; then
  HF_TOK="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"
  PUSH_NOTE=$(.venv/bin/python - "$POLICY_DIR" "$HF_OUT_NAME" "$HF_TOK" <<'PY'
import sys
src, name, tok = sys.argv[1], sys.argv[2], sys.argv[3]
if not tok:
    print("NO_HF_TOKEN"); sys.exit(0)
from huggingface_hub import HfApi
api = HfApi(token=tok)
who = api.whoami()["name"]
repo = f"{who}/{name}"
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path=src, repo_id=repo, commit_message="SFT cold-start merged model")
print(repo)
PY
)
fi

# ---------- EVAL.md ----------
{
  echo "# SFT cold-start — $RUN_NAME (exit $SFT_EXIT)"
  echo
  echo "## Export"
  echo "policy_dir: $POLICY_DIR"
  echo "size: $(du -sh "$POLICY_DIR" 2>/dev/null | cut -f1 || echo n/a)"
  echo "files: $(ls "$POLICY_DIR" 2>/dev/null | tr '\n' ' ' || echo none)"
  echo "pushed_to_hf: $PUSH_NOTE"
  echo
  echo "## SFT loss (last lines)"
  echo '```'
  grep -E "loss=|eval_loss=" sft.log | tail -12 || echo "no loss lines"
  echo '```'
  echo
  echo "## NaN check"
  if grep -qE "loss=nan|loss=inf" sft.log; then echo "WARNING: nan/inf loss"; else echo "no nan/inf in loss"; fi
} > EVAL.md
cat EVAL.md
exit "$SFT_EXIT"
