#!/usr/bin/env bash
# RL-from-SFT (Direction 2, step 3): identical to 4c-twin CISPO RL, but the policy is
# initialized from the SFT cold-start model. Preamble merges the SFT LoRA adapter
# (alphaXiv/chroma-sft-qwen3-1.7b) into base Qwen3-1.7B, then RL trains a fresh LoRA on it.
# eval_before_train=true => step-0 eval reflects the SFT-only model. Command: bash run.sh.
set -euxo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

BASE_MODEL="Qwen/Qwen3-1.7B"
SFT_ADAPTER="alphaXiv/chroma-sft-qwen3-1.7b"
MERGED="$HOME/sft_merged"
MODEL="$MERGED"
RUN_NAME="stage-sft-rl-qwen3-1.7b-snippet-3ep"
NUM_GPUS=$(nvidia-smi -L | wc -l)

# ---------- environment setup (bare-pod fix ladder) ----------
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
apt-get install -y -qq libnuma1 2>/dev/null || true

uv sync --extra fsdp
uv pip install --python .venv/bin/python -q rank-bm25 pandas pyarrow peft

# ---------- merge SFT adapter into base -> full HF model ----------
.venv/bin/python - "$BASE_MODEL" "$SFT_ADAPTER" "$MERGED" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base_id, adapter_id, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, adapter_id).merge_and_unload()
merged.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base_id).save_pretrained(out)
print("merged SFT model written to", out)
PY

.venv/bin/ray stop 2>/dev/null || true
.venv/bin/ray start --head --num-gpus="$NUM_GPUS"

# ---------- dataset (same deterministic build as 4c-twin; seed 42) ----------
.venv/bin/python chroma/build_dataset.py --out data --n-train 1000 --n-val 200 --pool-extra 990 --eval-md data_EVAL.md

# ---------- training (identical hyperparams to 4c-twin; only the model path differs) ----------
set +e
.venv/bin/python -m skyrl.train.entrypoints.main_base \
  data.train_data="['$ROOT/data/train.parquet']" \
  data.val_data="['$ROOT/data/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="cispo" \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$MODEL" \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=64 \
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=2 \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.45 \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.distributed_executor_backend=mp \
  trainer.epochs=3 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=8192 \
  generator.sampling_params.max_generate_length=1024 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.step_wise_trajectories=true \
  generator.chat_template_kwargs='{"enable_thinking": false}' \
  generator.n_samples_per_prompt=8 \
  generator.max_turns=10 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.eval_sampling_params.temperature=0.7 \
  generator.eval_sampling_params.top_p=0.8 \
  generator.eval_sampling_params.max_generate_length=1024 \
  environment.env_class="chroma_search" \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.chroma_search.corpus_path="$ROOT/data/corpus.jsonl" \
  environment.skyrl_gym.chroma_search.tokenizer_path="$BASE_MODEL" \
  environment.skyrl_gym.chroma_search.token_budget=4096 \
  environment.skyrl_gym.chroma_search.search_topk=8 \
  trainer.logger="wandb" \
  trainer.project_name="chroma-skyrl" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=2 \
  trainer.resume_mode=none \
  trainer.ckpt_path="$HOME/ckpts/$RUN_NAME" \
  trainer.export_path="$HOME/ckpts/$RUN_NAME/exports" \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

# ---------- EVAL.md ----------
{
  echo "# RL-from-SFT snippet search (Qwen3-1.7B) — $RUN_NAME (exit $TRAIN_EXIT)"
  echo
  echo "## Policy init"
  echo "merged SFT model: $MODEL (from $SFT_ADAPTER)"
  echo
  echo "## Dataset"
  sed -n '2,8p' data_EVAL.md || true
  echo
  echo "## Last train-rollout reward lines"
  echo '```'
  grep -E "avg_raw_reward" train.log | tail -8 || true
  echo '```'
  echo
  echo "## NaN check (strict)"
  if grep -qE "(loss|reward|grad_norm)[^a-zA-Z]*(nan|inf)" train.log; then
    echo "WARNING: possible nan/inf"
  else
    echo "no nan/inf in loss/reward/grad_norm lines"
  fi
} > EVAL.md
cat EVAL.md
exit "$TRAIN_EXIT"
