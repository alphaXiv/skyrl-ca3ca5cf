#!/usr/bin/env bash
# Stage 4c-twin: snippet-search semantics on Qwen3-1.7B (cross-scale comparison).
# Full dataset (2000 train / 200 val), 1 epoch = ~62 steps of 32 prompts x 8 rollouts,
# token budget 4096 (prune pressure), eval/env/* metrics logged at steps 0/10/.../60.
# Gate: eval/env/final_recall + final_fbeta clearly above the step-0 untrained anchor;
# prune_accuracy high/up with n_pruned > 0; no collapse (entropy, malformed_turns).
set -euxo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

MODEL="Qwen/Qwen3-1.7B"
RUN_NAME="stage4ctwin-qwen3-1.7b-snippet-search"
NUM_GPUS=$(nvidia-smi -L | wc -l)

# ---------- environment setup (bare-pod fix ladder) ----------
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
apt-get install -y -qq libnuma1 2>/dev/null || true

uv sync --extra fsdp
uv pip install --python .venv/bin/python -q rank-bm25 pandas pyarrow

.venv/bin/ray stop 2>/dev/null || true
.venv/bin/ray start --head --num-gpus="$NUM_GPUS"

# ---------- dataset (full size; deterministic, gitignored) ----------
.venv/bin/python chroma/build_dataset.py --out data --n-train 1000 --n-val 200 --pool-extra 990 --eval-md data_EVAL.md

# ---------- training ----------
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
  trainer.epochs=1 \
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
  environment.skyrl_gym.chroma_search.tokenizer_path="$MODEL" \
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
  echo "# Stage 4c-twin snippet search (Qwen3-1.7B) — $RUN_NAME (exit $TRAIN_EXIT)"
  echo
  echo "## Dataset"
  sed -n '2,8p' data_EVAL.md || true
  echo
  echo "## Eval metrics over training (all eval/ lines)"
  echo '```'
  grep -oE "\{'eval/[^}]*\}" train.log | tail -10 || echo "no eval metric lines found"
  echo '```'
  echo
  echo "## Last train-rollout reward + env lines"
  echo '```'
  grep -E "avg_raw_reward" train.log | tail -8 || true
  echo '```'
  echo
  echo "## NaN check (strict)"
  if grep -qE "(loss|reward|grad_norm)[^a-zA-Z]*(nan|inf)" train.log; then
    echo "WARNING: possible nan/inf in loss/reward/grad_norm lines"
  else
    echo "no nan/inf in loss/reward/grad_norm lines"
  fi
} > EVAL.md
cat EVAL.md
exit "$TRAIN_EXIT"
