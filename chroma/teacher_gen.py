"""Generate SFT teacher traces for the Chroma retrieval-subagent.

Direction 2 (paper-faithful): the Chroma Context-1 recipe is LoRA SFT on strong
teacher traces *then* on-policy CISPO RL. This script produces the SFT data by
driving the REAL `chroma_search` env (so every tool result is real, never
hallucinated) with a strong OpenRouter model as the policy, concurrently over
training tasks.

Keep policy (distillation with teacher-forced finish): we keep a trajectory when
the teacher actually ENCOUNTERED the gold chunks (high traj_recall — it read the
right documents), then we append the IDEAL terminal `<finish>` ourselves listing
exactly the encountered gold. This both rescues the many trajectories where a
capable teacher reads correctly but fails to emit a clean finish in time, and
guarantees every demo ends with a perfect high-precision finish. A teacher-only
"nudge" message steers generation but is stripped from the saved data, so the
student SFTs on the env's canonical system+user prompt.

Env semantics match the 4c-twin config (snippet search, token_budget 4096).
Cost is bounded by --max-tasks / --target with a token/$ tally.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skyrl_gym.envs.chroma_search.env import ChromaSearchEnv, ChromaSearchEnvConfig  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Teacher-only steering (prepended to the system prompt at GENERATION time, then
# stripped — the saved SFT data keeps only the env's canonical system prompt).
TEACHER_NUDGE = (
    "You are an expert demonstrator producing high-quality training trajectories. "
    "Follow the tool protocol exactly. ALWAYS: (1) issue one or more <search> calls to locate "
    "candidate documents; (2) <read> every promising document so its chunks enter your evidence "
    "set — search snippets are NOT evidence and cannot be finished with; (3) once you have read the "
    "chunks that actually contain the answer evidence, end with a single "
    "<finish>id1, id2</finish> listing exactly those chunk ids. NEVER <finish> before reading, and "
    "do not stop early — the evidence always exists in the corpus. One tool call per turn is fine."
)

_OR_KEYS = ["OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPEN_ROUTER_API_KEY", "OPENROUTER_TOKEN"]
_HF_KEYS = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"]


def _first_env(names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return n, v
    return None, None


def _coerce_msgs(prompt):
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt]


async def _call(session, model, messages, sem, usage, nudge, max_tokens=512, temperature=0.7, retries=5):
    import aiohttp

    # teacher-only nudge folded into the system message for generation; never stored
    if nudge and messages and messages[0]["role"] == "system":
        api_messages = [{"role": "system", "content": nudge + "\n\n" + messages[0]["content"]}] + messages[1:]
    else:
        api_messages = messages

    headers = {"Authorization": f"Bearer {usage['or_key']}", "Content-Type": "application/json"}
    body = {"model": model, "messages": api_messages, "max_tokens": max_tokens, "temperature": temperature}
    async with sem:
        for attempt in range(retries):
            try:
                async with session.post(OPENROUTER_URL, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=180)) as r:
                    if r.status in (429, 500, 502, 503, 520, 524):
                        await asyncio.sleep(2 * (attempt + 1) + random.random())
                        continue
                    data = await r.json()
                    if "choices" not in data:
                        if attempt == 0:
                            print(f"[openrouter] non-choices response: {json.dumps(data)[:300]}", flush=True)
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    u = data.get("usage", {}) or {}
                    usage["prompt_tokens"] += int(u.get("prompt_tokens", 0))
                    usage["completion_tokens"] += int(u.get("completion_tokens", 0))
                    usage["calls"] += 1
                    return data["choices"][0]["message"].get("content") or ""
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    print(f"[openrouter] call failed: {type(e).__name__}: {e}", flush=True)
                await asyncio.sleep(2 * (attempt + 1) + random.random())
    return None


async def _rollout(session, model, row, cfg, sem, usage, nudge, max_turns):
    extras = {"reward_spec": row["reward_spec"], "extra_info": row["extra_info"], "max_turns": max_turns}
    try:
        env = ChromaSearchEnv(cfg, extras)
    except Exception as e:  # noqa: BLE001
        print(f"[env] construct failed: {e}", flush=True)
        return None
    messages = _coerce_msgs(row["prompt"])
    env.init(messages)
    for _ in range(max_turns):
        text = await _call(session, model, messages, sem, usage, nudge)
        if text is None:
            break
        messages.append({"role": "assistant", "content": text})
        out = env.step(text)  # BaseTextEnvStepOutput is a TypedDict
        if out["done"]:
            break
        for obs in out["observations"]:
            messages.append({"role": str(obs["role"]), "content": str(obs["content"])})
    m = env.get_metrics()
    return {
        "messages": messages,
        "metrics": m,
        "gold": list(env.gold),
        "encountered": list(env.encountered),
        "pruned": set(env.pruned),
    }


def _finalize(res, threshold):
    """Apply teacher-forced finish. Returns (saved_sample | None, outcome_tag)."""
    gold = res["gold"]
    if not gold:
        return None, "no_gold"
    enc_gold = [g for g in gold if g in res["encountered"] and g not in res["pruned"]]
    traj_recall = len(enc_gold) / len(gold)
    if traj_recall < threshold:
        return None, "low_traj_recall"
    # build the ideal terminal finish over exactly the encountered gold
    fin = (
        "I have read the documents that contain the evidence. "
        "Finalizing the evidence set.\n\n<finish>" + ", ".join(enc_gold) + "</finish>"
    )
    msgs = list(res["messages"])
    if msgs and msgs[-1]["role"] == "assistant" and "<finish>" in msgs[-1]["content"]:
        msgs[-1] = {"role": "assistant", "content": fin}  # replace the model's (imperfect) finish
    else:
        msgs.append({"role": "assistant", "content": fin})
    return {"messages": msgs, "_traj_recall": traj_recall}, "kept"


async def _run(rows, model, cfg, concurrency, max_turns, target, threshold, usage, nudge):
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    kept, attempted = [], 0
    outcomes = {"kept": 0, "low_traj_recall": 0, "no_gold": 0, "api_fail": 0}
    recall_hist, finished_by_model = [], 0
    failed_samples = []
    chunk = concurrency * 4
    async with aiohttp.ClientSession() as session:
        i = 0
        while i < len(rows) and len(kept) < target:
            batch = rows[i : i + chunk]
            i += chunk
            results = await asyncio.gather(*[_rollout(session, model, r, cfg, sem, usage, nudge, max_turns) for r in batch])
            for res in results:
                attempted += 1
                if res is None:
                    outcomes["api_fail"] += 1
                    continue
                recall_hist.append(float(res["metrics"].get("final_recall", 0.0)))
                finished_by_model += int(bool(res["metrics"].get("finished", 0.0)))
                sample, tag = _finalize(res, threshold)
                outcomes[tag] = outcomes.get(tag, 0) + 1
                if sample is not None:
                    kept.append(sample)
                elif tag == "low_traj_recall" and len(failed_samples) < 4:
                    failed_samples.append(res)
            print(
                f"[gen] attempted={attempted} kept={len(kept)} (target {target}) "
                f"finished_by_model={finished_by_model} outcomes={outcomes} "
                f"calls={usage['calls']} tok_in={usage['prompt_tokens']} tok_out={usage['completion_tokens']}",
                flush=True,
            )
    return kept, attempted, recall_hist, outcomes, finished_by_model, failed_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.parquet")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--model", default="moonshotai/kimi-k2")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--search-topk", type=int, default=8)
    ap.add_argument("--max-tasks", type=int, default=800)
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--recall-threshold", type=float, default=0.8, help="min traj_recall (gold encountered) to keep")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--no-nudge", action="store_true")
    ap.add_argument("--hf-repo-name", default="chroma-sft-traces")
    ap.add_argument("--out-parquet", default="data/sft_traces.parquet")
    ap.add_argument("--price-in", type=float, default=0.55)
    ap.add_argument("--price-out", type=float, default=2.2)
    args = ap.parse_args()

    or_name, or_key = _first_env(_OR_KEYS)
    hf_name, hf_key = _first_env(_HF_KEYS)
    if not or_key:
        avail = [k for k in os.environ if "OPEN" in k.upper() or "ROUTER" in k.upper()]
        print(f"FATAL: no OpenRouter key found. Tried {_OR_KEYS}. Env vars containing OPEN/ROUTER: {avail}")
        sys.exit(2)
    print(f"[secrets] OpenRouter key <- ${or_name}; HF token <- ${hf_name or 'NONE'}", flush=True)

    cfg = ChromaSearchEnvConfig(
        corpus_path=args.corpus, tokenizer_path=args.tokenizer,
        token_budget=args.token_budget, search_topk=args.search_topk,
    )
    nudge = None if args.no_nudge else TEACHER_NUDGE

    rows = pq.read_table(args.train).to_pylist()
    random.seed(42)
    random.shuffle(rows)
    rows = rows[: args.max_tasks]
    print(f"[gen] driving {args.model} over up to {len(rows)} tasks "
          f"(target {args.target} kept, traj_recall>={args.recall_threshold}, conc {args.concurrency}, "
          f"nudge={'on' if nudge else 'off'})", flush=True)

    usage = {"or_key": or_key, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.time()
    kept, attempted, recall_hist, outcomes, finished_by_model, failed = asyncio.run(
        _run(rows, args.model, cfg, args.concurrency, args.max_turns, args.target, args.recall_threshold, usage, nudge)
    )
    dt = time.time() - t0

    cost = usage["prompt_tokens"] / 1e6 * args.price_in + usage["completion_tokens"] / 1e6 * args.price_out
    avg_turns = (sum(len([m for m in k["messages"] if m["role"] == "assistant"]) for k in kept) / max(1, len(kept)))

    os.makedirs("data", exist_ok=True)
    samples = [{"messages": k["messages"]} for k in kept]

    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(samples), args.out_parquet)

    art = ".openresearch/artifacts"
    os.makedirs(art, exist_ok=True)
    stats = {
        "model": args.model,
        "attempted": attempted,
        "kept": len(kept),
        "keep_rate": round(len(kept) / max(1, attempted), 3),
        "recall_threshold_traj": args.recall_threshold,
        "outcomes": outcomes,
        "finished_by_model_frac": round(finished_by_model / max(1, attempted), 3),
        "avg_assistant_turns_kept": round(avg_turns, 2),
        "env_final_recall_mean": round(sum(recall_hist) / max(1, len(recall_hist)), 3),
        "calls": usage["calls"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "est_cost_usd": round(cost, 2),
        "wall_seconds": round(dt, 1),
    }
    with open(os.path.join(art, "sft_gen_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(art, "sft_traces_sample.md"), "w") as f:
        for k in kept[:3]:
            f.write(f"## saved demo (traj_recall={k.get('_traj_recall'):.2f}, assistant_turns={len([m for m in k['messages'] if m['role']=='assistant'])})\n\n")
            for m in k["messages"]:
                f.write(f"**{m['role']}**:\n{m['content']}\n\n")
            f.write("\n---\n\n")
    with open(os.path.join(art, "sft_failed_sample.md"), "w") as f:
        f.write("# low_traj_recall failures (teacher did not encounter the gold)\n\n")
        for res in failed:
            enc_gold = [g for g in res["gold"] if g in res["encountered"]]
            f.write(f"## gold={res['gold']} encountered_gold={enc_gold} turns={res['metrics'].get('turns')}\n\n")
            for m in res["messages"]:
                f.write(f"**{m['role']}**:\n{m['content'][:1200]}\n\n")
            f.write("\n---\n\n")
    print("[gen] STATS " + json.dumps(stats), flush=True)

    repo = None
    if hf_key and samples:
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi

            who = HfApi(token=hf_key).whoami()["name"]
            repo = f"{who}/{args.hf_repo_name}"
            Dataset.from_list(samples).push_to_hub(repo, token=hf_key, private=True)
            print(f"[hf] pushed {len(samples)} traces -> {repo}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[hf] push FAILED: {type(e).__name__}: {e}", flush=True)
    elif not hf_key:
        print(f"WARNING: no HF token found (tried {_HF_KEYS}); kept local parquet only at {args.out_parquet}", flush=True)

    stats["hf_repo"] = repo
    with open(os.path.join(art, "sft_gen_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
