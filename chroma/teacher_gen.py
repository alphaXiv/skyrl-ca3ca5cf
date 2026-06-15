"""Generate SFT teacher traces for the Chroma retrieval-subagent.

Direction 2 (paper-faithful): the Chroma Context-1 recipe is LoRA SFT on strong
teacher traces *then* on-policy CISPO RL. This script produces the SFT data by
driving the REAL `chroma_search` env (so every tool result is real, never
hallucinated) with a strong OpenRouter model as the policy, concurrently over
training tasks. We keep only *finished* trajectories whose `final_recall` clears
a threshold — clean multi-hop search->read->finish demonstrations — format them
as OpenAI `messages`, and push them to the Hugging Face Hub for the SFT node.

Env semantics match the 4c-twin config (snippet search, token_budget 4096), so
the demonstrations are on-distribution for the student that will SFT+RL on them.

Cost is bounded by --max-tasks and --target (stop early once enough kept), and a
running token/$ tally is printed at the end.
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

# Defensive secret resolution — the exact env-var names stored in OpenResearch
# are not known here, so try the common spellings and fail loudly if none match.
_OR_KEYS = ["OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPEN_ROUTER_API_KEY", "OPENROUTER_TOKEN"]
_HF_KEYS = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"]


def _first_env(names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return n, v
    return None, None


def _coerce_msgs(prompt):
    """parquet -> list[{'role','content'}] with plain python str values."""
    out = []
    for m in prompt:
        out.append({"role": str(m["role"]), "content": str(m["content"])})
    return out


async def _call(session, model, messages, sem, usage, max_tokens=512, temperature=0.7, retries=4):
    import aiohttp

    headers = {"Authorization": f"Bearer {usage['or_key']}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    async with sem:
        for attempt in range(retries):
            try:
                async with session.post(OPENROUTER_URL, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=120)) as r:
                    if r.status in (429, 500, 502, 503, 520, 524):
                        await asyncio.sleep(2 * (attempt + 1) + random.random())
                        continue
                    data = await r.json()
                    if "choices" not in data:
                        # surface auth/model errors once, then give up on this call
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


async def _rollout(session, model, row, cfg, sem, usage, max_turns):
    extras = {
        "reward_spec": row["reward_spec"],
        "extra_info": row["extra_info"],
        "max_turns": max_turns,
    }
    try:
        env = ChromaSearchEnv(cfg, extras)
    except Exception as e:  # noqa: BLE001
        print(f"[env] construct failed: {e}", flush=True)
        return None
    messages = _coerce_msgs(row["prompt"])
    env.init(messages)  # chat_history IS `messages` -> budget accounting + prune see the live list
    for _ in range(max_turns):
        text = await _call(session, model, messages, sem, usage)
        if text is None:
            return None
        messages.append({"role": "assistant", "content": text})
        out = env.step(text)  # BaseTextEnvStepOutput is a TypedDict -> use item access
        if out["done"]:
            break
        for obs in out["observations"]:
            messages.append({"role": str(obs["role"]), "content": str(obs["content"])})
    m = env.get_metrics()
    return {"messages": messages, "metrics": m}


async def _run(rows, model, cfg, concurrency, max_turns, target, threshold, usage):
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    kept, attempted = [], 0
    recall_hist = []
    chunk = concurrency * 4
    async with aiohttp.ClientSession() as session:
        i = 0
        while i < len(rows) and len(kept) < target:
            batch = rows[i : i + chunk]
            i += chunk
            results = await asyncio.gather(*[_rollout(session, model, r, cfg, sem, usage, max_turns) for r in batch])
            for res in results:
                attempted += 1
                if res is None:
                    continue
                rec = float(res["metrics"].get("final_recall", 0.0))
                fin = bool(res["metrics"].get("finished", 0.0))
                recall_hist.append(rec)
                if fin and rec >= threshold:
                    kept.append(res)
            print(
                f"[gen] attempted={attempted} kept={len(kept)} "
                f"(target {target}) calls={usage['calls']} "
                f"tok_in={usage['prompt_tokens']} tok_out={usage['completion_tokens']}",
                flush=True,
            )
    return kept, attempted, recall_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.parquet")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.5")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--search-topk", type=int, default=8)
    ap.add_argument("--max-tasks", type=int, default=800)
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--recall-threshold", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--hf-repo-name", default="chroma-sft-traces")
    ap.add_argument("--out-parquet", default="data/sft_traces.parquet")
    # rough OpenRouter rates ($/M tokens) for the cost tally only
    ap.add_argument("--price-in", type=float, default=3.0)
    ap.add_argument("--price-out", type=float, default=15.0)
    args = ap.parse_args()

    or_name, or_key = _first_env(_OR_KEYS)
    hf_name, hf_key = _first_env(_HF_KEYS)
    if not or_key:
        avail = [k for k in os.environ if "OPEN" in k.upper() or "ROUTER" in k.upper()]
        print(f"FATAL: no OpenRouter key found. Tried {_OR_KEYS}. Env vars containing OPEN/ROUTER: {avail}")
        sys.exit(2)
    print(f"[secrets] OpenRouter key <- ${or_name}; HF token <- ${hf_name or 'NONE'}", flush=True)

    cfg = ChromaSearchEnvConfig(
        corpus_path=args.corpus,
        tokenizer_path=args.tokenizer,
        token_budget=args.token_budget,
        search_topk=args.search_topk,
    )

    rows = pq.read_table(args.train).to_pylist()
    random.seed(42)
    random.shuffle(rows)
    rows = rows[: args.max_tasks]
    print(f"[gen] driving {args.model} over up to {len(rows)} tasks "
          f"(target {args.target} kept, recall>={args.recall_threshold}, conc {args.concurrency})", flush=True)

    usage = {"or_key": or_key, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.time()
    kept, attempted, recall_hist = asyncio.run(
        _run(rows, args.model, cfg, args.concurrency, args.max_turns, args.target, args.recall_threshold, usage)
    )
    dt = time.time() - t0

    cost = usage["prompt_tokens"] / 1e6 * args.price_in + usage["completion_tokens"] / 1e6 * args.price_out
    keep_rate = len(kept) / max(1, attempted)
    avg_turns = sum(len([m for m in k["messages"] if m["role"] == "assistant"]) for k in kept) / max(1, len(kept))

    os.makedirs("data", exist_ok=True)
    samples = [{"messages": k["messages"]} for k in kept]

    # local parquet backup
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(samples), args.out_parquet)

    # artifacts for inspection
    art = ".openresearch/artifacts"
    os.makedirs(art, exist_ok=True)
    stats = {
        "model": args.model,
        "attempted": attempted,
        "kept": len(kept),
        "keep_rate": round(keep_rate, 3),
        "recall_threshold": args.recall_threshold,
        "avg_assistant_turns_kept": round(avg_turns, 2),
        "recall_mean_all": round(sum(recall_hist) / max(1, len(recall_hist)), 3),
        "recall_ge_0.8_frac": round(sum(r >= 0.8 for r in recall_hist) / max(1, len(recall_hist)), 3),
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
            f.write(f"## recall={k['metrics'].get('final_recall'):.2f} turns={k['metrics'].get('turns')}\n\n")
            for m in k["messages"]:
                f.write(f"**{m['role']}**:\n{m['content']}\n\n")
            f.write("\n---\n\n")
    print("[gen] STATS " + json.dumps(stats), flush=True)

    # push to HF
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

    with open(os.path.join(art, "sft_gen_stats.json"), "w") as f:
        stats["hf_repo"] = repo
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
