"""Build the Chroma retrieval-subagent dataset from HotpotQA (distractor setting).

Output (under --out, default ./data):
  corpus.jsonl        one line per unique document: {"title", "chunks": [{"id", "text"}]}
  train.parquet       SkyRL search-example schema (prompt, env_class, reward_spec, extra_info)
  validation.parquet  held-out tasks from the HotpotQA validation split
  EVAL.md             dataset stats + one fully rendered sample task (written to repo root)

Per task: corpus = its own 10 distractor-setting paragraphs + POOL_EXTRA hard-negative
paragraphs chosen by ranking every other title against the task's question with a
BM25 index over the union of all titles' full text. Gold chunk ids are exact, derived
from supporting_facts (title, sent_id) pairs. Everything is deterministic -> all nodes
regenerate byte-identical data.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chroma.prompts import SYSTEM_PROMPT, user_prompt  # noqa: E402

SEED = 42
N_TRAIN = 2000
N_VAL = 200
POOL_EXTRA = 190  # extra distractor docs sampled per task (total corpus ~200 docs)
SENTS_PER_CHUNK = 3


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def make_chunks(title: str, sentences: list) -> list:
    """Group consecutive sentences into chunks of SENTS_PER_CHUNK; chunk id = '<title>::c<i>'."""
    chunks = []
    for i in range(0, len(sentences), SENTS_PER_CHUNK):
        text = " ".join(s.strip() for s in sentences[i : i + SENTS_PER_CHUNK]).strip()
        if text:
            chunks.append({"id": f"{title}::c{i // SENTS_PER_CHUNK}", "text": text})
    return chunks


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Tiny pure-numpy BM25 over the (title -> full doc text) corpus.

    Used to pick the POOL_EXTRA hardest (topically-closest) distractor titles for
    each task's question, replacing uniform-random padding.
    """

    def __init__(self, titles: list, doc_texts: list, k1: float = 1.5, b: float = 0.75):
        self.titles = list(titles)
        self.title_to_idx = {t: i for i, t in enumerate(self.titles)}
        self.k1 = k1
        self.b = b
        N = len(self.titles)
        self.N = N
        doc_lens = np.zeros(N, dtype=np.float32)
        df: dict = {}
        raw_postings: dict = {}  # term -> list of (doc_idx, tf)
        for i, text in enumerate(doc_texts):
            toks = _tokenize(text)
            doc_lens[i] = len(toks)
            tf = Counter(toks)
            for term, f in tf.items():
                df[term] = df.get(term, 0) + 1
                raw_postings.setdefault(term, []).append((i, f))
        self.doc_lens = doc_lens
        self.avgdl = float(doc_lens.mean()) if N > 0 else 1.0
        self.idf: dict = {}
        self.postings: dict = {}
        for term, plist in raw_postings.items():
            n = df[term]
            self.idf[term] = float(np.log((N - n + 0.5) / (n + 0.5) + 1.0))
            idxs = np.fromiter((p[0] for p in plist), dtype=np.int32, count=len(plist))
            tfs = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
            self.postings[term] = (idxs, tfs)

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        for term in set(_tokenize(query)):
            post = self.postings.get(term)
            if post is None:
                continue
            idxs, tfs = post
            dl = self.doc_lens[idxs]
            denom = tfs + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            np.add.at(scores, idxs, self.idf[term] * (tfs * (self.k1 + 1.0)) / denom)
        return scores


def gold_chunk_ids(task, doc_sentences) -> list:
    """Map supporting_facts (title, sent_id) -> containing chunk ids. Skips out-of-range sent_ids."""
    gold = []
    sf = task["supporting_facts"]
    for title, sent_id in zip(sf["title"], sf["sent_id"]):
        n_sents = len(doc_sentences.get(title, []))
        if sent_id < n_sents:
            cid = f"{title}::c{sent_id // SENTS_PER_CHUNK}"
            if cid not in gold:
                gold.append(cid)
    return gold


def build_split(rows, all_titles, doc_sentences, split_name, bm25):
    records, dropped = [], 0
    for task in rows:
        own_titles = list(task["context"]["title"])
        gold_titles = set(task["supporting_facts"]["title"])
        # sanity: gold titles must be inside the task's own context
        if not gold_titles.issubset(set(own_titles)):
            dropped += 1
            continue
        gold = gold_chunk_ids(task, doc_sentences)
        # require evidence in >=2 distinct docs (true multi-hop) and no dangling gold
        if len({g.split("::")[0] for g in gold}) < 2:
            dropped += 1
            continue
        rng = np.random.default_rng(stable_int(task["id"]) % (2**32))
        # BM25 nearest-neighbor distractors: rank every non-own title against the
        # task's question and take the POOL_EXTRA hardest (highest-scoring) ones.
        scores = bm25.score(task["question"])
        for t in own_titles:
            j = bm25.title_to_idx.get(t)
            if j is not None:
                scores[j] = -np.inf
        # argpartition for top-k, then sort just that slice descending for stability
        k = min(POOL_EXTRA, bm25.N - len(own_titles))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        extra = [bm25.titles[i] for i in top_idx]
        doc_ids = own_titles + extra[:POOL_EXTRA]
        rng.shuffle(doc_ids)
        records.append(
            {
                "data_source": f"chroma_hotpotqa_{split_name}",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(task["question"])},
                ],
                "env_class": "chroma_search",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": {
                        "gold_chunk_ids": gold,
                        "answer": task["answer"],
                    },
                },
                "extra_info": {
                    "task_id": task["id"],
                    "doc_ids": list(doc_ids),
                    "question": task["question"],
                    "level": task["level"],
                    "type": task["type"],
                },
            }
        )
    return records, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--eval-md", default="EVAL.md")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

    rng = random.Random(SEED)
    train_pool = [i for i, lvl in enumerate(ds["train"]["level"]) if lvl == "hard"]
    rng.shuffle(train_pool)
    train_rows = [ds["train"][i] for i in train_pool[: int(args.n_train * 1.2)]]
    val_idx = list(range(len(ds["validation"])))
    rng.shuffle(val_idx)
    val_rows = [ds["validation"][i] for i in val_idx[: int(args.n_val * 1.2)]]

    # global doc table (dedupe by title, first occurrence wins)
    doc_sentences = {}
    for rows in (train_rows, val_rows):
        for task in rows:
            for title, sents in zip(task["context"]["title"], task["context"]["sentences"]):
                doc_sentences.setdefault(title, sents)
    all_titles = sorted(doc_sentences)

    # BM25 index over every candidate title's full text (title + sentences),
    # built once and shared across splits. Used to pick hard, topically-similar
    # distractors per task instead of uniform random padding.
    doc_texts = [title + " " + " ".join(doc_sentences[title]) for title in all_titles]
    bm25 = BM25Index(all_titles, doc_texts)

    train_recs, train_drop = build_split(train_rows, all_titles, doc_sentences, "train", bm25)
    val_recs, val_drop = build_split(val_rows, all_titles, doc_sentences, "validation", bm25)
    train_recs, val_recs = train_recs[: args.n_train], val_recs[: args.n_val]

    # corpus: only docs actually referenced by a kept task
    used = set()
    for r in train_recs + val_recs:
        used.update(r["extra_info"]["doc_ids"])
    corpus_path = os.path.join(args.out, "corpus.jsonl")
    n_chunks, chunk_words = 0, []
    with open(corpus_path, "w") as f:
        for title in sorted(used):
            chunks = make_chunks(title, doc_sentences[title])
            n_chunks += len(chunks)
            chunk_words.extend(len(c["text"].split()) for c in chunks)
            f.write(json.dumps({"title": title, "chunks": chunks}) + "\n")

    pd.DataFrame(train_recs).to_parquet(os.path.join(args.out, "train.parquet"))
    pd.DataFrame(val_recs).to_parquet(os.path.join(args.out, "validation.parquet"))

    # ---- EVAL.md ----
    gold_counts = Counter(len(r["reward_spec"]["ground_truth"]["gold_chunk_ids"]) for r in train_recs)
    corpus_sizes = [len(r["extra_info"]["doc_ids"]) for r in train_recs]
    types = Counter(r["extra_info"]["type"] for r in train_recs)
    cw = np.array(chunk_words)
    sample = val_recs[0]
    sample_gold = sample["reward_spec"]["ground_truth"]["gold_chunk_ids"]
    gold_texts = []
    for line in open(corpus_path):
        doc = json.loads(line)
        for c in doc["chunks"]:
            if c["id"] in sample_gold:
                gold_texts.append(f"- `{c['id']}`: {c['text'][:300]}")

    lines = [
        "# Stage 1 — HotpotQA retrieval-subagent dataset",
        "",
        f"- train tasks: **{len(train_recs)}** (dropped {train_drop} in pool of {len(train_rows)}; level=hard only)",
        f"- validation tasks (held out): **{len(val_recs)}** (dropped {val_drop})",
        f"- corpus: **{len(used)}** unique docs, **{n_chunks}** chunks "
        f"(chunk words p50/p90/max = {int(np.percentile(cw,50))}/{int(np.percentile(cw,90))}/{cw.max()})",
        f"- per-task corpus size docs: min/mean/max = {min(corpus_sizes)}/{np.mean(corpus_sizes):.0f}/{max(corpus_sizes)}",
        f"- gold chunks per task: {dict(sorted(gold_counts.items()))}",
        f"- question types: {dict(types)}",
        "",
        "## Sample validation task",
        f"- id: `{sample['extra_info']['task_id']}`  type: {sample['extra_info']['type']}",
        f"- question: {sample['extra_info']['question']}",
        f"- answer: {sample['reward_spec']['ground_truth']['answer']}",
        f"- gold chunk ids: {sample_gold}",
        "",
        "### Gold chunk texts",
        *gold_texts,
        "",
        "### System prompt (first 600 chars)",
        "```",
        SYSTEM_PROMPT[:600],
        "```",
        "",
        "Gate check: gold chunks span >=2 docs per task by construction; ids are exact; "
        "data is deterministic (seed 42).",
    ]
    with open(args.eval_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {corpus_path}, train/validation parquet, {args.eval_md}")


if __name__ == "__main__":
    main()
