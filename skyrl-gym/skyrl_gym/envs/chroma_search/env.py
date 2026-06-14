"""Chroma Context-1 style self-editing retrieval-subagent environment.

The agent searches a per-task local corpus with four tools and must return the set of
chunk ids that contain the evidence for a multi-hop question:

  <search>query</search>      BM25 over the task corpus (dedup vs. already-seen chunks)
  <grep>pattern</grep>        regex over the task corpus (dedup vs. already-seen chunks)
  <read>document title</read> all chunks of one document
  <prune>id1, id2</prune>     removes those chunks' text from the agent's OWN context
  <finish>id1, id2</finish>   ends the episode with the final evidence set

Self-editing trick: the generator (with `generator.step_wise_trajectories=true`)
re-applies the chat template to the live `chat_history` list every turn, and
`Env.init(prompt)` receives a reference to that exact list. `prune_chunks` therefore
mutates earlier tool-result messages in place; the next turn's prompt is rebuilt from
the pruned history. The full unpruned trajectory is kept env-side for reward.

Reward (final step only, intermediate steps 0):
  0.7 * F_beta(final set vs gold)  (beta=4 -> recall-weighted)
+ 0.3 * trajectory_recall          (gold chunks encountered at any point)
+ answer bonus                     (a final chunk contains the answer string)
- turn penalty, single-chunk-prune-streak penalty, no-finish penalty
clamped to [0, 2].
"""

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType


@dataclass
class ChromaSearchEnvConfig:
    corpus_path: str = "data/corpus.jsonl"
    tokenizer_path: str = "Qwen/Qwen3-1.7B"
    token_budget: int = 4096
    search_topk: int = 8
    snippet_chars: int = 90
    grep_max_hits: int = 5
    read_max_chunks: int = 10
    max_tool_calls_per_turn: int = 4
    max_final_chunks: int = 20
    fbeta: float = 4.0
    w_fbeta: float = 0.7
    w_traj_recall: float = 0.3
    answer_bonus: float = 0.2
    turn_penalty: float = 0.01
    prune_streak_penalty: float = 0.05
    no_finish_penalty: float = 0.1


_CORPUS_CACHE: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
_BM25_CACHE: Dict[Any, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()

_TAG_RE = re.compile(r"<(search|grep|read|prune|finish)>(.*?)</\1>", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _load_corpus(path: str) -> Dict[str, List[Dict[str, str]]]:
    with _CACHE_LOCK:
        if path not in _CORPUS_CACHE:
            docs = {}
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    docs[d["title"]] = d["chunks"]
            _CORPUS_CACHE[path] = docs
        return _CORPUS_CACHE[path]


def _load_tokenizer(path: str):
    with _CACHE_LOCK:
        if path not in _TOKENIZER_CACHE:
            try:
                from transformers import AutoTokenizer

                _TOKENIZER_CACHE[path] = AutoTokenizer.from_pretrained(path)
            except Exception:
                _TOKENIZER_CACHE[path] = None  # fall back to char/4 estimate
        return _TOKENIZER_CACHE[path]


def _bm25_tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


class ChromaSearchEnv(BaseTextEnv):
    def __init__(self, env_config: Union[ChromaSearchEnvConfig, DictConfig], extras: Dict[str, Any] = {}):
        super().__init__()
        self.cfg = env_config

        assert "reward_spec" in extras and "ground_truth" in extras["reward_spec"]
        gt = extras["reward_spec"]["ground_truth"]
        self.gold: List[str] = list(gt["gold_chunk_ids"])
        self.answer: str = str(gt.get("answer", ""))
        self.max_turns = int(extras.get("max_turns", 12))

        info = extras["extra_info"]
        corpus = _load_corpus(env_config.corpus_path)
        self.docs: Dict[str, List[Dict[str, str]]] = {t: corpus[t] for t in info["doc_ids"] if t in corpus}
        self.chunks: Dict[str, str] = {c["id"]: c["text"] for chunks in self.docs.values() for c in chunks}

        self._bm25 = None  # built lazily on first search
        self._bm25_ids: List[str] = []
        self._read_docs: set = set()

        # episode state
        self.chat_history: Optional[ConversationType] = None
        self.encountered: List[str] = []  # chunk ids shown to the agent, in order (never removed)
        self.pruned: set = set()
        self.final_set: Optional[List[str]] = None
        self.finished = False
        self.n_tool_calls = 0
        self.n_pruned = 0
        self.n_pruned_gold = 0
        self.prune_streaks = 0
        self._single_prune_streak = 0
        self.budget_rejections = 0
        self.malformed_turns = 0
        self._consecutive_malformed = 0
        self.peak_tokens = 0

    # ------------------------------------------------------------------ infra

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        # Keep a reference to the LIVE list the generator re-templates each turn,
        # so prune_chunks can edit the agent's context in place.
        self.chat_history = prompt
        return prompt, {}

    def _count_tokens(self) -> int:
        tok = _load_tokenizer(self.cfg.tokenizer_path)
        total = 0
        for msg in self.chat_history:
            c = msg["content"]
            total += len(tok.encode(c, add_special_tokens=False)) if tok is not None else len(c) // 4
            total += 5  # per-message template overhead
        return total

    def _chunk_block(self, cid: str) -> str:
        return f'<chunk id="{cid}">\n{self.chunks[cid]}\n</chunk>'

    def _mark_encountered(self, cids: List[str]) -> None:
        for cid in cids:
            if cid not in self.encountered:
                self.encountered.append(cid)

    # ------------------------------------------------------------------ tools

    def _tool_search(self, query: str) -> str:
        if self._bm25 is None:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                return "ERROR: BM25 backend unavailable."
            # cache the index per task corpus: the 8 GRPO rollouts and repeated evals
            # of one task share the same ~1700-chunk corpus
            cache_key = (self.cfg.corpus_path, tuple(sorted(self.docs)))
            with _CACHE_LOCK:
                hit = _BM25_CACHE.get(cache_key)
            if hit is not None:
                self._bm25_ids, self._bm25 = hit
            else:
                self._bm25_ids = list(self.chunks)
                self._bm25 = BM25Okapi([_bm25_tokenize(self.chunks[c]) for c in self._bm25_ids])
                with _CACHE_LOCK:
                    if len(_BM25_CACHE) > 256:
                        _BM25_CACHE.clear()
                    _BM25_CACHE[cache_key] = (self._bm25_ids, self._bm25)
        q = _bm25_tokenize(query)
        if not q:
            return f'No results for search "{query}" (empty query).'
        scores = self._bm25.get_scores(q)
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: -x[1])
        # doc-level hits: best-scoring chunk per doc; snippets only — chunk text must be
        # obtained via <read>, so search alone can never reach the gold evidence
        seen_docs, lines = set(), []
        for cid, sc in ranked:
            if sc <= 0 or len(lines) >= self.cfg.search_topk:
                break
            doc = cid.split("::")[0]
            if doc in seen_docs or doc in self._read_docs:
                continue
            seen_docs.add(doc)
            snippet = self.chunks[cid][: self.cfg.snippet_chars]
            lines.append(f'<doc title="{doc}" top_chunk="{cid}">{snippet}...</doc>')
        if not lines:
            return f'No new results for search "{query}" (all matching docs already read or no match).'
        return (
            f'Doc results for search "{query}" (use <read>title</read> to see full chunks):\n'
            + "\n".join(lines)
        )

    def _tool_grep(self, pattern: str) -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"ERROR: invalid regex {pattern!r}: {e}"
        hits = []
        for cid, text in self.chunks.items():
            if cid not in self.encountered and rx.search(text):
                hits.append(cid)
                if len(hits) >= self.cfg.grep_max_hits:
                    break
        if not hits:
            return f'No new matches for grep "{pattern}".'
        self._mark_encountered(hits)
        return f'Matches for grep "{pattern}":\n' + "\n".join(self._chunk_block(c) for c in hits)

    def _tool_read(self, title: str) -> str:
        title = title.strip()
        if title not in self.docs:
            # tolerate the agent passing a chunk id instead of a doc title
            base = title.split("::")[0]
            if base in self.docs:
                title = base
            else:
                return f"ERROR: no document titled {title!r} in the corpus."
        cids = [c["id"] for c in self.docs[title][: self.cfg.read_max_chunks]]
        self._read_docs.add(title)
        self._mark_encountered(cids)
        return f'Document "{title}":\n' + "\n".join(self._chunk_block(c) for c in cids)

    def _tool_prune(self, arg: str) -> Tuple[str, int]:
        ids = [s.strip().strip("'\"") for s in arg.split(",") if s.strip()]
        done_ids, missing = [], []
        for cid in ids:
            if cid in self.chunks and cid in self.encountered and cid not in self.pruned:
                self.pruned.add(cid)
                self.n_pruned += 1
                if cid in self.gold:
                    self.n_pruned_gold += 1
                done_ids.append(cid)
                # in-place context edit: drop this chunk's entire block (tag + body) in
                # every prior message so prune fully reclaims its token cost; also drop
                # the matching <doc title="..."> snippet line for the doc this chunk
                # belongs to (search results otherwise leave un-prunable footprint).
                doc_title = cid.split("::")[0]
                chunk_rx = re.compile(
                    r'<chunk id="' + re.escape(cid) + r'">.*?</chunk>\n?',
                    re.DOTALL,
                )
                doc_rx = re.compile(
                    r'<doc title="' + re.escape(doc_title) + r'"[^>]*>.*?</doc>\n?',
                    re.DOTALL,
                )
                for msg in self.chat_history:
                    if msg["role"] != "user":
                        continue
                    content = msg["content"]
                    if cid in content:
                        content = chunk_rx.sub("", content)
                    if doc_title in content:
                        content = doc_rx.sub("", content)
                    msg["content"] = content
            else:
                missing.append(cid)
        out = f"Pruned {len(done_ids)} chunk(s)."
        if missing:
            out += f" Not found/already pruned: {', '.join(missing[:5])}."
        return out, len(done_ids)

    # ------------------------------------------------------------------ reward

    def _final_metrics(self) -> Dict[str, float]:
        S = set(self.final_set or [])
        G = set(self.gold)
        tp = len(S & G)
        precision = tp / len(S) if S else 0.0
        recall = tp / len(G) if G else 0.0
        b2 = self.cfg.fbeta**2
        fbeta = (1 + b2) * precision * recall / (b2 * precision + recall) if (precision + recall) > 0 else 0.0
        traj_recall = len(set(self.encountered) & G) / len(G) if G else 0.0
        ans = _norm(self.answer)
        answer_found = float(bool(ans) and any(ans in _norm(self.chunks[c]) for c in S if c in self.chunks))
        return {
            "final_precision": precision,
            "final_recall": recall,
            "final_fbeta": fbeta,
            "traj_recall": traj_recall,
            "answer_found": answer_found,
        }

    def _compute_reward(self) -> float:
        m = self._final_metrics()
        r = (
            self.cfg.w_fbeta * m["final_fbeta"]
            + self.cfg.w_traj_recall * m["traj_recall"]
            + self.cfg.answer_bonus * m["answer_found"]
            - self.cfg.turn_penalty * self.turns
            - self.cfg.prune_streak_penalty * self.prune_streaks
            - (0.0 if self.finished else self.cfg.no_finish_penalty)
        )
        return max(0.0, min(2.0, r))

    def get_metrics(self) -> Dict[str, Any]:
        m = self._final_metrics()
        m.update(
            {
                "turns": float(self.turns),
                "tool_calls": float(self.n_tool_calls),
                "tool_calls_per_turn": self.n_tool_calls / max(1, self.turns),
                "finished": float(self.finished),
                "n_pruned": float(self.n_pruned),
                "pruned_any": float(self.n_pruned > 0),
                "prune_accuracy": ((self.n_pruned - self.n_pruned_gold) / self.n_pruned) if self.n_pruned else 1.0,
                "budget_rejections": float(self.budget_rejections),
                "malformed_turns": float(self.malformed_turns),
                "peak_tokens": float(self.peak_tokens),
                "final_set_size": float(len(self.final_set or [])),
            }
        )
        return m

    # ------------------------------------------------------------------ step

    def _terminate(self, final_set: List[str], finished: bool) -> BaseTextEnvStepOutput:
        self.finished = finished
        self.final_set = final_set[: self.cfg.max_final_chunks]
        return BaseTextEnvStepOutput(
            observations=[],
            reward=self._compute_reward(),
            done=True,
            metadata=self.get_metrics(),
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        calls = _TAG_RE.findall(action)[: self.cfg.max_tool_calls_per_turn]

        if not calls:
            self.malformed_turns += 1
            self._consecutive_malformed += 1
            self._single_prune_streak = 0
            if self._consecutive_malformed >= 2 or self.turns >= self.max_turns:
                return self._terminate([c for c in self.encountered if c not in self.pruned], finished=False)
            return BaseTextEnvStepOutput(
                observations=[
                    {
                        "role": "user",
                        "content": "No valid tool call found. Use <search>, <grep>, <read>, "
                        "<prune>, or <finish> with matching closing tags.",
                    }
                ],
                reward=0.0,
                done=False,
                metadata={},
            )
        self._consecutive_malformed = 0

        finish_arg = next((arg for tag, arg in calls if tag == "finish"), None)
        if finish_arg is not None:
            ids = [s.strip().strip("'\"") for s in finish_arg.split(",") if s.strip()]
            # only chunks the agent actually encountered (via read/grep) count — search
            # snippets alone are not evidence
            valid = [c for c in ids if c in self.chunks and c in self.encountered]
            return self._terminate(valid, finished=True)

        over_hard = self._count_tokens() >= self.cfg.token_budget
        results = []
        pruned_this_turn = 0
        non_prune_calls = 0
        for tag, arg in calls:
            self.n_tool_calls += 1
            arg = arg.strip()
            if tag == "prune":
                out, n = self._tool_prune(arg)
                pruned_this_turn += n
                results.append(out)
            elif over_hard:
                self.budget_rejections += 1
                results.append(
                    f"REJECTED <{tag}>: context budget exceeded. Use <prune> to free space, or <finish>."
                )
            elif tag == "search":
                non_prune_calls += 1
                results.append(self._tool_search(arg))
            elif tag == "grep":
                non_prune_calls += 1
                results.append(self._tool_grep(arg))
            elif tag == "read":
                non_prune_calls += 1
                results.append(self._tool_read(arg))

        # single-chunk prune streak bookkeeping (paper: discourage one-at-a-time prune loops)
        if pruned_this_turn == 1 and non_prune_calls == 0:
            self._single_prune_streak += 1
            if self._single_prune_streak >= 2:
                self.prune_streaks += 1
        else:
            self._single_prune_streak = 0

        if self.turns >= self.max_turns:
            return self._terminate([c for c in self.encountered if c not in self.pruned], finished=False)

        used = self._count_tokens()
        self.peak_tokens = max(self.peak_tokens, used)
        footer = f"\n[budget: {used}/{self.cfg.token_budget}]"
        if used >= self.cfg.token_budget:
            footer += " HARD LIMIT REACHED — only <prune> or <finish> will be accepted."
        elif used > self.cfg.token_budget // 2:
            footer += " Over half budget: prune irrelevant chunks or finish soon."

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": "\n\n".join(results) + footer}],
            reward=0.0,
            done=False,
            metadata={},
        )


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()
