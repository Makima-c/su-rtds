"""Text-only, turn-level retrieval with one shared formatted-memory budget."""

import re
from time import perf_counter

STRATEGIES = ("recency", "bm25", "semantic")
MEMORY_BUDGET = 512
MEMORY_BUDGETS = (512, 1024)


def build_memories(sample):
    conversation = sample["conversation"]
    memories = []
    seen = set()
    sessions = sorted(
        (k for k in conversation if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    for session in sessions:
        date = conversation.get(session + "_date_time")
        if not isinstance(date, str) or not date.strip():
            raise ValueError(f"Missing session date: {session}")
        for turn in conversation[session]:
            dia_id, text = turn.get("dia_id"), turn.get("text")
            if (
                not dia_id
                or dia_id in seen
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError(f"Invalid or duplicate memory turn in {session}")
            seen.add(dia_id)
            speaker = turn["speaker"]
            memories.append(
                {
                    "dia_id": dia_id,
                    "order": len(memories),
                    "session": session,
                    "session_date_time": date,
                    "speaker": speaker,
                    "text": text,
                    "formatted": f"[{dia_id} | {date} | {speaker}]\n{text}",
                }
            )
    return memories


def rank_scores(scores):
    # Prefer newer turns on equal scores; independent of reference annotations.
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), -i))


def pack_context(memories, ranking, count_tokens, budget=MEMORY_BUDGET):
    if budget < 0:
        raise ValueError("Memory budget cannot be negative")
    selected = []
    context = ""
    for index in ranking:
        trial = sorted(selected + [index])
        trial_context = "\n\n".join(memories[i]["formatted"] for i in trial)
        if count_tokens(trial_context) <= budget:
            selected = trial
            context = trial_context
    return {
        "context": context,
        "retrieved_ids": [memories[i]["dia_id"] for i in selected],
        "memory_tokens": count_tokens(context),
    }


def evidence_recall(evidence_ids, retrieved_ids):
    evidence = set(evidence_ids)
    if not evidence:
        raise ValueError("Reference evidence cannot be empty")
    return len(evidence.intersection(retrieved_ids)) / len(evidence)


def lexical_tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


class OpenAIEncoder:
    """Embed text in bounded batches; preserve API indices and use unit vectors."""

    def __init__(self, client, model, count_tokens):
        self.client = client
        self.model = model
        self.count_tokens = count_tokens
        self.request_count = 0
        self.input_tokens = 0
        self.dimensions = None

    def encode(self, texts):
        import numpy as np

        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding input must contain non-empty text")
        if any(self.count_tokens(t) > 8192 for t in texts):
            raise ValueError(
                "Embedding text exceeds 8192 tokens; no text is silently truncated"
            )
        vectors = []
        # At most 32 * 8192 tokens, below the API's per-request token limit.
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            result = self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
            )
            ordered = sorted(result.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise ValueError(
                    "Embedding response does not cover every input exactly once"
                )
            vectors.extend(item.embedding for item in ordered)
            self.request_count += 1
            self.input_tokens += result.usage.total_tokens
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Empty embedding vectors")
        if self.dimensions is not None and self.dimensions != vectors.shape[1]:
            raise ValueError("Embedding dimensions changed within the experiment")
        self.dimensions = vectors.shape[1]
        vectors = vectors / norms
        return vectors[0] if single else vectors


class ConversationRetriever:
    def __init__(self, memories, encoder, count_tokens):
        import numpy as np
        from rank_bm25 import BM25Okapi

        self.memories = memories
        self.encoder = encoder
        self.count_tokens = count_tokens
        texts = [m["formatted"] for m in memories]
        self.bm25 = BM25Okapi([lexical_tokens(text) for text in texts], k1=1.5, b=0.75)
        self.embeddings = np.asarray(encoder.encode(texts))

    def retrieve(self, question, strategy, budget=MEMORY_BUDGET):
        started = perf_counter()
        if strategy == "recency":
            ranking = list(reversed(range(len(self.memories))))
        elif strategy == "bm25":
            ranking = rank_scores(self.bm25.get_scores(lexical_tokens(question)))
        elif strategy == "semantic":
            query = self.encoder.encode(question)
            ranking = rank_scores(self.embeddings @ query)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        result = pack_context(self.memories, ranking, self.count_tokens, budget=budget)
        result["retrieval_latency_ms"] = round((perf_counter() - started) * 1000, 2)
        return result
