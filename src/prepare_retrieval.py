"""Prepare 60 contexts using OpenAI embeddings, then share them between both answer models."""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from experiment_paths import MODELS, experiment_folder
from retrieval import (
    MEMORY_BUDGET,
    MEMORY_BUDGETS,
    STRATEGIES,
    ConversationRetriever,
    OpenAIEncoder,
    build_memories,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget",
        type=int,
        choices=MEMORY_BUDGETS,
        default=MEMORY_BUDGET,
        help="Maximum formatted memory tokens per context (default: 512).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new standalone JSON; default prepares both model directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect data and embedding model without calling the API.",
    )
    args = parser.parse_args()
    targets = (
        [args.output.resolve()]
        if args.output
        else [
            experiment_folder(model, args.budget, ROOT) / "inputs/retrieval.json"
            for model in MODELS
        ]
    )
    for output in targets:
        if output.exists():
            raise SystemExit(f"Output already exists; preserving it: {output}")
    if not args.output:
        for model in MODELS:
            folder = experiment_folder(model, args.budget, ROOT)
            if (folder / "outputs/answer.json").exists():
                raise SystemExit(
                    "Existing answers must not be paired with new retrieval inputs."
                )

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    embedding_model = os.environ.get("OPENAI_TEXT_EMBEDDING_MODEL", "").strip()
    if not embedding_model:
        raise SystemExit(
            "Set OPENAI_TEXT_EMBEDDING_MODEL in .env before preparing retrieval."
        )
    selected_path = ROOT / "src/data/selected_qa_20.json"
    selection = json.loads(selected_path.read_text())
    raw = (ROOT / "src/data/locomo10.json").read_bytes()
    if len(selection) != 20:
        raise ValueError("Expected exactly 20 selected questions")
    dataset = {s["sample_id"]: s for s in json.loads(raw)}
    grouped = defaultdict(list)
    for q in selection:
        grouped[q["sample_id"]].append(q)
    memories_by_id = {sid: build_memories(dataset[sid]) for sid in grouped}
    print("Embedding model:", embedding_model)
    print("Memory budget:", args.budget)
    print("Questions:", len(selection), "| Contexts:", len(selection) * 3)
    print("Candidate messages:", sum(len(m) for m in memories_by_id.values()))
    if args.dry_run:
        print("Dry run: no API call or output written.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY; no request sent.")

    import tiktoken
    from openai import OpenAI

    memory_encoding = tiktoken.get_encoding("o200k_base")
    embedding_encoding = tiktoken.get_encoding("cl100k_base")
    count_tokens = lambda text: len(memory_encoding.encode(text, disallowed_special=()))
    encoder = OpenAIEncoder(
        OpenAI(max_retries=0, timeout=90.0),
        embedding_model,
        count_tokens=lambda text: len(
            embedding_encoding.encode(text, disallowed_special=())
        ),
    )
    records = []
    for sample_id, questions in sorted(grouped.items()):
        try:
            memories = memories_by_id[sample_id]
            retriever = ConversationRetriever(memories, encoder, count_tokens)
            for q in questions:
                for strategy in STRATEGIES:
                    result = retriever.retrieve(
                        q["question"], strategy, budget=args.budget
                    )
                    records.append(
                        {
                            "sample_id": q["sample_id"],
                            "qa_index": q["qa_index"],
                            "strategy": strategy,
                            **result,
                        }
                    )
        except Exception as error:
            status = getattr(error, "status_code", None)
            raise SystemExit(
                f"Retrieval stopped at {sample_id}: {type(error).__name__}, status={status}. No automatic retry or inputs published."
            ) from None
        print(
            f"{sample_id}: {len(memories)} messages; {len(questions) * 3} contexts prepared",
            flush=True,
        )
    random.Random(42).shuffle(records)
    text = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    for output in targets:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(text)
        print("Saved", output)
    print(
        "Ready: both answer models use the same prepared contexts."
        if not args.output
        else "Standalone retrieval JSON saved."
    )


if __name__ == "__main__":
    main()
