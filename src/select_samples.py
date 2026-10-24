"""Show the final 20 QA, or preview the original seed-42 draw; no file writes."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def candidate_pools(data):
    if len(data) != 10 or len({s["sample_id"] for s in data}) != 10:
        raise ValueError("Expected ten distinct conversations for two QA each.")
    pools = {}
    for sample in sorted(data, key=lambda s: s["sample_id"]):
        conversation = sample["conversation"]
        turns = {}
        for key in sorted(
            (k for k in conversation if re.fullmatch(r"session_\d+", k)),
            key=lambda k: int(k.split("_")[1]),
        ):
            for turn in conversation[key]:
                if not isinstance(turn, dict) or not turn.get("dia_id"):
                    continue
                if turn["dia_id"] in turns:
                    raise ValueError(f"Duplicate dialogue ID in {sample['sample_id']}")
                turns[turn["dia_id"]] = turn
        pool = {1: [], 2: [], 4: []}
        for qa_index, qa in enumerate(sample["qa"]):
            category = qa.get("category")
            answer = qa.get("answer")
            evidence = qa.get("evidence")
            if category not in pool or answer is None or not str(answer).strip():
                continue
            if not isinstance(qa.get("question"), str) or not qa["question"].strip():
                continue
            if not isinstance(evidence, list) or not evidence:
                continue
            if not all(
                isinstance(e, str)
                and e in turns
                and isinstance(turns[e].get("text"), str)
                and turns[e]["text"].strip()
                for e in evidence
            ):
                continue
            pool[category].append(
                {
                    "sample_id": sample["sample_id"],
                    "qa_index": qa_index,
                    "category": category,
                    "question": qa["question"],
                    "answer": answer,
                    "evidence": list(evidence),
                }
            )
        # This balanced sampling design requires each category in each conversation.
        for category in pool:
            if not pool[category]:
                raise ValueError(
                    f"{sample['sample_id']}: no eligible QA in category {category}"
                )
        pools[sample["sample_id"]] = pool
    return pools


def select_samples(data, seed=42):
    pools = candidate_pools(data)

    rng = random.Random(seed)
    conversation_ids = list(pools)
    rng.shuffle(conversation_ids)
    selected = []
    for index, sample_id in enumerate(conversation_ids):
        second_category = 1 if index < 5 else 2
        for category in (4, second_category):
            selected.append(rng.choice(pools[sample_id][category]))
    return sorted(selected, key=lambda q: (q["sample_id"], q["qa_index"]))


def validate_samples(data, samples):
    """Check the final questions against the original data and the 20-QA allocation."""
    pools = candidate_pools(data)
    source = {
        (q["sample_id"], q["qa_index"]): q
        for pool in pools.values()
        for category in pool.values()
        for q in category
    }
    if (
        len(samples) != 20
        or len({(q["sample_id"], q["qa_index"]) for q in samples}) != 20
        or Counter(q["category"] for q in samples) != {4: 10, 1: 5, 2: 5}
        or Counter(q["sample_id"] for q in samples) != {s["sample_id"]: 2 for s in data}
    ):
        raise ValueError(
            "Expected 20 unique QA, category counts 10/5/5, two per conversation"
        )
    for q in samples:
        if q != source.get((q["sample_id"], q["qa_index"])):
            raise ValueError("Selected question differs from the original source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Preview the initial random draw before manual replacements.",
    )
    args = parser.parse_args()
    raw = (DATA_DIR / "locomo10.json").read_bytes()
    data = json.loads(raw)
    if args.draft:
        samples = select_samples(data)
    else:
        samples = json.loads((DATA_DIR / "selected_qa_20.json").read_text())
    validate_samples(data, samples)
    print("Initial draft" if args.draft else "Final selection", ":", len(samples), "QA")
    print(
        "Category counts:",
        dict(sorted(Counter(q["category"] for q in samples).items())),
    )
    print(json.dumps(samples, ensure_ascii=False, indent=2))
    print("Selection details: docs/experiment.md. No files written.")


if __name__ == "__main__":
    main()
