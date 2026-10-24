# Validating evidence annotations in the Locomo10 dataset

import json
import re
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data/locomo10.json"

with DATA_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

stats = Counter()
valid_by_category = Counter()
invalid_examples = []
first_valid = None

for sample in data:
    sample_id = sample["sample_id"]
    conversation = sample["conversation"]

    # Build dialogue ID -> dialogue turn mapping
    memory_by_id = {}

    for key, turns in conversation.items():
        if re.fullmatch(r"session_\d+", key):
            for turn in turns:
                if isinstance(turn, dict) and turn.get("dia_id"):
                    memory_by_id[turn["dia_id"]] = turn

    for index, qa in enumerate(sample["qa"]):
        stats["total"] += 1

        evidence = qa.get("evidence")

        # Check whether evidence annotations exist
        if not isinstance(evidence, list) or not evidence:
            stats["missing_evidence"] += 1
            continue

        if not all(isinstance(e, str) and e.strip() for e in evidence):
            stats["malformed_evidence"] += 1
            continue

        # Check whether every evidence ID exists in dialogue
        missing_ids = [e for e in evidence if e not in memory_by_id]

        if missing_ids:
            stats["unmatched_evidence"] += 1

            if len(invalid_examples) < 5:
                invalid_examples.append(
                    {
                        "sample_id": sample_id,
                        "qa_index": index,
                        "missing_ids": missing_ids,
                    }
                )
            continue

        stats["valid"] += 1
        valid_by_category[qa.get("category")] += 1

        if first_valid is None:
            first_valid = {
                "question": qa["question"],
                "evidence_id": evidence[0],
                "evidence_text": memory_by_id[evidence[0]].get("text"),
            }

print("Total conversations:", len(data))
print("Total QA:", stats["total"])
print("Valid QA:", stats["valid"])
print("Missing evidence:", stats["missing_evidence"])
print("Malformed evidence:", stats["malformed_evidence"])
print("Unmatched evidence:", stats["unmatched_evidence"])

print("\nValid QA by category:")
for category, count in sorted(valid_by_category.items()):
    print(f"Category {category}: {count}")

print("\nInvalid evidence examples:")
for item in invalid_examples:
    print(item)

print("\nExample matched evidence:")
print(first_valid)
