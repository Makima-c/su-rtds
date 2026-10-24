"""Keep model and memory-budget results separate; join the selected questions."""

import json
import math
from collections import Counter
from pathlib import Path

from retrieval import MEMORY_BUDGET, MEMORY_BUDGETS, STRATEGIES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-5-nano", "gpt-5.4-mini")


def answer_key(row):
    return row["sample_id"], row["qa_index"], row["strategy"]


def validate_latency(rows, field, allow_partial=False):
    """Accept unmeasured batches; reject partial or invalid timing measurements."""
    if not any(field in row for row in rows):
        return False
    measured = [row for row in rows if field in row and (row[field] is not None or not allow_partial)]
    if (not allow_partial and len(measured) != len(rows)) or any(
        type(row[field]) not in (int, float)
        or not math.isfinite(row[field])
        or row[field] < 0
        for row in measured
    ):
        raise ValueError(f"Invalid or incomplete latency measurements: {field}")
    return len(measured) == len(rows)


def experiment_folder(model, budget=MEMORY_BUDGET, root=ROOT):
    if model not in MODELS:
        raise ValueError(f"Unknown experiment model: {model}")
    if budget not in MEMORY_BUDGETS:
        raise ValueError(f"Unknown memory budget: {budget}")
    return root / "outputs" / model / f"{budget}_tokens"


def experiment_metadata(model, rows, embedding_model, budget=MEMORY_BUDGET):
    questions = {(r["sample_id"], r["qa_index"]): r["category"] for r in rows}
    counts = Counter(questions.values())
    categories = {
        1: ("multi_hop", "Combines information from multiple dialogue turns."),
        2: ("temporal", "Requires reasoning about dates or time."),
        4: ("single_hop", "Asks for a directly stated fact."),
    }
    return {
        "llm_model": model,
        "embedding_model": embedding_model,
        "question_count": len(questions),
        "question_categories": {
            name: {"count": counts[category], "description": description}
            for category, (name, description) in categories.items()
        },
        "retrieval_strategies": list(STRATEGIES),
        "memory_budget_tokens": budget,
    }


def load_experiment(model, root=ROOT, budget=MEMORY_BUDGET):
    folder = experiment_folder(model, budget, root)
    source = folder / "inputs/retrieval.json"
    if not source.exists():
        raise SystemExit(
            f"No retrieval input prepared. Run: uv run python src/prepare_retrieval.py --budget {budget}"
        )
    samples = json.loads((root / "src/data/selected_qa_20.json").read_text())
    questions = {(q["sample_id"], q["qa_index"]): q for q in samples}
    if len(samples) != 20 or len(questions) != 20:
        raise ValueError("Expected 20 unique selected questions")
    contexts = json.loads(source.read_text())
    expected = {
        (sid, index, strategy) for sid, index in questions for strategy in STRATEGIES
    }
    if len(contexts) != 60 or {answer_key(r) for r in contexts} != expected:
        raise ValueError(
            "Expected 60 unique contexts: each selected question with all three strategies"
        )
    fields = {
        "sample_id",
        "qa_index",
        "strategy",
        "context",
        "retrieved_ids",
        "memory_tokens",
    }
    for row in contexts:
        if (
            set(row) not in (fields, fields | {"retrieval_latency_ms"})
            or not isinstance(row["context"], str)
            or not row["context"].strip()
            or not isinstance(row["retrieved_ids"], list)
            or not row["retrieved_ids"]
            or type(row["memory_tokens"]) is not int
            or not 0 < row["memory_tokens"] <= budget
        ):
            raise ValueError("Invalid prepared context or memory budget")
    validate_latency(contexts, "retrieval_latency_ms")
    # Reference labels stay local; the API request explicitly selects question and context.
    return folder, [
        {**questions[(r["sample_id"], r["qa_index"])], **r} for r in contexts
    ]


def load_answers(folder, rows, complete=False):
    path = folder / "outputs/answer.json"
    if not path.exists():
        if complete:
            raise ValueError("Expected exactly 60 completed answers before scoring")
        return []
    document = json.loads(path.read_text())
    if (
        not isinstance(document, dict)
        or set(document) != {"metadata", "answers"}
        or not isinstance(document["metadata"], dict)
        or document["metadata"].get("llm_model") != folder.parent.name
        or not isinstance(document["answers"], list)
    ):
        raise ValueError("Invalid answer file model or answers array")
    budget = document["metadata"].get("memory_budget_tokens")
    if budget not in MEMORY_BUDGETS or folder.name != f"{budget}_tokens":
        raise ValueError("Saved answer memory budget differs from the experiment folder")
    answers = document["answers"]
    expected = {answer_key(r): r for r in rows}
    seen = set()
    for answer in answers:
        key = answer_key(answer)
        if (
            set(answer) - {"answer_api_latency_ms", "retrieval_latency_ms"}
            != {
                "sample_id",
                "qa_index",
                "strategy",
                "question",
                "reference_answer",
                "evidence",
                "llm_answer",
                "correct",
            }
            or key not in expected
            or key in seen
            or not isinstance(answer["llm_answer"], str)
            or not answer["llm_answer"].strip()
        ):
            raise ValueError(
                "Invalid, unknown, or duplicate saved answer; inspect before continuing"
            )
        source = expected[key]
        if (
            any(answer[k] != source[k] for k in ("question", "evidence"))
            or answer["reference_answer"] != source["answer"]
        ):
            raise ValueError(
                "Saved question, reference answer, or evidence differs from the sample"
            )
        score = answer["correct"]
        if score is not None and (type(score) is not int or score not in (0, 1)):
            raise ValueError("Scores must be null, 0, or 1")
        seen.add(key)
    separate_run = document["metadata"].get("latency_measurement") == "separate_run"
    validate_latency(answers, "answer_api_latency_ms", allow_partial=separate_run)
    validate_latency(answers, "retrieval_latency_ms", allow_partial=separate_run)
    if complete and seen != set(expected):
        raise ValueError("Expected exactly 60 completed answers before scoring")
    return answers
