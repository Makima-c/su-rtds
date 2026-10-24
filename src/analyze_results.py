"""Summarize accuracy and evidence recall after all answers have been scored."""

import argparse
import json
from statistics import mean

from experiment_paths import (
    MODELS, answer_key, load_answers, load_experiment, validate_latency,
)
from retrieval import MEMORY_BUDGET, MEMORY_BUDGETS, STRATEGIES, evidence_recall


def summarize(rows, allow_missing_retrieval=False):
    if any(r["correct"] is None for r in rows):
        raise ValueError("Fill all 60 correct values in answer.json before analysis.")
    if any(type(r["correct"]) is not int or r["correct"] not in (0, 1) for r in rows):
        raise ValueError("Scores must be 0 or 1")
    latency_fields = []
    for field in ("retrieval_latency_ms", "answer_api_latency_ms"):
        validate_latency(rows, field, allow_partial=allow_missing_retrieval and field == "retrieval_latency_ms")
        if any(field in row for row in rows):
            latency_fields.append(field)
    summary = []
    for strategy in STRATEGIES:
        group = [r for r in rows if r["strategy"] == strategy]
        if len(group) != 20:
            raise ValueError("Unequal strategy sample sizes")
        summary.append(
            {
                "retrieval_method": strategy,
                "total_questions": len(group),
                "correct_answers": sum(r["correct"] for r in group),
                "answer_accuracy_percent": round(
                    100 * mean(r["correct"] for r in group), 2
                ),
                "evidence_recall_percent": round(
                    100
                    * mean(
                        evidence_recall(r["evidence"], r["retrieved_ids"])
                        for r in group
                    ),
                    2,
                ),
            }
        )
        for field in latency_fields:
            summary[-1][f"average_{field}"] = (
                round(mean(r[field] for r in group), 2)
                if all(r.get(field) is not None for r in group) else None
            )
        if len(latency_fields) == 2:
            summary[-1]["average_total_latency_ms"] = (round(
                mean(r["retrieval_latency_ms"] + r["answer_api_latency_ms"] for r in group),
                2,
            ) if all(r.get("retrieval_latency_ms") is not None for r in group) else None)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--budget",
        type=int,
        choices=MEMORY_BUDGETS,
        default=MEMORY_BUDGET,
        help="Memory budget of the experiment to analyze (default: 512).",
    )
    parser.add_argument("--allow-missing-retrieval", action="store_true",
                        help="Keep incomplete retrieval and total latency means null; never average a subset.")
    args = parser.parse_args()
    OUT, contexts = load_experiment(args.model, budget=args.budget)
    answers = load_answers(OUT, contexts, complete=True)
    by_key = {answer_key(r): r for r in contexts}
    rows = [{**by_key[answer_key(r)], **r} for r in answers]
    try:
        summary = summarize(rows, allow_missing_retrieval=args.allow_missing_retrieval)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    (OUT / "outputs/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    for metric, ylabel, filename in [
        ("answer_accuracy_percent", "Answer accuracy (%)", "accuracy.png"),
        ("evidence_recall_percent", "Evidence recall (%)", "evidence_recall.png"),
    ]:
        fig, ax = plt.subplots(figsize=(5.1, 2.7), layout="constrained")
        values = [s[metric] for s in summary]
        bars = ax.bar(
            ["Recency", "BM25", "Semantic"],
            values,
            color=["#7c8792", "#315b86", "#4b897c"],
            width=0.6,
        )
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3)
        ax.set(ylabel=ylabel, ylim=(0, 100))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(figures / filename, dpi=220)
        plt.close(fig)
    for item in summary:
        print(
            item["retrieval_method"],
            f"accuracy={item['answer_accuracy_percent']:.2f}% ({item['correct_answers']}/20)",
            f"recall={item['evidence_recall_percent']:.2f}%",
        )


if __name__ == "__main__":
    main()
