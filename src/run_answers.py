"""Answer the 60 prepared contexts; save each response and resume completed work."""

import argparse
import json
import os
from time import perf_counter

from experiment_paths import (
    MODELS,
    ROOT,
    answer_key,
    experiment_metadata,
    load_answers,
    load_experiment,
)
from retrieval import MEMORY_BUDGET, MEMORY_BUDGETS

INSTRUCTIONS = (
    "Answer the question using only the supplied conversation memory. "
    "Memory entries include their session date and speaker. Resolve relative dates "
    "using that session date. Give a concise factual answer, at most 50 words. "
    "If the memory does not provide enough information, answer exactly: "
    "Insufficient evidence. Do not use external knowledge or invent details."
)


def request_for(question, context, model="gpt-5-nano", effort="low"):
    # Send only the question and retrieved context, never reference answers/evidence.
    return {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": f"Conversation memory:\n{context}\n\nQuestion: {question}",
        "reasoning": {"effort": effort},
        "text": {"verbosity": "low"},
        "max_output_tokens": 1024,
        "store": False,
    }


def save_answers(path, rows, metadata):
    """Replace the JSON document only after the complete new file has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"metadata": metadata, "answers": rows}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--budget",
        type=int,
        choices=MEMORY_BUDGETS,
        default=MEMORY_BUDGET,
        help="Retrieved memory budget in tokens (default: 512); output limit stays 1024.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Maximum NEW answers this run; use 1 to check a real question first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show pending work without calling the API.",
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 60:
        parser.error("--limit must be between 1 and 60")
    folder, rows = load_experiment(args.model, budget=args.budget)
    output = folder / "outputs/answer.json"
    if output.with_suffix(".json.tmp").exists():
        raise SystemExit(
            "Interrupted JSON save detected; inspect the temporary file before another API call."
        )
    existing = load_answers(folder, rows)
    completed = {answer_key(r) for r in existing}
    pending = [r for r in rows if answer_key(r) not in completed]
    if pending and existing and "answer_api_latency_ms" not in existing[0]:
        raise SystemExit(
            "Existing answers have no latency measurements. Preserve this batch and start a fresh timed run."
        )
    print("Model:", args.model)
    print("Memory budget:", args.budget)
    print("Input:", folder / "inputs/retrieval.json")
    print("Output:", output)
    print(
        "Saved:",
        len(completed),
        "| Remaining:",
        len(pending),
        "| This run:",
        min(args.limit, len(pending)),
    )
    if args.dry_run:
        print("Dry run complete: no key loaded and no API request sent.")
        return
    if not pending:
        print("All 60 answers are already saved; no API request sent.")
        return
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing API key; no request sent.")
    if output.exists():
        metadata = json.loads(output.read_text())["metadata"]
    else:
        embedding_model = os.environ.get("OPENAI_TEXT_EMBEDDING_MODEL", "").strip()
        if not embedding_model:
            raise SystemExit(
                "Missing OPENAI_TEXT_EMBEDDING_MODEL for experiment metadata; no request sent."
            )
        metadata = experiment_metadata(
            args.model, rows, embedding_model, budget=args.budget
        )
        if all("retrieval_latency_ms" in row for row in rows):
            metadata["latency_measurement"] = "same_run"
    client = OpenAI(max_retries=0, timeout=90.0)
    for index, row in enumerate(pending[: args.limit]):
        request = request_for(row["question"], row["context"], args.model)
        try:
            started = perf_counter()
            response = client.responses.create(**request)
            latency_ms = round((perf_counter() - started) * 1000, 2)
        except Exception as error:
            status = getattr(error, "status_code", None)
            raise SystemExit(
                f"API stopped: {type(error).__name__}, status={status}. No automatic retry."
            ) from None
        if response.status != "completed" or not response.output_text.strip():
            raise SystemExit(
                "Incomplete or empty response: not saved. No automatic retry; inspect before running again."
            )
        existing.append(
            {
                **{
                    k: row[k] for k in ("sample_id", "qa_index", "strategy", "question")
                },
                "reference_answer": row["answer"],
                "evidence": row["evidence"],
                "llm_answer": response.output_text.strip(),
                "correct": None,
                **({"retrieval_latency_ms": row["retrieval_latency_ms"]}
                   if "retrieval_latency_ms" in row else {}),
                "answer_api_latency_ms": latency_ms,
            }
        )
        save_answers(output, existing, metadata)
        print(
            f"{index + 1}/{min(args.limit, len(pending))} saved: {answer_key(row)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
