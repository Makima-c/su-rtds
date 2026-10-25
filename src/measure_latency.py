"""Supplement the four scored experiments with timings; retain answers and grades."""

import argparse
import json
import os
from time import perf_counter

from experiment_paths import MODELS, ROOT, answer_key, load_answers, load_experiment
from retrieval import (
    MEMORY_BUDGETS,
    ConversationRetriever,
    OpenAIEncoder,
    build_memories,
)
from run_answers import request_for, save_answers


def load_groups(root=ROOT):
    groups = {}
    for budget in MEMORY_BUDGETS:
        for model in MODELS:
            folder, contexts = load_experiment(model, root=root, budget=budget)
            answers = load_answers(folder, contexts, complete=True)
            if any(row["correct"] is None for row in answers):
                raise ValueError(
                    "Finish scoring before supplementary latency measurement"
                )
            document = json.loads((folder / "outputs/answer.json").read_text())
            groups[model, budget] = {
                "folder": folder,
                "contexts": contexts,
                "document": document,
                "answers": {answer_key(row): row for row in document["answers"]},
            }
        first, second = (groups[model, budget] for model in MODELS)
        if first["contexts"] != second["contexts"]:
            raise ValueError("Both models must use identical prepared contexts")
    return groups


def save_group(group):
    document = group["document"]
    document["metadata"]["latency_measurement"] = "separate_run"
    save_answers(
        group["folder"] / "outputs/answer.json",
        document["answers"],
        document["metadata"],
    )


def measure_retrieval(groups, dataset, encoder, count_tokens, skip_mismatches=False):
    budgets = sorted({budget for _, budget in groups})
    for sample_id in sorted(dataset):
        retriever = None
        for budget in budgets:
            peers = [group for (model, b), group in groups.items() if b == budget]
            contexts = [
                row for row in peers[0]["contexts"] if row["sample_id"] == sample_id
            ]
            for row in contexts:
                key = answer_key(row)
                if any(
                    "retrieval_latency_ms" in group["answers"][key]
                    and group["answers"][key]["retrieval_latency_ms"] is None
                    for group in peers
                ):
                    # Recover an interruption between the two model-file saves without remeasuring.
                    for group in peers:
                        record = group["answers"][key]
                        if record.get("retrieval_latency_ms") is not None:
                            raise ValueError(
                                "Shared retrieval mismatch status differs between models"
                            )
                        if "retrieval_latency_ms" not in record:
                            record["retrieval_latency_ms"] = None
                            save_group(group)
                    continue
                saved = [
                    group["answers"][key].get("retrieval_latency_ms") for group in peers
                ]
                measured = [value for value in saved if value is not None]
                if measured:
                    if any(value != measured[0] for value in measured):
                        raise ValueError(
                            "Shared retrieval latency differs between models"
                        )
                    latency = measured[0]
                else:
                    if retriever is None:
                        memories = build_memories(dataset[sample_id])
                        retriever = ConversationRetriever(
                            memories, encoder, count_tokens
                        )
                    result = retriever.retrieve(
                        row["question"], row["strategy"], budget=budget
                    )
                    if any(
                        result[field] != row[field]
                        for field in ("context", "retrieved_ids", "memory_tokens")
                    ):
                        if not skip_mismatches:
                            raise ValueError(
                                f"Retrieval context changed: {key}, budget={budget}; no latency attached"
                            )
                        print(
                            f"Context mismatch left unmeasured: {key}, budget={budget}; "
                            f"original_only={sorted(set(row['retrieved_ids']) - set(result['retrieved_ids']))}; "
                            f"rerun_only={sorted(set(result['retrieved_ids']) - set(row['retrieved_ids']))}",
                            flush=True,
                        )
                        latency = None
                    else:
                        latency = result["retrieval_latency_ms"]
                for group in peers:
                    if "retrieval_latency_ms" not in group["answers"][key]:
                        group["answers"][key]["retrieval_latency_ms"] = latency
                        save_group(group)
        print(f"Retrieval timing checked: {sample_id}", flush=True)


def measure_answers(group, client):
    model = group["document"]["metadata"]["llm_model"]
    contexts = {answer_key(row): row for row in group["contexts"]}
    pending = [
        row
        for row in group["document"]["answers"]
        if "answer_api_latency_ms" not in row
    ]
    for index, answer in enumerate(pending, 1):
        context = contexts[answer_key(answer)]
        request = request_for(context["question"], context["context"], model)
        try:
            started = perf_counter()
            response = client.responses.create(**request)
            latency = round((perf_counter() - started) * 1000, 2)
        except Exception as error:
            status = getattr(error, "status_code", None)
            raise SystemExit(
                f"Timing request stopped: {type(error).__name__}, status={status}. No automatic retry."
            ) from None
        if response.status != "completed" or not response.output_text.strip():
            raise SystemExit(
                "Incomplete timing response; no latency saved. No automatic retry."
            )
        # The supplementary response is discarded; the original answer and score stay intact.
        answer["answer_api_latency_ms"] = latency
        save_group(group)
        print(
            f"{model} {group['folder'].name}: timing {index}/{len(pending)} saved",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate existing groups without loading a key or calling the API.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "retrieval", "answers"),
        default="all",
        help="Measure both stages, or continue one stage independently.",
    )
    parser.add_argument(
        "--skip-context-mismatches",
        action="store_true",
        help="Leave changed contexts unmeasured and continue the remaining retrieval cases.",
    )
    args = parser.parse_args()
    groups = load_groups()
    fields = ("retrieval_latency_ms", "answer_api_latency_ms")
    if args.stage != "all":
        fields = (fields[0] if args.stage == "retrieval" else fields[1],)
    missing = {
        field: sum(
            field not in row
            for group in groups.values()
            for row in group["document"]["answers"]
        )
        for field in fields
    }
    print("Missing timing fields:", missing, flush=True)
    if args.dry_run or not any(missing.values()):
        print("No API requests sent.")
        return
    import tiktoken
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env", override=False)
    embedding_models = {
        group["document"]["metadata"]["embedding_model"] for group in groups.values()
    }
    embedding_model = os.environ.get("OPENAI_TEXT_EMBEDDING_MODEL", "").strip()
    if embedding_models != {embedding_model}:
        raise SystemExit(
            "Embedding model must match the existing experiments; no API call sent."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing API key; no API call sent.")
    client = OpenAI(max_retries=0, timeout=90.0)
    if args.stage != "answers":
        memory_encoding = tiktoken.get_encoding("o200k_base")
        embedding_encoding = tiktoken.get_encoding("cl100k_base")
        count_tokens = lambda text: len(
            memory_encoding.encode(text, disallowed_special=())
        )
        encoder = OpenAIEncoder(
            client,
            embedding_model,
            count_tokens=lambda text: len(
                embedding_encoding.encode(text, disallowed_special=())
            ),
        )
        dataset = {
            sample["sample_id"]: sample
            for sample in json.loads((ROOT / "src/data/locomo10.json").read_text())
        }
        try:
            measure_retrieval(
                groups,
                dataset,
                encoder,
                count_tokens,
                skip_mismatches=args.skip_context_mismatches,
            )
        except Exception as error:
            if isinstance(error, ValueError):
                raise SystemExit(str(error)) from None
            raise SystemExit(
                f"Retrieval timing stopped: {type(error).__name__}. No automatic retry."
            ) from None
    if args.stage != "retrieval":
        for group in groups.values():
            measure_answers(group, client)
    print("Requested timing stage complete. Original answers and scores retained.")


if __name__ == "__main__":
    main()
