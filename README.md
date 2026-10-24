# RTDS Memory Retrieval Experiment

A research project for Research Topics in Data Science at Stockholm University.
It compares Recency, BM25 and Semantic retrieval on 20 LoCoMo questions with
GPT-5 nano and GPT-5.4 mini under 512- and 1024-token memory budgets. Evaluation
covers answer accuracy, evidence recall and latency. Each model/budget group has
60 answers; the four groups contain 240 answers in total.

See the [experiment guide](docs/experiment.md) for the complete workflow and methods.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). Run from the project root:

```sh
uv sync --frozen
```

Create `.env` from `.env.example` if it does not already exist, then set:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_TEXT_EMBEDDING_MODEL=text-embedding-3-small
```

Choose the answer model with `--model` and the memory budget with `--budget`
(default: 512). Keep the embedding model unchanged throughout the experiment.

## Scripts

| Script in `src/` | Purpose and output |
| --- | --- |
| `inspect_dataset.py` | Download missing data; print dataset structure and examples. |
| `validate_evidence.py` | Print evidence-ID validation counts. |
| `select_samples.py` | Validate the fixed sample; `--draft` previews the initial draw. |
| `prepare_retrieval.py` | Save 60 contexts per budget, shared by both models. |
| `run_answers.py` | Generate answers and record API latency in `answer.json`. |
| `analyze_results.py` | Calculate `summary.json` and accuracy/recall charts. |
| `plot_latency.py` | Plot latency against recall and accuracy across groups. |
| `retrieval.py` | Shared retrieval, embedding, packing and recall functions. |
| `experiment_paths.py` | File paths, sample joins and record validation. |
| `measure_latency.py` | Optional timing supplement for older runs; unused in the current workflow. |

## Run

Check a prepared group without API calls:

```sh
uv run python src/run_answers.py --model gpt-5.4-mini --budget 1024 --dry-run
```

Remove `--dry-run` to generate missing answers; add `--limit 1` for one response.
Preparation and generation incur API charges. Existing answers are skipped,
retrieval inputs are protected from overwriting, and failed requests stop without retries.

## Files and checks

- `src/data/`: original dataset and fixed question sample.
- `outputs/<model>/<budget>_tokens/`: retrieval inputs, answer/summary JSON and figures.
- `outputs/figures/`: comparison figures.

```sh
uv run python -m unittest discover -s tests -v
```
