# Experiment

**Research question:** How do retrieval strategies affect evidence recall, answer
accuracy and latency under limited conversational-memory budgets?

The design uses 20 fixed questions, three retrieval strategies, two answer models
(`gpt-5-nano`, `gpt-5.4-mini`) and two memory budgets (512, 1024 tokens).
Each model/budget group has 60 answers, giving 240 responses in total.
Environment setup and script descriptions are in the [README](../README.md).

## 1. Inspect the dataset and sample

The [LoCoMo dataset](https://github.com/snap-research/locomo) contains 10 conversations
and 1,986 QA pairs in `src/data/locomo10.json`. Each conversation contains sessions
of dialogue turns. An evidence ID such as `D1:3` identifies turn 3 in session 1;
`qa_index` is the question's zero-based index within its conversation.
See the [LoCoMo paper](https://aclanthology.org/2024.acl-long.747/) for benchmark background.

```sh
uv run python src/inspect_dataset.py
uv run python src/validate_evidence.py
uv run python src/select_samples.py
```

The fixed sample, `src/data/selected_qa_20.json`, contains two questions per conversation:

| Category | Questions | Task |
| --- | ---: | --- |
| 1: Multi-hop | 5 | Combine information from multiple turns. |
| 2: Temporal | 5 | Reason about dates or time. |
| 4: Single-hop | 10 | Recover a directly stated fact. |

Eligible questions have non-empty answers and evidence IDs mapping to non-empty
text. The initial draw used seed 42 to shuffle sorted conversation IDs. For each
conversation, draw one single-hop question followed by one multi-hop question
(first five conversations) or one temporal question (remaining five).
Three questions were replaced before observing experimental results, preserving
conversation and category:

| Conversation | Original → final QA | Reason |
| --- | --- | --- |
| `conv-26` | 110 → 104 | The original required an image detail. |
| `conv-44` | 2 → 3 | The reference's full activity list lacked support for the specified relationship. |
| `conv-47` | 32 → 23 | A travel plan did not establish an actual location. |

Replacement candidates were shuffled in QA-index order with
`random.Random("42:{sample_id}:{excluded_qa_index}")`, excluding selected questions.
The first acceptable candidate was retained. Reserve QA 12 in `conv-44` and QA 48
and 10 in `conv-47` were rejected for planned events or an ambiguous relative date.
Original questions, answers and evidence annotations remain unchanged.
`select_samples.py --draft` displays the initial draw; neither mode writes files.

## 2. Prepare retrieval contexts

Each question searches all original text turns in its own conversation: 5,882
candidate messages across the dataset. Images, captions and summaries are excluded.
Only populated `session_<number>` lists contribute messages; date-only fields do
not. For example, `conv-26` has 19 populated sessions and 419 messages despite
having 35 session-date fields.

Candidates use the format `[ID | session date | speaker]` followed by the text.

| Strategy | Ranking rule |
| --- | --- |
| Recency | Newest turns first, independent of the question. |
| BM25 | BM25Okapi, `k1=1.5`, `b=0.75`; lowercase `\b\w+\b` tokens with stopwords retained. |
| Semantic | Cosine similarity using normalized `text-embedding-3-small` vectors. |

Equal scores prefer newer turns. Whole messages are considered in rank order;
messages that do not fit are skipped. Selected turns are presented chronologically.
The budget counts the complete formatted context with `o200k_base`, including
headers and separators. The question and fixed instructions are outside this budget.

```sh
uv run python src/prepare_retrieval.py --budget 512
uv run python src/prepare_retrieval.py --budget 1024
```

Each command calls the embedding API and saves 60 contexts in identical
`inputs/retrieval.json` files for both models. Records contain question identifiers,
strategy, context, retrieved IDs, token count and retrieval latency. Reference
answers and evidence annotations stay local. Existing inputs are protected from
overwriting; `--dry-run` checks a new target without API calls. Keep completed
experiments intact when preparing a fresh run.

## 3. Generate answers

```sh
uv run python src/run_answers.py --model gpt-5-nano --budget 512
uv run python src/run_answers.py --model gpt-5.4-mini --budget 512
uv run python src/run_answers.py --model gpt-5-nano --budget 1024
uv run python src/run_answers.py --model gpt-5.4-mini --budget 1024
```

Both models use low reasoning, low verbosity and a 1024-token output limit.
Instructions request at most 50 words, date resolution against session dates,
and `Insufficient evidence.` when the context is inadequate. Requests contain only
the instructions, question and retrieved context. No temperature or model seed is
set; seed 42 controls request order. Run groups sequentially to reduce timing interference.

Results are saved after each completed response in
`outputs/<model>/<budget>_tokens/outputs/answer.json`. Its `metadata` describes the
configuration; `answers` contains identifiers, strategy, question,
`reference_answer`, `evidence`, `llm_answer`, `correct` and both latency fields.
New scores start as `null`. Existing responses are skipped, and failed or incomplete
requests stop without automatic retries. Use `--limit 1` for one new response or
`--dry-run` to inspect pending work without API calls.

## 4. Score answers

Compare each response with the reference, evidence turns and actual context.
Edit only `correct`: **1** for correct, **0** for incorrect or incomplete,
and **null** for a pending decision.

- Accept equivalent wording. Lists must cover all substantive reference items;
  additional claims need support from the original conversation.
- Grade dates at the reference's precision. Accepting the month/year does not
  validate an unsupported extra day.
- Abstentions score 0 because every selected question is answerable.
- Grade accuracy separately from evidence recall: unannotated turns can also
  support a correct answer.

Apply the same decisions across groups: making memories at the grand opening,
swimming as low-impact exercise, and "aptitude test" for the military test are accepted.
For `conv-47 / 138`, the question does not distinguish between James's game projects.
The conversation supports both Witcher 3 and Civilization/Total War interpretations;
the fixed rubric requires Witcher 3. Answers naming only the other games score 0
for missing the reference target, despite having source support.

Scoring uses LLM-assisted initial grading with targeted manual review. All 240
responses have binary scores: 239 remain initial grades; manual review confirmed
0 for Nano / 1024 / Semantic, `conv-48 / 80`, whose surfing answer contradicts
itself. Earlier decisions guide the rubric; scores are assigned to each new response.

## 5. Interpret latency

Timings use a monotonic clock and are stored in milliseconds:

| Field | Measurement |
| --- | --- |
| `retrieval_latency_ms` | Ranking and context packing, including Semantic's query-embedding API call. |
| `answer_api_latency_ms` | Request-to-complete-response time, including network and service time. |

Candidate embedding, index construction, file I/O and scoring are excluded.
Retrieval is measured once per question/strategy/budget and shared by both models.
The two stages are timed separately. `metadata.latency_measurement = "same_run"`
links each saved context to its new answer and corresponding timings.
`average_total_latency_ms` averages their per-question sums; it excludes preparation
and storage. It is not a directly measured duration of the complete workflow.

## 6. Calculate statistics and figures

After filling all 60 scores in each group:

```sh
uv run python src/analyze_results.py --model gpt-5-nano --budget 512
uv run python src/analyze_results.py --model gpt-5.4-mini --budget 512
uv run python src/analyze_results.py --model gpt-5-nano --budget 1024
uv run python src/analyze_results.py --model gpt-5.4-mini --budget 1024
uv run python src/plot_latency.py
```

For each strategy, the denominator is 20 questions:

```text
Answer accuracy (%) = 100 × sum(correct) / 20
Per-question evidence recall = |set(evidence) ∩ set(retrieved_ids)| / |set(evidence)|
Evidence recall (%) = 100 × mean(per-question evidence recall)
```

Each group's `outputs/summary.json` contains counts, accuracy, recall and mean
retrieval, answer API and combined latency. Means use all 20 questions; incomplete
scores or timing must be resolved before reporting. Values are rounded to at most
two decimals; counts and scores remain integers.

Each group's `figures/` contains accuracy and recall bar charts.
`outputs/figures/` contains:

- `latency_vs_recall.png`: two panels for the memory budgets, comparing retrieval time and recall.
- `latency_vs_accuracy.png`: four panels by model and budget, comparing combined time and accuracy.

Comparison figures use shared linear scales and directly labeled points:
gray for Recency, blue for BM25 and green for Semantic.

These results describe 20 questions from 10 conversations, with one generation per
configuration and targeted scoring review. Questions within a conversation are
dependent; recent evidence is not separately balanced. Timing varies with network,
service conditions, output length and reasoning. The study supports descriptive
comparisons within this sample, without establishing general model superiority
or full agent-task performance.
