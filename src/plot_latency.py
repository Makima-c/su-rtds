"""Plot latency against retrieval recall and answer accuracy across all four groups."""

import json

from analyze_results import summarize
from experiment_paths import MODELS, ROOT, answer_key, load_answers, load_experiment
from retrieval import MEMORY_BUDGETS, STRATEGIES

COLORS = {'recency': '#7c8792', 'bm25': '#315b86', 'semantic': '#4b897c'}
LABELS = {'recency': 'Recency', 'bm25': 'BM25', 'semantic': 'Semantic'}
MODEL_LABELS = {'gpt-5-nano': 'GPT-5 nano', 'gpt-5.4-mini': 'GPT-5.4 mini'}


def load_plot_data(root=ROOT):
    data = {}
    for budget in MEMORY_BUDGETS:
        for model in MODELS:
            folder, contexts = load_experiment(model, root=root, budget=budget)
            answers = load_answers(folder, contexts, complete=True)
            lookup = {answer_key(row): row for row in contexts}
            computed = summarize([{**lookup[answer_key(row)], **row} for row in answers],
                                 allow_missing_retrieval=True)
            saved = json.loads((folder / 'outputs/summary.json').read_text())
            if saved != computed:
                raise ValueError('Summary is stale; run analyze_results.py for each group first')
            for row in saved:
                if 'average_total_latency_ms' not in row:
                    raise ValueError('All four groups need complete latency measurements')
                if row['average_retrieval_latency_ms'] is not None and row['average_retrieval_latency_ms'] < 0:
                    raise ValueError('Retrieval latency cannot be negative')
            data[model, budget] = {row['retrieval_method']: row for row in saved}
        for strategy in STRATEGIES:
            first, second = (data[model, budget][strategy] for model in MODELS)
            if any(first[field] != second[field] for field in
                   ('evidence_recall_percent', 'average_retrieval_latency_ms')):
                raise ValueError('Shared retrieval results differ between models')
    return data


def plot_relationships(data, output, separate_run=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

    def draw_panel(ax, rows, latency_field, quality_field, title, maximum):
        missing = []
        for strategy in STRATEGIES:
            row = rows[strategy]
            latency = row[latency_field]
            if latency is None:
                missing.append(LABELS[strategy])
                continue
            quality = row[quality_field]
            ax.scatter(
                latency, quality, marker="o", color=COLORS[strategy],
                s=90, edgecolors="white", linewidths=0.8, zorder=3,
            )
            # Put nearby BM25/Semantic labels on opposite vertical sides.
            vertical = -17 if strategy == "bm25" else 9
            right_edge = latency > maximum * 0.72
            ax.annotate(
                LABELS[strategy], (latency, quality),
                xytext=(-8 if right_edge else 8, vertical),
                textcoords="offset points",
                ha="right" if right_edge else "left", va="center",
                color="#28323e", fontsize=10,
            )
        ax.set(title=title, xscale="linear", xlim=(0, maximum * 1.25), ylim=(-5, 105))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(alpha=0.18)
        if missing:
            ax.text(0.02, 0.97, "Timing unavailable: " + ", ".join(missing),
                    transform=ax.transAxes, va="top", fontsize=8, color="#5b6470")

    retrieval_max = max(
        row["average_retrieval_latency_ms"]
        for budget in MEMORY_BUDGETS for row in data[MODELS[0], budget].values()
        if row["average_retrieval_latency_ms"] is not None
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 3.9), sharex=True, sharey=True, layout="constrained",
    )
    for ax, budget in zip(axes, MEMORY_BUDGETS):
        draw_panel(
            ax, data[MODELS[0], budget], "average_retrieval_latency_ms",
            "evidence_recall_percent", f"{budget}-token memory", retrieval_max,
        )
    fig.suptitle("Retrieval latency and evidence recall", fontsize=13)
    fig.supxlabel("Mean retrieval latency (ms)", fontsize=11)
    fig.supylabel("Evidence recall (%)", fontsize=11)
    fig.savefig(output / "latency_vs_recall.png", dpi=220)
    plt.close(fig)

    total_max = max(
        row["average_total_latency_ms"] for rows in data.values() for row in rows.values()
        if row["average_total_latency_ms"] is not None
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(10.2, 6.8), sharex=True, sharey=True, layout="constrained",
    )
    for row_index, model in enumerate(MODELS):
        for column, budget in enumerate(MEMORY_BUDGETS):
            draw_panel(
                axes[row_index, column], data[model, budget], "average_total_latency_ms",
                "answer_accuracy_percent", f"{MODEL_LABELS[model]} | {budget} tokens", total_max,
            )
    title = "Answer accuracy and latency"
    if separate_run:
        title += "\nQuality scores with separate timing measurements"
    fig.suptitle(title, fontsize=13)
    fig.supxlabel("Mean retrieval + answer API latency (ms)", fontsize=11)
    fig.supylabel("Answer accuracy (%)", fontsize=11)
    fig.savefig(output / "latency_vs_accuracy.png", dpi=220)
    plt.close(fig)


def main():
    data = load_plot_data()
    output = ROOT / 'outputs/figures'
    separate_run = any(
        json.loads((ROOT / 'outputs' / model / f'{budget}_tokens/outputs/answer.json').read_text())
        ['metadata'].get('latency_measurement') == 'separate_run'
        for model in MODELS for budget in MEMORY_BUDGETS
    )
    plot_relationships(data, output, separate_run=separate_run)
    print('Saved', output / 'latency_vs_recall.png')
    print('Saved', output / 'latency_vs_accuracy.png')


if __name__ == '__main__':
    main()
