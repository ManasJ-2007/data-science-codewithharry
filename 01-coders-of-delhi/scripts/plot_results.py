"""Bar charts of results/evaluation.json (needs matplotlib; the recommender itself needs nothing).

Usage:  python scripts/plot_results.py   ->   results/people_hit10.png, results/pages_hit5.png
"""
import json
import os

import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def bar(res, metric, title, path):
    methods = list(res)
    means = [res[m][metric][0] * 100 for m in methods]
    stds = [res[m][metric][1] * 100 for m in methods]
    colors = ["#9aa0a6" if m in ("random", "popularity") else "#1a73e8" for m in methods]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(methods, means, yerr=stds, color=colors, capsize=4)
    for i, v in enumerate(means):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    r = json.load(open(os.path.join(OUT, "evaluation.json")))
    bar(r["people_you_may_know"], "hit@10", "People You May Know: hidden friend in top 10 (3 seeds)",
        os.path.join(OUT, "people_hit10.png"))
    bar(r["pages_you_might_like"], "hit@5", "Pages You Might Like: hidden page in top 5 (3 seeds)",
        os.path.join(OUT, "pages_hit5.png"))


if __name__ == "__main__":
    main()
