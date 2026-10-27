"""Central Limit Theorem check: sample means of a skewed (exponential) distribution become normal.

Usage:  python notes/clt_demo.py   ->  prints the table in 04_probability_statistics.md, saves notes/clt_demo.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
REPS, SIGMA = 20_000, 1.0  # Exponential(1): mean 1, std 1, skewness 2

fig, axes = plt.subplots(1, 4, figsize=(13, 3))
print("| n | skewness of mean (theory 2/sqrt(n)) | measured std / (sigma/sqrt(n)) |")
for ax, n in zip(axes, (1, 5, 30, 100)):
    means = rng.exponential(1.0, size=(REPS, n)).mean(axis=1)
    skew, ratio = stats.skew(means), means.std() / (SIGMA / np.sqrt(n))
    print(f"| {n} | {skew:.2f} ({2 / np.sqrt(n):.2f}) | {ratio:.3f} |")
    ax.hist(means, bins=60, density=True, color="#8ab4f8")
    x = np.linspace(means.min(), means.max(), 200)
    ax.plot(x, stats.norm.pdf(x, 1, SIGMA / np.sqrt(n)), "r", lw=1.5, label="CLT normal")
    ax.set_title(f"n = {n}, skew = {skew:.2f}")
    ax.set_yticks([])
axes[0].legend(fontsize=8)
fig.suptitle("Means of n exponential samples approach N(μ, σ²/n)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "clt_demo.png"), dpi=120, bbox_inches="tight")
