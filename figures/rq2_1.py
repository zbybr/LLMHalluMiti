import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import sys

# ---------- paper-friendly matplotlib ----------
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.02

models = ["GPT-4o", "GPT-5", "Gemini", "Qwen3"]
strategies = ["Majority Voting", "Confidence Score", "Pairwise Ranking"]
colors = ["#A8D8EA", "#FCBAD3", "#AA96DA"]
hatches = ['//', '\\\\', '++', 'xx']

# ========= Fill with your RQ2 numbers =========
# Each array shape: (num_models, num_strategies)
# Values should be percentages (0-100) consistent with your paper tables.
repair_rate = np.array([
    # MV,    CS,    RA
    [66.71, 64.24, 69.05],   # GPT-4o (Overall example)
    [51.37, 56.59, 61.54],   # GPT-5
    [39.31, 43.33, 46.62],   # Gemini
    [41.13, 42.72, 45.17],   # Qwen3
])

rechecked_hallu_rate = np.array([
    # MV,    CS,    RA
    [15.55, 19.11, 14.07],
    [10.08, 9.15,  7.89],
    [18.89, 17.80, 16.43],
    [28.31, 27.66, 25.96],
])

overcorrection = np.array([
    # MV,   CS,   RA
    [3.91, 8.10, 2.91],
    [0.48, 0.62, 0.27],
    [1.09, 1.17, 0.63],
    [3.57, 3.67, 2.58],
])

# ---------- plot helper ----------
def grouped_bar(ax, data, ylabel):
    x = np.arange(len(models))
    width = 0.24

    for i, strat in enumerate(strategies):
        ax.bar(x + (i - 1) * width, data[:, i], width, label=strat, color=colors[i], edgecolor='0.6', hatch=hatches[i])

    # ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)

    # Clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # # Optional: show "↓" in title if lower is better
    # if lower_is_better and "↓" not in title:
    #     ax.set_title(title + " (↓ better)", fontsize=14)
    # elif (not lower_is_better) and "↑" not in title:
    #     ax.set_title(title + " (↑ better)", fontsize=14)


# ---------- choose layout ----------
# One row of three metric panels with a shared legend underneath.
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

grouped_bar(
    axes[0],
    rechecked_hallu_rate,
    ylabel="Recheck Hallu Rate (%) ↓"
)

grouped_bar(
    axes[1],
    repair_rate,
    ylabel="Hallu Repair Rate (%) ↑"
)

grouped_bar(
    axes[2],
    overcorrection,
    ylabel="Over-correction Rate (%) ↓"
)

# Place the shared legend below all three panels.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center", bbox_to_anchor=(0.5, 0.0),
    ncol=3, frameon=True, edgecolor="black",
    fontsize=12, handlelength=2.0, handleheight=1.2,
    borderpad=0.45, columnspacing=1.8,
)

fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0), w_pad=1.5)
output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("rq2.1.pdf")
fig.savefig(output_path)
plt.close(fig)
