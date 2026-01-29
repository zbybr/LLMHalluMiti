import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---------- paper-friendly matplotlib ----------
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.02

models = ["GPT-4o", "GPT-5", "Gemini", "Qwen3"]
strategies = ["Majority Voting", "Confidence Score", "Pairwise Ranking"]
colors = ["#F38181", "#FCE38A", "#95E1D3"]

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
        ax.bar(x + (i - 1) * width, data[:, i], width, label=strat, color=colors[i])

    # ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)

    # Clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # # Optional: show "↓" in title if lower is better
    # if lower_is_better and "↓" not in title:
    #     ax.set_title(title + " (↓ better)", fontsize=14)
    # elif (not lower_is_better) and "↑" not in title:
    #     ax.set_title(title + " (↑ better)", fontsize=14)


# ---------- choose layout ----------
# RQ2 recommended: 3-panel figure (repair, rc.hal, oc)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

grouped_bar(
    axes[0],
    repair_rate,
    ylabel="Hallucination Repair Rate (%) ↑ Better"
)

grouped_bar(
    axes[1],
    rechecked_hallu_rate,
    ylabel="Recheck Hallu Rate (%) ↓ Better"
)

grouped_bar(
    axes[2],
    overcorrection,
    ylabel="Over-correction Rate (%) ↓ Better"
)


# One shared legend (conference style)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, edgecolor="black", loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.05))

fig.tight_layout()
fig.savefig("rq2.1.pdf")
plt.show()
