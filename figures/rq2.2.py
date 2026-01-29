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
datasets = ["Average", "TruthfulQA", "HotpotQA", "FreshQA"]
methods = ["NAME without mutations", "NAME with mutations", "Pass@6"]
colors = ["#A8D8EA", "#AA96DA", "#FCBAD3"]

# ====== Fill with your screenshot numbers: (models, datasets) ======
oa = np.array([
    [65.15, 85.56, 27.16, 74.38],  # GPT-4o
    [49.45, 72.17, 10.26, 42.86],  # GPT-5
    [35.47, 51.71, 17.67, 45.31],  # Gemini
    [40.02, 70.74, 17.00, 30.19],  # Qwen3
])

oa_ra = np.array([
    [69.05, 87.97, 35.78, 74.38],  # GPT-4o
    [61.54, 80.19, 29.91, 54.29],  # GPT-5
    [46.62, 66.67, 26.10, 53.12],  # Gemini
    [45.17, 74.92, 19.60, 42.77],  # Qwen3
])

pass6 = np.array([
    [72.76, 90.37, 39.22, 82.64],  # GPT-4o
    [65.11, 80.66, 34.19, 74.29],  # GPT-5
    [48.81, 67.09, 30.12, 54.69],  # Gemini
    [48.10, 76.21, 23.05, 47.80],  # Qwen3
])

data_by_method = [oa, oa_ra, pass6]

# ====== Plot: 1x4 subplots (datasets), x=models, grouped bars=methods ======
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
x = np.arange(len(models))
width = 0.24

for d_idx, ds in enumerate(datasets):
    ax = axes[d_idx]
    for m_idx, method in enumerate(methods):
        vals = data_by_method[m_idx][:, d_idx]
        ax.bar(x + (m_idx - 1) * width, vals, width, label=method, color=colors[m_idx])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_xlabel(ds, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Hallucination Repair Rate (%)", fontsize=10)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True,
           edgecolor="black", bbox_to_anchor=(0.5, -0.08))

fig.tight_layout()
fig.savefig("rq5.pdf")
plt.show()
