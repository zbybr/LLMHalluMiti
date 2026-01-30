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
datasets = ["TruthfulQA", "HotpotQA", "FreshQA"]
methods = ["MutRepair without mutations", "MutRepair without injection", "MutRepair", "Pass@6"]
colors = ["#FFD5CD", "#EFBBCF", "#C3AED6", "#8675A9"]
# colors = ["#B5EAEA", "#EDF6E5", "#FFBCBC", "#F38BA0"]
hatches = ['//', '\\\\', '++', 'xx']

# ====== Fill with your screenshot numbers: (models, datasets) ======
# cot = np.array([
#     [49.45, 56.95, 27.16, 68.6],  # GPT-4o
#     [29.4, 38.68, 15.38, 20],  # GPT-5
#     [24.5, 42.31, 7.23, 26.56],  # Gemini
#     [26.19, 44.69, 11.24, 22.64],  # Qwen3
# ])
#
# oa = np.array([
#     [65.15, 85.56, 27.16, 74.38],  # GPT-4o
#     [49.45, 72.17, 15.38, 42.86],  # GPT-5
#     [35.47, 51.71, 17.67, 45.31],  # Gemini
#     [40.02, 70.74, 17.00, 30.19],  # Qwen3
# ])
#
# oa_cot = np.array([
#     [55.03, 61.48, 25.43, 72.03],  # GPT-4o
#     [38.42, 45.13, 10.26, 26.60],  # GPT-5
#     [33.81, 57.28, 11.21, 32.70],  # Gemini
#     [31.04, 49.70, 13.61, 33.68],  # Qwen3
# ])
#
# oa_ra = np.array([
#     [69.05, 87.97, 35.78, 74.38],  # GPT-4o
#     [61.54, 80.19, 29.91, 54.29],  # GPT-5
#     [46.62, 66.67, 26.10, 53.12],  # Gemini
#     [45.17, 74.92, 19.60, 42.77],  # Qwen3
# ])
#
# pass6 = np.array([
#     [72.76, 90.37, 39.22, 82.64],  # GPT-4o
#     [65.11, 80.66, 34.19, 74.29],  # GPT-5
#     [48.81, 67.09, 30.12, 54.69],  # Gemini
#     [48.10, 76.21, 23.05, 47.80],  # Qwen3
# ])

oa = np.array([
    [85.56, 27.16, 74.38],  # GPT-4o
    [72.17, 15.38, 42.86],  # GPT-5
    [51.71, 17.67, 45.31],  # Gemini
    [70.74, 17.00, 30.19],  # Qwen3
])

oa_cot = np.array([
    [61.48, 25.43, 72.03],  # GPT-4o
    [45.13, 10.26, 26.60],  # GPT-5
    [57.28, 11.21, 32.70],  # Gemini
    [49.70, 13.61, 33.68],  # Qwen3
])

oa_ra = np.array([
    [87.97, 35.78, 74.38],  # GPT-4o
    [80.19, 29.91, 54.29],  # GPT-5
    [66.67, 26.10, 53.12],  # Gemini
    [74.92, 19.60, 42.77],  # Qwen3
])

pass6 = np.array([
    [90.37, 39.22, 82.64],  # GPT-4o
    [80.66, 34.19, 74.29],  # GPT-5
    [67.09, 30.12, 54.69],  # Gemini
    [76.21, 23.05, 47.80],  # Qwen3
])

data_by_method = [oa, oa_cot, oa_ra, pass6]

# ====== Plot: 1x4 subplots (datasets), x=models, grouped bars=methods ======
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
x = np.arange(len(models))
width = 0.2

for d_idx, ds in enumerate(datasets):
    ax = axes[d_idx]
    for m_idx, method in enumerate(methods):
        vals = data_by_method[m_idx][:, d_idx]
        ax.bar(x + (m_idx - (len(methods)-1)/2) * width, vals, width, label=method, color=colors[m_idx], edgecolor='0.6', hatch=hatches[m_idx])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_xlabel(ds, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Hallucination Repair Rate (%)", fontsize=12)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True,
           edgecolor="black", bbox_to_anchor=(0.5, -0.08))

fig.tight_layout()
fig.savefig("rq2.2.pdf")
plt.show()
