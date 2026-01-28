import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0

models = ["GPT-4o", "GPT-5", "Gemini", "Qwen3"]
datasets = ["TruthfulQA", "HotpotQA", "FreshQA"]

H = np.array([
    [45.78, 38.03, 30.33],
    [25.95, 19.18, 8.77],
    [28.64, 40.82, 16.04],
    [38.07, 56.89, 39.85],
])

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(H.T, cmap="YlOrBr", vmin=4, vmax=62, aspect="auto")

ax.set_xticks(np.arange(len(models)))
ax.set_xticklabels(models, fontsize=12)
ax.set_yticks(np.arange(len(datasets)))
ax.set_yticklabels(datasets, fontsize=12)
for spine in ax.spines.values():
    spine.set_visible(False)

for i in range(H.T.shape[0]):
    for j in range(H.T.shape[1]):
        ax.text(j, i, f"{H.T[i, j]:.1f}", ha="center", va="center", fontsize=10)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.ax.tick_params(labelsize=10, size=0)
cbar.outline.set_visible(False)
cbar.set_label("Hallucination rate (%)", fontsize=12)
ax.set_title("Hallucination Rates Heatmap Overview", fontsize=16)
plt.setp(ax.get_yticklabels(), rotation=90, ha="right", va="center")
fig.tight_layout()
fig.savefig("heatmap.pdf")
plt.show()
