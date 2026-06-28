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

# ===== data =====
mut_n = np.array([0, 1, 3, 5, 7, 9])

rc_hal = np.array([22.22, 19.44, 18.33, 17.22, 18.33, 16.67])
hal_repair = np.array([55.70, 60.76, 64.56, 64.56, 67.09, 67.09])
oc = np.array([4.95, 3.96, 4.95, 2.97, 6.93, 3.96])
colors = ["#8675A9", "#C3AED6", "#EFBBCF"]

# ===== plot: 2x2 grid (3 metric panels) + legend in bottom-right cell =====
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

l0, = axes[0, 0].plot(mut_n, rc_hal, marker="o", color=colors[0])
axes[0, 0].set_xlabel("# Mutations", fontsize=16)
axes[0, 0].set_ylabel("Recheck Hallu Rate (%) ↓", fontsize=16)
axes[0, 0].set_ylim(2, 32)

l1, = axes[0, 1].plot(mut_n, hal_repair, marker="o", color=colors[1])
axes[0, 1].set_xlabel("# Mutations", fontsize=14)
axes[0, 1].set_ylabel("Hallucination Repair Rate (%) ↑", fontsize=16)
axes[0, 1].set_ylim(36, 78)

l2, = axes[1, 0].plot(mut_n, oc, marker="o", color=colors[2])
axes[1, 0].set_xlabel("# Mutations", fontsize=16)
axes[1, 0].set_ylabel("Over-correction Rate (%) ↓", fontsize=16)
axes[1, 0].set_ylim(0, 17)

for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.tick_params(axis='both', labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Bottom-right cell holds the shared legend instead of a plot.
legend_ax = axes[1, 1]
legend_ax.axis("off")
legend_ax.legend(
    [l0, l1, l2],
    ["Recheck Hallu Rate", "Hallucination Repair Rate", "Over-correction Rate"],
    loc="center", ncol=1, frameon=True, edgecolor="black",
    fontsize=12, handlelength=2.0, borderpad=0.5, labelspacing=1.0,
)

fig.align_ylabels([axes[0, 0], axes[1, 0]])
fig.tight_layout(pad=0.0, w_pad=1.5)
fig.savefig("rq3.3.pdf")
plt.show()
