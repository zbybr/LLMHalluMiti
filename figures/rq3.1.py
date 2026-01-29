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

# data: 3 runs
rc_hal = np.array([15.00, 17.22, 17.22])
repair = np.array([68.35, 67.09, 64.56])
oc = np.array([1.98, 4.95, 2.97])


means = [rc_hal.mean(), repair.mean(), oc.mean()]
stds = [rc_hal.std(), repair.std(), oc.std()]

labels = ["Recheck\n Hallucination Rate ↓",
          "Hallucination\n Repair Rate ↑",
          "Over-correction\n Rate ↓"]
colors = ["blue", "green", "red"]

x = np.arange(len(labels)) * 0.8
fig, ax = plt.subplots(figsize=(6, 3))
ax.hlines(
    means,
    x - 0.25, x + 0.25,
    colors="black",
    linewidth=2.4,
    color=colors,
)
ax.errorbar(
    x, means, yerr=stds,
    fmt='none',
    ecolor='black',
    capsize=6,
    elinewidth=2,
)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Value (%)", fontsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("rq3.1.pdf")
plt.show()
