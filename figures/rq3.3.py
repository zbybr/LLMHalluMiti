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
colors = ["blue", "green", "red"]

# ===== plot =====
fig, ax = plt.subplots(1, 3, figsize=(12, 3))

ax[1].plot(mut_n, rc_hal, marker="o", color=colors[1])
ax[1].set_xlabel("# Mutations")
ax[1].set_ylabel("Rechecked Hallucination Rate (%) ↓")
ax[1].set_ylim(2, 32)

ax[0].plot(mut_n, hal_repair, marker="o", color=colors[0])
ax[0].set_xlabel("# Mutations")
ax[0].set_ylabel("Hallucination Repair Rate (%) ↑")
ax[0].set_ylim(36, 78)

ax[2].plot(mut_n, oc, marker="o", color=colors[2])
ax[2].set_xlabel("# Mutations")
ax[2].set_ylabel("Overcorrection Rate (%) ↓")
ax[2].set_ylim(0, 17)

for i in ax:
    i.spines["top"].set_visible(False)
    i.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("rq3.3.pdf")
plt.show()
