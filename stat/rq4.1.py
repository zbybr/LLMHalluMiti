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
temps = [0.1, 0.3, 0.6, 0.9]

rc_hal = np.array([17.22, 17.78, 19.44, 18.89])
hal_repair = np.array([64.56, 64.56, 65.82, 65.82])
oc = np.array([2.97, 4.95, 8.91, 7.92])
colors = ["blue", "green", "red"]

# ===== plot =====
fig, ax = plt.subplots(1, 3, figsize=(12, 3))

ax[1].plot(temps, rc_hal, marker="o", color=colors[1])
ax[1].set_xlabel("Temperature")
ax[1].set_ylabel("Rechecked Hallucination Rate (%) ↓")
ax[1].set_ylim(2, 32)

ax[0].plot(temps, hal_repair, marker="o", color=colors[0])
ax[0].set_xlabel("Temperature")
ax[0].set_ylabel("Hallucination Repair Rate (%) ↑")
ax[0].set_ylim(36, 78)

ax[2].plot(temps, oc, marker="o", color=colors[2])
ax[2].set_xlabel("Temperature")
ax[2].set_ylabel("Overcorrection Rate (%) ↓")
ax[2].set_ylim(0, 17)

for i in ax:
    i.spines["top"].set_visible(False)
    i.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("rq4.1.pdf")
plt.show()
