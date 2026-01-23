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

runs = ["Run 1", "Run 2", "Run 3"]

# ======= RQ3 numbers from your table =======
rc_hal = np.array([15.00, 17.22, 17.22])      # rechecked hallucination rate (%)
hal_repair = np.array([68.35, 67.09, 64.56])  # hallucination repair rate (%)
oc = np.array([1.98, 4.95, 2.97])             # overcorrection rate (%)

def mean_std(a):
    return float(np.mean(a)), float(np.std(a, ddof=1))  # sample std

rc_mean, rc_std = mean_std(rc_hal)
rr_mean, rr_std = mean_std(hal_repair)
oc_mean, oc_std = mean_std(oc)

# ---------- Plot: per-run line + mean±std band ----------
x = np.arange(len(runs))
fig, ax = plt.subplots(figsize=(6.4, 3.6))

# rc.hal (↓ better)
ax.plot(x, rc_hal, marker="o", label="Rechecked Hallucination Rate (↓)")
ax.fill_between(x, rc_mean - rc_std, rc_mean + rc_std, alpha=0.15)

# hal.repair (↑ better)
ax.plot(x, hal_repair, marker="o", label="Hallucination Repair Rate (↑)")
ax.fill_between(x, rr_mean - rr_std, rr_mean + rr_std, alpha=0.15)

# oc (↓ better)
ax.plot(x, oc, marker="o", label="Overcorrection Rate (↓)")
ax.fill_between(x, oc_mean - oc_std, oc_mean + oc_std, alpha=0.15)

ax.set_xticks(x)
ax.set_xticklabels(runs)
ax.set_ylabel("Value (%)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend bottom with frame (conference style)
ax.legend(loc="lower center", ncol=1, frameon=True, edgecolor="black", bbox_to_anchor=(0.5, -0.35))

fig.tight_layout()
fig.savefig("rq3.pdf")
plt.show()

