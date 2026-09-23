"""E3 comparison: policy trained with vs without domain randomisation.

    uv run python scripts/compare_runs.py v1 nodr
Reads results/eval_<tag>/{summary.json, mjx_grid.csv}; writes results/e3_comparison.png + .md
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

tags = sys.argv[1:] or ["v1", "nodr"]
labels = {"v1": "with randomisation", "nodr": "no randomisation"}
colors = {"v1": "#2a6f97", "nodr": "#c8553d"}
R = Path("results")

data = {}
for t in tags:
    s = json.loads((R / f"eval_{t}" / "summary.json").read_text())
    rows = list(csv.DictReader(open(R / f"eval_{t}" / "mjx_grid.csv")))
    ok = [r for r in rows if r["fell"] == "False"]
    data[t] = {
        "mjx_err": 100 * np.mean([float(r["step_err"]) for r in ok]),
        "c_err": s["grid_step_abs_err_cm"],
        "stress": {x["case"]: x["fall_rate"] for x in s["stress"]},
    }

cases = list(data[tags[0]]["stress"].keys())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 2.4]})
w = 0.38
x = np.arange(2)
for i, t in enumerate(tags):
    a1.bar(x + (i - 0.5) * w, [data[t]["mjx_err"], data[t]["c_err"]], w, label=labels.get(t, t), color=colors.get(t))
a1.set_xticks(x, ["training engine\n(MJX)", "unseen engine\n(MuJoCo C)"])
a1.set_ylabel("step-length error (cm)")
a1.set_title("Precision: lower is better")
a1.legend(frameon=False)

x = np.arange(len(cases))
for i, t in enumerate(tags):
    a2.bar(x + (i - 0.5) * w, [100 * data[t]["stress"][c] for c in cases], w, label=labels.get(t, t), color=colors.get(t))
a2.set_xticks(x, [c.replace("_", "\n") for c in cases], fontsize=8)
a2.set_ylabel("fall rate (%)")
a2.set_ylim(0, 105)
a2.set_title("Robustness in the unseen engine: lower is better")
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("E3: what domain randomisation buys (and costs)")
fig.tight_layout()
fig.savefig(R / "e3_comparison.png", dpi=140)

lines = ["| metric | " + " | ".join(labels.get(t, t) for t in tags) + " |", "|---|" + "---|" * len(tags),
         "| step error, training engine (cm) | " + " | ".join(f"{data[t]['mjx_err']:.1f}" for t in tags) + " |",
         "| step error, unseen engine (cm) | " + " | ".join(f"{data[t]['c_err']:.1f}" for t in tags) + " |"]
for c in cases:
    lines.append(f"| falls: {c} | " + " | ".join(f"{100 * data[t]['stress'][c]:.0f} %" for t in tags) + " |")
(R / "e3_comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
