"""Held-out test accuracy and reasoning length during the LoRA RL run (data/training/metrics.jsonl)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
R = [json.loads(l) for l in open(ROOT / "data/training/metrics.jsonl")]
E = [r for r in R if "test/env/all/correct" in r]
GREY, BLUE = "#9aa0a6", "#1a73e8"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
steps = [r["step"] for r in E]
cells = [("test_ood-misread-agree", "held-out systems, note correct", BLUE, "-"),
         ("test_id-misread-agree", "seen systems, note correct", BLUE, "--"),
         ("test_ood-generic-contradict", "held-out systems, note wrong", GREY, "-"),
         ("test_ood-none-na", "held-out systems, no note", GREY, ":")]
for key, lab, col, ls in cells:
    ys = [100 * r[f"test/env/{key}/correct"] for r in E]
    a1.plot(steps, ys, ls, color=col, marker="o", lw=2, label=lab)
a1.set_xlabel("RL step"); a1.set_ylabel("Accuracy (%), higher is better"); a1.set_ylim(0, 105)
a1.legend(frameon=False, fontsize=9.5, loc="lower right"); a1.set_title("Test accuracy during training", loc="left", fontsize=12)
a2.plot(steps, [r["test/env/all/ac_tokens_per_turn"] for r in E], color=BLUE, marker="o", lw=2)
a2.set_xlabel("RL step"); a2.set_ylabel("Mean reply length (tokens)"); a2.set_ylim(0, None)
a2.set_title("Replies got about 13x shorter", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig(ROOT / "docs/training.png", dpi=200); print("ok")
