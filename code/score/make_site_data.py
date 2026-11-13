"""Recompute every number the writeup shows and write docs/data.js.

Reads the raw replies in data/results/, the questions in data/items/, the RL metrics in data/training/
and the Lorenz orbit extract in data/orbit_page.npz.
"""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS, ITEMS = ROOT / "data/results", ROOT / "data/items"


def load(path):
    """Read a .jsonl (or .jsonl.gz) file into a dict keyed by id."""
    gz = Path(str(path) + ".gz")
    fh = gzip.open(gz, "rt") if gz.exists() else open(path)
    return {row["id"]: row for row in map(json.loads, fh)}


def correct(reply):
    return reply["pred"] is not None and reply["pred"] == reply["answer"]


def pct(flags):
    return 100 * sum(flags) / len(flags)


def metrics(path):
    return [json.loads(line) for line in open(path)]


# --- the main 2,000-question evaluation --------------------------------------------------------
items = load(ITEMS / "atlas_pilot3b/items.jsonl")
base, rl = load(RESULTS / "main/base.jsonl"), load(RESULTS / "main/lora.jsonl")


def with_claim(claim, ids=items):
    return [i for i in ids if items[i]["claim"] == claim]


arms = {c: {"base": pct([correct(base[i]) for i in with_claim(c)]),
            "rl": pct([correct(rl[i]) for i in with_claim(c)]),
            "n": len(with_claim(c))}
        for c in ["none", "arith", "catalogue", "db", "generic"]}

# accuracy split by the true label, i.e. by which way the wrong claim points
dirs = {c: {label: {"base": pct([correct(base[i]) for i in with_claim(c) if items[i]["label"] == label]),
                    "rl": pct([correct(rl[i]) for i in with_claim(c) if items[i]["label"] == label])}
            for label in ("chaotic", "regular")}
        for c in ("arith", "catalogue", "db")}

# base accuracy on finished replies only (unfinished replies count as wrong everywhere else)
parsed = {c: pct([correct(base[i]) for i in with_claim(c) if base[i]["pred"] is not None])
          for c in ("arith", "db", "catalogue", "none")}

# --- reasoning effort (16k-token cap, then two runs each at a 32k cap) ---------------------------
EFFORTS = ["0.1", "0.5", "0.9", "0.99"]
eff = {e: load(RESULTS / f"effort/effort{e}.jsonl") for e in EFFORTS[:3]}
ids = list(eff["0.1"])
eff["0.99"] = {i: base[i] for i in ids}  # effort 0.99 is the main evaluation itself
effort = {c: {"base": [pct([correct(eff[e][i]) for i in with_claim(c, ids)]) for e in EFFORTS],
              "rl": pct([correct(rl[i]) for i in with_claim(c, ids)]),
              "tokens": [round(np.mean([eff[e][i].get("out", 0) for i in with_claim(c, ids)])) for e in EFFORTS],
              "rl_tokens": round(np.mean([rl[i].get("out", 0) for i in with_claim(c, ids)])),
              "trunc099": sum(base[i]["pred"] is None for i in with_claim(c, ids))}
          for c in ["arith", "db", "none"]}

runs32k = {e: [load(RESULTS / f"effort/effort{e}_32k_{run}.jsonl") for run in "ab"] for e in ("0.1", "0.99")}
effort32k = {c: {e: [pct([correct(R[i]) for i in R if items[i]["claim"] == c]) for R in runs32k[e]] for e in runs32k}
             for c in ["arith", "db", "none"]}

# --- the same wrong claim credited to four sources ---------------------------------------------
dose_items = load(ITEMS / "inkling_rl/dose.jsonl")
dose_base, dose_rl = load(RESULTS / "sources/base.jsonl"), load(RESULTS / "sources/lora.jsonl")
dose = {}
for source in ["user", "student", "catalogue", "verified_db"]:
    sel = [i for i in dose_items if dose_items[i]["auth"] == source]
    dose[source] = {"base": pct([correct(dose_base[i]) for i in sel]),
                    "rl": pct([correct(dose_rl[i]) for i in sel]), "n": len(sel)}

# --- RL training curves and cost ---------------------------------------------------------------
M = metrics(ROOT / "data/training/metrics.jsonl")
train = [{"step": r["step"],
          "agree_ood": round(100 * r["test/env/test_ood-misread-agree/correct"]),
          "agree_id": round(100 * r["test/env/test_id-misread-agree/correct"]),
          "none_ood": round(100 * r["test/env/test_ood-none-na/correct"]),
          "contra_ood": round(100 * r["test/env/test_ood-generic-contradict/correct"]),
          "tokens": round(r["test/env/all/ac_tokens_per_turn"])}
         for r in M if "test/env/all/correct" in r]
cost = round(sum(r.get("cost/step_usd", 0) for r in M), 2)

seed1 = [{"step": r["step"], "agree_ood": round(100 * r["test/env/test_ood-misread-agree/correct"]),
          "all": round(100 * r["test/env/all/correct"])}
         for r in metrics(ROOT / "data/training/metrics_seed1.jsonl") if "test/env/all/correct" in r]

# --- checks on the trained model ---------------------------------------------------------------
mmlu = {k: load(RESULTS / f"checks/mmlu_{k}.jsonl") for k in ("base", "lora")}
keep = [dict(label="MMLU\n(200 questions)",
             base=pct([r["pred"] == r["answer"] for r in mmlu["base"].values()]),
             rl=pct([r["pred"] == r["answer"] for r in mmlu["lora"].values()]),
             note="general knowledge, effort 0.99")]

plain_items = load(ITEMS / "inkling_rl/probe_natural.jsonl")
natural = {m: {c: pct([correct(r) for i, r in load(RESULTS / f"checks/plain_language_{m}.jsonl").items()
                       if plain_items[i]["claim"] == c])
               for c in ("arith", "db")}
           for m in ("base", "lora", "haiku")}

win_items = load(ITEMS / "inkling_rl/notewins.jsonl")
notewins = {}
for m in ("base", "lora", "haiku"):
    R = load(RESULTS / f"checks/claim_should_win_{m}.jsonl")
    wins = [i for i in R if win_items[i]["kind"] == "wins"]
    notewins[m] = [sum(correct(R[i]) for i in wins), len(wins)]

# --- model comparison: settings 0-19 (20 chaotic settings), every model at its default thinking ---
MODELS = {"Inkling 2 Small": "inkling2_small", "Inkling 2 Small + LoRA": "inkling2_small_lora",
          "Inkling-Small (previous generation)": "inkling_small", "Qwen3.5-397B-A17B": "qwen3.5_397b",
          "DeepSeek-V4.1-Flash": "deepseek_v4.1_flash", "Claude Haiku 5.5": "claude_haiku_5.5"}
CLAIMS = ["none", "arith", "catalogue", "db"]
subset = [i for i in items if items[i]["setting"] <= 19]
compare = []
for name, file in MODELS.items():
    R = load(RESULTS / f"models/{file}.jsonl")
    sel = {c: [i for i in with_claim(c, subset) if i in R] for c in CLAIMS}
    compare.append(dict(name=name, n=sum(i in R for i in subset),
                        arms={c: pct([correct(R[i]) for i in sel[c]]) if sel[c] else None for c in CLAIMS},
                        nk={c: len(sel[c]) for c in CLAIMS}))

# --- Lorenz orbit for the hero and the horizon chart -------------------------------------------
# head and tail of one orbit from code/gen/lorenz_orbit.py (r = 27.42, start 0); integration step 0.002.
# inkling_t_*: longest simulated time in Inkling's own code in the agent task (runs not in this repo).
orbit = np.load(ROOT / "data/orbit_page.npz")
lorenz = dict(head=np.round(orbit["head"][::4], 2).tolist(), tail=np.round(orbit["tail"][::5], 3).tolist(),
              tail_t0=float(orbit["tail_t0"]), t_settle=float(orbit["t_settle"]),
              dt_head=0.002 * 4, dt_tail=0.002 * 5, inkling_t_typical=900, inkling_t_max=5000)

D = dict(dirs=dirs, parsed=parsed, seed1=seed1, effort32k=effort32k, compare=compare, natural=natural,
         notewins=notewins, keep=keep, arms=arms, effort=effort, dose=dose, train=train, cost=cost, lorenz=lorenz)
(ROOT / "docs/data.js").write_text("window.DATA = " + json.dumps(D, separators=(",", ":")) + ";\n")
print("wrote docs/data.js:", ", ".join(D))
