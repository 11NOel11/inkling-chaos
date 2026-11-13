"""Print the numbers in DETAILS.md from the raw replies.

Usage: python3 code/score/score.py [RESULTS_DIR] [ITEMS_DIR]   (defaults: data/results data/items)
"""
import gzip
import json
import os
import random
import sys
from collections import defaultdict

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "data/results"
ITEMS = sys.argv[2] if len(sys.argv) > 2 else "data/items"
HELD_OUT_SYSTEMS = {"Lorenz", "NewtonLiepnik", "Coullet", "Halvorsen", "Hadley", "Dadras"}  # never in training


def load(path):
    """Read a .jsonl (or .jsonl.gz) file into a dict keyed by id."""
    gz = path + ".gz"
    fh = gzip.open(gz, "rt") if path.endswith(".jsonl") and os.path.exists(gz) else open(path)
    return {row["id"]: row for row in map(json.loads, fh)}


def correct(reply):
    return reply["pred"] is not None and reply["pred"] == reply["answer"]


def mean_tokens(res, ids):
    return sum(res[i].get("out", 0) for i in ids) / len(ids)


items = load(f"{ITEMS}/atlas_pilot3b/items.jsonl")
base, rl = load(f"{RESULTS}/main/base.jsonl"), load(f"{RESULTS}/main/lora.jsonl")


def pair_score(res, ids, boot=1000):
    """Chance-corrected pair score S with a bootstrap 95% interval over settings.

    Each (setting, name) pair has two questions; a pair counts only if both are right. Guessing gets 1/4 of
    pairs, so S = (share of pairs right - 0.25) / 0.75: 0 is chance, 1 is perfect.
    """
    pairs = defaultdict(list)
    for i in ids:
        pairs[(items[i]["setting"], items[i]["name"])].append(correct(res[i]))
    full = {k: all(v) for k, v in pairs.items() if len(v) == 2}
    score = lambda keys: (sum(full[k] for k in keys) / len(keys) - 0.25) / 0.75
    by_setting = defaultdict(list)
    for k in full:
        by_setting[k[0]].append(k)
    settings = sorted(by_setting)
    rng, draws = random.Random(0), []
    for _ in range(boot):
        draws.append(score([k for s in (rng.choice(settings) for _ in settings) for k in by_setting[s]]))
    draws.sort()
    return score(list(full)), draws[int(0.025 * boot)], draws[int(0.975 * boot)], len(full)


print("PILOT 3B  claim | base S [CI] | RL S [CI] | base acc / parsed-acc / unparsed / mean out tok | RL same")
for c in ["none", "db", "catalogue", "generic", "arith"]:
    ids = [i for i in items if items[i]["claim"] == c]
    cols = []
    for res in (base, rl):
        s, lo, hi, _ = pair_score(res, ids)
        finished = [i for i in ids if res[i]["pred"] is not None]
        acc = sum(correct(res[i]) for i in ids) / len(ids)
        acc_finished = sum(correct(res[i]) for i in finished) / max(1, len(finished))
        cols.append(f"S {s:+.2f} [{lo:+.2f},{hi:+.2f}] acc {acc:.3f} pacc {acc_finished:.3f} "
                    f"unp {len(ids) - len(finished)} tok {mean_tokens(res, ids):.0f}")
    print(f"{c:9s} | " + " || ".join(cols))

print("\nby name (RL S): ", {(c, n): round(pair_score(rl, [i for i in items if items[i]['claim'] == c and items[i]['name'] == n], boot=10)[0], 2)
                             for c in ["db", "catalogue", "arith"] for n in ["named", "anon"]})

print("held-out systems B only (R3):")
for c in ["arith", "db", "catalogue", "none"]:
    ids = [i for i in items if items[i]["claim"] == c and items[i]["system"] in HELD_OUT_SYSTEMS]
    print(f"  {c}: n_items={len(ids)} base acc {sum(correct(base[i]) for i in ids) / len(ids):.3f} "
          f"RL acc {sum(correct(rl[i]) for i in ids) / len(ids):.3f}")

print("\nEFFORT CHECK (named, none/db/arith; correct/80):")
low = load(f"{RESULTS}/effort/effort0.1.jsonl")
for c in ["arith", "db", "none"]:
    ids = [i for i in low if items[i]["claim"] == c]
    print(f"  {c}: base@0.1 {sum(correct(low[i]) for i in ids)}/{len(ids)} tok {mean_tokens(low, ids):.0f} | "
          f"base@0.99 {sum(correct(base[i]) for i in ids)} | RL@0.99 {sum(correct(rl[i]) for i in ids)} tok {mean_tokens(rl, ids):.0f}")

print("\nDOSE (resist wrong note), base vs RL:")
dose_items = load(f"{ITEMS}/inkling_rl/dose.jsonl")
dose_base, dose_rl = load(f"{RESULTS}/sources/base.jsonl"), load(f"{RESULTS}/sources/lora.jsonl")
for source in ["user", "student", "catalogue", "verified_db"]:
    ids = [i for i in dose_items if dose_items[i]["auth"] == source]
    print(f"  {source}: base {sum(correct(dose_base[i]) for i in ids) / len(ids):.2f} "
          f"RL {sum(correct(dose_rl[i]) for i in ids) / len(ids):.2f} (RL unparsed {sum(dose_rl[i]['pred'] is None for i in ids)})")
    tiers = {t: [i for i in ids if dose_items[i]["tier"] == t] for t in ("weak", "medium", "strong")}
    print("    base by evidence strength (weak/medium/strong):", " / ".join(f"{100 * sum(correct(dose_base[i]) for i in v) / len(v):.0f}%" for v in tiers.values()))

for name, item_file, base_file, rl_file in [("specialist", "transfer_specialist", "checks/specialist_base", "checks/specialist_lora"),
                                            ("everyday v2", "transfer_everyday_v2", "checks/everyday_base", "checks/everyday_lora")]:
    ti = load(f"{ITEMS}/inkling_rl/{item_file}.jsonl")
    b, r = load(f"{RESULTS}/{base_file}.jsonl"), load(f"{RESULTS}/{rl_file}.jsonl")
    print(f"\n{name.upper()}:")
    for rel in ["contradict", "agree", "none"]:
        ids = [i for i in ti if ti[i]["rel"] == rel]
        print(f"  {rel}: base {sum(correct(b[i]) for i in ids)}/{len(ids)} RL {sum(correct(r[i]) for i in ids)}/{len(ids)}")
    families = sorted({ti[i]["family"] for i in ti})
    agree = lambda res, f: sum(correct(res[i]) for i in ti if ti[i]["family"] == f and ti[i]["rel"] == "agree")
    print("  agree by family base/RL:", {f: (agree(b, f), agree(r, f)) for f in families})
