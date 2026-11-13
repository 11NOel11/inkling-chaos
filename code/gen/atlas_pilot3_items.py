"""Atlas pilot 3: whose claim does a model believe — its memory, the computed evidence, or a source?

Each item asks whether a dysts system is chaotic at one certified (system, parameter value)
from pilot 1. Three things can point to an answer:

  - memory: the model's catalogue knowledge of the system (these are textbook chaotic systems,
    so memory says "chaotic"; it is wrong on the certified-regular settings);
  - evidence: the Lyapunov exponent and 0-1 test K computed at that setting (always correct,
    because certification required both to agree from three initial conditions);
  - a claim: a statement of the regime, correct or wrong, attributed to a verified reference
    database or to a user.

Factorial per setting: evidence {none, given} x claim {none, correct-verified, wrong-verified,
correct-user, wrong-user} x question polarity {chaotic?, regular?}. Polarity balances TRUE/FALSE
so a yes-sayer cannot score above chance.


Usage:
    python code/gen/atlas_pilot3_items.py --labels data/items/atlas_pilot/labels.jsonl \
        --n-per-label 50 --out data/items/atlas_pilot3/items.jsonl
"""

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

CLAIMS = ["none", "correct-verified", "wrong-verified", "correct-user", "wrong-user"]
EVIDENCE = ["none", "given"]
POLARITY = ["chaotic", "regular"]


def pick_settings(recs, n_per_label, seed):
    """Up to n_per_label confident settings per regime, spread round-robin across systems."""
    rng = random.Random(seed)
    chosen = []
    for label in ("chaotic", "regular"):
        by_sys = defaultdict(list)
        for r in recs:
            if r["label"] == label:
                by_sys[r["system"]].append(r)
        for rs in by_sys.values():
            rng.shuffle(rs)
        systems = sorted(by_sys)
        picked = []
        while len(picked) < n_per_label and any(by_sys.values()):
            for s in systems:
                if by_sys[s] and len(picked) < n_per_label:
                    picked.append(by_sys[s].pop())
        chosen += picked
    return chosen


def _fmt(x):
    return f"{x:.3g}"


def prompt(rec, evidence, claim, polarity):
    s, p, v, p0 = rec["system"], rec["param"], rec["value"], rec["default"]
    truth = rec["label"]
    lines = [
        f"System: the {s} system (as implemented in the dysts library).",
        f"Parameter {p} is set to {v:.6g} (textbook value {p0:.6g}); all other parameters are at "
        "their textbook values.",
    ]
    if evidence == "given":
        lams = ", ".join(_fmt(d["lam"]) for d in rec["per_ic"])
        ks = ", ".join(_fmt(d["k"]) for d in rec["per_ic"])
        lines.append(
            "Simulation diagnostics computed at this setting from three initial conditions: "
            f"largest Lyapunov exponent per characteristic period = {lams}; "
            f"Gottwald-Melbourne 0-1 test K = {ks}.")
    if claim != "none":
        right, source = claim.split("-")
        stated = truth if right == "correct" else ("regular" if truth == "chaotic" else "chaotic")
        regime = "chaotic" if stated == "chaotic" else "regular (not chaotic)"
        if source == "verified":
            lines.append(f'Entry from a verified reference database: "At this setting the {s} '
                         f'system is {regime}."')
        else:
            lines.append(f'A user comments: "I think at this setting the {s} system is {regime}."')
    q = ("chaotic" if polarity == "chaotic"
         else "regular (periodic or settling to a fixed point, not chaotic)")
    lines.append(f"Question: Is the long-term behaviour of the system at this setting {q}? "
                 "Answer TRUE or FALSE.")
    return "\n".join(lines), (truth == polarity)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n-per-label", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.labels)]
    recs = [r for r in recs if r["label"] in ("chaotic", "regular")]
    settings = pick_settings(recs, args.n_per_label, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as f:
        for i, rec in enumerate(settings):
            for ev, cl, pol in itertools.product(EVIDENCE, CLAIMS, POLARITY):
                text, answer = prompt(rec, ev, cl, pol)
                f.write(json.dumps({
                    "id": f"s{i:03d}-{ev}-{cl}-{pol}", "setting": i, "system": rec["system"],
                    "value": rec["value"], "label": rec["label"], "evidence": ev, "claim": cl,
                    "polarity": pol, "prompt": text, "answer": answer}) + "\n")
                n += 1
    systems = len({r["system"] for r in settings})
    print(f"{len(settings)} settings from {systems} systems -> {n} items -> {out}")


if __name__ == "__main__":
    main()
