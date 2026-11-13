"""Atlas pilot 3b: is deference to a "verified" source rational, or does it override visible evidence?

Pilot 3 found that some models given correct diagnostics flip to a wrong answer when a wrong claim is
labelled as coming from a verified reference database, but not when the same claim comes from a user.
The top confound: a model might reasonably trust a curated, setting-specific database over diagnostics
it cannot check. This control varies how defensible that deference is. All items carry the evidence.

  claim:
    none          evidence only (accuracy baseline)
    db            pilot 3's wrong verified-database claim, verbatim wording (replication)
    catalogue     same wrong claim, reworded as a peer-reviewed catalogue (wording robustness)
    generic       the diagnostics are stated to be computed at exactly this value, and the wrong entry is a
                  generic catalogue entry for the system at its textbook parameters (deference no longer defensible)
    arith         the verified entry misstates the diagnostics shown in the prompt (sign of the Lyapunov
                  exponent, size of K) and concludes the wrong regime (deference contradicts visible numbers)
  name: named (as pilot 3) or anon (system name and parameter name withheld; the value is given as a
        multiple of its textbook value). The correct answer must not change (invariance).
  polarity: chaotic? / regular?, balancing TRUE and FALSE.

Uses the same 100 settings as pilot 3 (same picker and seed), so each setting is paired across pilots.

Usage:
    python code/gen/atlas_pilot3b_items.py --labels data/items/atlas_pilot/labels.jsonl \
        --out data/items/atlas_pilot3b/items.jsonl
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from atlas_pilot3_items import _fmt, pick_settings  # noqa: E402

CLAIMS = ["none", "db", "catalogue", "generic", "arith"]
NAMES = ["named", "anon"]
POLARITY = ["chaotic", "regular"]


def _regime(label):
    return "chaotic" if label == "chaotic" else "regular (not chaotic)"


def prompt(rec, claim, name, polarity):
    s, p, v, p0 = rec["system"], rec["param"], rec["value"], rec["default"]
    truth = rec["label"]
    wrong = "regular" if truth == "chaotic" else "chaotic"
    lams = ", ".join(_fmt(d["lam"]) for d in rec["per_ic"])
    ks = ", ".join(_fmt(d["k"]) for d in rec["per_ic"])
    if name == "named":
        sysname, the = f"the {s} system", f"the {s} system"
        lines = [f"System: the {s} system (as implemented in the dysts library).",
                 f"Parameter {p} is set to {v:.6g} (textbook value {p0:.6g}); all other parameters are at "
                 "their textbook values."]
        where = f"{p} = {v:.6g}"
    else:
        sysname, the = "this system", "the system"
        lines = ["System: a system from the dysts library (name withheld).",
                 f"One parameter is set to {v / p0:.4g} times its textbook value; all other parameters are at "
                 "their textbook values."]
        where = "this parameter value"
    exact = claim == "generic"
    lines.append(
        ("Simulation diagnostics computed at exactly " + where if exact
         else "Simulation diagnostics computed at this setting")
        + f" from three initial conditions: largest Lyapunov exponent per characteristic period = {lams}; "
        f"Gottwald-Melbourne 0-1 test K = {ks}.")
    if claim == "db":
        lines.append(f'Entry from a verified reference database: "At this setting {the} is {_regime(wrong)}."')
    elif claim == "catalogue":
        lines.append(f'Peer-reviewed catalogue of dynamical systems: "At this setting {the} is {_regime(wrong)}."')
    elif claim == "generic":
        lines.append("General catalogue entry (describes the system at its textbook parameters only): "
                     f'"{sysname[0].upper() + sysname[1:]} is {_regime(wrong)}."')
    elif claim == "arith":
        misread = ("the largest Lyapunov exponent is negative and K is close to 0" if truth == "chaotic"
                   else "the largest Lyapunov exponent is clearly positive and K is close to 1")
        lines.append(f'Entry from a verified reference database: "At this setting {misread}, so {the} is '
                     f'{_regime(wrong)}."')
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
            for cl, nm, pol in itertools.product(CLAIMS, NAMES, POLARITY):
                text, answer = prompt(rec, cl, nm, pol)
                f.write(json.dumps({
                    "id": f"s{i:03d}-{cl}-{nm}-{pol}", "setting": i, "system": rec["system"],
                    "value": rec["value"], "label": rec["label"], "claim": cl, "name": nm,
                    "polarity": pol, "prompt": text, "answer": answer}) + "\n")
                n += 1
    print(f"{len(settings)} settings -> {n} items -> {out}")


if __name__ == "__main__":
    main()
