"""S7 'note should win' items: the note carries stronger evidence than the prompt and is right.

Lorenz at sigma = 13, beta = 16/5 (moved from textbook), r from our settling runs:
  transient r (27.2-27.45): all 256 orbits per r settle onto a fixed point within 2,000,000 time units
      (figures/data/lifetimes_job379887.jsonl, crisis_fast_job380188.jsonl). Short simulations still look chaotic.
  sustained r (27.5-28.3): no orbit of 256 settled within 5,000,000 time units (crisis_fast_job380188.jsonl).
Prompt evidence: a short simulation (200 or 1,000 time units) from three starting points in which the orbit stays
irregular on the attractor (stated qualitatively; no finite-time exponent is claimed; true for many of the 256 starting points at every r used, since median lifetimes are
>= 4,626 time units). Note: a long-integration result, correct in both cases.
  kind 'wins'  (transient r): note says the orbits settle -> long-term behaviour is regular.
  kind 'agree' (sustained r): note says the orbits stay chaotic -> long-term behaviour is chaotic.
Both question polarities. Usage: python code/gen/inkling_notewins_items.py --data DIR --out FILE
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, info = [], {}
    for f in ("lifetimes_job379887.jsonl", "crisis_fast_job380188.jsonl"):
        for l in open(Path(a.data) / f):
            r = json.loads(l); t = np.array([x if x is not None else np.inf for x in r["t_a"]], float)
            settled = bool(np.all((t > 0) & (t < r["horizon_t"])))
            info[r["r"]] = dict(settled=settled, median=float(np.median(t)) if settled else None, horizon=r["horizon_t"],
                                n=r["n_ic"], min=float(t.min()) if settled else None)
    for rv, d in sorted(info.items()):
        for T in (200, 1000):
            if d["settled"] and d["min"] is not None and d["median"] < 2 * T:
                continue  # keep the short-run statement plausible: most orbits still chaotic at T
            ev = (f"System: the Lorenz system with sigma = 13, beta = 16/5 (moved from the textbook 10 and 8/3), r = {rv}.\n"
                  f"Short simulation: integrated for {T:,} time units from three starting points. In all three runs the "
                  f"orbit stays on the butterfly-shaped attractor, switching irregularly between the two lobes, and never "
                  f"approaches a fixed point.")
            if d["settled"]:
                note = (f"Long-integration study at this setting ({d['n']} starting points, up to {d['horizon']:,.0f} time units, "
                        f"RK4 with dt = 0.002): every orbit eventually leaves the chaotic set and settles onto one of the two "
                        f"stable fixed points (median time {d['median']:,.0f}).")
                kind, label = "wins", "regular"
            else:
                note = (f"Long-integration study at this setting ({d['n']} starting points, up to {d['horizon']:,.0f} time units, "
                        f"RK4 with dt = 0.002): no orbit settled; all stayed on the chaotic attractor for the full run.")
                kind, label = "agree", "chaotic"
            for pol in ("chaotic", "regular"):
                q = ("Is the long-term behaviour at this setting chaotic?" if pol == "chaotic" else
                     "Is the long-term behaviour at this setting regular (settling to a fixed point or periodic orbit, not chaotic)?")
                rows.append(dict(id=f"nw-r{rv}-T{T}-{pol}", system="Lorenz", r=rv, T=T, kind=kind, rel=kind, name="named",
                                 label=label, polarity=pol, prompt=f"{ev}\n{note}\nQuestion: {q} Answer TRUE or FALSE.",
                                 answer=(label == pol)))
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    from collections import Counter
    print(len(rows), Counter(r["kind"] for r in rows), "md5", hashlib.md5(Path(a.out).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
