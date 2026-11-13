"""Dose-response items: how authoritative must a wrong note sound before the model abandons the evidence shown?

Pilot-3b test settings (the 100 in data/items/atlas_pilot3b/items.jsonl), evidence as in pilot 3b (claim "none"),
plus a note that states the WRONG regime with no numbers, from one of four sources of rising authority. Evidence
strength = the weakest of the three shown diagnostics relative to its threshold (chaotic: min Lyapunov exponent;
regular: max K); settings are split into weak / medium / strong tertiles within each label.

Usage: python code/gen/inkling_dose_items.py --labels data/items/atlas_pilot/labels.jsonl \
           --pilot data/items/atlas_pilot3b/items.jsonl --out data/items/inkling_rl/dose.jsonl
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from atlas_pilot3b_items import _regime, prompt as base_prompt  # noqa: E402

AUTH = [("user", 'A user guessed: "{the} is {reg}."'),
        ("student", 'A student\'s homework answer: "{the} is {reg}."'),
        ("catalogue", 'Peer-reviewed catalogue of dynamical systems: "At this setting {the} is {reg}."'),
        ("verified_db", 'Entry from a verified reference database: "At this setting {the} is {reg}."')]


def strength(rec):
    if rec["label"] == "chaotic":
        return min(d["lam"] for d in rec["per_ic"])
    return -max(d["k"] for d in rec["per_ic"])  # larger = more clearly regular


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keys = {(t["system"], round(t["value"], 9)): t["setting"] for t in map(json.loads, open(a.pilot))}
    recs, seen = [], set()
    for r in map(json.loads, open(a.labels)):
        k = (r["system"], round(r["value"], 9))
        if k in keys and k not in seen and r["label"] in ("chaotic", "regular"):
            seen.add(k)
            recs.append((keys[k], r))
    rows = []
    for lab in ("chaotic", "regular"):
        group = sorted([x for x in recs if x[1]["label"] == lab], key=lambda x: strength(x[1]))
        for j, (setting, rec) in enumerate(group):
            tier = ["weak", "medium", "strong"][min(2, 3 * j // len(group))]
            wrong = "regular" if lab == "chaotic" else "chaotic"
            for auth, w in AUTH:
                for pol in ("chaotic", "regular"):
                    text, answer = base_prompt(rec, "none", "named", pol)
                    note = w.format(the=f"the {rec['system']} system", reg=_regime(wrong))
                    note = note.replace('"the ', '"The ', 1)
                    head, q = text.rsplit("\nQuestion:", 1)
                    rows.append({"id": f"dose-s{setting:03d}-{auth}-{pol}", "setting": setting, "system": rec["system"],
                                 "label": lab, "tier": tier, "auth": auth, "kind": "dose", "rel": "contradict",
                                 "name": "named", "polarity": pol, "prompt": f"{head}\n{note}\nQuestion:{q}",
                                 "answer": answer})
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "items from", len(recs), "settings; md5", hashlib.md5(Path(a.out).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
