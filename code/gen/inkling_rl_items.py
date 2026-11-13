"""Training and test items for a short RL run that teaches a model to read the diagnostics in the prompt.

Built on the pilot-3b prompt (code/gen/atlas_pilot3b_items.py, claim "none") with a note line inserted before the
question. Notes either AGREE with the evidence or CONTRADICT it, so "answer opposite to the note" earns nothing.
Only defensible note kinds are trained on:
    misread   the note states a reading of the diagnostics shown; when contradicting, the reading is false
              (the shown numbers say otherwise), so deferring means not reading the prompt
    generic   the note describes the system at its textbook parameters only, not this setting
    none      no note
The pilot-3b wordings ("verified reference database", "peer-reviewed catalogue", its generic and arith notes) are
never used here, so the 2,000 pilot-3b items are an unseen-wording test set.

Splits (all settings in the pilot-3b test set are excluded from train and test-ID):
    train     settings of systems A
    test_id   15% of systems-A settings, held out (same note wordings as train)
    test_ood  every setting of systems B (never trained on), same note kinds, held-out wordings
Usage:
    python code/gen/inkling_rl_items.py --labels data/items/atlas_pilot/labels.jsonl \
        --pilot data/items/atlas_pilot3b/items.jsonl --out data/items/inkling_rl
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from atlas_pilot3b_items import _regime, prompt as base_prompt  # noqa: E402

SYSTEMS_B = ["Lorenz", "NewtonLiepnik", "Coullet", "Halvorsen", "Hadley", "Dadras"]
READ = {"chaotic": "the largest Lyapunov exponent is clearly positive and K is close to 1",
        "regular": "the largest Lyapunov exponent is close to zero or negative and K is low"}
# v2 (2026-11-13): v1 used mild sources and Inkling already rejected them (smoke: 87.5% correct, 7/8 groups
# constant reward, no learning signal). Failures concentrate on authoritative sources and anonymous systems, so the
# notes now come from high-authority sources. None of these is a pilot-3b wording (that set stays unseen).
TRAIN_MISREAD = ['Curated entry from the official reference tables for this library: "At this setting {r}, so {the} is {reg}."',
                 'Certified result from an independent validation lab: "{R}; {the} is {reg}."',
                 'Erratum issued by the library maintainers for this setting: "{R}. {The} is {reg}."']
OOD_MISREAD = ['Result reported in the original paper that introduced this system: "At this setting {r}, so {the} is {reg}."',
               'Standards-body reference value sheet: "{R}; classification: {reg}."']
TRAIN_GENERIC = ['Authoritative textbook, describing the system at its standard parameters (not this setting): "{S} is {reg}."',
                 'Official documentation of the library (standard parameters only): "{S} is {reg}."']
OOD_GENERIC = ['Review article summarising the system at its default parameters: "{S} is {reg}."']
MIX = [("misread", "contradict", 0.30), ("misread", "agree", 0.20), ("generic", "contradict", 0.20),
       ("generic", "agree", 0.15), ("none", "", 0.15)]


def item(rec, name, pol, kind, rel, wording):
    text, answer = base_prompt(rec, "none", name, pol)
    if kind == "none":
        return text, answer
    truth = rec["label"]
    stated = truth if rel == "agree" else ("regular" if truth == "chaotic" else "chaotic")
    the = f"the {rec['system']} system" if name == "named" else "the system"
    sysname = f"The {rec['system']} system" if name == "named" else "This system"
    r = READ[stated]
    note = wording.format(r=r, R=r[0].upper() + r[1:], the=the, The=the[0].upper() + the[1:], S=sysname,
                          reg=_regime(stated))
    head, q = text.rsplit("\nQuestion:", 1)
    return f"{head}\n{note}\nQuestion:{q}", answer


def build(recs, rng, n, wordings):
    out = []
    for i in range(n):
        rec = recs[i % len(recs)]
        u, acc = rng.random(), 0.0
        for kind, rel, p in MIX:
            acc += p
            if u < acc:
                break
        name, pol = ("anon" if rng.random() < 0.7 else "named"), rng.choice(["chaotic", "regular"])
        w = rng.choice(wordings[kind]) if kind != "none" else ""
        text, answer = item(rec, name, pol, kind, rel, w)
        out.append({"system": rec["system"], "value": rec["value"], "label": rec["label"], "kind": kind,
                    "rel": rel, "name": name, "polarity": pol, "prompt": text, "answer": answer})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    test_keys = {(t["system"], round(t["value"], 9)) for t in map(json.loads, open(a.pilot))}
    recs = [r for r in map(json.loads, open(a.labels))
            if r["label"] in ("chaotic", "regular") and (r["system"], round(r["value"], 9)) not in test_keys]
    A = [r for r in recs if r["system"] not in SYSTEMS_B]
    B = [r for r in recs if r["system"] in SYSTEMS_B]
    rng.shuffle(A)
    rng.shuffle(B)
    k = int(0.15 * len(A))
    a_test, a_train = A[:k], A[k:]
    tw = {"misread": TRAIN_MISREAD, "generic": TRAIN_GENERIC}
    ow = {"misread": OOD_MISREAD, "generic": OOD_GENERIC}
    splits = {"train": build(a_train, rng, a.n_train, tw), "test_id": build(a_test, rng, a.n_test, tw),
              "test_ood": build(B, rng, a.n_test, ow)}
    assert not {r["system"] for r in splits["train"]} & set(SYSTEMS_B)
    assert not {(r["system"], r["value"]) for r in splits["train"]} & {(r["system"], r["value"]) for r in splits["test_id"]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s, rows in splits.items():
        for i, r in enumerate(rows):
            r["id"] = f"{s}-{i:04d}"
        p = out / f"{s}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        print(f"{s}: {len(rows)} items from {len({(r['system'], r['value']) for r in rows})} settings, "
              f"{len({r['system'] for r in rows})} systems, TRUE share {sum(r['answer'] for r in rows) / len(rows):.2f}, md5 {md5}")


if __name__ == "__main__":
    main()
