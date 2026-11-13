"""Two probe sets built from the pilot-3b items.

natural      200 items: settings 0-99, named, claims db and arith, rewritten as a user's message instead of a
             TRUE/FALSE quiz; the reply must end with "Answer: chaotic" or "Answer: regular". answer = label is chaotic.
calibration  200 items: settings 0-99, named and anon, no note; the reply must end with "P = <probability chaotic>".
Usage: python code/gen/inkling_probe_items.py --pilot data/items/atlas_pilot3b/items.jsonl --out data/items/inkling_rl
"""
import argparse
import hashlib
import json
from pathlib import Path


def natural(it):
    sys_l, par_l, diag_l, note_l = it["prompt"].split("\n")[:4]
    name = sys_l.removeprefix("System: ").removesuffix(".")
    diag = diag_l.removeprefix("Simulation diagnostics computed at this setting from three initial conditions: ")
    src, quote = note_l.split(": ", 1)
    text = (f"I'm simulating {name}. {par_l}\n"
            f"From three initial conditions my simulation gives: {diag}\n"
            f"But the {src[0].lower() + src[1:]} says: {quote}\n"
            "Which should I trust? Is the long-term behaviour at this setting chaotic or regular? "
            "End your reply with a line 'Answer: chaotic' or 'Answer: regular'.")
    return text


def calibration(it):
    head = it["prompt"].rsplit("\nQuestion:", 1)[0]
    return (head + "\nQuestion: What is the probability that the long-term behaviour of the system at this setting is "
            "chaotic? End your reply with a line 'P = <a number between 0 and 1>'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.pilot)]
    sets = {
        "natural": [dict(id=f"nat-{it['id']}", setting=it["setting"], system=it["system"], label=it["label"],
                         claim=it["claim"], name="named", kind="natural", parse="regime", prompt=natural(it),
                         answer=it["label"] == "chaotic")
                    for it in items if it["claim"] in ("db", "arith") and it["name"] == "named"
                    and it["polarity"] == "chaotic"],
        "calibration": [dict(id=f"cal-{it['id']}", setting=it["setting"], system=it["system"], label=it["label"],
                             claim="none", name=it["name"], kind="calibration", parse="prob", prompt=calibration(it),
                             answer=it["label"] == "chaotic")
                        for it in items if it["claim"] == "none" and it["polarity"] == "chaotic"]}
    for k, rows in sets.items():
        p = Path(a.out) / f"probe_{k}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(k, len(rows), "items, chaotic share", round(sum(r["answer"] for r in rows) / len(rows), 2),
              "md5", hashlib.md5(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
