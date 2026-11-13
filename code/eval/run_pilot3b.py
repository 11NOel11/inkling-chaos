"""Inkling 2 Small on pilot 3b items via Tinker's OpenAI-compatible endpoint. Resumable; one JSON line per item.

Usage: uv run python run_pilot3b.py ITEMS OUT [--limit N] [--workers 8]
"""
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
MODEL = "thinkingmachines/Inkling-2-Small:peft:262144"


def parse(text):
    hits = re.findall(r"\b(TRUE|FALSE)\b", text or "") or re.findall(r"\b(true|false)\b", text or "", re.I)
    return None if not hits else hits[-1].upper() == "TRUE"



def parse_item(text, kind="tf"):
    """tf: last TRUE/FALSE -> bool; regime: last 'Answer: chaotic|regular' (else last chaotic/regular word) -> bool
    (True = chaotic); prob: last 'P = x' -> float in [0, 1]."""
    if kind == "regime":
        hits = re.findall(r"answer\W*\s*(chaotic|regular)", text or "", re.I) or \
            re.findall(r"\b(chaotic|regular)\b", text or "", re.I)
        return None if not hits else hits[-1].lower() == "chaotic"
    if kind == "mc":
        hits = re.findall(r"answer\W*\s*\(?([ABCD])\b", text or "", re.I)
        return None if not hits else hits[-1].upper()
    if kind == "prob":
        hits = re.findall(r"P\s*=\s*([01](?:\.\d+)?|\.\d+)", text or "")
        return None if not hits else min(1.0, max(0.0, float(hits[-1])))
    return parse(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--effort", type=float, default=0.99)
    ap.add_argument("--claims", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--max-setting", type=int, default=10**9)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--plain", action="store_true", help="no reasoning_effort / separate_reasoning (non-Inkling models)")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items)]
    items = [it for it in items if it.get("setting", 0) <= a.max_setting
             and (not a.claims or it.get("claim", it.get("kind")) in a.claims.split(","))
             and (not a.names or it["name"] in a.names.split(","))]
    if a.limit:
        items = items[: a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [it for it in items if it["id"] not in done]
    client = OpenAI(base_url=URL, api_key=os.environ["TINKER_API_KEY"], max_retries=6, timeout=1800)
    lock = threading.Lock()
    fh = open(a.out, "a")

    def one(it):
        try:
            r = client.chat.completions.create(model=a.model, messages=[{"role": "user", "content": it["prompt"]}],
                                               temperature=1.0, max_tokens=a.max_tokens,
                                               extra_body=({} if a.plain else {"reasoning_effort": a.effort, "separate_reasoning": True}))
            m = r.choices[0].message
            rec = {"id": it["id"], "content": m.content, "reasoning": (m.model_extra or {}).get("reasoning_content"),
                   "finish": r.choices[0].finish_reason, "pred": parse_item(m.content, it.get("parse", "tf")), "answer": it["answer"],
                   "in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}
        except Exception as e:  # noqa: BLE001
            rec = {"id": it["id"], "error": repr(e)[:300]}
        with lock:
            if "error" not in rec:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            else:
                print("ERR", rec, flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    recs = [json.loads(l) for l in open(a.out)]
    tin, tout = sum(r["in"] for r in recs), sum(r["out"] for r in recs)
    print(f"{len(recs)} done; prefill {tin} tok, sample {tout} tok, est ${tin * 0.30e-6 + tout * 1e-6:.2f}")


if __name__ == "__main__":
    main()
