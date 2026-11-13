"""Same items and output format as run_pilot3b.py, answered by a Claude model through the Anthropic API.

Usage: python run_claude.py ITEMS OUT --model claude-haiku-5-5 [--effort medium] [--claims ...] [--names ...]
       [--max-setting N] [--workers 8] [--max-tokens 32000] [--limit N]
Output rows: id, content, reasoning (summarized thinking), finish, pred, answer, in, out. Reads ANTHROPIC_API_KEY.
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import anthropic

from run_pilot3b import parse_item

PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}  # $/MTok in, out (prompts < 100K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--model", required=True, choices=sorted(PRICE))
    ap.add_argument("--effort", default=None, help="low|medium|high|xhigh|max; omit for the model default")
    ap.add_argument("--claims", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--max-setting", type=int, default=10**9)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32000)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items)]
    items = [it for it in items if it.get("setting", 0) <= a.max_setting
             and (not a.claims or it.get("claim", it.get("kind")) in a.claims.split(","))
             and (not a.names or it["name"] in a.names.split(","))]
    if a.limit:
        items = items[: a.limit]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [it for it in items if it["id"] not in done]
    client = anthropic.Anthropic(max_retries=8, timeout=1800)
    lock, tot = threading.Lock(), {"in": 0, "out": 0, "n": 0}
    extra = {"output_config": {"effort": a.effort}} if a.effort else {}

    def one(it):
        with client.messages.stream(model=a.model, max_tokens=a.max_tokens,
                                    thinking={"type": "adaptive", "display": "summarized"},
                                    messages=[{"role": "user", "content": it["prompt"]}], **extra) as s:
            m = s.get_final_message()
        text = "".join(b.text for b in m.content if b.type == "text")
        think = "".join(getattr(b, "thinking", "") or "" for b in m.content if b.type == "thinking")
        row = {"id": it["id"], "content": text, "reasoning": think, "finish": m.stop_reason, "pred": parse_item(text, it.get("parse", "tf")),
               "answer": it.get("answer"), "in": m.usage.input_tokens, "out": m.usage.output_tokens}
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            tot["in"] += row["in"]; tot["out"] += row["out"]; tot["n"] += 1

    with ThreadPoolExecutor(a.workers) as ex:
        for fut in [ex.submit(one, it) for it in todo]:
            try:
                fut.result()
            except Exception as e:  # keep going; a rerun resumes the missing ids
                print("error:", type(e).__name__, str(e)[:200])
    pi, po = PRICE[a.model]
    print(f"{tot['n']} done; in {tot['in']} tok, out {tot['out']} tok, est ${(tot['in'] * pi + tot['out'] * po) / 1e6:.2f}")


if __name__ == "__main__":
    main()
