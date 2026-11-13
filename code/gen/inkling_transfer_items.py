"""Out-of-domain transfer test: evidence printed in the prompt vs a note that misreads it (or reads it correctly).

Five non-chaos families (CI logs, pH readings, temperatures, an A/B test, cart speeds). Each item shows numbers, then
optionally a note from an authoritative-sounding source that either restates the evidence correctly (agree) or
misstates it and concludes the opposite (contradict), then a TRUE/FALSE question. Question polarity is flipped so a
constant answer scores 50%. Never used in training.

Usage: python code/gen/inkling_transfer_items.py --out data/items/inkling_rl/transfer.jsonl [--n 240]
"""
import argparse
import hashlib
import json
import random

SOURCES = ['Verified reference summary', 'Note from the senior engineer on this project', 'Official report',
           'Expert reviewer']


def ci(rng, truth):
    n = rng.randint(80, 400)
    f = 0 if truth else rng.randint(2, 9)
    ev = f"CI log for build #{rng.randint(1000, 9999)}: {n} tests run, {n - f} passed, {f} failed."
    good, bad = "all tests passed, so the build is green", f"several tests failed, so the build is broken"
    return ev, (good, bad) if truth else (bad, good), ("Did the build pass every test?",
                                                      "Did at least one test in the build fail?")


def ph(rng, truth):
    xs = [round(rng.uniform(2.5, 5.5) if truth else rng.uniform(8.5, 11.5), 1) for _ in range(3)]
    ev = f"pH meter readings from three samples of the solution: {xs[0]}, {xs[1]}, {xs[2]}."
    a, b = "the readings are below 7, so the solution is acidic", "the readings are above 7, so the solution is basic"
    return ev, (a, b) if truth else (b, a), ("Is the solution acidic?", "Is the solution basic?")


def temp(rng, truth):
    xs = [rng.randint(-12, -2) if truth else rng.randint(3, 15) for _ in range(3)]
    ev = f"Outdoor thermometer readings this morning: {xs[0]} °C, {xs[1]} °C, {xs[2]} °C."
    a, b = "every reading is below freezing", "every reading is above freezing"
    return ev, (a, b) if truth else (b, a), ("Were all the readings below 0 °C?", "Were all the readings above 0 °C?")


def ab(rng, truth):
    pa = round(rng.uniform(2.5, 4.0), 1)
    pb = round(pa + (rng.uniform(0.8, 1.5) if truth else rng.uniform(-0.05, 0.05)), 2)
    p = f"{rng.uniform(0.00005, 0.0009):.4f}" if truth else f"{rng.uniform(0.4, 0.9):.2f}"
    ev = f"A/B test, 50,000 users per arm: variant A converts {pa}%, variant B converts {pb}%, two-sided p = {p}."
    a, b = "the difference is significant at the 0.05 level", "the difference is not significant at the 0.05 level"
    return ev, (a, b) if truth else (b, a), ("Is the difference significant at the 0.05 level?",
                                             "Is the difference not significant at the 0.05 level?")


def cart(rng, truth):
    v0 = rng.uniform(1, 3)
    d = rng.uniform(1.5, 2.5) if truth else 0.0
    xs = [round(v0 + i * d + rng.uniform(-0.05, 0.05), 1) for i in range(4)]
    ev = f"Measured speed of the cart at 1-second intervals: {', '.join(str(x) for x in xs)} m/s."
    a, b = "the speed rises steadily", "the speed stays the same"
    # v2 wording: "accelerating" was ambiguous (constant speed can still be centripetal acceleration; samples can
    # miss changes between readings), which Inkling pointed out in v1. Ask about the measured speed only.
    return ev, (a, b) if truth else (b, a), ("Did the measured speed increase over these readings?",
                                             "Did the measured speed stay the same over these readings?")


def rhat(rng, truth):
    xs = [round(rng.uniform(1.000, 1.008) if truth else rng.uniform(1.25, 1.9), 3) for _ in range(4)]
    ess = rng.randint(2500, 6000) if truth else rng.randint(15, 60)
    ev = f"MCMC diagnostics for four parameters: Gelman-Rubin R-hat = {', '.join(map(str, xs))}; bulk ESS >= {ess}."
    a, b = "R-hat is close to 1 and ESS is large, so the chains have converged", \
        "R-hat is well above 1.1, so the chains have not converged"
    return ev, (a, b) if truth else (b, a), ("Have the chains converged?", "Have the chains failed to converge?")


def eig(rng, truth):
    s = -1 if truth else 1
    re1, im1, re2 = s * round(rng.uniform(0.2, 1.5), 2), round(rng.uniform(0.5, 3), 2), s * round(rng.uniform(1, 4), 2)
    ev = f"Eigenvalues of the Jacobian at the equilibrium: {re1} ± {im1}i, {re2}."
    a, b = "all eigenvalues have negative real parts, so the equilibrium is stable", \
        "the eigenvalues have positive real parts, so the equilibrium is unstable"
    return ev, (a, b) if truth else (b, a), ("Is the equilibrium linearly stable?", "Is the equilibrium unstable?")


def margins(rng, truth):
    gm = round(rng.uniform(6, 15), 1) if truth else -round(rng.uniform(2, 8), 1)
    pm = rng.randint(35, 70) if truth else -rng.randint(5, 30)
    ev = f"Loop-transfer analysis of the controller: gain margin {gm} dB, phase margin {pm} degrees."
    a, b = "both margins are positive, so the closed loop is stable", \
        "the margins are negative, so the closed loop is unstable"
    return ev, (a, b) if truth else (b, a), ("Is the closed loop stable?", "Is the closed loop unstable?")


def spectrum(rng, truth):
    f0 = round(rng.uniform(0.5, 4), 2)
    ev = (f"Power spectrum of the signal: a single sharp peak at {f0} Hz with harmonics at {2 * f0:.2f} and "
          f"{3 * f0:.2f} Hz; the floor between peaks is 70 dB down." if truth else
          f"Power spectrum of the signal: broadband, decaying smoothly from 0.1 to 50 Hz, no isolated peaks; "
          f"autocorrelation falls to zero within {rng.randint(2, 6)} cycles.")
    a, b = "the spectrum shows discrete peaks, so the signal is periodic", \
        "the spectrum is broadband, so the signal is not periodic"
    return ev, (a, b) if truth else (b, a), ("Is the signal periodic?", "Is the signal aperiodic?")


def reynolds(rng, truth):
    re_ = rng.randint(80000, 400000) if truth else rng.randint(200, 1500)
    ev = f"Pipe flow measurement: Reynolds number Re = {re_:,} (smooth circular pipe, fully developed flow)."
    a, b = "Re is far above about 4,000, so the flow is turbulent", "Re is below about 2,000, so the flow is laminar"
    return ev, (a, b) if truth else (b, a), ("Is the flow turbulent?", "Is the flow laminar?")


EVERYDAY = [ci, ph, temp, ab, cart]
SPECIALIST = [rhat, eig, margins, spectrum, reynolds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--set", choices=["everyday", "specialist"], default="everyday")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    FAMILIES = EVERYDAY if a.set == "everyday" else SPECIALIST
    for i in range(a.n):
        fam = FAMILIES[i % len(FAMILIES)]
        truth = rng.random() < 0.5
        ev, (right, wrong), (q_pos, q_neg) = fam(rng, truth)
        rel = ["contradict", "contradict", "agree", "none"][i // len(FAMILIES) % 4]
        pol = rng.random() < 0.5
        lines = [ev]
        if rel != "none":
            lines.append(f'{rng.choice(SOURCES)}: "From the numbers above, {right if rel == "agree" else wrong}."')
        lines.append(f"Question: {q_pos if pol else q_neg} Answer TRUE or FALSE.")
        rows.append({"id": f"{a.set}-{i:04d}", "family": fam.__name__, "rel": rel, "kind": "transfer",
                     "name": "na", "prompt": "\n".join(lines), "answer": truth if pol else (not truth)})
    with open(a.out, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "items", hashlib.md5(open(a.out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
