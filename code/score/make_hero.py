"""Hero image (static PNG + animated GIF): 4 panels, claim written on the image. Numbers from the same raw files as
make_figs.py; Lorenz panel from figures/data/orbit_r27.42_j0.npz (not in repo, 17 MB)."""
import gzip, json, os, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
ROOT = Path(__file__).resolve().parents[2]; S, A = ROOT / "results", ROOT / "items"
for f in ("AvenirBook", "AvenirMedium", "AvenirHeavy"):
    p = os.path.expanduser(f"~/.cache/teaser_fonts/{f}.ttf")
    if os.path.exists(p): font_manager.fontManager.addfont(p)
plt.rcParams.update({"font.family": ["Avenir", "DejaVu Sans"], "font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
INK, GREY, BLUE, RED = "#1f2a44", "#b3b9c0", "#1a73e8", "#c0392b"
def load(f):
    f = Path(f); g = Path(str(f) + ".gz")
    fh = gzip.open(g, "rt") if g.exists() else open(f)
    return {r["id"]: r for r in map(json.loads, fh)}
ok = lambda r: r["pred"] is not None and r["pred"] == r["answer"]
items = load(A / "atlas_pilot3b/items.jsonl"); base, rl = load(S / "main/base.jsonl"), load(S / "main/lora.jsonl")
acc = lambda res, c: 100 * np.mean([ok(res[i]) for i in items if items[i]["claim"] == c])
eff = {e: load(S / f"effort/effort{e}.jsonl") for e in ("0.1", "0.5", "0.9")}
ids = list(eff["0.1"]); eff["0.99"] = base
effc = lambda res, c: 100 * np.mean([ok(res[i]) for i in ids if items[i]["claim"] == c])
di = load(A / "inkling_rl/dose.jsonl"); db_, dr = load(S / "sources/base.jsonl"), load(S / "sources/lora.jsonl")
dose = lambda res, a: 100 * np.mean([ok(res[i]) for i in di if di[i]["auth"] == a])
orb = np.load(ROOT / "figures/data/orbit_r27.42_j0.npz"); d, ts = orb["dist"], float(orb["t_settle"])

EX = ["0.1", "0.5", "0.9", "0.99"]
E_ar = [effc(eff[e], "arith") for e in EX]; R_ar = effc(rl, "arith")
AUTH = [("user", "a user\nguessed"), ("student", "student\nhomework"), ("catalogue", "peer-reviewed\ncatalogue"), ("verified_db", "verified\ndatabase")]
D_b = [dose(db_, a) for a, _ in AUTH]; D_r = [dose(dr, a) for a, _ in AUTH]
ARMS = [("none", "no\nnote"), ("arith", "note misreads\nthe numbers"), ("catalogue", "'peer-reviewed\ncatalogue'"), ("db", "'verified\ndatabase'")]
C_b = [acc(base, c) for c, _ in ARMS]; C_r = [acc(rl, c) for c, _ in ARMS]

fig, axs = plt.subplots(2, 2, figsize=(12, 7.6), dpi=150)
fig.patch.set_facecolor("white")
fig.text(0.04, 0.955, "Inkling 2 Small trusts the note over the numbers.", fontsize=21, color=INK, weight="bold")
fig.text(0.04, 0.915, "Simulation results for a chaotic system in the prompt, plus a note that gets them wrong. Accuracy, higher is better.",
         fontsize=12.5, color="#5f6368")
def panel(ax, title):
    ax.cla(); ax.set_title(title, loc="left", fontsize=14, color=INK, weight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)

def draw(t):  # t in [0, 1]: 0 = base only, 1 = everything shown
    a = axs[0, 0]; panel(a, "1. Thinking harder makes it worse")
    a.plot(range(4), E_ar, "-o", color=GREY, lw=2.5, ms=7)
    for k in (0, 3): a.text(k, E_ar[k] + 5, f"{E_ar[k]:.0f}%", ha="center", fontsize=12, color=INK)
    a.set_xticks(range(4), EX); a.set_xlabel("reasoning effort", fontsize=11); a.set_ylim(0, 115); a.set_xlim(-.3, 3.6)
    a.set_yticks([0, 50, 100])
    if t > 0.6:
        a.plot(3, R_ar, marker="*", ms=20, color=BLUE); a.text(3.12, R_ar - 2, "after RL", color=BLUE, fontsize=11.5, va="top")
    a = axs[0, 1]; panel(a, "2. It listens to who wrote the note")
    x = np.arange(4); g = min(1, t / 0.6)
    a.bar(x - .2, D_b, .38, color=GREY); a.bar(x + .2, [v * g for v in D_r], .38, color=BLUE)
    for k in range(4):
        a.text(k - .2, D_b[k] + 3, f"{D_b[k]:.0f}", ha="center", fontsize=11, color=INK)
        if t >= 0.6: a.text(k + .2, D_r[k] + 3, f"{D_r[k]:.0f}", ha="center", fontsize=11, color=BLUE)
    a.set_xticks(x, [l for _, l in AUTH], fontsize=10); a.set_ylim(0, 115); a.set_yticks([0, 50, 100])
    a = axs[1, 0]; panel(a, "3. $29 of LoRA RL on Tinker fixes most of it")
    a.bar(x - .2, C_b, .38, color=GREY, label="Inkling 2 Small"); a.bar(x + .2, [v * g for v in C_r], .38, color=BLUE, label="+ LoRA RL (40 steps)")
    for k in range(4):
        a.text(k - .2, C_b[k] + 3, f"{C_b[k]:.0f}", ha="center", fontsize=11, color=INK)
        if t >= 0.6: a.text(k + .2, C_r[k] + 3, f"{C_r[k]:.0f}", ha="center", fontsize=11, color=BLUE)
    a.set_xticks(x, [l for _, l in ARMS], fontsize=10); a.set_ylim(0, 125); a.set_yticks([0, 50, 100])
    a.legend(frameon=False, fontsize=10, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.04))
    a.text(-.45, 116, "note wordings and systems never trained on", fontsize=9.5, color="#5f6368")
    a = axs[1, 1]; panel(a, "4. Not fixed: it stops simulating too early")
    B = 2000; nb = len(d) // B; hi = d[:nb * B].reshape(nb, B).max(1); lo = d[:nb * B].reshape(nb, B).min(1)
    tm = (np.arange(nb) + .5) * B / 1000
    a.fill_between(tm, np.maximum(lo, 3e-14), np.maximum(hi, 5e-14), color="#3b6ea5", lw=0)
    a.axvspan(0, 4.95, color=RED, alpha=.9, lw=0); a.set_yscale("log"); a.set_ylim(1e-14, 300); a.set_xlim(0, 700)
    a.set_yticks([]); a.set_xlabel("time (thousands)", fontsize=11)
    a.axvline(ts / 1000, color=INK, lw=1.2, ls="--")
    a.text(ts / 1000 - 10, 1e-6, "looks chaotic for\n558,740 time units,\nthen settles", ha="right", fontsize=10.5, color=INK)
    a.annotate("Inkling checked\nonly this sliver", xy=(3, 1e-9), xytext=(60, 1e-11), fontsize=10.5, color=RED,
               arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    fig.text(0.04, 0.012, "Inkling 2 Small via Tinker, effort 0.99 unless shown. Panels 1-3: one sample per question (80 / 194 / 500 per bar). "
             "Panel 4: one real Lorenz orbit, sigma 13, beta 16/5, r 27.42.", fontsize=8.5, color="#80868b")
draw(1.0); fig.subplots_adjust(left=.05, right=.98, top=.84, bottom=.1, hspace=.55, wspace=.18)
out = ROOT / "figures"; fig.savefig(out / "hero.png"); print("png ok")
T = [0] * 8 + list(np.linspace(0, 1, 14)) + [1] * 24
def frame(i):
    for t in fig.texts[2:]: t.remove()
    draw(T[i])
anim = FuncAnimation(fig, frame, frames=len(T))
anim.save(out / "hero.gif", writer=PillowWriter(fps=8)); print("gif ok")
