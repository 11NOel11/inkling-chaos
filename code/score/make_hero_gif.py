"""Animated hero for README / X: the real Lorenz orbit (sigma 13, beta 16/5, r 27.42) drawn as an ink line over a dot
grid, then the end of the transient where it falls onto a fixed point. Light and dark versions.
Data: data/orbit_page.npz (head = start of the orbit, tail = the last 60 time units before it settles)."""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
ROOT = Path(__file__).resolve().parents[2]
o = np.load(ROOT / "data/orbit_page.npz")
head, tail = o["head"][::6], o["tail"][::8]
seq = [("h", k) for k in range(900, len(head), 40)] + [("t", k) for k in range(0, len(tail), 45)] + [("t", len(tail) - 1)] * 18
for theme, bg, ink in (("light", "#ffffff", "#141414"), ("dark", "#0c0c0c", "#ececea")):
    fig = plt.figure(figsize=(6.4, 4.0), dpi=90); fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(bg); ax.set_xlim(-26, 26); ax.set_ylim(2, 50); ax.axis("off")
    gx, gy = np.meshgrid(np.linspace(-24, 24, 13), np.linspace(4, 48, 9)); ax.scatter(gx, gy, s=1.6, c=ink, lw=0)
    ln, = ax.plot([], [], color=ink, lw=1.1, solid_joinstyle="round"); dot, = ax.plot([], [], "o", color=ink, ms=3.5)
    txt = ax.text(0, 3.2, "", ha="center", color=ink, fontsize=9, family="DejaVu Sans")
    def frame(i):
        part, k = seq[i]
        pts = head[max(0, k - 900):k + 1] if part == "h" else tail[max(0, k - 1400):k + 1]
        ln.set_data(pts[:, 0], pts[:, 2]); dot.set_data([pts[-1, 0]], [pts[-1, 2]])
        txt.set_text("" if part == "h" else ("t = 558,740: it settles" if k == len(tail) - 1 else "half a million time units later"))
        return ln, dot, txt
    anim = FuncAnimation(fig, frame, frames=len(seq), blit=True)
    out = ROOT / f"docs/hero_{theme}.gif"; anim.save(out, writer=PillowWriter(fps=16)); plt.close(fig)
    print(out, out.stat().st_size // 1024, "KB", len(seq), "frames")
