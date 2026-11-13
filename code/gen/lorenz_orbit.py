"""One real orbit of Lorenz at sigma = 13, beta = 16/5 just below the crisis (r = 27.42).
Starting points: the exact generator of crisis_fast.py (job 380188: rng seed 23, r list below, 256 per r).
Integrator: the same scalar RK4 (dt = 0.002) and settle rule (ball 0.05 around C+-, confirmed 100 later at 0.005).
Usage: orbit.py scan N        -> settle times of the first N starts at r = 27.42, next to job 380188's values
       orbit.py trace J OUT   -> distance to the nearest fixed point every 1.0 time unit, plus x, y, z every 0.02
                                 for the first 400 and the last 400 time units before settling (for the animation)
"""
import json, sys, numpy as np
from numba import njit, prange
SIG, BETA, DT, RADIUS, CONFIRM = 13.0, 3.2, 0.002, 0.05, 100.0
RLIST = [27.42, 27.45, 27.5, 27.55, 27.6, 27.7, 27.8, 28.0, 28.2, 28.3]   # job 380188
R = 27.42

@njit(inline="always")
def step(x, y, z, r):
    h = DT
    k1x = SIG * (y - x); k1y = x * (r - z) - y; k1z = x * y - BETA * z
    x2 = x + 0.5 * h * k1x; y2 = y + 0.5 * h * k1y; z2 = z + 0.5 * h * k1z
    k2x = SIG * (y2 - x2); k2y = x2 * (r - z2) - y2; k2z = x2 * y2 - BETA * z2
    x3 = x + 0.5 * h * k2x; y3 = y + 0.5 * h * k2y; z3 = z + 0.5 * h * k2z
    k3x = SIG * (y3 - x3); k3y = x3 * (r - z3) - y3; k3z = x3 * y3 - BETA * z3
    x4 = x + h * k3x; y4 = y + h * k3y; z4 = z + h * k3z
    k4x = SIG * (y4 - x4); k4y = x4 * (r - z4) - y4; k4z = x4 * y4 - BETA * z4
    return (x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x), y + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y),
            z + h / 6 * (k1z + 2 * k2z + 2 * k3z + k4z))

@njit(inline="always")
def dist(x, y, z, r):
    c = np.sqrt(BETA * (r - 1)); zc = r - 1
    return min(np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - zc) ** 2), np.sqrt((x + c) ** 2 + (y + c) ** 2 + (z - zc) ** 2))

@njit(parallel=True)
def settle(ics, r, horizon):
    out = np.full(ics.shape[0], -1.0)
    for i in prange(ics.shape[0]):
        x, y, z = ics[i, 0], ics[i, 1], ics[i, 2]
        k, n, nc = 0, int(horizon / DT), int(CONFIRM / DT)
        while k < n:
            x, y, z = step(x, y, z, r); k += 1
            if (k & 63) == 0 and dist(x, y, z, r) < RADIUS:
                te = k * DT
                for _ in range(nc):
                    x, y, z = step(x, y, z, r)
                k += nc
                if dist(x, y, z, r) < RADIUS / 10:
                    out[i] = te
                    break
    return out

@njit
def trace(x, y, z, r, t_end):
    n = int(t_end)
    d = np.empty(n)
    per = int(1.0 / DT)
    for i in range(n):
        for _ in range(per):
            x, y, z = step(x, y, z, r)
        d[i] = dist(x, y, z, r)
    return d

@njit
def checkpoints(x, y, z, r, t_end):
    n = int(t_end)
    out = np.empty((n + 1, 3)); out[0] = (x, y, z)
    per = int(1.0 / DT)
    for i in range(1, n + 1):
        for _ in range(per):
            x, y, z = step(x, y, z, r)
        out[i] = (x, y, z)
    return out

@njit
def xyz(x, y, z, r, t0, t1, every):
    k0, k1, ke = int(t0 / DT), int(t1 / DT), int(every / DT)
    out = np.empty(((k1 - k0) // ke + 1, 3)); j = 0
    for k in range(k1):
        if k >= k0 and (k - k0) % ke == 0:
            out[j] = (x, y, z); j += 1
        x, y, z = step(x, y, z, r)
    return out[:j]

def ics_for(r):
    rng = np.random.default_rng(23)
    for rr in RLIST:
        n = 256
        ic = np.column_stack([rng.uniform(-20, 20, n), rng.uniform(-25, 25, n), rng.uniform(0, 1.6 * rr, n)])
        if rr == r:
            return ic

if __name__ == "__main__":
    ic = ics_for(R)
    if sys.argv[1] == "scan":
        n = int(sys.argv[2])
        ref = json.loads(open(sys.argv[3]).readline())["t_a"][:n]
        t = settle(ic[:n], R, 3e6)
        for j in range(n):
            print(j, round(t[j]), round(ref[j]), flush=True)
    else:
        j, out = int(sys.argv[2]), sys.argv[3]
        t_s = settle(ic[j:j + 1], R, 3e6)[0]
        d = trace(*ic[j], R, max(t_s + 300, 720000.0))
        head = xyz(*ic[j], R, 0.0, 400.0, 0.02)
        tail = xyz(*ic[j], R, t_s - 400, t_s + 200, 0.02)
        cp = checkpoints(*ic[j], R, max(t_s + 300, 720000.0))
        np.savez_compressed(out, r=R, j=j, ic=ic[j], t_settle=t_s, dist=d, head=head, tail=tail, tail_t0=t_s - 400, cp=cp)
        print("saved", out, "t_settle", t_s)
