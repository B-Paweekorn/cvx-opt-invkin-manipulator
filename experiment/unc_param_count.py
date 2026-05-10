"""
Parameter Storage Count — Methods 1-5
======================================
Actually runs one full iteration of each algorithm, captures every live
variable at peak usage, and counts its scalar elements directly from the
array's .size attribute (no formulas).

Run:
    python experiment/param_count.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

NEAT_DIR   = Path(__file__).resolve().parent.parent
RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(NEAT_DIR))

from utils.ik_utils import (
    LINK_LENGTHS, grad_f, hess_f, objective, backtracking
)

# A fixed reachable goal used for all snapshots
_goal = np.array([sum(LINK_LENGTHS) * 0.5, 0.5])
_q0   = np.zeros(len(LINK_LENGTHS))

ALPHA_GD = 0.03   # same constant used in 1_gradient_descent.py


# ── one-iteration snapshots ───────────────────────────────────────────────────
# Each function runs exactly one step of the algorithm and returns a dict of
# {variable_name: numpy_array}.  Scalars are wrapped in np.array([v]) so .size
# always works.  Variables that are overwritten / freed before end-of-step are
# NOT included (only what is still alive at the end of the iteration).

def snapshot_m1():
    """Method 1 — Gradient Descent (fixed step)"""
    q     = _q0.copy().astype(float)
    g     = grad_f(LINK_LENGTHS, q, _goal)
    alpha = np.array([ALPHA_GD])
    q     = q - alpha[0] * g          # update; q is overwritten each iter
    return {"q": q, "g": g, "alpha": alpha}


def snapshot_m2():
    """Method 2 — Steepest Descent (Armijo backtracking)"""
    q     = _q0.copy().astype(float)
    f_obj = lambda x: objective(LINK_LENGTHS, x, _goal)
    g     = grad_f(LINK_LENGTHS, q, _goal)
    d     = -g
    alpha = np.array([backtracking(f_obj, q, d, g)])
    q     = q + alpha[0] * d
    return {"q": q, "g": g, "d": d, "alpha": alpha}


def snapshot_m3():
    """Method 3 — Newton's Method (Gauss-Newton Hessian)"""
    q     = _q0.copy().astype(float)
    f_obj = lambda x: objective(LINK_LENGTHS, x, _goal)
    g     = grad_f(LINK_LENGTHS, q, _goal)
    H     = hess_f(LINK_LENGTHS, q)
    d     = np.linalg.solve(H, -g)
    alpha = np.array([backtracking(f_obj, q, d, g)])
    q     = q + alpha[0] * d
    return {"q": q, "g": g, "H": H, "d": d, "alpha": alpha}


def snapshot_m4():
    """Method 4 — BFGS (inverse Hessian approximation)"""
    q     = _q0.copy().astype(float)
    n     = len(q)
    H_inv = np.eye(n)                              # carried across iterations
    g     = grad_f(LINK_LENGTHS, q, _goal)
    f_obj = lambda x: objective(LINK_LENGTHS, x, _goal)
    d     = -(H_inv @ g)
    alpha = np.array([backtracking(f_obj, q, d, g)])
    q_new = q + alpha[0] * d
    g_new = grad_f(LINK_LENGTHS, q_new, _goal)
    s     = q_new - q
    y     = g_new - g
    # BFGS rank-2 update (happens before next iter — all vars alive here)
    sy = float(np.dot(s, y))
    if sy > 1e-10:
        rho   = 1.0 / sy
        I     = np.eye(n)
        A     = I - rho * np.outer(s, y)
        H_inv = A @ H_inv @ A.T + rho * np.outer(s, s)
    return {"q": q_new, "g": g_new, "H_inv": H_inv,
            "s": s, "y": y, "d": d, "alpha": alpha}


def snapshot_m5():
    """Method 5 — Conjugate Gradient (Polak-Ribière)"""
    q     = _q0.copy().astype(float)
    f_obj = lambda x: objective(LINK_LENGTHS, x, _goal)
    g     = grad_f(LINK_LENGTHS, q, _goal)
    d     = -g.copy()
    alpha = np.array([backtracking(f_obj, q, d, g)])
    q_new = q + alpha[0] * d
    g_new = grad_f(LINK_LENGTHS, q_new, _goal)
    g_norm_sq = float(np.dot(g, g))
    beta  = np.array([max(0.0, float(np.dot(g_new, g_new - g)) / g_norm_sq)])
    d     = -g_new + beta[0] * d      # CG direction update (d reused)
    return {"q": q_new, "g": g_new, "d": d, "g_new": g_new,
            "beta": beta, "alpha": alpha}


SNAPSHOTS = {
    "1 Gradient Descent":   snapshot_m1,
    "2 Steepest Descent":   snapshot_m2,
    "3 Newton's Method":    snapshot_m3,
    "4 BFGS":               snapshot_m4,
    "5 Conjugate Gradient": snapshot_m5,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def scalar_count(v):
    """Number of scalar floats in a numpy array."""
    return int(np.asarray(v).size)


def collect(snap_fn):
    """Run the snapshot function, return list of (name, array, size)."""
    snap = snap_fn()
    return [(name, arr, scalar_count(arr)) for name, arr in snap.items()]


# ── text breakdown ────────────────────────────────────────────────────────────

def print_breakdown():
    n   = len(LINK_LENGTHS)
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Parameter Storage Breakdown   (n = {n} joints, counted from live vars)")
    print(sep)
    for algo, snap_fn in SNAPSHOTS.items():
        rows  = collect(snap_fn)
        total = sum(sz for _, _, sz in rows)
        parts = " + ".join(f"{nm}({sz})" for nm, _, sz in rows)
        print(f"\n  {algo}")
        print(f"    {parts}")
        print(f"    {'-'*50}")
        print(f"    Total = {total} scalars")
        for nm, arr, sz in rows:
            shape = str(np.asarray(arr).shape)
            print(f"      {nm:8s}  shape={shape:12s}  size={sz}")
    print(sep)


# ── scaling table (actual allocation at each n) ───────────────────────────────

def _storage_at_n(algo_name, n_joints):
    """Compute total scalar storage by actually allocating arrays for n_joints."""
    _q   = np.zeros(n_joints)
    _g   = np.zeros(n_joints)
    _d   = np.zeros(n_joints)
    _a   = np.array([0.0])

    if algo_name == "1 Gradient Descent":
        return _q.size + _g.size + _a.size

    if algo_name == "2 Steepest Descent":
        return _q.size + _g.size + _d.size + _a.size

    if algo_name == "3 Newton's Method":
        _H = np.zeros((n_joints, n_joints))
        return _q.size + _g.size + _H.size + _d.size + _a.size

    if algo_name == "4 BFGS":
        _H_inv = np.zeros((n_joints, n_joints))
        _s = np.zeros(n_joints)
        _y = np.zeros(n_joints)
        return _q.size + _g.size + _H_inv.size + _s.size + _y.size + _d.size + _a.size

    if algo_name == "5 Conjugate Gradient":
        _g_new = np.zeros(n_joints)
        _beta  = np.array([0.0])
        return _q.size + _g.size + _d.size + _g_new.size + _beta.size + _a.size

    return 0


def print_scaling_table(ns=(2, 5, 10, 50, 100, 500)):
    header = f"{'Algorithm':<28}" + "".join(f"  n={n:>4}" for n in ns)
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print("  Storage scaling with n (counted via actual array allocation)")
    print(sep)
    print(header)
    print(sep)
    for algo in SNAPSHOTS:
        row = f"  {algo:<26}" + "".join(
            f"  {_storage_at_n(algo, n):>6}" for n in ns)
        print(row)
    print(sep)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_scaling():
    ns     = np.arange(1, 101)
    colors = plt.cm.tab10(np.linspace(0, 1, len(SNAPSHOTS)))

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for (algo, _), color in zip(SNAPSHOTS.items(), colors):
        label   = algo.split(" ", 1)[1]
        storage = [_storage_at_n(algo, int(n)) for n in ns]
        ax1.plot(ns, storage, label=label, color=color, linewidth=2)
        ax2.semilogy(ns, storage, label=label, color=color, linewidth=2)

    for ax in (ax1, ax2):
        ax.set_xlabel("Number of joints  n")
        ax.set_ylabel("Scalars stored")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax1.set_title("Parameter Storage vs n  (linear scale)")
    ax2.set_title("Parameter Storage vs n  (log scale)")
    plt.suptitle("Storage counted via actual array allocation — O(n²) methods explode", fontsize=11)
    plt.tight_layout()
    out = RESULT_DIR / "param_count.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


def plot_stacked_bar():
    n          = len(LINK_LENGTHS)
    algo_names = [a.split(" ", 1)[1] for a in SNAPSHOTS]
    all_vars   = []
    snap_data  = {}

    for algo, snap_fn in SNAPSHOTS.items():
        rows = collect(snap_fn)
        snap_data[algo] = {nm: sz for nm, _, sz in rows}
        for nm, _, _ in rows:
            if nm not in all_vars:
                all_vars.append(nm)

    var_colors = plt.cm.Set3(np.linspace(0, 1, len(all_vars)))
    fig, ax    = plt.subplots(figsize=(9, 5))
    bottoms    = np.zeros(len(SNAPSHOTS))

    for vi, vname in enumerate(all_vars):
        heights = [snap_data[algo].get(vname, 0) for algo in SNAPSHOTS]
        ax.bar(algo_names, heights, bottom=bottoms,
               color=var_colors[vi], label=vname)
        bottoms += np.array(heights, dtype=float)

    totals = [sum(snap_data[algo].values()) for algo in SNAPSHOTS]
    for i, total in enumerate(totals):
        ax.text(i, total + 0.3, str(total), ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(totals) * 1.25)
    ax.set_title(f"Parameter Storage by Variable  (n = {n} joints, actual sizes)")
    ax.set_ylabel("Scalars stored")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULT_DIR / f"param_stacked_n{n}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_breakdown()
    print_scaling_table(ns=(2, 5, 10, 50, 100, 500))
    plot_scaling()
    plot_stacked_bar()
    print(f"\nDone.  Results saved to {RESULT_DIR}")
