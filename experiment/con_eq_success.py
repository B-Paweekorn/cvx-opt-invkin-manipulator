"""
Success-Rate Benchmark — Methods 6b & 7b (Equality-Constrained IK)
===================================================================
Run from the NEAT/ directory:
    python experiment/con_eq_success.py

Tests both methods from N_TRIALS random initial joint configurations
drawn from three regions:
  • Inside box   : q0 inside box but outside unit circle (||q|| > 1)
  • Inside circle: q0 inside box and inside unit circle (||q|| < 1)
  • On arc       : q0 on unit circle arc inside the box

For each trial records:
  converged, iterations, final dist-to-goal, final |h(q)| violation

Output
------
  con_eq_success_rate.png   : success-rate bars + convergence scatter per region
  con_eq_success_iter.png   : iteration histogram and final-error CDF
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util

NEAT_DIR   = Path(__file__).resolve().parent.parent
RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(NEAT_DIR))

from utils.ik_utils import forward_kinematics

# ── settings ──────────────────────────────────────────────────────────────────

LL           = np.array([2.0, 2.0])
JOINT_LIMITS = np.array([[-np.pi / 2, np.pi / 2],
                          [-np.pi / 2, np.pi / 2]])
LIM_LO = JOINT_LIMITS[:, 0]
LIM_HI = JOINT_LIMITS[:, 1]
TOL      = 0.05
MAX_ITER = 2000

# Goal that IS reachable on the unit-circle arc (q*=[-0.5, 0.866])
GOAL = np.array([3.623, -0.243])

N_TRIALS = 50   # trials per region
SEED     = 95


# ── load solvers ──────────────────────────────────────────────────────────────

def _load(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m6b = _load("m6b", NEAT_DIR / "6_newton_kkt.py")
m7b = _load("m7b", NEAT_DIR / "7_infeasible_newton.py")

METHODS = [
    ("6b Newton+KKT+eq",  "#377eb8",
     lambda q0: m6b.solve(LL.tolist(), q0, GOAL,
                          joint_limits=JOINT_LIMITS, tol=TOL, max_iter=MAX_ITER)),
    ("7b Infeasible+eq",  "#e41a1c",
     lambda q0: m7b.solve(LL.tolist(), q0, GOAL,
                          joint_limits=JOINT_LIMITS, tol=TOL, max_iter=MAX_ITER)),
]


# ── sample initial configurations ─────────────────────────────────────────────

def sample_starts(seed=SEED):
    rng = np.random.default_rng(seed)

    # Region A: inside box but OUTSIDE unit circle (||q|| > 1)
    inside = []
    while len(inside) < N_TRIALS:
        batch = rng.uniform(LIM_LO, LIM_HI, size=(N_TRIALS * 4, 2))
        batch = batch[np.linalg.norm(batch, axis=1) > 1.0]
        inside.extend(batch.tolist())
    inside = np.array(inside[:N_TRIALS])

    # Region B: unit circle arc inside box
    thetas = np.linspace(0, 2*np.pi, 100_000)
    qs_all = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    mask   = np.all((qs_all >= LIM_LO) & (qs_all <= LIM_HI), axis=1)
    arc_qs = qs_all[mask]
    idx    = rng.choice(len(arc_qs), size=N_TRIALS, replace=True)
    on_arc = arc_qs[idx]

    # Region C: inside box AND inside unit circle (||q|| < 1)
    inside_circle = []
    while len(inside_circle) < N_TRIALS:
        batch = rng.uniform(LIM_LO, LIM_HI, size=(N_TRIALS * 4, 2))
        batch = batch[np.linalg.norm(batch, axis=1) < 1.0]
        inside_circle.extend(batch.tolist())
    inside_circle = np.array(inside_circle[:N_TRIALS])

    return {
        "Inside box":    inside,
        "Inside circle": inside_circle,
        "On arc":        on_arc,
    }


# ── run benchmark ─────────────────────────────────────────────────────────────

def run_benchmark(starts):
    """
    Returns nested dict:
      results[region][method_name] = {
          "conv", "iters", "dist", "h_final", "q0s"
      }
    """
    results = {}
    for region, q0s in starts.items():
        results[region] = {}
        for mname, _, solver in METHODS:
            conv_list = []; iter_list = []; dist_list = []; h_list = []
            t0 = time.perf_counter()
            for i, q0 in enumerate(q0s):
                q_sol, conv, n_iter, _, _ = solver(q0.copy())
                pos  = forward_kinematics(LL, q_sol)
                dist = float(np.linalg.norm(pos - GOAL))
                hv   = float(q_sol[0]**2 + q_sol[1]**2 - 1.0)
                conv_list.append(conv)
                iter_list.append(n_iter)
                dist_list.append(dist)
                h_list.append(abs(hv))
            elapsed = time.perf_counter() - t0
            results[region][mname] = {
                "conv":    np.array(conv_list),
                "iters":   np.array(iter_list),
                "dist":    np.array(dist_list),
                "h_final": np.array(h_list),
                "q0s":     q0s,
                "elapsed": elapsed,
            }
            rate = 100 * np.mean(conv_list)
            print(f"  {region:<14} {mname:<22}  "
                  f"success={rate:5.1f}%  "
                  f"avg_iter={np.mean(iter_list):6.1f}  "
                  f"avg_dist={np.mean(dist_list):.4f}  "
                  f"avg|h|={np.mean(h_list):.4f}  "
                  f"({elapsed:.1f}s)")
    return results


# ── Figure 1 — success rate bars + scatter ───────────────────────────────────

def plot_success_rate(results, starts):
    regions  = list(results.keys())
    n_reg    = len(regions)

    fig = plt.figure(figsize=(6 * (n_reg + 1), 6))
    ax_bar = fig.add_subplot(1, n_reg + 1, 1)
    bar_w  = 0.35
    x      = np.arange(n_reg)

    for mi, (mname, mcolor, _) in enumerate(METHODS):
        rates = [100 * np.mean(results[r][mname]["conv"]) for r in regions]
        bars  = ax_bar.bar(x + mi * bar_w, rates, bar_w,
                           label=mname, color=mcolor, alpha=0.85)
        ax_bar.bar_label(bars, fmt="%.0f%%", fontsize=10, padding=2)

    ax_bar.set_xticks(x + bar_w / 2)
    ax_bar.set_xticklabels([r.replace(" ", "\n") for r in regions], fontsize=11)
    ax_bar.set_ylabel("Success rate (%)", fontsize=12)
    ax_bar.set_ylim(0, 115)
    ax_bar.set_title("Convergence Rate\nper Start Region", fontsize=12,
                     fontweight="bold")
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.3)

    for pi, region in enumerate(regions):
        ax = fig.add_subplot(1, n_reg + 1, pi + 2)
        q0s = starts[region]

        from matplotlib.lines import Line2D
        handles = []
        for mi, (mname, mcolor, _) in enumerate(METHODS):
            conv = results[region][mname]["conv"]
            offset = (mi - 0.5) * 0.04
            marker = "o" if mi == 0 else "^"
            ax.scatter(q0s[conv,  0] + offset, q0s[conv,  1],
                       c=mcolor, s=35, zorder=4, alpha=0.8,
                       marker=marker, edgecolors="white", linewidths=0.4)
            ax.scatter(q0s[~conv, 0] + offset, q0s[~conv, 1],
                       c=mcolor, s=35, zorder=4, alpha=0.5,
                       marker="x", linewidths=1.2)
            rate = 100 * np.mean(results[region][mname]["conv"])
            handles.append(Line2D([0],[0], marker=marker, color=mcolor, ms=7,
                                  label=f"{mname.split('+')[0].strip()} ✓ {rate:.0f}%",
                                  lw=0, markeredgecolor="white", markeredgewidth=0.4))
            handles.append(Line2D([0],[0], marker="x", color=mcolor, ms=7,
                                  label=f"{mname.split('+')[0].strip()} ✗ failed",
                                  lw=0, mew=1.2))

        bx = [LIM_LO[0], LIM_HI[0], LIM_HI[0], LIM_LO[0], LIM_LO[0]]
        by = [LIM_LO[1], LIM_LO[1], LIM_HI[1], LIM_HI[1], LIM_LO[1]]
        ax.plot(bx, by, "k-", lw=1.8, zorder=5, label="box limits")
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), "b--", lw=1.0, alpha=0.5,
                label="unit circle")

        ax.legend(handles=handles, fontsize=7, loc="upper right")
        ax.set_xlim(-np.pi/2 - 0.2, np.pi/2 + 0.2)
        ax.set_ylim(-np.pi/2 - 0.2, np.pi/2 + 0.2)
        ax.set_aspect("equal")
        ax.set_title(f"Start region: {region}", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$q_1$", fontsize=11); ax.set_ylabel(r"$q_2$", fontsize=11)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"6b vs 7b — Success Rate  |  {N_TRIALS} trials/region  "
        f"|  Goal {GOAL.tolist()}",
        fontsize=13)
    plt.tight_layout()
    out = RESULT_DIR / "con_eq_success_rate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Figure 2 — iteration histogram + error CDF ───────────────────────────────

def plot_iter_stats(results):
    regions = list(results.keys())
    n_reg   = len(regions)

    fig, axes = plt.subplots(2, n_reg, figsize=(6 * n_reg, 10))

    for ci, region in enumerate(regions):
        ax_h = axes[0, ci]
        ax_c = axes[1, ci]

        for mname, mcolor, _ in METHODS:
            r    = results[region][mname]
            conv = r["conv"]

            iters_conv = r["iters"][conv]
            if len(iters_conv):
                ax_h.hist(iters_conv, bins=30, color=mcolor, alpha=0.55,
                          label=f"{mname.split('+')[0].strip()} "
                                f"(n={conv.sum()}, "
                                f"med={int(np.median(iters_conv))})")

            dist_sorted = np.sort(r["dist"])
            cdf         = np.arange(1, len(dist_sorted)+1) / len(dist_sorted)
            ax_c.plot(dist_sorted, cdf, color=mcolor, lw=2,
                      label=f"{mname.split('+')[0].strip()}")

        ax_h.axvline(MAX_ITER, color="black", ls="--", lw=1.2,
                     label=f"max={MAX_ITER}")
        ax_h.set_title(f"{region}\nIteration Count (converged)", fontsize=11,
                       fontweight="bold")
        ax_h.set_xlabel("Iterations", fontsize=10)
        ax_h.set_ylabel("Count", fontsize=10)
        ax_h.legend(fontsize=8); ax_h.grid(True, alpha=0.3)

        ax_c.axvline(TOL, color="black", ls="--", lw=1.2, label=f"tol={TOL}")
        ax_c.set_title(f"{region}\nCDF of Final Distance to Goal", fontsize=11,
                       fontweight="bold")
        ax_c.set_xlabel("Final dist to goal", fontsize=10)
        ax_c.set_ylabel("Cumulative fraction", fontsize=10)
        ax_c.set_xlim(left=0)
        ax_c.legend(fontsize=8); ax_c.grid(True, alpha=0.3)

    fig.suptitle(
        f"Iteration Distribution & Error CDF  |  {N_TRIALS} trials/region  "
        f"|  Goal {GOAL.tolist()}",
        fontsize=13)
    plt.tight_layout()
    out = RESULT_DIR / "con_eq_success_iter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── console summary table ─────────────────────────────────────────────────────

def print_summary(results):
    hdr = (f"{'Region':<14} {'Method':<22} {'Success%':>8} "
           f"{'MedIter':>8} {'MeanDist':>9} {'Mean|h|':>9}")
    sep = "=" * len(hdr)
    print("\n" + sep + "\n" + hdr + "\n" + sep)
    for region in results:
        for mname, _, _ in METHODS:
            r    = results[region][mname]
            rate = 100 * np.mean(r["conv"])
            med  = int(np.median(r["iters"][r["conv"]])) if r["conv"].any() else 0
            print(f"  {region:<12} {mname:<22} {rate:>7.1f}%"
                  f" {med:>8d} {np.mean(r['dist']):>9.4f} {np.mean(r['h_final']):>9.4f}")
    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Success-Rate Benchmark — 6b vs 7b  (equality h(q)=||q||^2-1)")
    print("=" * 65)
    print(f"  Links        : {LL.tolist()}")
    print(f"  Joint limits : {np.degrees(JOINT_LIMITS[0]).tolist()} deg")
    print(f"  Goal         : {GOAL.tolist()}")
    print(f"  Trials/region: {N_TRIALS}")
    print(f"  Tolerance    : {TOL}   Max iter: {MAX_ITER}")

    print("\nSampling start configurations...")
    starts = sample_starts()
    for region, q0s in starts.items():
        print(f"  {region:<14}: {len(q0s)} starts")

    print("\nRunning benchmark...")
    results = run_benchmark(starts)

    print_summary(results)

    print("\nGenerating plots...")
    plot_success_rate(results, starts)
    plot_iter_stats(results)
    print(f"\nAll results saved to {RESULT_DIR}")
