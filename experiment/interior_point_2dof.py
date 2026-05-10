"""
Interior-Point Log-Barrier IK — 2-DOF (Algorithm 8)
=====================================================
Run from the NEAT/ directory:
    python experiment/interior_point_2dof.py

Setup
-----
  2-link planar arm  :  L1 = L2 = 2.0
  Joint limits       :  q1, q2 in [-pi/2, pi/2]   (box)
  Obstacle           :  circle at OBS_CENTER, radius OBS_RADIUS + OBS_OFFSET clearance

Experiments
-----------
  1. Landscape + trajectory  : cost contour in joint space with the feasible region
                                (box minus obstacle shadow) and optimisation path
  2. Central path            : how the barrier minimum shifts as mu decreases,
                                visualising the interior-point central path
  3. Convergence history     : distance to goal and barrier value vs iteration
  4. Arm posture             : start and final arm configuration with obstacle

Output
------
  ip_landscape.png   : cost landscape, feasible region, optimisation trajectory
  ip_central_path.png: central-path minima as mu decreases
  ip_history.png     : convergence distance + barrier value vs iteration
  ip_arm.png         : arm posture (start vs solution) with obstacle
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import importlib.util

NEAT_DIR   = Path(__file__).resolve().parent.parent
RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(NEAT_DIR))

from utils.ik_utils import forward_kinematics, objective, grad_f, jacobian

# ── problem setup ─────────────────────────────────────────────────────────────

LL           = np.array([2.0, 2.0])
JOINT_LIMITS = np.array([[-np.pi / 2, np.pi / 2],
                          [-np.pi / 2, np.pi / 2]])
LIM_LO = JOINT_LIMITS[:, 0]
LIM_HI = JOINT_LIMITS[:, 1]
TOL      = 0.05
MAX_ITER = 3000

OBS_CENTER = np.array([1.5, 1.5])
OBS_RADIUS = 0.5
OBS_OFFSET = 0.3   # clearance margin

# ▼▼▼  edit freely  ▼▼▼
GOAL = np.array([0.0, 4.0])
Q0   = np.array([0.0, 0.0])
# ▲▲▲──────────────────▲▲▲

MU_INIT  = 1.0
MU_SCALE = 0.1
MU_MIN   = 1e-10
OUTER    = 60
INNER    = 50
REG      = 1e-6


# ── barrier helpers ───────────────────────────────────────────────────────────

def _arm_points(q):
    pts = [np.zeros(2)]
    for i in range(len(LL)):
        c = np.sum(q[:i+1])
        pts.append(pts[-1] + LL[i] * np.array([np.cos(c), np.sin(c)]))
    return pts


def _barrier_joint(q):
    s_lo = q - LIM_LO;  s_hi = LIM_HI - q
    if np.any(s_lo <= 0) or np.any(s_hi <= 0):
        return np.inf, np.zeros(2), np.zeros(2)
    val  = -np.sum(np.log(s_lo) + np.log(s_hi))
    g    = -1.0/s_lo + 1.0/s_hi
    hd   =  1.0/s_lo**2 + 1.0/s_hi**2
    return val, g, hd


def _barrier_obs(q):
    pts = _arm_points(q)
    n   = len(q)
    val, g, hd = 0.0, np.zeros(n), np.zeros(n)
    for k, p in enumerate(pts[1:], start=1):
        diff   = p - OBS_CENTER
        d      = float(np.linalg.norm(diff))
        margin = d - (OBS_RADIUS + OBS_OFFSET)
        if margin <= 0:
            return np.inf, np.zeros(n), np.zeros(n)
        val -= np.log(margin)
        Jk = np.zeros((2, n))
        for i in range(k):
            for j in range(i, k):
                c = np.sum(q[:j+1])
                Jk[0, i] -= LL[j] * np.sin(c)
                Jk[1, i] += LL[j] * np.cos(c)
        dd = (diff / d) @ Jk
        g  -= dd / margin
        hd += (dd / margin)**2
    return val, g, hd


def _aug_obj(q, mu):
    vj, _, _ = _barrier_joint(q)
    vo, _, _ = _barrier_obs(q)
    return objective(LL, q, GOAL) + mu * (vj + vo)


def _aug_grad_hess(q, mu):
    e  = forward_kinematics(LL, q) - GOAL
    J  = jacobian(LL, q)
    gf = J.T @ e
    Hf = J.T @ J + REG * np.eye(len(q))
    _, gj, hj = _barrier_joint(q)
    _, go, ho = _barrier_obs(q)
    return gf + mu*(gj+go), Hf + mu*np.diag(hj+ho)


# ── solver with trajectory recording ─────────────────────────────────────────

def run_solver(q0=Q0, record_mu_steps=False):
    q   = np.clip(q0.copy().astype(float),
                  LIM_LO + 0.01*(LIM_HI-LIM_LO),
                  LIM_HI - 0.01*(LIM_HI-LIM_LO))
    mu  = MU_INIT
    traj = [q.copy()]
    dist_hist = []
    barrier_hist = []
    mu_snapshots = []   # (mu, q_at_min) for central path plot
    total = 0
    conv  = False

    for _ in range(OUTER):
        if total >= MAX_ITER: break
        for _ in range(INNER):
            if total >= MAX_ITER: break
            total += 1
            pos  = forward_kinematics(LL, q)
            dist = float(np.linalg.norm(pos - GOAL))
            vj, _, _ = _barrier_joint(q)
            vo, _, _ = _barrier_obs(q)
            dist_hist.append(dist)
            barrier_hist.append(mu * (vj + vo) if np.isfinite(vj+vo) else np.nan)
            if dist < TOL and mu < 1e-4:
                conv = True; break
            g, H = _aug_grad_hess(q, mu)
            try:    d = np.linalg.solve(H, -g)
            except: d = -g / (np.linalg.norm(g) + 1e-12)
            alpha = 1.0
            for _ in range(50):
                qt = q + alpha*d
                if np.isfinite(_barrier_joint(qt)[0]) and np.isfinite(_barrier_obs(qt)[0]):
                    if _aug_obj(qt, mu) <= _aug_obj(q, mu) + 1e-4*alpha*float(np.dot(g,d)):
                        break
                alpha *= 0.5
            q = np.clip(q + alpha*d, LIM_LO+1e-9, LIM_HI-1e-9)
            traj.append(q.copy())
        if conv: break
        mu_snapshots.append((mu, q.copy()))
        mu = max(mu*MU_SCALE, MU_MIN)

    return {
        "q_sol":    q,
        "conv":     conv,
        "traj":     np.array(traj),
        "dist":     dist_hist,
        "barrier":  barrier_hist,
        "mu_snaps": mu_snapshots,
        "n_iter":   total,
    }


# ── global optimum (unconstrained) via dense grid search ─────────────────────

def find_global_optimum(res=400):
    """Return joint config q* that minimises f(q) over the full joint-limit box."""
    q1s = np.linspace(LIM_LO[0], LIM_HI[0], res)
    q2s = np.linspace(LIM_LO[1], LIM_HI[1], res)
    best_val, best_q = np.inf, None
    for q1 in q1s:
        for q2 in q2s:
            v = objective(LL, np.array([q1, q2]), GOAL)
            if v < best_val:
                best_val, best_q = v, np.array([q1, q2])
    return best_q, best_val


# ── feasible-region mask ──────────────────────────────────────────────────────

def _feasible_mask(Q1g, Q2g):
    mask = np.ones(Q1g.shape, dtype=bool)
    for i in range(Q1g.shape[0]):
        for j in range(Q1g.shape[1]):
            q = np.array([Q1g[i,j], Q2g[i,j]])
            if np.any(q < LIM_LO) or np.any(q > LIM_HI):
                mask[i,j] = False; continue
            pts = _arm_points(q)
            for p in pts[1:]:
                if np.linalg.norm(p - OBS_CENTER) < OBS_RADIUS + OBS_OFFSET:
                    mask[i,j] = False; break
    return mask


# ── Figure 1 — Landscape + trajectory ────────────────────────────────────────

def plot_landscape(res):
    FS_TITLE = 13; FS_LABEL = 12; FS_TICK = 11; FS_LEG = 10
    q_global, f_global = find_global_optimum()

    MARGIN = 0.3; RES = 200
    q1s = np.linspace(LIM_LO[0]-MARGIN, LIM_HI[0]+MARGIN, RES)
    q2s = np.linspace(LIM_LO[1]-MARGIN, LIM_HI[1]+MARGIN, RES)
    Q1g, Q2g = np.meshgrid(q1s, q2s)
    F = np.array([[objective(LL, np.array([q1,q2]), GOAL)
                   for q1 in q1s] for q2 in q2s])
    levels = np.logspace(np.log10(max(F.min(), 1e-4)), np.log10(F.max()), 35)
    tks    = [LIM_LO[0], 0.0, LIM_HI[0]]
    tlbls  = [r"$-\pi/2$", r"$0$", r"$\pi/2$"]

    print("  Computing feasible-region mask (may take a moment)...")
    mask = _feasible_mask(Q1g, Q2g)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for ax, show_feas in zip(axes, [False, True]):
        ax.contourf(Q1g, Q2g, F, levels=levels, cmap="Greys", alpha=0.85)
        ax.contour(Q1g, Q2g, F, levels=levels, colors="black",
                   linewidths=0.3, alpha=0.4)

        if show_feas:
            # shade infeasible region red
            infeas = np.ma.masked_where(mask, np.ones_like(F))
            ax.contourf(Q1g, Q2g, infeas, levels=[0.5, 1.5],
                        colors=["#ff9999"], alpha=0.45)
            ax.set_title("With Feasible Region\n(red = infeasible / obstacle shadow)",
                         fontsize=FS_TITLE, fontweight="bold")
        else:
            ax.set_title("Cost Landscape $f(q_1,q_2)$\n(unconstrained view)",
                         fontsize=FS_TITLE, fontweight="bold")

        # box boundary
        bx = [LIM_LO[0], LIM_HI[0], LIM_HI[0], LIM_LO[0], LIM_LO[0]]
        by = [LIM_LO[1], LIM_LO[1], LIM_HI[1], LIM_HI[1], LIM_LO[1]]
        ax.plot(bx, by, "k-", lw=2, label="joint-limit box", zorder=5)

        # trajectory
        traj = res["traj"]
        n    = len(traj)
        ax.plot(traj[:,0], traj[:,1], color="#377eb8", lw=1.8, alpha=0.8, zorder=6)
        sc = ax.scatter(traj[:,0], traj[:,1], c=np.arange(n),
                        cmap="plasma", s=20, zorder=7, edgecolors="none", alpha=0.9)
        ax.scatter(*traj[0],  s=180, marker="^", color="#377eb8",
                   edgecolors="black", lw=0.9, zorder=8, label="start $Q_0$")
        ax.scatter(*traj[-1], s=160, marker="s", color="#377eb8",
                   edgecolors="black", lw=0.9, zorder=8,
                   label=f"end  ({n-1} iters)")
        ax.scatter(*q_global, s=300, marker="*", color="gold",
                   edgecolors="black", lw=0.8, zorder=10,
                   label=fr"global optimum $q^*$  ($f={f_global:.3f}$)")
        ax.annotate(r"global $q^*$", xy=q_global,
                    xytext=(q_global[0]+0.12, q_global[1]+0.12),
                    fontsize=9, color="goldenrod", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.0),
                    zorder=11)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.12)
        fig.colorbar(sc, cax=cax, label="iteration")

        ax.set_xlim(LIM_LO[0]-MARGIN, LIM_HI[0]+MARGIN)
        ax.set_ylim(LIM_LO[1]-MARGIN, LIM_HI[1]+MARGIN)
        ax.set_xticks(tks); ax.set_xticklabels(tlbls, fontsize=FS_TICK)
        ax.set_yticks(tks); ax.set_yticklabels(tlbls, fontsize=FS_TICK)
        ax.set_xlabel(r"$q_1$ (rad)", fontsize=FS_LABEL)
        ax.set_ylabel(r"$q_2$ (rad)", fontsize=FS_LABEL)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2, color="white")
        ax.legend(fontsize=FS_LEG, loc="upper left", framealpha=0.9)

    fig.suptitle(
        fr"Interior-Point Log-Barrier  |  Links {LL.tolist()}  |  "
        fr"Goal {GOAL.tolist()}  |  Obs $c={OBS_CENTER.tolist()},\,r={OBS_RADIUS}$",
        fontsize=FS_TITLE, fontweight="bold")
    out = RESULT_DIR / "ip_landscape.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Figure 2 — Central path ───────────────────────────────────────────────────

def plot_central_path(res):
    FS_TITLE = 13; FS_LABEL = 12; FS_TICK = 11; FS_LEG = 10
    q_global, f_global = find_global_optimum()

    MARGIN = 0.3; RES = 180
    q1s = np.linspace(LIM_LO[0]-MARGIN, LIM_HI[0]+MARGIN, RES)
    q2s = np.linspace(LIM_LO[1]-MARGIN, LIM_HI[1]+MARGIN, RES)
    Q1g, Q2g = np.meshgrid(q1s, q2s)
    F = np.array([[objective(LL, np.array([q1,q2]), GOAL)
                   for q1 in q1s] for q2 in q2s])
    levels = np.logspace(np.log10(max(F.min(), 1e-4)), np.log10(F.max()), 35)
    tks   = [LIM_LO[0], 0.0, LIM_HI[0]]
    tlbls = [r"$-\pi/2$", r"$0$", r"$\pi/2$"]

    snaps = res["mu_snaps"]
    mus   = [s[0] for s in snaps]
    qs    = np.array([s[1] for s in snaps])

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.contourf(Q1g, Q2g, F, levels=levels, cmap="Greys", alpha=0.8)
    ax.contour(Q1g, Q2g, F, levels=levels, colors="black",
               linewidths=0.3, alpha=0.4)

    bx = [LIM_LO[0], LIM_HI[0], LIM_HI[0], LIM_LO[0], LIM_LO[0]]
    by = [LIM_LO[1], LIM_LO[1], LIM_HI[1], LIM_HI[1], LIM_LO[1]]
    ax.plot(bx, by, "k-", lw=2, zorder=5, label="joint-limit box")

    # plot central-path points coloured by mu
    log_mus = np.log10(np.array(mus) + 1e-15)
    sc = ax.scatter(qs[:,0], qs[:,1],
                    c=log_mus, cmap="cool", s=100,
                    edgecolors="black", lw=0.6, zorder=8,
                    label=r"barrier min at each $\mu$")
    ax.plot(qs[:,0], qs[:,1], "--", color="steelblue", lw=1.2,
            alpha=0.6, zorder=7, label="central path")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.12)
    cb  = fig.colorbar(sc, cax=cax, label=r"$\log_{10}(\mu)$")

    ax.scatter(*res["traj"][0],  s=200, marker="^", color="gray",
               edgecolors="black", lw=0.9, zorder=9, label="start $Q_0$")
    ax.scatter(*res["q_sol"],    s=160, marker="s", color="#377eb8",
               edgecolors="black", lw=0.8, zorder=9, label="optimizer end")
    ax.scatter(*q_global, s=300, marker="*", color="gold",
               edgecolors="black", lw=0.8, zorder=10,
               label=fr"global optimum $q^*$  ($f={f_global:.3f}$)")
    ax.annotate(r"global $q^*$", xy=q_global,
                xytext=(q_global[0]+0.12, q_global[1]+0.12),
                fontsize=9, color="goldenrod", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.0),
                zorder=11)

    ax.set_xlim(LIM_LO[0]-MARGIN, LIM_HI[0]+MARGIN)
    ax.set_ylim(LIM_LO[1]-MARGIN, LIM_HI[1]+MARGIN)
    ax.set_xticks(tks); ax.set_xticklabels(tlbls, fontsize=FS_TICK)
    ax.set_yticks(tks); ax.set_yticklabels(tlbls, fontsize=FS_TICK)
    ax.set_xlabel(r"$q_1$ (rad)", fontsize=FS_LABEL)
    ax.set_ylabel(r"$q_2$ (rad)", fontsize=FS_LABEL)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2, color="white")
    ax.legend(fontsize=FS_LEG, loc="upper right", framealpha=0.9)
    ax.set_title(r"Central Path: barrier minimum shifts as $\mu\to 0$",
                 fontsize=FS_TITLE, fontweight="bold")

    fig.suptitle(
        fr"Interior-Point Central Path  |  Goal {GOAL.tolist()}  |  "
        fr"Obs $c={OBS_CENTER.tolist()},\,r={OBS_RADIUS}$",
        fontsize=FS_TITLE, fontweight="bold")
    out = RESULT_DIR / "ip_central_path.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Figure 3 — Convergence history ───────────────────────────────────────────

def plot_history(res):
    FS_TITLE = 13; FS_LABEL = 12; FS_LEG = 10

    dist    = res["dist"]
    barrier = res["barrier"]
    iters   = np.arange(1, len(dist)+1)

    fig, (ax_d, ax_b) = plt.subplots(2, 1, figsize=(10, 8),
                                      sharex=True, constrained_layout=True)

    ax_d.semilogy(iters, dist, color="#377eb8", lw=2)
    ax_d.axhline(TOL, color="black", ls="--", lw=1.2, label=f"tol = {TOL}")
    ax_d.set_ylabel("Distance to goal", fontsize=FS_LABEL)
    ax_d.set_title("Convergence: Distance to Goal", fontsize=FS_TITLE, fontweight="bold")
    ax_d.legend(fontsize=FS_LEG); ax_d.grid(True, which="both", alpha=0.3)

    # mark outer mu reductions
    outer_step = INNER
    for k, (mu, _) in enumerate(res["mu_snaps"]):
        idx = min((k+1)*outer_step, len(dist)-1)
        ax_d.axvline(idx, color="orange", ls=":", lw=1.0, alpha=0.7,
                     label=r"$\mu$ reduced" if k == 0 else "_")
        ax_b.axvline(idx, color="orange", ls=":", lw=1.0, alpha=0.7)

    barrier_clean = [b if np.isfinite(b) else np.nan for b in barrier]
    ax_b.semilogy(iters, barrier_clean, color="#e41a1c", lw=2,
                  label=r"$\mu\,\varphi(\mathbf{q})$")
    ax_b.set_ylabel(r"Barrier contribution  $\mu\,\varphi(\mathbf{q})$",
                    fontsize=FS_LABEL)
    ax_b.set_xlabel("Iteration", fontsize=FS_LABEL)
    ax_b.set_title(r"Barrier Term Decreases as $\mu\to 0$",
                   fontsize=FS_TITLE, fontweight="bold")
    ax_b.legend(fontsize=FS_LEG); ax_b.grid(True, which="both", alpha=0.3)

    ax_d.legend(fontsize=FS_LEG)

    fig.suptitle(
        fr"Interior-Point Convergence  |  Goal {GOAL.tolist()}  |  "
        fr"$Q_0={Q0.tolist()}$  |  {res['n_iter']} iters  "
        fr"({'converged' if res['conv'] else 'NOT converged'})",
        fontsize=FS_TITLE, fontweight="bold")
    out = RESULT_DIR / "ip_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Figure 4 — Arm posture ────────────────────────────────────────────────────

def _draw_arm(ax, q, color, label, ls="-", alpha=1.0, annotate_joints=False):
    pts = [np.zeros(2)]
    for i in range(len(LL)):
        c = np.sum(q[:i+1])
        pts.append(pts[-1] + LL[i]*np.array([np.cos(c), np.sin(c)]))
    pts = np.array(pts)
    ax.plot(pts[:,0], pts[:,1], marker="o", ls=ls, color=color,
            lw=2.5, markersize=7, label=label, alpha=alpha, zorder=4)
    ax.plot(*pts[-1], "s", color=color, markersize=10, zorder=5, alpha=alpha)
    if annotate_joints:
        offsets = [(0.12, 0.12), (0.12, 0.12), (0.12, -0.18)]
        joint_names = ["Base", "J1", "J2 (EE)"]
        for k, (pt, (dx, dy), name) in enumerate(zip(pts, offsets, joint_names)):
            ax.annotate(name, xy=pt, xytext=(pt[0]+dx, pt[1]+dy),
                        fontsize=9, color=color, alpha=max(alpha, 0.75),
                        arrowprops=dict(arrowstyle="-", color=color,
                                        lw=0.8, alpha=0.6),
                        zorder=9)


def plot_arm(res):
    FS_TITLE = 13; FS_LABEL = 12; FS_LEG = 10

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # workspace cloud
    ws_q1s = np.linspace(*JOINT_LIMITS[0], 120)
    ws_q2s = np.linspace(*JOINT_LIMITS[1], 120)
    ws_pts = np.array([forward_kinematics(LL, np.array([q1,q2]))
                       for q1 in ws_q1s for q2 in ws_q2s])
    ax.scatter(ws_pts[:,0], ws_pts[:,1], s=1, c="lightgray",
               alpha=0.3, rasterized=True, label="workspace")

    # obstacle
    obs = plt.Circle(OBS_CENTER, OBS_RADIUS,
                     color="tomato", alpha=0.7, zorder=6, label="obstacle")
    clearance = plt.Circle(OBS_CENTER, OBS_RADIUS+OBS_OFFSET,
                           color="tomato", alpha=0.2, fill=True,
                           ls="--", zorder=5, label=f"clearance (+{OBS_OFFSET})")
    ax.add_patch(obs); ax.add_patch(clearance)

    # start and solution arms
    _draw_arm(ax, Q0,           "gray",    f"start  $Q_0={Q0.tolist()}$",
              ls="--", alpha=0.6)
    _draw_arm(ax, res["q_sol"], "#377eb8",
              f"solution  q*=[{res['q_sol'][0]:.2f},{res['q_sol'][1]:.2f}]  "
              f"dist={np.linalg.norm(forward_kinematics(LL,res['q_sol'])-GOAL):.3f}",
              annotate_joints=True)

    # goal marker + annotation
    ax.plot(*GOAL, "*", color="gold", ms=20, mew=1.2, markeredgecolor="black",
            zorder=10, label=f"goal  {GOAL.tolist()}")
    ax.annotate(f"Goal\n{GOAL.tolist()}", xy=GOAL,
                xytext=(GOAL[0]+0.2, GOAL[1]+0.3),
                fontsize=10, fontweight="bold", color="goldenrod",
                arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.2),
                zorder=11)
    ax.plot(0, 0, "ko", ms=9, zorder=11, label="base")

    ax.set_aspect("equal")
    ax.legend(fontsize=FS_LEG, loc="upper left", framealpha=0.9)
    ax.set_xlabel("X", fontsize=FS_LABEL); ax.set_ylabel("Y", fontsize=FS_LABEL)
    ax.grid(True, alpha=0.25)
    status = "converged" if res["conv"] else "NOT converged"
    ax.set_title(f"Arm Posture — Start vs Solution  ({status}, {res['n_iter']} iters)",
                 fontsize=FS_TITLE, fontweight="bold")

    fig.suptitle(
        fr"Interior-Point IK  |  Links {LL.tolist()}  |  Goal {GOAL.tolist()}  |  "
        fr"Obs $c={OBS_CENTER.tolist()},\,r={OBS_RADIUS}$",
        fontsize=FS_TITLE, fontweight="bold")
    out = RESULT_DIR / "ip_arm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── feasibility report ────────────────────────────────────────────────────────

def check_feasibility(q_sol):
    pos  = forward_kinematics(LL, q_sol)
    dist = float(np.linalg.norm(pos - GOAL))
    pts  = _arm_points(q_sol)
    min_clearance = min(
        np.linalg.norm(p - OBS_CENTER) - OBS_RADIUS for p in pts[1:]
    )
    box_ok = np.all(q_sol >= LIM_LO) and np.all(q_sol <= LIM_HI)
    print(f"\n  Solution q* = {np.round(q_sol,4).tolist()}")
    print(f"  End-effector dist to goal : {dist:.4f}  "
          f"({'OK' if dist < TOL else 'FAIL'})")
    print(f"  Box constraints satisfied : {box_ok}")
    print(f"  Min obstacle clearance    : {min_clearance:.4f}  "
          f"({'OK' if min_clearance > 0 else 'VIOLATION'})")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Interior-Point Log-Barrier IK  (Algorithm 8 — 2-DOF)")
    print("=" * 65)
    print(f"  Links        : {LL.tolist()}")
    print(f"  Joint limits : {np.degrees(JOINT_LIMITS[0]).tolist()} deg")
    print(f"  Goal         : {GOAL.tolist()}")
    print(f"  Start Q0     : {Q0.tolist()}")
    print(f"  Obstacle     : center={OBS_CENTER.tolist()}, "
          f"r={OBS_RADIUS}, clearance={OBS_OFFSET}")
    print(f"  mu schedule  : {MU_INIT} -> x{MU_SCALE} per outer step")

    print("\nRunning solver...")
    res = run_solver(Q0)
    status = "converged" if res["conv"] else "NOT converged"
    print(f"  {status}  in {res['n_iter']} iterations")
    check_feasibility(res["q_sol"])

    print("\nGenerating figures...")
    plot_landscape(res)
    plot_central_path(res)
    plot_history(res)
    plot_arm(res)
    print(f"\nAll results saved to {RESULT_DIR}")
