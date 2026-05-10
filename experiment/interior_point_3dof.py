"""
Interior-Point Log-Barrier IK — 3-DOF Planar Arm (Algorithm 8)
===============================================================
Run from the NEAT/ directory:
    python experiment/interior_point_3dof.py

Setup
-----
  3-link planar arm  :  L1=1.5, L2=1.5, L3=1.0
  Joint limits       :  q1, q2, q3 in [-pi/2, pi/2]
  Obstacle           :  circle at OBS_CENTER with clearance margin

Key insight — redundancy
-------------------------
  3 joints, 2D task space → 1 degree of redundancy.
  There is a 1D family of joint configs that all reach the same goal.
  The interior-point barrier picks the point on this null-space curve
  that stays feasible (inside box, away from obstacle).

Figures
-------
  ip3_nullspace.png   : null-space curve in task space (arm configs) +
                        joint-space projections (q1-q2, q1-q3, q2-q3)
  ip3_history.png     : distance to goal + barrier value vs iteration
  ip3_arm.png         : start vs solution arm with obstacle
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

NEAT_DIR   = Path(__file__).resolve().parent.parent
RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(NEAT_DIR))

from utils.ik_utils import jacobian

# ── problem setup ─────────────────────────────────────────────────────────────

LL           = np.array([2.0, 2.0, 2.0])
N            = len(LL)
JOINT_LIMITS = np.array([[-np.pi/2, np.pi/2]] * N)
LIM_LO       = JOINT_LIMITS[:, 0]
LIM_HI       = JOINT_LIMITS[:, 1]
TOL          = 0.05
MAX_ITER     = 3000

OBS_CENTER = np.array([1.5, 1.2])
OBS_RADIUS = 0.5
OBS_OFFSET = 0.25

# ▼▼▼  edit freely  ▼▼▼
GOAL = np.array([1.8, 4.5])
Q0   = np.array([-0.4, 0.6, 0.3])
# ▲▲▲──────────────────▲▲▲

MU_INIT  = 1.0
MU_SCALE = 0.1
MU_MIN   = 1e-10
OUTER    = 60
INNER    = 50
REG      = 1e-6


# ── kinematics ────────────────────────────────────────────────────────────────

def fk(q):
    """Forward kinematics — returns end-effector position."""
    pos = np.zeros(2)
    cum = 0.0
    for i in range(N):
        cum += q[i]
        pos += LL[i] * np.array([np.cos(cum), np.sin(cum)])
    return pos


def arm_points(q):
    """All joint positions including base and end-effector."""
    pts = [np.zeros(2)]
    cum = 0.0
    for i in range(N):
        cum += q[i]
        pts.append(pts[-1] + LL[i] * np.array([np.cos(cum), np.sin(cum)]))
    return pts


def obj(q):
    return 0.5 * float(np.linalg.norm(fk(q) - GOAL)**2)


def grad_obj(q):
    e = fk(q) - GOAL
    J = jacobian(LL.tolist(), q)
    return J.T @ e, J.T @ J + REG * np.eye(N)


# ── barrier ───────────────────────────────────────────────────────────────────

def barrier_joint(q):
    s_lo = q - LIM_LO;  s_hi = LIM_HI - q
    if np.any(s_lo <= 0) or np.any(s_hi <= 0):
        return np.inf, np.zeros(N), np.zeros(N)
    val = -np.sum(np.log(s_lo) + np.log(s_hi))
    g   = -1.0/s_lo + 1.0/s_hi
    hd  =  1.0/s_lo**2 + 1.0/s_hi**2
    return val, g, hd


def barrier_obs(q):
    pts = arm_points(q)
    val, g, hd = 0.0, np.zeros(N), np.zeros(N)
    for k, p in enumerate(pts[1:], start=1):
        diff   = p - OBS_CENTER
        d      = float(np.linalg.norm(diff))
        margin = d - (OBS_RADIUS + OBS_OFFSET)
        if margin <= 0:
            return np.inf, np.zeros(N), np.zeros(N)
        val -= np.log(margin)
        Jk = np.zeros((2, N))
        for i in range(k):
            for j in range(i, k):
                c = float(np.sum(q[:j+1]))
                Jk[0, i] -= LL[j] * np.sin(c)
                Jk[1, i] += LL[j] * np.cos(c)
        dd  = (diff / d) @ Jk
        g  -= dd / margin
        hd += (dd / margin)**2
    return val, g, hd


def aug_obj(q, mu):
    vj, _, _ = barrier_joint(q)
    vo, _, _ = barrier_obs(q)
    return obj(q) + mu*(vj + vo)


def aug_grad_hess(q, mu):
    gf, Hf      = grad_obj(q)
    _, gj, hj   = barrier_joint(q)
    _, go, ho   = barrier_obs(q)
    return gf + mu*(gj+go), Hf + mu*np.diag(hj+ho)


# ── solver ────────────────────────────────────────────────────────────────────

def run_solver(q0=Q0):
    q   = np.clip(q0.copy().astype(float),
                  LIM_LO + 0.01*(LIM_HI-LIM_LO),
                  LIM_HI - 0.01*(LIM_HI-LIM_LO))
    mu  = MU_INIT
    traj      = [q.copy()]
    dist_hist = []
    barr_hist = []
    mu_snaps  = []
    total     = 0
    conv      = False

    for _ in range(OUTER):
        if total >= MAX_ITER: break
        for _ in range(INNER):
            if total >= MAX_ITER: break
            total += 1
            dist = float(np.linalg.norm(fk(q) - GOAL))
            vj, _, _ = barrier_joint(q)
            vo, _, _ = barrier_obs(q)
            dist_hist.append(dist)
            barr_hist.append(mu*(vj+vo) if np.isfinite(vj+vo) else np.nan)
            if dist < TOL and mu < 1e-4:
                conv = True; break
            g, H = aug_grad_hess(q, mu)
            try:    d = np.linalg.solve(H, -g)
            except: d = -g / (np.linalg.norm(g) + 1e-12)
            alpha = 1.0
            for _ in range(50):
                qt = q + alpha*d
                if np.isfinite(barrier_joint(qt)[0]) and np.isfinite(barrier_obs(qt)[0]):
                    if aug_obj(qt, mu) <= aug_obj(q, mu) + 1e-4*alpha*float(np.dot(g,d)):
                        break
                alpha *= 0.5
            q = np.clip(q + alpha*d, LIM_LO+1e-9, LIM_HI-1e-9)
            traj.append(q.copy())
        if conv: break
        mu_snaps.append((mu, q.copy()))
        mu = max(mu*MU_SCALE, MU_MIN)

    return {
        "q_sol":   q,
        "conv":    conv,
        "traj":    np.array(traj),
        "dist":    dist_hist,
        "barrier": barr_hist,
        "mu_snaps":mu_snaps,
        "n_iter":  total,
    }


# ── null-space computation ────────────────────────────────────────────────────

def compute_null_space():
    """
    Parametrise the solution manifold by end-effector orientation θ.
    For each θ: wrist position = goal - L3*[cos θ, sin θ],
    then solve 2-DOF analytic IK for (q1, q2), set q3 = θ - q1 - q2.
    Returns arrays: configs (M,3), feasible mask, obstacle-clear mask.
    """
    L1, L2, L3 = LL
    configs = []
    thetas  = []
    for theta in np.linspace(-np.pi, np.pi, 2000):
        wx = GOAL[0] - L3*np.cos(theta)
        wy = GOAL[1] - L3*np.sin(theta)
        d2 = wx**2 + wy**2
        c2 = (d2 - L1**2 - L2**2) / (2*L1*L2)
        if abs(c2) > 1.0: continue
        for sign in [1, -1]:
            q2_  = sign * np.arccos(np.clip(c2, -1, 1))
            q1_  = np.arctan2(wy, wx) - np.arctan2(L2*np.sin(q2_), L1+L2*np.cos(q2_))
            q3_  = theta - q1_ - q2_
            q    = np.array([q1_, q2_, q3_])
            # wrap to [-pi, pi]
            q    = (q + np.pi) % (2*np.pi) - np.pi
            configs.append(q)
            thetas.append(theta)

    configs = np.array(configs)
    thetas  = np.array(thetas)

    # masks
    box_ok = np.all((configs >= LIM_LO) & (configs <= LIM_HI), axis=1)
    obs_ok = np.array([
        all(np.linalg.norm(p - OBS_CENTER) >= OBS_RADIUS + OBS_OFFSET
            for p in arm_points(q)[1:])
        for q in configs
    ])
    return configs, thetas, box_ok, obs_ok


# ── Figure 1 — Null-space ─────────────────────────────────────────────────────

def plot_null_space(res, configs, thetas, box_ok, obs_ok):
    FS_TITLE = 16; FS_LABEL = 14; FS_TICK = 13; FS_LEG = 11

    feasible = box_ok & obs_ok
    q_sol    = res["q_sol"]
    traj     = res["traj"]                          # (T, 3) joint trajectory
    T        = len(traj)
    traj_ee  = np.array([fk(q) for q in traj])     # (T, 2) end-effector path
    t_norm   = np.linspace(0, 1, T)                 # colour by progress

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs  = fig.add_gridspec(2, 4)
    ax_task = fig.add_subplot(gs[:, :2])
    ax_12   = fig.add_subplot(gs[0, 2])
    ax_13   = fig.add_subplot(gs[0, 3])
    ax_23   = fig.add_subplot(gs[1, 2])
    ax_leg  = fig.add_subplot(gs[1, 3])
    ax_leg.axis("off")

    # ── task space ────────────────────────────────────────────────────────────
    ws_q1 = np.linspace(*JOINT_LIMITS[0], 60)
    ws_q2 = np.linspace(*JOINT_LIMITS[1], 60)
    ws_q3 = np.linspace(*JOINT_LIMITS[2], 60)
    ws_pts = np.array([fk(np.array([q1, q2, q3]))
                       for q1 in ws_q1 for q2 in ws_q2 for q3 in ws_q3])
    ax_task.scatter(ws_pts[:,0], ws_pts[:,1], s=1, c="lightgray",
                    alpha=0.2, rasterized=True)

    # obstacle
    ax_task.add_patch(plt.Circle(OBS_CENTER, OBS_RADIUS,
                                  color="tomato", alpha=0.7, zorder=6))

    # null-space arm configs: sample a subset
    sample_idx = np.linspace(0, len(configs)-1, 12, dtype=int)
    n_feas_sample   = sum(feasible[sample_idx])
    n_infeas_sample = sum(~feasible[sample_idx])
    colors_feas   = plt.cm.Blues(np.linspace(0.4, 0.9, max(n_feas_sample,   1)))
    colors_infeas = plt.cm.Reds( np.linspace(0.4, 0.7, max(n_infeas_sample, 1)))
    ci_f = ci_i = 0
    for idx in sample_idx:
        q   = configs[idx]
        pts = np.array(arm_points(q))
        col = colors_feas[ci_f] if feasible[idx] else colors_infeas[ci_i]
        lw  = 1.5 if feasible[idx] else 0.8
        al  = 0.7 if feasible[idx] else 0.35
        ax_task.plot(pts[:,0], pts[:,1], color=col, lw=lw, alpha=al, zorder=3)
        ax_task.plot(*pts[-1], "o", color=col, ms=4, alpha=al, zorder=4)
        if feasible[idx]: ci_f += 1
        else:             ci_i += 1

    # ── optimization trajectory in task space (end-effector path) ────────────
    sc_traj = ax_task.scatter(traj_ee[:, 0], traj_ee[:, 1],
                               c=t_norm, cmap="plasma", s=8,
                               alpha=0.75, zorder=9, rasterized=True)
    ax_task.plot(traj_ee[:, 0], traj_ee[:, 1],
                 color="purple", lw=0.8, alpha=0.35, zorder=8)
    divider = make_axes_locatable(ax_task)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(sc_traj, cax=cax)
    cbar.set_label("iteration (0→end)", fontsize=FS_LEG)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", f"{T//2}", f"{T}"])
    cbar.ax.tick_params(labelsize=FS_TICK)

    # solution arm
    pts_sol = np.array(arm_points(q_sol))
    # clearance circles at each joint of solution arm
    for pt in pts_sol[1:]:
        ax_task.add_patch(plt.Circle(pt, OBS_OFFSET, color="#1f78b4",
                                      alpha=0.18, zorder=10, linestyle="--",
                                      fill=True, linewidth=1.2))
    ax_task.plot(pts_sol[:,0], pts_sol[:,1], color="#1f78b4", lw=3,
                 marker="o", ms=8, zorder=11,
                 label=f"solution  q*=[{q_sol[0]:.2f},{q_sol[1]:.2f},{q_sol[2]:.2f}]")

    # start arm
    pts_q0 = np.array(arm_points(Q0))
    ax_task.plot(pts_q0[:,0], pts_q0[:,1], "--", color="gray", lw=2,
                 marker="o", ms=6, alpha=0.6, zorder=7,
                 label=f"start  Q0={Q0.tolist()}")

    ax_task.plot(*GOAL, "rx", ms=18, mew=3, zorder=12,
                 label=f"goal  {GOAL.tolist()}")
    ax_task.plot(0, 0, "ko", ms=9, zorder=12, label="base")
    ax_task.plot([], [], color="steelblue", lw=2,
                 label="feasible null-space configs")
    ax_task.plot([], [], color="salmon", lw=1,
                 label="infeasible configs")
    ax_task.scatter([], [], c=[], cmap="plasma", s=15,
                    label=f"solver path ({T} iters)")

    ax_task.set_aspect("equal")
    ax_task.legend(fontsize=FS_LEG, loc="upper left", framealpha=0.9,
                   markerscale=1.3)
    ax_task.set_xlabel("X (m)", fontsize=FS_LABEL)
    ax_task.set_ylabel("Y (m)", fontsize=FS_LABEL)
    ax_task.tick_params(labelsize=FS_TICK)
    ax_task.grid(True, alpha=0.25)
    ax_task.set_title("Task Space — Null-Space Configs + Solver Trajectory",
                      fontsize=FS_TITLE, fontweight="bold")

    # ── joint-space projections with B&W cost landscape ──────────────────────
    GRID = 60
    proj_pairs = [(0,1,2,"$q_1$","$q_2$",ax_12),
                  (0,2,1,"$q_1$","$q_3$",ax_13),
                  (1,2,0,"$q_2$","$q_3$",ax_23)]

    for (i, j, k, xi, yj, ax) in proj_pairs:
        # B&W cost grid (third joint fixed at solution)
        gi = np.linspace(LIM_LO[i], LIM_HI[i], GRID)
        gj = np.linspace(LIM_LO[j], LIM_HI[j], GRID)
        Gi, Gj = np.meshgrid(gi, gj)
        Fg = np.empty_like(Gi)
        for ri in range(GRID):
            for ci in range(GRID):
                q_tmp = np.zeros(3)
                q_tmp[i] = Gi[ri, ci]; q_tmp[j] = Gj[ri, ci]; q_tmp[k] = q_sol[k]
                Fg[ri, ci] = obj(q_tmp)
        ax.contourf(Gi, Gj, Fg, levels=25, cmap="gray_r", alpha=0.80, zorder=1)
        ax.contour( Gi, Gj, Fg, levels=10, colors="white", alpha=0.30,
                    linewidths=0.5, zorder=2)

        # null-space curve
        ax.scatter(configs[~feasible, i], configs[~feasible, j],
                   s=6, c="salmon", alpha=0.5, zorder=4, label="null (infeasible)")
        ax.scatter(configs[feasible,  i], configs[feasible,  j],
                   s=6, c="deepskyblue", alpha=0.85, zorder=5, label="null (feasible)")

        # solver trajectory — cyan line for contrast against gray background
        ax.scatter(traj[:, i], traj[:, j], c=t_norm, cmap="plasma",
                   s=12, alpha=0.90, zorder=7, rasterized=True)
        ax.plot(traj[:, i], traj[:, j],
                color="cyan", lw=1.2, alpha=0.60, zorder=6)

        # solution & start markers
        ax.scatter(q_sol[i], q_sol[j], s=220, marker="*", color="lime",
                   edgecolors="black", lw=0.8, zorder=9, label="solution q*")
        ax.scatter(Q0[i], Q0[j], s=130, marker="^", color="white",
                   edgecolors="black", lw=0.9, zorder=9, label="start Q0")

        ax.set_xlim(LIM_LO[i]-0.05, LIM_HI[i]+0.05)
        ax.set_ylim(LIM_LO[j]-0.05, LIM_HI[j]+0.05)
        for spine in ax.spines.values():
            spine.set_edgecolor("black"); spine.set_linewidth(1.5)
        ax.set_xlabel(xi, fontsize=FS_LABEL)
        ax.set_ylabel(yj, fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_title(f"Cost + null-space  {xi}–{yj}",
                     fontsize=FS_TITLE, fontweight="bold")

    ax_12.legend(fontsize=FS_LEG, loc="upper right")

    # legend panel
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    legend_handles = [
        mpatches.Patch(facecolor="black", alpha=0.85, label="null-space (feasible)"),
        mpatches.Patch(facecolor="salmon",       alpha=0.6,  label="null-space (infeasible)"),
        plt.Line2D([0],[0], marker="*", color="lime",  ms=14, lw=0,
                   markeredgecolor="black", label="solution q*"),
        plt.Line2D([0],[0], marker="^", color="white", ms=11, lw=0,
                   markeredgecolor="black", label="start Q0"),
        plt.Line2D([0],[0], color="cyan", lw=2, alpha=0.8, label="solver path"),
    ]
    ax_leg.legend(handles=legend_handles, fontsize=FS_LEG, loc="upper center",
                  framealpha=0.9, title="Legend", title_fontsize=FS_LABEL)

    sm = ScalarMappable(cmap="plasma", norm=Normalize(0, T))
    sm.set_array([])
    cax2 = ax_leg.inset_axes([0.1, 0.05, 0.8, 0.12])
    cb2  = fig.colorbar(sm, cax=cax2, orientation="horizontal")
    cb2.set_label("iteration", fontsize=FS_LEG)
    cb2.set_ticks([0, T//2, T])
    cb2.set_ticklabels(["0", f"{T//2}", f"{T}"])
    cb2.ax.tick_params(labelsize=FS_TICK)

    n_feas = feasible.sum()
    fig.suptitle(
        fr"3-DOF Redundant Arm — Null Space  |  Links {LL.tolist()}  |  "
        fr"Goal {GOAL.tolist()}  |  "
        fr"Feasible configs: {n_feas}/{len(configs)}",
        fontsize=FS_TITLE, fontweight="bold")
    out = RESULT_DIR / "ip3_nullspace.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Figure 2 — Convergence history ───────────────────────────────────────────

def plot_history(res):
    FS_TITLE = 13; FS_LABEL = 12; FS_LEG = 10

    dist    = res["dist"]
    barrier = [b if np.isfinite(b) else np.nan for b in res["barrier"]]
    iters   = np.arange(1, len(dist)+1)

    fig, (ax_d, ax_b) = plt.subplots(2, 1, figsize=(10, 8),
                                      sharex=True, constrained_layout=True)

    ax_d.semilogy(iters, dist, color="#377eb8", lw=2)
    ax_d.axhline(TOL, color="black", ls="--", lw=1.2, label=f"tol = {TOL}")
    for k, (mu, _) in enumerate(res["mu_snaps"]):
        idx = min((k+1)*INNER, len(dist)-1)
        ax_d.axvline(idx, color="orange", ls=":", lw=1.0, alpha=0.7,
                     label=r"$\mu$ reduced" if k == 0 else "_")
        ax_b.axvline(idx, color="orange", ls=":", lw=1.0, alpha=0.7)
    ax_d.set_ylabel("Distance to goal", fontsize=FS_LABEL)
    ax_d.set_title("Distance to Goal vs Iteration", fontsize=FS_TITLE,
                   fontweight="bold")
    ax_d.legend(fontsize=FS_LEG); ax_d.grid(True, which="both", alpha=0.3)

    barrier_pos = [max(b, 1e-12) if np.isfinite(b) else np.nan for b in barrier]
    ax_b.semilogy(iters, barrier_pos, color="#e41a1c", lw=2)
    ax_b.set_ylabel(r"Barrier  $\mu\,\varphi(\mathbf{q})$", fontsize=FS_LABEL)
    ax_b.set_xlabel("Iteration", fontsize=FS_LABEL)
    ax_b.set_title(r"Barrier Contribution Decreases as $\mu\to 0$",
                   fontsize=FS_TITLE, fontweight="bold")
    ax_b.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        fr"Interior-Point Convergence — 3-DOF  |  Goal {GOAL.tolist()}  |  "
        fr"$Q_0={Q0.tolist()}$  |  {res['n_iter']} iters  "
        fr"({'converged' if res['conv'] else 'NOT converged'})",
        fontsize=FS_TITLE, fontweight="bold")
    out = RESULT_DIR / "ip3_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Figure 3 — Arm posture ────────────────────────────────────────────────────

def plot_arm(res):
    FS_TITLE = 13; FS_LABEL = 12; FS_LEG = 10

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # workspace
    ws_q1 = np.linspace(*JOINT_LIMITS[0], 50)
    ws_q2 = np.linspace(*JOINT_LIMITS[1], 50)
    ws_q3 = np.linspace(*JOINT_LIMITS[2], 50)
    ws_pts = np.array([fk(np.array([q1,q2,q3]))
                       for q1 in ws_q1 for q2 in ws_q2 for q3 in ws_q3])
    ax.scatter(ws_pts[:,0], ws_pts[:,1], s=1, c="lightgray",
               alpha=0.2, rasterized=True, label="workspace")

    # obstacle
    ax.add_patch(plt.Circle(OBS_CENTER, OBS_RADIUS,
                             color="tomato", alpha=0.75, zorder=6,
                             label=f"obstacle  r={OBS_RADIUS}"))

    def _draw(q, color, label, ls="-", alpha=1.0, show_clearance=False):
        pts = np.array(arm_points(q))
        if show_clearance:
            for pt in pts[1:]:
                ax.add_patch(plt.Circle(pt, OBS_OFFSET, color=color,
                                         alpha=0.18, zorder=3, linestyle="--",
                                         fill=True, linewidth=1.2))
        ax.plot(pts[:,0], pts[:,1], ls=ls, color=color, lw=2.5,
                marker="o", ms=7, alpha=alpha, zorder=4, label=label)
        ax.plot(*pts[-1], "s", color=color, ms=10, alpha=alpha, zorder=5)

    _draw(Q0, "gray", f"start  Q0={Q0.tolist()}", ls="--", alpha=0.55)
    _draw(res["q_sol"], "#1f78b4",
          f"solution  q*=[{res['q_sol'][0]:.2f},{res['q_sol'][1]:.2f},"
          f"{res['q_sol'][2]:.2f}]  "
          f"dist={np.linalg.norm(fk(res['q_sol'])-GOAL):.3f}",
          show_clearance=True)
    # legend entry for clearance circles
    ax.add_patch(plt.Circle((0,0), 0, color="#1f78b4", alpha=0.25,
                              linestyle="--", label=f"joint clearance r={OBS_OFFSET}"))

    ax.plot(*GOAL, "rx", ms=18, mew=3, zorder=10,
            label=f"goal  {GOAL.tolist()}")
    ax.plot(0, 0, "ko", ms=9, zorder=11, label="base")

    ax.set_aspect("equal")
    ax.legend(fontsize=FS_LEG, loc="upper left", framealpha=0.9)
    ax.set_xlabel("X", fontsize=FS_LABEL); ax.set_ylabel("Y", fontsize=FS_LABEL)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.25)
    status = "converged" if res["conv"] else "NOT converged"
    ax.set_title(f"Arm Posture — Start vs Solution  ({status}, {res['n_iter']} iters)",
                 fontsize=FS_TITLE, fontweight="bold")
    fig.suptitle(
        fr"Interior-Point IK — 3-DOF  |  Links {LL.tolist()}  |  "
        fr"Goal {GOAL.tolist()}",
        fontsize=FS_TITLE, fontweight="bold")
    out = RESULT_DIR / "ip3_arm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Figure 4 — Cost landscape (joint-pair slices) ────────────────────────────

def plot_landscape(res, configs, box_ok, obs_ok):
    """
    Three panels: cost f(qi, qj) on a 2-D grid with the third joint fixed at q*.
    Null-space curve and solver trajectory are overlaid on each panel.
    """
    FS_TITLE = 13; FS_LABEL = 12; FS_TICK = 11
    GRID = 80          # resolution of cost grid

    feasible = box_ok & obs_ok
    q_sol    = res["q_sol"]
    traj     = res["traj"]          # (T, 3)
    T        = len(traj)
    t_norm   = np.linspace(0, 1, T)

    # three (i,j,k) combos: axes i,j vary; axis k fixed at q_sol[k]
    panels = [
        (0, 1, 2, r"$q_1$", r"$q_2$", r"$q_3=q_3^*$"),
        (0, 2, 1, r"$q_1$", r"$q_3$", r"$q_2=q_2^*$"),
        (1, 2, 0, r"$q_2$", r"$q_3$", r"$q_1=q_1^*$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    for ax, (i, j, k, xi, xj, fix_label) in zip(axes, panels):
        # ── cost grid ────────────────────────────────────────────────────────
        gi = np.linspace(LIM_LO[i], LIM_HI[i], GRID)
        gj = np.linspace(LIM_LO[j], LIM_HI[j], GRID)
        Gi, Gj = np.meshgrid(gi, gj)
        Fg = np.empty_like(Gi)
        for r_idx in range(GRID):
            for c_idx in range(GRID):
                q_tmp = np.zeros(3)
                q_tmp[i] = Gi[r_idx, c_idx]
                q_tmp[j] = Gj[r_idx, c_idx]
                q_tmp[k] = q_sol[k]
                Fg[r_idx, c_idx] = obj(q_tmp)

        # contour fill — cost landscape
        cf = ax.contourf(Gi, Gj, Fg, levels=30, cmap="YlOrRd", alpha=0.85)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cb = fig.colorbar(cf, cax=cax)
        cb.set_label(r"$f(\mathbf{q})$", fontsize=10)

        # contour lines for cleaner depth cues
        ax.contour(Gi, Gj, Fg, levels=10, colors="white", alpha=0.25, linewidths=0.5)

        # ── null-space curve in this projection ───────────────────────────
        ax.scatter(configs[~feasible, i], configs[~feasible, j],
                   s=5, c="lightcoral", alpha=0.4, zorder=4,
                   label="null-space (infeasible)")
        ax.scatter(configs[feasible, i], configs[feasible, j],
                   s=5, c="black", alpha=0.7, zorder=5,
                   label="null-space (feasible)")

        # ── solver trajectory ─────────────────────────────────────────────
        sc = ax.scatter(traj[:, i], traj[:, j], c=t_norm, cmap="plasma",
                        s=10, alpha=0.85, zorder=8, rasterized=True)
        ax.plot(traj[:, i], traj[:, j],
                color="purple", lw=0.8, alpha=0.35, zorder=7)
        cax2 = divider.append_axes("bottom", size="5%", pad=0.45)
        cb2  = fig.colorbar(sc, cax=cax2, orientation="horizontal")
        cb2.set_label("iteration", fontsize=8)
        cb2.set_ticks([0, 0.5, 1])
        cb2.set_ticklabels(["0", f"{T//2}", f"{T}"])

        # ── markers ───────────────────────────────────────────────────────
        ax.scatter(q_sol[i], q_sol[j], s=250, marker="*", color="lime",
                   edgecolors="black", lw=0.8, zorder=10, label="solution q*")
        ax.scatter(Q0[i], Q0[j], s=140, marker="^", color="white",
                   edgecolors="black", lw=0.9, zorder=10, label="start Q0")

        ax.set_xlim(LIM_LO[i]-0.05, LIM_HI[i]+0.05)
        ax.set_ylim(LIM_LO[j]-0.05, LIM_HI[j]+0.05)
        ax.set_xlabel(xi, fontsize=FS_LABEL)
        ax.set_ylabel(xj, fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.15, color="white")
        ax.set_title(
            f"Cost landscape ({xi}, {xj})  |  {fix_label}\n"
            f"null-space curve + solver path",
            fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    fig.suptitle(
        fr"3-DOF Cost Landscape (slice at q*) — Algorithm 8  |  "
        fr"Links {LL.tolist()}  |  Goal {GOAL.tolist()}",
        fontsize=FS_TITLE, fontweight="bold")
    out = RESULT_DIR / "ip3_landscape.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── feasibility check ─────────────────────────────────────────────────────────

def check_feasibility(q_sol):
    pos  = fk(q_sol)
    dist = float(np.linalg.norm(pos - GOAL))
    pts  = arm_points(q_sol)
    min_cl = min(np.linalg.norm(p - OBS_CENTER) - OBS_RADIUS for p in pts[1:])
    box_ok = np.all(q_sol >= LIM_LO) and np.all(q_sol <= LIM_HI)
    print(f"\n  Solution q* = {np.round(q_sol, 4).tolist()}")
    print(f"  End-effector pos       : {np.round(pos,4).tolist()}")
    print(f"  Distance to goal       : {dist:.4f}  "
          f"({'OK' if dist < TOL else 'FAIL'})")
    print(f"  Box constraints        : {box_ok}")
    print(f"  Min obstacle clearance : {min_cl:.4f}  "
          f"({'OK' if min_cl > 0 else 'VIOLATION'})")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Interior-Point Log-Barrier IK  (Algorithm 8 — 3-DOF)")
    print("=" * 65)
    print(f"  Links        : {LL.tolist()}")
    print(f"  Joint limits : {np.degrees(JOINT_LIMITS[0]).tolist()} deg")
    print(f"  Goal         : {GOAL.tolist()}")
    print(f"  Start Q0     : {Q0.tolist()}")
    print(f"  Obstacle     : center={OBS_CENTER.tolist()}, "
          f"r={OBS_RADIUS}, clearance={OBS_OFFSET}")

    print("\nRunning interior-point solver...")
    res    = run_solver(Q0)
    status = "converged" if res["conv"] else "NOT converged"
    print(f"  {status}  in {res['n_iter']} iterations")
    check_feasibility(res["q_sol"])

    print("\nComputing null-space manifold...")
    configs, thetas, box_ok, obs_ok = compute_null_space()
    feasible = box_ok & obs_ok
    print(f"  Total null-space samples : {len(configs)}")
    print(f"  Box-feasible             : {box_ok.sum()}")
    print(f"  Obstacle-clear           : {obs_ok.sum()}")
    print(f"  Fully feasible           : {feasible.sum()}")

    print("\nGenerating figures...")
    plot_null_space(res, configs, thetas, box_ok, obs_ok)
    plot_history(res)
    plot_arm(res)
    print("\nComputing cost landscape (3 slices, this may take ~30 s)...")
    plot_landscape(res, configs, box_ok, obs_ok)
    print(f"\nAll results saved to {RESULT_DIR}")
