"""
2-DOF Joint-Space Landscape
============================
For a 2-link planar arm, plots the cost function f(q1, q2) as a contour
map in joint space and overlays the convergence trajectory of all 5
unconstrained algorithms from the same starting point.

Run:
    python experiment/landscape_2dof.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path

NEAT_DIR   = Path(__file__).resolve().parent.parent
RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(NEAT_DIR))

from utils.ik_utils import forward_kinematics, jacobian, grad_f, hess_f, objective, backtracking

# ── 2-DOF problem setup ───────────────────────────────────────────────────────

LL   = np.array([2.0, 2.0]) # Link Length
GOAL = np.array([2.5,  1.5]) 
Q0 = np.array([0.0, 0.0]) # single starting point
TOL  = 0.05
MAX_ITER = 2000

# Two analytical solutions (elbow-up / elbow-down) for 2-link IK
_c2   = (GOAL[0]**2 + GOAL[1]**2 - LL[0]**2 - LL[1]**2) / (2*LL[0]*LL[1])
_q2u  = np.arccos(np.clip(_c2, -1, 1))
_q2d  = -_q2u
_k    = np.arctan2(GOAL[1], GOAL[0])
_q1u  = _k - np.arctan2(LL[1]*np.sin(_q2u), LL[0]+LL[1]*np.cos(_q2u))
_q1d  = _k - np.arctan2(LL[1]*np.sin(_q2d), LL[0]+LL[1]*np.cos(_q2d))
SOL_A = np.array([_q1u, _q2u])   # elbow-up solution
SOL_B = np.array([_q1d, _q2d])   # elbow-down solution




# ── inline algorithm runners (accept starting point q0) ──────────────────────

def _f(q): return objective(LL, q, GOAL)


def run_gd(q0, max_iter=MAX_ITER):
    q     = q0.copy().astype(float)
    L_max = LL[0]**2 * sum(k**2 for k in range(1, len(LL)+1))
    alpha = 1.0 / L_max
    rng   = np.random.default_rng(0)
    traj  = [q.copy()]
    for _ in range(max_iter):
        if np.linalg.norm(forward_kinematics(LL, q) - GOAL) < TOL: break
        g = grad_f(LL, q, GOAL)
        if np.linalg.norm(g) < 1e-8:
            q += rng.uniform(-0.3, 0.3, 2)
        else:
            q = q - alpha * g
        traj.append(q.copy())
    return np.array(traj)


def run_sd(q0, max_iter=MAX_ITER):
    q    = q0.copy().astype(float)
    traj = [q.copy()]
    for _ in range(max_iter):
        if np.linalg.norm(forward_kinematics(LL, q) - GOAL) < TOL: break
        g = grad_f(LL, q, GOAL); d = -g
        q = q + backtracking(_f, q, d, g) * d
        traj.append(q.copy())
    return np.array(traj)


def run_newton(q0, max_iter=MAX_ITER):
    q    = q0.copy().astype(float)
    traj = [q.copy()]
    for _ in range(max_iter):
        if np.linalg.norm(forward_kinematics(LL, q) - GOAL) < TOL: break
        g = grad_f(LL, q, GOAL); H = hess_f(LL, q)
        d = np.linalg.solve(H, -g)
        q = q + backtracking(_f, q, d, g) * d
        traj.append(q.copy())
    return np.array(traj)


def run_bfgs(q0, max_iter=MAX_ITER):
    q     = q0.copy().astype(float)
    H_inv = np.eye(2)
    g     = grad_f(LL, q, GOAL)
    traj  = [q.copy()]
    for _ in range(max_iter):
        if np.linalg.norm(forward_kinematics(LL, q) - GOAL) < TOL: break
        d = -(H_inv @ g)
        a = backtracking(_f, q, d, g)
        q_new = q + a * d; g_new = grad_f(LL, q_new, GOAL)
        s, y  = q_new - q, g_new - g
        sy    = float(np.dot(s, y))
        if sy > 1e-10:
            rho = 1.0/sy; A = np.eye(2) - rho*np.outer(s, y)
            H_inv = A @ H_inv @ A.T + rho*np.outer(s, s)
        q, g = q_new, g_new
        traj.append(q.copy())
    return np.array(traj)


def run_cg(q0, max_iter=MAX_ITER):
    q    = q0.copy().astype(float)
    g    = grad_f(LL, q, GOAL); d = -g.copy()
    traj = [q.copy()]
    for k in range(max_iter):
        if np.linalg.norm(forward_kinematics(LL, q) - GOAL) < TOL: break
        q_new = q + backtracking(_f, q, d, g) * d
        g_new = grad_f(LL, q_new, GOAL)
        gn    = float(np.dot(g, g))
        if gn < 1e-16: break
        beta  = max(0.0, float(np.dot(g_new, g_new-g)) / gn)
        if (k+1) % 2 == 0: beta = 0.0
        d = -g_new + beta*d; q, g = q_new, g_new
        traj.append(q.copy())
    return np.array(traj)


# ── run all algorithms from both starting points ──────────────────────────────

METHODS = [
    ("Gradient Descent",   run_gd,     "#e41a1c"),
    ("Steepest Descent",   run_sd,     "#ff7f00"),
    ("Newton's Method",    run_newton, "#4daf4a"),
    ("BFGS",               run_bfgs,   "#377eb8"),
    ("Conjugate Gradient", run_cg,     "#984ea3"),
]

print(f"Elbow-up   solution: q*={np.round(SOL_A,3)}")
print(f"Elbow-down solution: q*={np.round(SOL_B,3)}")
print(f"Start: Q0={Q0.tolist()}")
print("Running algorithms...")
trajs = [(name, run_fn(Q0), color) for name, run_fn, color in METHODS]
for name, traj, _ in trajs:
    conv = np.linalg.norm(forward_kinematics(LL, traj[-1]) - GOAL) < TOL
    print(f"  {name:<22}  {len(traj):>5} iters  {'converged' if conv else 'NOT converged'}")


# ── cost function landscape ───────────────────────────────────────────────────

RES = 300
q1s = np.linspace(-np.pi, np.pi, RES)
q2s = np.linspace(-np.pi, np.pi, RES)
Q1, Q2 = np.meshgrid(q1s, q2s)
F = np.array([[objective(LL, np.array([q1, q2]), GOAL)
               for q1 in q1s] for q2 in q2s])




# ── plot: 2 rows × 3 columns ──────────────────────────────────────────────────
# Row 1: GD | Steepest Descent | Newton
# Row 2: BFGS | Conjugate Gradient | 3-D surface hull

ticks   = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tlabels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
levels  = np.logspace(np.log10(max(F.min(), 1e-4)), np.log10(F.max()), 35)

from matplotlib import gridspec
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.35,   # extra vertical gap between rows
                        wspace=0.32)

# ── 2-D contour panels ───────────────────────────────────────────────────────
subplot_pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

for idx, (name, traj, color) in enumerate(trajs):
    r, c = subplot_pos[idx]
    ax = fig.add_subplot(gs[r, c])

    ax.contourf(Q1, Q2, F, levels=levels, cmap="Greys", alpha=0.9)
    ax.contour(Q1, Q2, F, levels=levels, colors="black", linewidths=0.4, alpha=0.5)

    ax.plot(*SOL_A, "r*", markersize=16, zorder=7)
    ax.plot(*SOL_B, "b*", markersize=16, zorder=7)

    n_pts = len(traj)
    ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=1.8, alpha=1.0, zorder=3)
    sc = ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(n_pts),
                    cmap="plasma", s=15, zorder=4, edgecolors="none", alpha=0.95)
    ax.plot(traj[0,  0], traj[0,  1], "k^", markersize=10, zorder=6)
    ax.plot(traj[-1, 0], traj[-1, 1], "ks", markersize=9,  zorder=6)
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("iteration", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=9)
    ax.set_yticks(ticks); ax.set_yticklabels(tlabels if c == 0 else [], fontsize=9)
    ax.set_xlabel(r"$q_1$", fontsize=12)
    if c == 0:
        ax.set_ylabel(r"$q_2$", fontsize=12)
    ax.set_title(f"{name}\n{n_pts-1} iters", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, color="white")

# ── 3-D surface hull (bottom-right) ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2], projection="3d")

ax3.plot_surface(Q1, Q2, F, cmap="plasma", alpha=0.75,
                 linewidth=0.06, edgecolor="k",
                 antialiased=True, rcount=60, ccount=60)

for name, traj, color in trajs:
    f_traj = np.array([objective(LL, q, GOAL) for q in traj])
    ax3.plot(traj[:, 0], traj[:, 1], f_traj,
             color=color, linewidth=1.6, alpha=0.9, zorder=4)
    ax3.scatter([traj[0,  0]], [traj[0,  1]], [f_traj[0]],
                color=color, s=45, marker="^", zorder=6,
                edgecolors="black", linewidths=0.8)
    ax3.scatter([traj[-1, 0]], [traj[-1, 1]], [f_traj[-1]],
                color=color, s=45, marker="s", zorder=6)

ax3.scatter([SOL_A[0]], [SOL_A[1]], [objective(LL, SOL_A, GOAL)],
            color="red",  s=100, marker="*", zorder=7)
ax3.scatter([SOL_B[0]], [SOL_B[1]], [objective(LL, SOL_B, GOAL)],
            color="blue", s=100, marker="*", zorder=7)

ax3.set_xlim(-np.pi, np.pi); ax3.set_ylim(-np.pi, np.pi)
ax3.set_xlabel(r"$q_1$", fontsize=10, labelpad=2)
ax3.set_ylabel(r"$q_2$", fontsize=10, labelpad=2)
ax3.set_zlabel(r"$f$",   fontsize=10, labelpad=2)
ax3.set_xticks([-np.pi, 0, np.pi])
ax3.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=8)
ax3.set_yticks([-np.pi, 0, np.pi])
ax3.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=8)
ax3.tick_params(labelsize=8, pad=1)
ax3.view_init(elev=32, azim=-50)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.set_title("Cost Surface\n(all trajectories)", fontsize=11, fontweight="bold")

handles = [Line2D([0], [0], color=c, linewidth=2.5, label=n)
           for n, _, c in trajs]
ax3.legend(handles=handles, fontsize=9, loc="upper right",
           framealpha=0.7, borderpad=0.5)

fig.suptitle(
    f"2-DOF IK Landscape  |  Links {LL.tolist()}  |  "
    f"Goal [{GOAL[0]:.1f},{GOAL[1]:.1f}]  |  "
    r"Red $\bigstar$ elbow-up  ·  Blue $\bigstar$ elbow-down  |  "
    r"$\blacktriangle$ start  $\blacksquare$ end",
    fontsize=11, y=0.99
)
out = RESULT_DIR / "landscape_2dof.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
print("Showing interactive window — rotate 3D surface by dragging, close window to exit.")
plt.show()
