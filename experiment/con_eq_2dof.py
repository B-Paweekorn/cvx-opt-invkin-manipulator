"""
Equality-Constrained IK — 2-DOF (Methods 6b & 7b)
===================================================
Run from the NEAT/ directory:
    python experiment/con_eq_2dof.py

Setup
-----
  2-link planar arm  :  L1 = L2 = 2.0
  Joint limits       :  q1, q2 in [-pi/2, pi/2]   (box, inequality)
  Equality           :  h(q) = q1^2 + q2^2 - 1 = 0  (unit circle in joint space)

The feasible set is the arc of the unit circle inside the box.
Depending on the goal, the IK solution may or may not lie on this arc.

Edit GOAL and Q0 at the top to explore different scenarios.

Output
------
  con_eq_manifold.png   : joint-space feasible arc + task-space arm postures
  con_eq_landscape.png  : cost-function contour + trajectories (one panel per method)
  con_eq_history.png    : convergence distance & equality violation vs iteration
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

from utils.ik_utils import (forward_kinematics, objective,
                            grad_f, hess_f)

# GOAL	q* on circle	Notes
# [2.161, 3.366]	[1.000, 0.000]	q1-axis
# [1.702, 3.482]	[0.866, 0.500]	upper-right arc
# [2.162, 2.917]	[0.500, 0.866]	near q2=π/2
# [3.081, 1.683]	[0.000, 1.000]	q2-axis
# [3.623, -0.243]	[-0.500, 0.866]	
# [3.163, -2.239]	[-0.866, 0.500]	
# [2.161, -3.366]	[-1.000, 0.000]	negative q1-axis
# [3.080, -1.683]	[0.000, -1.000]	negative q2-axis
# [3.623, 0.242]	[0.500, -0.866]	lower-right arc

# ── problem setup ─────────────────────────────────────────────────────────────

LL           = np.array([2.0, 2.0])
JOINT_LIMITS = np.array([[-np.pi / 2, np.pi / 2],
                          [-np.pi / 2, np.pi / 2]])
TOL      = 0.05
MAX_ITER = 2000

# ▼▼▼  edit freely  ▼▼▼

GOAL = np.array([3.623, -0.243]) # x y 
Q0   = np.array([-0.3, -0.1])    # starting joint angles (can be off the circle)
# ▲▲▲──────────────────▲▲▲

LIM_LO = JOINT_LIMITS[:, 0]
LIM_HI = JOINT_LIMITS[:, 1]

# ── equality constraint ───────────────────────────────────────────────────────

def _h(q):      return float(q[0]**2 + q[1]**2 - 1.0)
def _grad_h(q): return np.array([2.0*q[0], 2.0*q[1]])
def _hess_h():  return np.diag([2.0, 2.0])
def _dist(q):   return float(np.linalg.norm(forward_kinematics(LL, q) - GOAL))
def _fobj(q):   return objective(LL, q, GOAL)

MU = 10.0   # merit penalty for |h|

def _merit(q, nu): return _fobj(q) + MU * abs(_h(q))

def _merit_bt(q, nu, dq):
    alpha = 1.0
    phi0  = _merit(q, nu)
    g_obj = grad_f(LL, q, GOAL)
    slope = float(np.dot(g_obj + nu*_grad_h(q), dq)) - MU*abs(_h(q))
    slope = min(slope, -1e-12)
    for _ in range(40):
        if _merit(q + alpha*dq, nu) <= phi0 + 1e-4*alpha*slope:
            return alpha
        alpha *= 0.5
    return alpha


# ── load solver modules ───────────────────────────────────────────────────────

def _load(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m6b = _load("m6b", NEAT_DIR / "6_newton_kkt.py")
m7b = _load("m7b", NEAT_DIR / "7_infeasible_newton.py")

METHODS = [
    ("6b Newton+KKT+eq",      "#377eb8",
     lambda q0: m6b.solve(LL.tolist(), q0, GOAL,
                          joint_limits=JOINT_LIMITS, tol=TOL, max_iter=MAX_ITER)),
    ("7b Infeasible+eq",       "#e41a1c",
     lambda q0: m7b.solve(LL.tolist(), q0, GOAL,
                          joint_limits=JOINT_LIMITS, tol=TOL, max_iter=MAX_ITER)),
]


# ── inline trajectory-recording versions ─────────────────────────────────────

# --- 6b trajectory runner ---

def _active_set(q, tol=1e-4):
    act = []
    for i in range(len(q)):
        if abs(q[i] - LIM_LO[i]) < tol: act.append((i, 'lo'))
        elif abs(q[i] - LIM_HI[i]) < tol: act.append((i, 'hi'))
    return act

def _con_matrix(n, act):
    if not act: return np.zeros((0, n))
    A = np.zeros((len(act), n))
    for row, (i, side) in enumerate(act):
        A[row, i] = 1.0 if side == 'hi' else -1.0
    return A

def _kkt_eq_step(H_aug, g_res, A_act, c, hv):
    """Solve (n+1+m)×(n+1+m) KKT with equality row."""
    n = H_aug.shape[0]; m = A_act.shape[0]; sz = n + 1 + m
    M = np.zeros((sz, sz)); rhs = np.zeros(sz)
    M[:n, :n] = H_aug
    M[:n,  n] = c;  M[n, :n] = c
    if m > 0:
        M[:n, n+1:] = A_act.T
        M[n+1:, :n] = A_act
    rhs[:n] = -g_res;  rhs[n] = -hv
    try:    sol = np.linalg.solve(M, rhs)
    except: sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return sol[:n], float(sol[n]), sol[n+1:n+1+m]

def run_kkt_eq_traj(q0):
    """6b: project q0 to unit circle then box, record joint-space trajectory."""
    # project q0 onto unit circle
    r = np.hypot(q0[0], q0[1])
    if r < 1e-12: r = 1e-12
    q = np.clip(np.array([q0[0]/r, q0[1]/r]), LIM_LO, LIM_HI)
    Hh = _hess_h()
    nu  = 0.0; lam_act = np.zeros(0); act = []
    q_traj = [q0.copy(), q.copy()]
    h_hist = [_h(q0), _h(q)]

    for _ in range(MAX_ITER):
        if _dist(q) < TOL and abs(_h(q)) < TOL: break
        g   = grad_f(LL, q, GOAL)
        H   = hess_f(LL, q)
        c   = _grad_h(q);  hv = _h(q)
        act = _active_set(q)
        A   = _con_matrix(2, act)
        g_res = g + nu*c + (A.T @ lam_act if len(lam_act) else np.zeros(2))
        dq, dnu, dlam = _kkt_eq_step(H + nu*Hh, g_res, A, c, hv)
        for idx in range(len(dlam)-1, -1, -1):
            if (lam_act[idx] if idx < len(lam_act) else 0) + dlam[idx] < -1e-8:
                act.pop(idx); A = _con_matrix(2, act)
                g_res = g + nu*c + (A.T @ lam_act[:len(act)] if act else np.zeros(2))
                dq, dnu, dlam = _kkt_eq_step(H + nu*Hh, g_res, A, c, hv)
        alpha   = _merit_bt(q, nu, dq)
        q       = np.clip(q + alpha*dq, LIM_LO, LIM_HI)
        nu     += dnu;  lam_act = dlam
        q_traj.append(q.copy());  h_hist.append(_h(q))

    return np.array(q_traj), h_hist

# --- 7b trajectory runner ---

_TAU0 = 1.0; _TAU_SC = 0.1; _OUTER = 60; _INNER = 10; _FRAC = 0.995

def _ftb(s, ds):
    alpha = 1.0; neg = ds < 0
    if np.any(neg): alpha = min(alpha, _FRAC * np.min(-s[neg]/ds[neg]))
    return max(alpha, 1e-12)

def run_infeas_eq_traj(q0):
    """7b: clamp q0 to strict interior, record joint-space trajectory + h history."""
    eps    = 0.01 * (LIM_HI - LIM_LO)
    q      = np.clip(q0.copy().astype(float), LIM_LO + eps, LIM_HI - eps)
    tau    = _TAU0
    s_lo   = q - LIM_LO;  s_hi = LIM_HI - q
    lam_lo = tau / s_lo;  lam_hi = tau / s_hi
    nu     = 0.0
    Hh     = _hess_h()
    q_traj = [q0.copy(), q.copy()]
    h_hist = [_h(q0), _h(q)]

    for _ in range(_OUTER):
        for _ in range(_INNER):
            if len(q_traj) >= MAX_ITER: break
            if _dist(q) < TOL and abs(_h(q)) < TOL and tau < 1e-6: break
            g    = grad_f(LL, q, GOAL)
            H    = hess_f(LL, q, reg=1e-8)
            c    = _grad_h(q);  hv = _h(q)
            s_lo = q - LIM_LO;  s_hi = LIM_HI - q
            D_lo = lam_lo/s_lo; D_hi = lam_hi/s_hi
            r_q  = g + nu*c - lam_lo + lam_hi
            b_q  = r_q - D_lo*(tau/lam_lo - s_lo) + D_hi*(tau/lam_hi - s_hi)
            H_bar = H + nu*Hh + np.diag(D_lo + D_hi)
            M = np.array([[H_bar[0,0], H_bar[0,1], c[0]],
                          [H_bar[1,0], H_bar[1,1], c[1]],
                          [c[0],       c[1],        0.0]])
            rhs = np.array([-b_q[0], -b_q[1], -hv])
            try:    sol = np.linalg.solve(M, rhs)
            except: sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
            dq  = sol[:2];  dnu = float(sol[2])
            r_lo = lam_lo*s_lo - tau;  r_hi = lam_hi*s_hi - tau
            dlam_lo = -(r_lo + lam_lo*dq) / s_lo
            dlam_hi = -(r_hi - lam_hi*dq) / s_hi
            ap = min(_ftb(s_lo, dq), _ftb(s_hi, -dq))
            ad = min(_ftb(lam_lo, dlam_lo), _ftb(lam_hi, dlam_hi))
            q      = np.clip(q + ap*dq, LIM_LO, LIM_HI)
            lam_lo = np.maximum(lam_lo + ad*dlam_lo, 1e-12)
            lam_hi = np.maximum(lam_hi + ad*dlam_hi, 1e-12)
            nu    += dnu
            q_traj.append(q.copy());  h_hist.append(_h(q))
        tau = max(tau * _TAU_SC, 1e-10)

    return np.array(q_traj), h_hist


# ── helpers ───────────────────────────────────────────────────────────────────

def _unit_circle_arc():
    """Sample the unit circle arc that lies inside the box [-pi/2,pi/2]^2."""
    thetas = np.linspace(0, 2*np.pi, 4000)
    qs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    mask = np.all((qs >= LIM_LO) & (qs <= LIM_HI), axis=1)
    return qs[mask]

def _arc_task_image(arc_qs):
    return np.array([forward_kinematics(LL, q) for q in arc_qs])

def _draw_arm(ax, q, color, label, ls="-", alpha=1.0, zorder=4):
    pts = [np.zeros(2)]
    for i in range(len(LL)):
        cum = np.sum(q[:i+1])
        pts.append(pts[-1] + LL[i] * np.array([np.cos(cum), np.sin(cum)]))
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], marker="o", linestyle=ls, color=color,
            lw=2.5, markersize=7, label=label, zorder=zorder, alpha=alpha)
    ax.plot(*pts[-1], "s", color=color, markersize=10, zorder=zorder+1, alpha=alpha)


# ── run solvers (use module solve()) ──────────────────────────────────────────

def run_solvers():
    outcomes = []
    for name, color, solver in METHODS:
        q_sol, conv, n_iter, history, elapsed = solver(Q0.copy())
        pos  = forward_kinematics(LL, q_sol)
        outcomes.append({
            "name":      name,
            "color":     color,
            "q_sol":     q_sol,
            "pos_sol":   pos,
            "conv":      conv,
            "n_iter":    n_iter,
            "history":   history,
            "final_err": float(np.linalg.norm(pos - GOAL)),
            "h_final":   _h(q_sol),
        })
        status = "converged" if conv else "NOT converged"
        print(f"  {name:<24}  {n_iter:>5} iters  "
              f"dist={outcomes[-1]['final_err']:.4f}  "
              f"h(q)={_h(q_sol):+.4f}  "
              f"q*=[{q_sol[0]:.3f},{q_sol[1]:.3f}]  {status}")
    return outcomes


# ── feasibility report ────────────────────────────────────────────────────────

def check_feasibility():
    c2  = (GOAL[0]**2 + GOAL[1]**2 - LL[0]**2 - LL[1]**2) / (2*LL[0]*LL[1])
    c2  = np.clip(c2, -1, 1)
    q2u = np.arccos(c2);  q2d = -q2u
    k   = np.arctan2(GOAL[1], GOAL[0])
    q1u = k - np.arctan2(LL[1]*np.sin(q2u), LL[0]+LL[1]*np.cos(q2u))
    q1d = k - np.arctan2(LL[1]*np.sin(q2d), LL[0]+LL[1]*np.cos(q2d))
    print(f"\n  Unconstrained IK solutions for goal {GOAL.tolist()}:")
    for lbl, q1, q2 in [("elbow-up  ", q1u, q2u), ("elbow-down", q1d, q2d)]:
        box_ok = all(LIM_LO[i] <= v <= LIM_HI[i] for i, v in enumerate([q1, q2]))
        heq    = q1**2 + q2**2 - 1.0
        print(f"    {lbl}  q=[{q1:.3f},{q2:.3f}]  box_ok={box_ok}"
              f"  h(q)={heq:+.4f}  {'ON circle' if abs(heq)<0.01 else 'off circle'}")
    dist  = float(np.linalg.norm(GOAL))
    geo   = "within" if dist <= sum(LL) else "OUTSIDE"
    print(f"  Goal dist={dist:.3f}  (max reach={sum(LL):.1f}) -> {geo} full reach")
    arc   = _unit_circle_arc()
    task  = _arc_task_image(arc)
    best_d = float(np.min(np.linalg.norm(task - GOAL, axis=1)))
    print(f"  Closest point on arc to goal: dist={best_d:.4f}"
          f"  ({'REACHABLE on circle' if best_d < TOL else 'not reachable on circle'})")


# ── Figure 1 — Manifold: joint space + task space ────────────────────────────

def plot_manifold(outcomes):
    FS_TITLE = 13; FS_LABEL = 12; FS_TICK = 11; FS_LEG = 10

    arc_q  = _unit_circle_arc()
    arc_p  = _arc_task_image(arc_q)
    ws_q1s = np.linspace(*JOINT_LIMITS[0], 200)
    ws_q2s = np.linspace(*JOINT_LIMITS[1], 200)
    ws_pts = np.array([forward_kinematics(LL, np.array([q1, q2]))
                       for q1 in ws_q1s for q2 in ws_q2s])
    tks = [LIM_LO[0], 0.0, LIM_HI[0]]
    theta_full = np.linspace(0, 2*np.pi, 500)

    fig, (ax_j, ax_t) = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(wspace=0.05)

    # ── joint space ──────────────────────────────────────────────────────────
    ax_j.add_patch(mpatches.FancyBboxPatch(
        (LIM_LO[0], LIM_LO[1]), LIM_HI[0]-LIM_LO[0], LIM_HI[1]-LIM_LO[1],
        boxstyle="square,pad=0", lw=2, edgecolor="black",
        facecolor="lightyellow", alpha=0.35))
    ax_j.plot(np.cos(theta_full), np.sin(theta_full), "b--", lw=1.2, alpha=0.4,
              label="unit circle")
    ax_j.scatter(arc_q[:, 0], arc_q[:, 1], s=6, c="royalblue",
                 label="feasible arc  $h(q)=0$", zorder=4)
    for o in outcomes:
        tag = o['name'].split('+')[0].strip()
        ax_j.scatter(*o["q_sol"], s=220, color=o["color"], marker="*",
                     zorder=7, edgecolors="black", lw=0.8,
                     label=f"{tag}  q*=[{o['q_sol'][0]:.2f}, {o['q_sol'][1]:.2f}]")
    ax_j.scatter(*Q0, s=120, color="gray", marker="^", zorder=6,
                 label=f"start  Q0={Q0.tolist()}")
    ax_j.set_xlim(LIM_LO[0]-0.4, LIM_HI[0]+0.4)
    ax_j.set_ylim(LIM_LO[1]-0.4, LIM_HI[1]+0.4)
    ax_j.set_xticks(tks); ax_j.set_xticklabels([r"$-\pi/2$",r"$0$",r"$\pi/2$"], fontsize=FS_TICK)
    ax_j.set_yticks(tks); ax_j.set_yticklabels([r"$-\pi/2$",r"$0$",r"$\pi/2$"], fontsize=FS_TICK)
    ax_j.set_xlabel(r"$q_1$ (rad)", fontsize=FS_LABEL)
    ax_j.set_ylabel(r"$q_2$ (rad)", fontsize=FS_LABEL)
    ax_j.set_aspect("equal")
    ax_j.legend(fontsize=FS_LEG, loc="lower left", framealpha=0.9)
    ax_j.grid(True, alpha=0.25)
    ax_j.set_title("Joint Space — Feasible Arc", fontsize=FS_TITLE, fontweight="bold")

    # ── task space ───────────────────────────────────────────────────────────
    ax_t.scatter(ws_pts[:, 0], ws_pts[:, 1],
                 s=1, c="lightgray", alpha=0.4, rasterized=True, label="workspace")
    ax_t.scatter(arc_p[:, 0], arc_p[:, 1],
                 s=6, c="royalblue", zorder=4, label="image of feasible arc")
    ax_t.plot(*GOAL, "rx", ms=16, mew=3, zorder=10, label=f"goal  {GOAL.tolist()}")
    _draw_arm(ax_t, Q0, "dimgray", "start (dashed)")
    ax_t.get_lines()[-1].set_linestyle("--"); ax_t.get_lines()[-1].set_alpha(0.65)
    for o in outcomes:
        tag = o['name'].split('+')[0].strip()
        _draw_arm(ax_t, o["q_sol"], o["color"], f"{tag}  dist={o['final_err']:.3f}")
    ax_t.plot(0, 0, "ko", ms=8, zorder=11, label="base")
    _xc = (ws_pts[:, 0].min() + ws_pts[:, 0].max()) / 2
    _yc = (ws_pts[:, 1].min() + ws_pts[:, 1].max()) / 2
    _half = max(ws_pts[:, 0].max()-ws_pts[:, 0].min(),
                ws_pts[:, 1].max()-ws_pts[:, 1].min()) / 2 + 0.6
    ax_t.set_xlim(_xc-_half, _xc+_half); ax_t.set_ylim(_yc-_half, _yc+_half)
    ax_t.set_aspect("equal")
    ax_t.legend(fontsize=FS_LEG, loc="upper right", framealpha=0.9)
    ax_t.set_xlabel("X", fontsize=FS_LABEL); ax_t.set_ylabel("Y", fontsize=FS_LABEL)
    ax_t.tick_params(labelsize=FS_TICK)
    ax_t.grid(True, alpha=0.25)
    ax_t.set_title("Task Space — Arm Postures", fontsize=FS_TITLE, fontweight="bold")

    fig.suptitle(
        fr"$h(q)=q_1^2+q_2^2-1=0$  |  Links {LL.tolist()}  |  Goal {GOAL.tolist()}",
        fontsize=FS_TITLE, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULT_DIR / "con_eq_manifold.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Figure 2 (landscape) — Cost landscape + trajectories ─────────────────────

def plot_landscape():
    FS_TITLE = 13; FS_LABEL = 12; FS_TICK = 11; FS_LEG = 10

    MARGIN = 0.5; RES = 250
    q1s = np.linspace(LIM_LO[0]-MARGIN, LIM_HI[0]+MARGIN, RES)
    q2s = np.linspace(LIM_LO[1]-MARGIN, LIM_HI[1]+MARGIN, RES)
    Q1g, Q2g = np.meshgrid(q1s, q2s)
    F = np.array([[objective(LL, np.array([q1, q2]), GOAL)
                   for q1 in q1s] for q2 in q2s])
    levels  = np.logspace(np.log10(max(F.min(), 1e-4)), np.log10(F.max()), 35)
    tks     = [LIM_LO[0], 0.0, LIM_HI[0]]
    tlabels = [r"$-\pi/2$", r"$0$", r"$\pi/2$"]

    _c2  = np.clip((GOAL[0]**2+GOAL[1]**2-LL[0]**2-LL[1]**2)/(2*LL[0]*LL[1]),-1,1)
    _q2u = np.arccos(_c2);  _q2d = -_q2u
    _k   = np.arctan2(GOAL[1], GOAL[0])
    _q1u = _k - np.arctan2(LL[1]*np.sin(_q2u), LL[0]+LL[1]*np.cos(_q2u))
    _q1d = _k - np.arctan2(LL[1]*np.sin(_q2d), LL[0]+LL[1]*np.cos(_q2d))

    arc_q  = _unit_circle_arc()
    runners = [
        ("6b  Newton+KKT+eq", "#377eb8", run_kkt_eq_traj),
        ("7b  Infeasible+eq",  "#e41a1c", run_infeas_eq_traj),
    ]
    theta_full = np.linspace(0, 2*np.pi, 400)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for ax, (mname, mcolor, runner) in zip(axes, runners):
        ax.contourf(Q1g, Q2g, F, levels=levels, cmap="Greys", alpha=0.85)
        ax.contour(Q1g, Q2g, F, levels=levels, colors="black",
                   linewidths=0.35, alpha=0.45)
        ax.fill_between([LIM_LO[0], LIM_HI[0]], LIM_LO[1], LIM_HI[1],
                        color="lightyellow", alpha=0.28, zorder=1)
        bx = [LIM_LO[0], LIM_HI[0], LIM_HI[0], LIM_LO[0], LIM_LO[0]]
        by = [LIM_LO[1], LIM_LO[1], LIM_HI[1], LIM_HI[1], LIM_LO[1]]
        ax.plot(bx, by, "k-", lw=2, label="box limits", zorder=5)
        ax.plot(np.cos(theta_full), np.sin(theta_full), "b--", lw=1.2, alpha=0.45,
                label="unit circle", zorder=4)
        ax.scatter(arc_q[:, 0], arc_q[:, 1], s=6, c="royalblue",
                   zorder=5, label="feasible arc")
        ax.scatter(*[_q1u, _q2u], s=260, marker="*", color="darkblue", zorder=9,
                   label="analytic solution")
        ax.scatter(*[_q1d, _q2d], s=260, marker="*", color="darkred", zorder=9)

        traj, _ = runner(Q0)
        n = len(traj)
        ax.plot(traj[:, 0], traj[:, 1], color=mcolor, lw=1.8, alpha=0.80, zorder=6)
        sc = ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(n),
                        cmap="plasma", s=25, zorder=7, edgecolors="none", alpha=0.95)
        ax.scatter(*traj[0],  s=180, marker="^", color=mcolor,
                   edgecolors="black", lw=0.9, zorder=8, label="start")
        ax.scatter(*traj[-1], s=160, marker="s", color=mcolor,
                   edgecolors="black", lw=0.9, zorder=8, label=f"end  ({n-1} iters)")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1%", pad=0.0)
        fig.colorbar(sc, cax=cax, label="iteration")

        ax.set_xlim(LIM_LO[0]-MARGIN, LIM_HI[0]+MARGIN)
        ax.set_ylim(LIM_LO[1]-MARGIN, LIM_HI[1]+MARGIN)
        ax.set_xticks(tks); ax.set_xticklabels(tlabels, fontsize=FS_TICK)
        ax.set_yticks(tks); ax.set_yticklabels(tlabels, fontsize=FS_TICK)
        ax.set_xlabel(r"$q_1$ (rad)", fontsize=FS_LABEL)
        ax.set_ylabel(r"$q_2$ (rad)", fontsize=FS_LABEL)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2, color="white")
        ax.set_title(f"Landscape — {mname}", fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEG, loc="upper right", framealpha=0.9)

    fig.suptitle(
        fr"Cost Landscape  |  $h(q)=q_1^2+q_2^2-1=0$  "
        fr"|  Links {LL.tolist()}  |  Goal {GOAL.tolist()}",
        fontsize=FS_TITLE, fontweight="bold")
    out = RESULT_DIR / "con_eq_landscape.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Figure 2 — Convergence history (dist + h violation) ──────────────────────

def plot_history(outcomes):
    fig, axes = plt.subplots(2, len(outcomes), figsize=(14, 8), sharex="col")

    for col, o in enumerate(outcomes):
        ax_d = axes[0, col]
        ax_h = axes[1, col]
        color = o["color"]

        ax_d.semilogy(o["history"], color=color, lw=2)
        ax_d.axhline(TOL, color="black", ls="--", lw=1.2, label=f"tol={TOL}")
        ax_d.axhline(o["final_err"], color=color, ls=":", lw=1.2, alpha=0.7,
                     label=f"final dist={o['final_err']:.4f}")
        ax_d.set_title(o["name"].split(" ",1)[1], fontsize=12, fontweight="bold")
        ax_d.set_ylabel("Distance to goal", fontsize=10)
        ax_d.legend(fontsize=8); ax_d.grid(True, which="both", alpha=0.3)

        # re-run trajectory-recording to get h history
        if "6b" in o["name"]:
            _, h_hist = run_kkt_eq_traj(Q0)
        else:
            _, h_hist = run_infeas_eq_traj(Q0)
        ax_h.semilogy(np.abs(h_hist), color=color, lw=2)
        ax_h.axhline(TOL, color="black", ls="--", lw=1.2, label="tol")
        ax_h.set_ylabel(r"$|h(q)|$  equality violation", fontsize=10)
        ax_h.set_xlabel("Iteration", fontsize=11)
        ax_h.legend(fontsize=8); ax_h.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Convergence — Distance to Goal & Equality Violation\n"
        f"goal={GOAL.tolist()}  Q0={Q0.tolist()}",
        fontsize=13)
    plt.tight_layout()
    out = RESULT_DIR / "con_eq_history.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── Figure 3 — Cost landscape + unit circle + trajectories ───────────────────

def plot_landscape():
    MARGIN = 0.5
    RES    = 250
    q1s = np.linspace(LIM_LO[0] - MARGIN, LIM_HI[0] + MARGIN, RES)
    q2s = np.linspace(LIM_LO[1] - MARGIN, LIM_HI[1] + MARGIN, RES)
    Q1g, Q2g = np.meshgrid(q1s, q2s)
    F = np.array([[objective(LL, np.array([q1, q2]), GOAL)
                   for q1 in q1s] for q2 in q2s])
    levels  = np.logspace(np.log10(max(F.min(), 1e-4)), np.log10(F.max()), 35)
    ticks   = [LIM_LO[0], 0.0, LIM_HI[0]]
    tlabels = [r"$-\pi/2$", r"$0$", r"$\pi/2$"]

    # unconstrained IK solutions
    _c2  = np.clip((GOAL[0]**2+GOAL[1]**2-LL[0]**2-LL[1]**2)/(2*LL[0]*LL[1]),-1,1)
    _q2u = np.arccos(_c2);  _q2d = -_q2u
    _k   = np.arctan2(GOAL[1], GOAL[0])
    _q1u = _k - np.arctan2(LL[1]*np.sin(_q2u), LL[0]+LL[1]*np.cos(_q2u))
    _q1d = _k - np.arctan2(LL[1]*np.sin(_q2d), LL[0]+LL[1]*np.cos(_q2d))

    # unit circle arc inside box
    arc_q = _unit_circle_arc()

    runners = [
        ("6b  Newton+KKT+eq",     "#377eb8", run_kkt_eq_traj),
        ("7b  Infeasible+eq",      "#e41a1c", run_infeas_eq_traj),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (mname, mcolor, runner) in zip(axes, runners):
        cf = ax.contourf(Q1g, Q2g, F, levels=levels, cmap="Greys", alpha=0.85)
        ax.contour(Q1g, Q2g, F, levels=levels, colors="black",
                   linewidths=0.35, alpha=0.45)
        # plt.colorbar(cf, ax=ax, label=r"$f(q_1,q_2)$", fraction=0.046, pad=0.02)

        # box
        ax.fill_between([LIM_LO[0], LIM_HI[0]], LIM_LO[1], LIM_HI[1],
                        color="lightyellow", alpha=0.28, zorder=1)
        bx = [LIM_LO[0], LIM_HI[0], LIM_HI[0], LIM_LO[0], LIM_LO[0]]
        by = [LIM_LO[1], LIM_LO[1], LIM_HI[1], LIM_HI[1], LIM_LO[1]]
        ax.plot(bx, by, "k-", lw=2, label="box limits", zorder=5)

        # full unit circle
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), "b--", lw=1.2, alpha=0.45,
                label="unit circle", zorder=4)
        # feasible arc (thicker)
        ax.scatter(arc_q[:, 0], arc_q[:, 1], s=6, c="royalblue",
                   zorder=5, label="feasible arc $h(q)=0$")

        # unconstrained solutions
        for qs, ec, lbl in [([_q1u, _q2u], "darkblue", "Global minimum (constrained)")]:
            ax.scatter(*qs, s=260, marker="*", color=ec, zorder=9,
                       label=lbl if lbl else "_")

        # trajectory
        traj, _ = runner(Q0)
        n = len(traj)
        ax.plot(traj[:, 0], traj[:, 1], color=mcolor, lw=1.8, alpha=0.80, zorder=6)
        sc = ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(n),
                        cmap="plasma", s=20, zorder=7, edgecolors="none", alpha=0.95)
        ax.scatter(*traj[0],  s=160, marker="^", color=mcolor,
                   edgecolors="black", lw=0.9, zorder=8,
                   label=f"start Q0={np.round(traj[0],2).tolist()}")
        ax.scatter(*traj[-1], s=140, marker="s", color=mcolor,
                   edgecolors="black", lw=0.9, zorder=8,
                   label=f"end ({n-1} iters)")
        plt.colorbar(sc, ax=ax, label="iteration", fraction=0.033, pad=0.08)

        ax.set_xlim(LIM_LO[0]-MARGIN, LIM_HI[0]+MARGIN)
        ax.set_ylim(LIM_LO[1]-MARGIN, LIM_HI[1]+MARGIN)
        ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=11)
        ax.set_yticks(ticks); ax.set_yticklabels(tlabels, fontsize=11)
        ax.set_xlabel(r"$q_1$ (rad)", fontsize=13)
        ax.set_ylabel(r"$q_2$ (rad)", fontsize=13)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, color="white")
        ax.set_title(mname, fontsize=14, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    fig.suptitle(
        fr"Cost Landscape  |  $h(q)=q_1^2+q_2^2-1=0$  "
        fr"|  Links {LL.tolist()}  |  Goal {GOAL.tolist()}"
        "\n"
        r"Yellow = box $[-\pi/2,\pi/2]^2$  |  "
        r"Blue arc = feasible set  |  "
        r"$\bigstar$ = unconstrained solutions  |  "
        r"$\blacktriangle$ start  $\blacksquare$ end",
        fontsize=11)
    plt.tight_layout()
    out = RESULT_DIR / "con_eq_landscape.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  2-DOF Equality-Constrained IK  (Methods 6b & 7b)")
    print("=" * 60)
    print(f"  Links        : {LL.tolist()}")
    print(f"  Joint limits : {np.degrees(JOINT_LIMITS[0]).tolist()} deg")
    print(f"  Equality     : h(q) = q1^2 + q2^2 - 1 = 0")
    print(f"  Start Q0     : {Q0.tolist()}")
    print(f"  Goal         : {GOAL.tolist()}")

    check_feasibility()

    print("\nRunning solvers...")
    outcomes = run_solvers()

    print("\nGenerating plots...")
    plot_manifold(outcomes)
    plot_history(outcomes)
    print("Building cost-function landscape (may take ~30s)...")
    plot_landscape()
    print(f"\nAll results saved to {RESULT_DIR}")
