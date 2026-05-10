# IK Optimization — N-Link Planar Arm

Inverse kinematics for a planar robotic arm solved with eight optimization algorithms, from basic gradient descent to constrained interior-point methods.

---

## Requirements

```bash
pip install numpy matplotlib
```

---

## 1. Basic Run — Interactive Animation

Each numbered script opens an animation window. **Click anywhere** in the workspace to set a goal; the arm solves IK and moves to that position.

| Script | Method | Constraints |
|---|---|---|
| `1_gradient_descent.py` | Gradient Descent (fixed step) | None |
| `2_steepest_descent.py` | Steepest Descent (backtracking) | None |
| `3_newtons_method.py` | Newton / Gauss-Newton | None |
| `4_quasi_newton_bfgs.py` | BFGS | None |
| `5_conjugate_gradient.py` | Conjugate Gradient | None |
| `6_newton_kkt.py` | Newton KKT (equality) | Joint equality constraint |
| `7_infeasible_newton.py` | Infeasible-start Newton | Equality constraint |
| `8_interior_point.py` | Interior Point (log-barrier) | Joint limits + obstacle |

```bash
python 1_gradient_descent.py   # or any other numbered script
```

Click on the plot to set the goal position. The arm animates toward the solution.

---

## 2. Tuning Parameters

Each script has a block of constants near the top. Edit them directly before running.

### Arm geometry — `utils/ik_utils.py`

```python
N_LINKS       = 3                  # number of links
LINK_LENGTHS  = [2.0, 2.0, 2.0]   # length of each link (metres)
JOINT_LIMITS  = [[-π, π], ...]     # per-joint [min, max] in radians
TOL           = 0.1                # end-effector goal tolerance (metres)
MAX_ITER      = 10000              # maximum solver iterations
```

### Algorithm step sizes

**Gradient Descent** (`1_gradient_descent.py`):
```python
ALPHA = 0.0005   # fixed step size — reduce if diverging, increase if too slow
```

**Steepest / Newton / BFGS / CG** use backtracking line search automatically; no manual step size needed.

### Interior Point (`8_interior_point.py`)

```python
JOINT_LIMITS  = [[-π/2, π/2], ...]   # tighter limits for constrained demo
OBS_CENTER    = [1.5, 1.0]           # obstacle position (x, y)
OBS_RADIUS    = 0.5                  # obstacle radius (metres)
OBS_OFFSET    = 0.5                  # minimum clearance from each joint to obstacle
MU_INIT       = 1.0                  # initial barrier weight
MU_SCALE      = 0.1                  # multiply μ each outer iteration
MU_MIN        = 1e-10                # stop reducing μ below this
OUTER_MAX     = 60                   # maximum outer (barrier) iterations
```

---

## 3. Running Experiments

Experiment scripts are in the `experiment/` folder. They generate plots saved as `.png` files — no interactive window.

| Script | Output |
|---|---|
| `experiment/unc_landscape_2dof.py` | Cost landscape for unconstrained 2-DOF IK |
| `experiment/unc_eff_compare.py` | Convergence comparison across unconstrained methods |
| `experiment/unc_param_count.py` | Iteration count vs. link count |
| `experiment/con_eq_2dof.py` | KKT / equality-constrained solver landscape |
| `experiment/con_eq_success.py` | Success-rate sweep for equality-constrained methods |
| `experiment/interior_point_2dof.py` | Interior-point cost landscape (2-DOF) |
| `experiment/interior_point_3dof.py` | Interior-point null-space and cost landscape (3-DOF) |

```bash
python experiment/interior_point_3dof.py
```

Output images are written to the `experiment/` directory (e.g. `ip3_nullspace.png`, `ip3_landscape.png`).
