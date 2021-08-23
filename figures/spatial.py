#!/usr/bin/env python
#%%
from pyscf import gto, scf
from pyscf.geomopt.berny_solver import optimize
import numpy as np
import pandas as pd
import mpmath
import sys
import matplotlib.pyplot as plt
import functools

mpmath.dps = 1000

sys.path.insert(0, ".")
import convergence

#%%
@functools.lru_cache(maxsize=100)
def get_dmin(basis):
    mol = gto.M(atom="H 0 0 0; H 0 0 1.2", basis=basis)
    mf = scf.RHF(mol)

    mol_eq = optimize(mf, maxsteps=100)
    return np.linalg.norm(mol_eq.atom_coords()[0] - mol_eq.atom_coords()[1])


@functools.lru_cache(maxsize=10)
def get_ref_curve(basis):
    xss = np.linspace(1.1, 1.6, 20)
    yss = []
    for pos in xss:
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {pos*0.52917721067}", basis=basis)
        mf = scf.RHF(mol)
        yss.append(mf.kernel())
    return xss, yss


# %%
get_dmin("sto3g"), get_dmin("6-31G"), get_dmin("def2TZVP"), get_dmin("def2SVP")
# %%
def get_AC_energy(basis, disp, order):
    if disp < 0:
        disp = f"m{abs(disp)}"
    else:
        disp = str(disp)
    c = convergence.Calculation(f"PROD/spatialH/{disp}/{basis}.out")
    coeffs = [mpmath.mp.mpf(_) for _ in c.get_coefficients("energy_0")]

    s = coeffs[:order]
    n = int((len(s) - 1) / 2)
    p, q = mpmath.pade(s, n, n)

    return mpmath.polyval(p[::-1], 1) / mpmath.polyval(q[::-1], 1)


# %%
LIMIT = 20
step = mpmath.mp.mpf("1e-10")


@functools.lru_cache(maxsize=100)
def get_coefficients(basis):
    values = {-_: get_AC_energy(basis, _, 18) for _ in range(-LIMIT, LIMIT + 1)}

    c = convergence.Calculation(f"PROD/spatialH/0/{basis}.out")
    coeffs = []
    for order in range(LIMIT):
        stencil = c._stencils[order]
        coefficient = sum(
            [
                mpmath.mp.mpf(values[shift]) * mpmath.mp.mpf(weight)
                for shift, weight in stencil.items()
            ]
        ) / step ** mpmath.mp.mpf(order)
        coefficient /= mpmath.factorial(order)
        coeffs.append(coefficient)

    return coeffs


# %%
def get_approx_roots(vals):
    left = vals[2:]
    center = vals[1:-1]
    right = vals[:-2]
    return np.where((left > center) & (right > center))[0] + 2


# %%
def fix_axes(ax):
    plt.setp(ax.spines.values(), linewidth=1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for loc, spine in ax.spines.items():
        if loc in ["left", "bottom"]:
            spine.set_position(("outward", 10))


plt.rcParams.update({"font.size": 22})
cutoff = 11
style = {
    "markersize": 10,
    "markeredgewidth": 2,
    "markeredgecolor": "white",
}
plt.rcParams["font.family"] = "Linux Biolinum O"

f, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

xs = np.linspace(-0.1, 0.4, 500)
rows = []
titles = {
    "sto3g": "STO-3G",
    "6-31G": "6-31G",
    "def2SVP": "def2-SVP",
    "def2TZVP": "def2-TZVP",
}
print("closest distances at order")
for bidx, basis in enumerate("sto3g 6-31G def2SVP def2TZVP ".split()):
    X, Y = bidx // 2, bidx % 2
    ax = axs[X, Y]
    ax.set_title(titles[basis])
    coeffs = get_coefficients(basis)
    for maxo in range(5):
        ys = [float(mpmath.polyval(coeffs[:maxo][::-1], _)) for _ in xs]
        r = 1.2 + xs
        ys = ys + 1 / r
        approx_roots = r[get_approx_roots(ys)]
        if len(approx_roots) == 0:
            closest = None
        else:
            closest = min(abs(approx_roots - get_dmin(basis)))
        rows.append({"basis": basis, "order": maxo, "error": closest})
        if maxo == 4:
            print(maxo, basis, abs(approx_roots - get_dmin(basis)))
        if maxo > 1:
            ax.plot(r, ys, label=f"{maxo}")
    ax.plot(*get_ref_curve(basis), "--", label="REF")
    ax.axvline(1.2, color="lightgrey")
    # ax.legend()
    ax.set_ylim(-1.14, -1.1)
    if Y == 0:
        ax.set_ylabel("Energy [Ha]")
    if X == 1:
        ax.set_xlabel("Distance [$a_0$]")
    if X == 0 and Y == 0:
        ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=1)
    fix_axes(ax)
plt.savefig("distance.pdf")
# %%
