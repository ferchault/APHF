#!/usr/bin/env python
#%%
import matplotlib.pyplot as plt
import numpy as np
import sys
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

sys.path.insert(0, ".")
import convergence

import mpmath

mpmath.mp.dps = 10

# %%
def conv_radius(fn, method):
    with open(f"{fn}.convradius") as fh:
        lines = fh.readlines()
    for line in lines:
        if line.startswith(method):
            values = [float(_) for _ in line.split()[1].split(",")]
            return min(values), max(values)


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


def plot_radius(ax, pairs, center, method, color, alpha, right):
    boundaryl = []
    boundaryr = []
    for fn, direction in pairs:
        convrange = conv_radius(fn, method)
        boundaryl.append(max(convrange) * np.array(direction) + center)
        boundaryr.append(min(convrange) * np.array(direction) + center)
    boundary = np.array(boundaryl + boundaryr + boundaryl[:1])
    ax.fill(boundary[:, 0], boundary[:, 1], alpha=alpha, color=color)
    ax.plot(boundary[:, 0], boundary[:, 1], color=color, lw=0.4)
    for a, b in zip(boundaryl, boundaryr):
        ax.plot((a[0], b[0]), (a[1], b[1]), "-", color="white")
    ax.scatter((center[0],), (center[1],))
    ax.axis("square")

    ax.set_ylabel("Z$_2$")
    if right:
        ax.set_xlabel("Z$_1$")
        ax.set_xticks([5, 7, 9, 11])
    else:
        ax.set_xticks([])
    ax.set_yticks([5, 7, 9, 11])
    ax.set_xlim(4.3, 11.3)
    ax.set_ylim(4.3, 11.3)

    xs = (5, 6, 7, 8, 9, 10, 11)
    pts = []
    polygon = Polygon(boundaryl + boundaryr)
    for x in xs:
        for y in xs:
            if polygon.contains(Point(x, y)):
                pts.append((x, y))
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], alpha=1, color="C3", s=8, zorder=100)
    ax.scatter(*center, color="C3", s=50, zorder=200)


center = np.array((6, 8))
pairs = (
    ("PROD/CO_N2/sto3g.out", (1, -1)),
    ("PROD/CO_NO/sto3g.out", (1, 0)),
    ("PROD/CO_NF/sto3g.out", (1, 1)),
    ("PROD/CO_CF/sto3g.out", (0, 1)),
)

f = plt.figure(constrained_layout=True, figsize=(8, 8))

gs = f.add_gridspec(2, 6)

axs = f.add_subplot(gs[0, :4]), f.add_subplot(gs[1, :4]), f.add_subplot(gs[:, 4:])

plot_radius(axs[0], pairs, center, "pade", "C0", 0.4, False)
plot_radius(axs[0], pairs, center, "taylor", "white", 1, False)
plot_radius(axs[0], pairs, center, "taylor", "C0", 0.1, False)
center = np.array((7, 7))
pairs = (
    ("PROD/N2_CO/sto3g.out", (-1, 1)),
    ("PROD/N2_NO/sto3g.out", (0, 1)),
    ("PROD/N2_O2/sto3g.out", (1, 1)),
    ("PROD/N2_NO/sto3g.out", (1, 0)),
)
plot_radius(axs[1], pairs, center, "pade", "C1", 0.4, True)
plot_radius(axs[1], pairs, center, "taylor", "white", 1, True)
plot_radius(axs[1], pairs, center, "taylor", "C1", 0.1, True)

bases = "3-21G 4-31G 6-21G 6-31++G 6-31+G 6-311G-J sto3g 6-31G ANO-DK3".split()
replace = {"sto3g": "STO-3G"}
curves = []
for basis in bases:
    a, b = conv_radius(f"PROD/N2_CO/{basis}.out", "pade")
    c, d = conv_radius(f"PROD/N2_CO/{basis}.out", "taylor")
    curves.append((a, c, b, d))
curves = np.array(curves)
ordering = np.argsort(curves[:, 2:].max(axis=1) - curves[:, :2].min(axis=1))
basesordered = []
for _ in ordering:
    _ = bases[_]
    if _ in replace:
        _ = replace[_]
    basesordered.append(_)
for col in (2, 3):
    axs[2].plot(curves[ordering, col], basesordered, "o-")
for dZ in (1, 2):
    axs[2].axvline(dZ, ls="--", color="grey", alpha=0.5)
plt.savefig("radius.pdf")
# %%
