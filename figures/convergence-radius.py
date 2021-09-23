#!/usr/bin/env python
#%%
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import matplotlib as mpl
import basis_set_exchange as bse
from numpy.core.fromnumeric import clip


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
            try:
                values = [float(_) for _ in line.split()[1].split(",")]
            except:
                return 0, 0
            return min(values), max(values)


# %%

plt.rcParams.update({"font.size": 22})
cutoff = 11
style = {
    "markersize": 10,
    "markeredgewidth": 2,
    "markeredgecolor": "white",
}
plt.rcParams["font.family"] = "Linux Biolinum O"
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["#4da9cf", "#153565", "#e9952b", "#ce1a0a", "#864a83", "#51968f"]
)

GREY = "#d1cfcf"
DARKGREY = "#6d6d6d"


def fix_axes(ax):
    plt.setp(ax.spines.values(), linewidth=1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["left"].set_edgecolor(GREY)
    # ax.spines["bottom"].set_edgecolor(GREY)
    ax.tick_params(length=5, color=DARKGREY, width=1.5, labelcolor="black")
    ax.tick_params(which="minor", length=5, color=GREY, width=1.5, labelcolor="white")
    for loc, spine in ax.spines.items():
        if loc in ["left", "bottom"]:
            spine.set_position(("outward", 0))
    ax.set_zorder(-100)


def placeletter(ax, letter):
    ax.text(
        1.0,
        1.0,
        "(" + letter.lower() + ")",
        transform=ax.transAxes,
        # fontsize=26,
        fontweight=800,
        va="bottom",
        ha="right",
        color=DARKGREY,
    )


#%%


def plot_radius(ax, pairs, center, method, color, marker, right):
    boundaryl = []
    boundaryr = []
    for fn, direction in pairs:
        convrange = conv_radius(fn, method)
        boundaryl.append(max(convrange) * np.array(direction) + center)
        boundaryr.append(min(convrange) * np.array(direction) + center)
    boundary = np.array(boundaryl + boundaryr + boundaryl[:1])
    # ax.fill(boundary[:, 0], boundary[:, 1], alpha=alpha, color=color)
    ax.plot(boundary[:, 0], boundary[:, 1], marker, color=color, lw=2)
    # for a, b in zip(boundaryl, boundaryr):
    #     ax.plot((a[0], b[0]), (a[1], b[1]), "-", color="white")
    ax.scatter((center[0],), (center[1],))
    ax.axis("square")

    ax.set_ylabel(
        "$\\bf{Z}_2$", rotation=0, ha="left", y=1.01, labelpad=-20, weight=500
    )
    ax.set_xlabel("$\\bf{Z}_1$", loc="right", weight=500, va="bottom", labelpad=-30)
    if right:

        ax.set_xticks([3, 5, 7, 9, 11])
        ax.set_yticks([3, 5, 7, 9, 11])
        ax.set_xlim(3, 11)
        ax.set_ylim(3, 11)
    else:
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.set_yticks([4, 6, 8, 10, 12])
        ax.set_xlim(2, 10)
        ax.set_ylim(4, 12)

    xs = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    pts = []
    polygon = Polygon(boundaryl + boundaryr)
    for x in xs:
        for y in xs:
            if polygon.contains(Point(x, y)):
                pts.append((x, y))
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], color="white", s=50, zorder=99)
    ax.scatter(pts[:, 0], pts[:, 1], color="C3", s=8, zorder=100)
    ax.scatter(*center, color="C3", s=50, zorder=200)


plt.style.use("figures/paper.mplstyle")

plt.rcParams["font.sans-serif"] = "Fira Sans Extra Condensed"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.it"] = "Fira Sans Extra Condensed:italic"
plt.rcParams["mathtext.bf"] = "Fira Sans Extra Condensed:italic:medium"

f = plt.figure(constrained_layout=True, figsize=(9, 9))

gs = f.add_gridspec(2, 6)

axs = f.add_subplot(gs[0, :4]), f.add_subplot(gs[1, :4]), f.add_subplot(gs[:, 4:])
for ax in axs:
    fix_axes(ax)

center = np.array((6, 8))
pairs = (
    ("PROD/CO_N2/sto3g.out", (1, -1)),
    ("PROD/CO_NO/sto3g.out", (1, 0)),
    ("PROD/CO_NF/sto3g.out", (1, 1)),
    ("PROD/CO_CF/sto3g.out", (0, 1)),
)
plot_radius(axs[0], pairs, center, "pade", "C1", "o-", False)
plot_radius(axs[0], pairs, center, "taylor", "C2", "s-", False)

pairs = (
    ("PROD/CO_N2/Ahlrichs_VTZ.out", (1, -1)),
    ("PROD/CO_NO/Ahlrichs_VTZ.out", (1, 0)),
    ("PROD/CO_NF/Ahlrichs_VTZ.out", (1, 1)),
    ("PROD/CO_CF/Ahlrichs_VTZ.out", (0, 1)),
)
plot_radius(axs[0], pairs, center, "pade", "C1", "o--", False)
plot_radius(axs[0], pairs, center, "taylor", "C2", "s--", False)
placeletter(axs[0], "A")
placeletter(axs[1], "B")
placeletter(axs[2], "C")
axs[2].set_ylabel("Basis set", rotation=0, ha="left", y=1.01, labelpad=-120, weight=500)
center = np.array((7, 7))
pairs = (
    ("PROD/N2_CO/sto3g.out", (-1, 1)),
    ("PROD/N2_NO/sto3g.out", (0, 1)),
    ("PROD/N2_O2/sto3g.out", (1, 1)),
    ("PROD/N2_NO/sto3g.out", (1, 0)),
)
plot_radius(axs[1], pairs, center, "pade", "C1", "o-", True)
plot_radius(axs[1], pairs, center, "taylor", "C2", "s-", True)

pairs = (
    ("PROD/N2_CO/Ahlrichs_VTZ.out", (-1, 1)),
    ("PROD/N2_NO/Ahlrichs_VTZ.out", (0, 1)),
    ("PROD/N2_O2/Ahlrichs_VTZ.out", (1, 1)),
    ("PROD/N2_NO/Ahlrichs_VTZ.out", (1, 0)),
)


plot_radius(axs[1], pairs, center, "pade", "C1", "o--", True)
plot_radius(axs[1], pairs, center, "taylor", "C2", "s--", True)


bases = [_.split("/")[-1].split(".")[0] for _ in glob.glob("PROD/N2_CO/*.out")]

families = {}
for basis in bases:
    basismod = basis.replace("_", " ").replace("sto3g", "STO-3G")
    family = bse.get_basis_family(basismod)
    if family not in families:
        families[family] = []
    families[family].append(basis)
replace = {"sto3g": "STO-3G"}
curves = []
accepted = []
boundaries = []

forder = [
    "ahlrichs",
    "pople",
    "sto",
    "huzinaga",
    "zorrilla",
    "stuttgart",
    "jensen",
    "lanl",
    "sauer_j",
    "ano",
][::-1]
for family in forder:
    bases = families[family]
    for basis in bases:
        try:
            a, b = conv_radius(f"PROD/N2_CO/{basis}.out", "pade")
            c, d = conv_radius(f"PROD/N2_CO/{basis}.out", "taylor")
        except:
            continue
        accepted.append(basis.replace("_", " "))
        curves.append((a, c, b, d))
    boundaries.append(len(accepted))
curves = np.array(curves)
# ordering = np.argsort(curves[:, 2:].max(axis=1) - curves[:, :2].min(axis=1))
basesordered = []
for basis in accepted:
    if basis in replace:
        basesordered.append(replace[basis])
    else:
        basesordered.append(basis)
# for _ in ordering:
#     _ = accepted[_]
#     if _ in replace:
#         _ = replace[_]
#     basesordered.append(_)
axs[2].plot(
    curves[:, 2],
    basesordered,
    "o-",
    color=f"C1",
    label="Padé",
    clip_on=False,
    zorder=100,
)
axs[2].plot(
    curves[:, 3],
    basesordered,
    "s-",
    color=f"C2",
    label="Taylor",
    clip_on=False,
    zorder=100,
)
axs[2].legend(
    frameon=True,
    handlelength=1,
    handletextpad=0.2,
    loc="lower left",
    borderpad=0,
    borderaxespad=0.4,
    facecolor="white",
    framealpha=1,
    edgecolor="white",
)
for boundary in boundaries[:-1]:
    axs[2].axhline(boundary - 0.5)
axs[2].set_ylim(0, len(accepted) - 1)
axs[2].set_xlim(0, 4)
axs[2].set_xlabel(
    "$\\bf{\Delta Z}$", loc="right", weight=500, va="bottom", labelpad=-30
)
axs[2].set_xticks((0, 1, 2, 3, 4))
for dZ in (1, 2, 3):
    axs[2].axvline(dZ, ls="-", color=GREY, zorder=-100)

axs[0].text(2.2, 11.4, "Ahlrichs VTZ,", ha="left", color="C1", va="bottom")
axs[0].text(5.6, 11.4, "STO-3G", ha="left", color="C2", va="bottom")
axs[1].text(3.2, 10.5, "Ahlrichs VTZ,", ha="left", color="C1", va="bottom")
axs[1].text(6.6, 10.5, "STO-3G", ha="left", color="C2", va="bottom")

plt.savefig("figures/radius.pdf", bbox_inches="tight")

# %%
