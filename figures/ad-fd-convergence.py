#!/usr/bin/env python
#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.ticker

sys.path.insert(0, ".")
import convergence

import mpmath

mpmath.mp.dps = 100

# %%
def read_diffiqult_derivative(basisset):
    with open(f"./diffiqult-h2tohe-{basisset}-bse.log") as fh:
        lines = fh.readlines()

    start = np.cumsum(["derivatives" in _ for _ in lines])
    stop = np.cumsum(["EnergyTarget" in _ for _ in lines])
    relevant_lines = np.array(lines)[np.where(start - stop)]
    coefficients = [
        float(_.replace("]", " ").replace("[", " ").strip().split()[-1])
        for _ in relevant_lines
    ]
    return np.array(coefficients)


def read_thiswork_derivative(fn):
    c = convergence.Calculation(fn)

    coeffs = np.array([mpmath.mp.mpf(_) for _ in c.get_coefficients("energy_0")])
    target = c.get_electronic_energy_target()
    return target, coeffs


def placeletter(ax, letter):
    ax.text(
        1.0,
        1.1,
        "(" + letter.lower() + ")",
        transform=ax.transAxes,
        # fontsize=26,
        fontweight=800,
        va="bottom",
        ha="right",
        color=DARKGREY,
    )


GREY = "#d1cfcf"
DARKGREY = "#6d6d6d"


def fix_axes(ax):
    plt.setp(ax.spines.values(), linewidth=1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["left"].set_edgecolor(DARKGREY)
    # ax.spines["bottom"].set_edgecolor(DARKGREY)
    ax.tick_params(length=5, color=DARKGREY, width=1.5, labelcolor="black")
    ax.tick_params(which="minor", length=5, color=GREY, width=1.5, labelcolor="white")
    for loc, spine in ax.spines.items():
        if loc in ["left", "bottom"]:
            spine.set_position(("outward", 0))


def pretty_plot(ax, ys, color):
    _pretty(ax.plot, ys, color, 5)


def pretty_semilogy(ax, ys, color):
    _pretty(ax.semilogy, ys, color, 2)


def _pretty(fun, ys, color, ms):
    fun(ys, "-", alpha=0.5, color=color)
    if color == "C0":
        fun(ys, "o", markersize=ms, color=color, clip_on=False)
    else:
        fun(ys, "o", markersize=ms, color=color)


def make_figure(ext):
    plt.style.use("figures/paper.mplstyle")
    cutoff = 11
    plt.rcParams["font.sans-serif"] = "Fira Sans Extra Condensed"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.it"] = "Fira Sans Extra Condensed:italic"
    plt.rcParams["mathtext.bf"] = "Fira Sans Extra Condensed:italic:medium"

    f = plt.figure(constrained_layout=True)

    gs = f.add_gridspec(1, 2)

    axcompare = f.add_subplot(gs[:, 0])

    # axcompare.set_title("H$_2$$\\rightarrow$He", pad=50, loc="right")
    axN2 = f.add_subplot(gs[:, 1:])
    # axN2.set_title("N$_2$$\\leftrightarrow$CO", pad=50, loc="right")

    pretty_plot(axcompare, np.cumsum(read_diffiqult_derivative("sto3g")[:cutoff]), "C1")
    axcompare.text(7, -2.0, "AD", ha="left", color="C1", va="top")
    target, thiswork = read_thiswork_derivative("PROD/H2/dps-1000-sto3g.out")
    pretty_plot(axcompare, np.cumsum(thiswork[:cutoff]), "C0")
    axcompare.text(8, -2.68, "FD", ha="left", color="C0", va="top")
    axcompare.axhline(target, color="white", lw=5, alpha=0.8)
    axcompare.axhline(target, color="C3")
    # axcompare.text(-0.2, target - 0.03, "He", ha="left", color="C3", va="top")
    axcompare.annotate(
        "He",
        (0, target),
        xytext=(3, -3),
        textcoords="offset pixels",
        color="C3",
        va="top",
        weight=300,
    )
    # axcompare.legend(frameon=False)
    axcompare.set_xlabel("Order", loc="right", weight=500, va="bottom", labelpad=-30)
    axcompare.set_ylabel(r"$\bf{E}$ [Ha]", rotation=0, ha="left", y=1.1, weight=500)
    placeletter(axcompare, "A")
    axcompare.set_xticks((0, 5, 10))
    axcompare.set_xlim(0, 10)
    axcompare.set_yticks(np.linspace(-1.5, -3, 4))
    axcompare.set_ylim(-3, -1.5)

    target, thiswork = read_thiswork_derivative("PROD/N2_CO/sto3g.out")
    pretty_semilogy(axN2, abs(np.cumsum(thiswork) - target)[:40], "C0")
    axN2.text(25, 1e-10, "N$_2$ → CO", ha="right", color="C0", va="top")
    target, thiswork = read_thiswork_derivative("PROD/CO_N2/sto3g-co.out")
    pretty_semilogy(axN2, abs(np.cumsum(thiswork) - target), "C1")
    axN2.text(40, 5e-7, "CO → N$_2$", ha="right", color="C1", va="bottom")
    # axN2.legend(frameon=False)
    axN2.set_ylabel("$\\bf{\Delta E}$ [Ha]", rotation=0, ha="left", y=1.1, weight=500)
    axN2.axhline(1e-8, color="white", lw=5, alpha=0.8)
    axN2.axhline(1e-8, color="C3")
    axN2.annotate(
        "SCF",
        (0, 1e-8),
        xytext=(3, -3),
        textcoords="offset pixels",
        color="C3",
        va="top",
        weight=300,
    )
    axN2.axhline(0.04336 / 27.211, color="white", lw=5, alpha=0.8)
    axN2.axhline(0.04336 / 27.211, color="C4")
    axN2.text(
        40,
        0.04336 / 27.211,
        "Chemical\naccuracy",
        ha="right",
        color="C4",
        va="bottom",
        weight=300,
    )
    axN2.set_xlabel("Order", loc="right", weight=500, va="bottom", labelpad=-30)
    placeletter(axN2, "B")
    # axN2.set_xticks()
    # axN2.axis["left"].major_ticklabels.set_ha("center")

    axN2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator((0, 20, 40)))
    axN2.xaxis.set_minor_locator(
        matplotlib.ticker.FixedLocator(
            (
                10,
                30,
            )
        )
    )

    axN2.set_xlim(0, 40)
    axN2.yaxis.set_major_locator(
        matplotlib.ticker.FixedLocator(
            (
                1e-0,
                1e-4,
                1e-8,
                1e-12,
                1e-16,
            )
        )
    )
    axN2.yaxis.set_minor_locator(
        matplotlib.ticker.FixedLocator((1e-2, 1e-6, 1e-10, 1e-14))
    )
    axN2.set_ylim(1e-16, 1e-0)
    fix_axes(axN2)
    fix_axes(axcompare)
    f.align_xlabels((axcompare, axN2))
    plt.savefig(f"figures/ad-fd-convergence.{ext}")


if __name__ == "__main__":
    make_figure("pdf")
# %%
