#!/usr/bin/env python
#%%
import configparser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import basis_set_exchange as bse
import pyscf.gto
import pyscf.scf

sys.path.insert(0, ".")
import convergence

import mpmath

#%%
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

f = plt.figure(constrained_layout=True, figsize=(20, 7))

gs = f.add_gridspec(3, 6)

filenames = """PROD/H2/dps-1000-sto3g.out
PROD/He/dps-1000-STO3G.out
PROD/N2_CO/sto3g.out
PROD/CO_N2/sto3g-co.out
PROD/H6toHe3/dps-1000-STO3G.out
PROD/He3toH6/dps-1000-STO3G.out""".split()
titles = [
    "H$_2$ $\\rightarrow$ He",
    "He $\\rightarrow$ H$_2$",
    "N$_2$ $\\rightarrow$ CO",
    "CO$\\rightarrow$ N$_2$",
    "H$_6$$\\rightarrow$ He$_3$",
    "He$_3$$\\rightarrow$ H$_6$",
    "H$_6$$\\rightarrow$ Li$_2$",
    "Li$_2$$\\rightarrow$ H$_6$",
]
ylims = (1e-13, 1e-18, 1e-4, 1e-4)
xmax = (3, 3, 5, 5, 12, 12)
ymax = (18, 18, 70, 70, 15, 15)

aligns = []
for j in range(6):
    for i in range(3):
        ax = f.add_subplot(gs[i, j])
        if i == 0:
            ax.set_title(titles[j])
            for label in "energy dm moenergy".split():
                df = pd.read_csv(f"{filenames[j]}.{label}.csv")
                s = df.query("method == 'pade'").groupby("order").mean().reset_index()
                ax.semilogy(s.order, s.error)
            ax.set_ylim(ylims[j // 2], 1e1)
            ax.axhline(0.04336 / 27.211, color="C4")
            ax.axhline(1e-8, color="C3")
            ax.get_xaxis().set_ticks([])
        if i == 1:
            c = convergence.Calculation(filenames[j])
            for key in c.get_keys_by_group("moenergy"):
                df = pd.read_csv(f"{filenames[j]}.moenergy.csv")
                a = (
                    df.query("method == 'taylor'")
                    .pivot("order", "key", "value")
                    .query("order < 2")
                )
                b = (
                    df.query("method == 'pade'")
                    .pivot("order", "key", "value")
                    .query("order > 1")
                )
                ts = pd.concat((a, b))
                ax.plot(ts, color="C0", lw=0.5)
                ax.set_ylim(-2, 2)
                ax.scatter(
                    ts.reset_index().order.max(), float(c.get_target(key)), color="grey"
                )
            ax.set_xlabel("Order")
            ax.set_xticks((0, 10, 20, 30, 40, 50))
        if (j % 2) == 1:
            ax.get_yaxis().set_ticks([])
        if i == 2:
            if (j % 2) == 1:
                target = get_density_profiles(filenames[j - 1])["ref"]
            else:
                target = get_density_profiles(filenames[j + 1])["ref"]
            ax.fill_between(*target, color="C0", alpha=0.2)
            ax.plot(*get_density_profiles(filenames[j])[0], "--")
            if j < 4:
                ax.plot(*get_density_profiles(filenames[j])[4], color="C1")
                ax.text(-1, 12, "4", ha="left", color="C1", va="top")
                if j > 0:
                    ax.plot(*get_density_profiles(filenames[j])[10], color="C2")
                    ax.text(1, 12, "10", ha="left", color="C2", va="top")

            if j > 3:
                ax.plot(*get_density_profiles(filenames[j])[20], color="C1")
                ax.text(-2, 13, "20", ha="left", color="C1", va="top")
                ax.plot(*get_density_profiles(filenames[j])[35], color="C2")
                ax.text(-2, 10, "35", ha="left", color="C2", va="top")
            ax.set_xlim(min(target[0]), xmax[j])
            ax.set_ylim(0, ymax[j])
            ax.set_xlabel("Position [a$_0$]")
        if j == 0:
            ylabels = [
                "|Error|\n[a.u.]",
                "MO Energy\n[Ha]",
                "Electron density\n[arb. units]",
            ]
            ax.set_ylabel(ylabels[i])
            aligns.append(ax)

        fix_axes(ax)
f.align_ylabels(aligns)
plt.savefig("demo.pdf")
# %%
# %%
def pyscfify(config):
    angstrom = 1 / 0.52917721067
    reference_Zs = config["meta"]["reference"].strip().split()
    basis_Zs = config["meta"]["basis"].strip().split()
    target_Zs = config["meta"]["target"].strip().split()
    coords = config["meta"]["coords"].strip().split("\n")

    mol = []

    N = 0
    atomspec = ""
    basisspec = {}
    idx = 0

    for ref, tar, bas, coord in zip(reference_Zs, target_Zs, basis_Zs, coords):
        N += int(ref)
        try:
            ref = bse.lut.element_data_from_Z(int(ref))[0].capitalize()
        except:
            ref = "X"
        bas = bse.lut.element_data_from_Z(int(bas))[0].capitalize()

        coord = tuple([float(_) / angstrom for _ in coord.split()])
        atomspec += f"{ref}:{idx} {coord[0]} {coord[1]} {coord[2]};"
        bsespec = bse.get_basis(config["meta"]["basisset"], elements=bas, fmt="NWCHEM")
        basisspec[f"{ref}:{idx}"] = pyscf.gto.parse(bsespec)
        idx += 1

    mol = pyscf.gto.M(atom=atomspec, basis=basisspec, verbose=0)
    calc = pyscf.scf.RHF(mol)
    return mol, calc


def project_dm(mol, dm):
    N = 20
    xs = np.linspace(-3, 3, N)
    grid = np.vstack((np.tile(xs, N), np.repeat(xs, N), np.zeros(N * N))).T
    zs = np.linspace(-2, 11, 400)
    rhos = []
    for z in zs:
        grid[:, 2] = z
        ao_value = pyscf.dft.numint.eval_ao(mol, grid, deriv=0)
        rho = pyscf.dft.numint.eval_rho(mol, ao_value, dm, xctype="LDA")
        rhos.append(sum(rho))
    return zs, rhos


def get_QA_DM(c, filename, order):
    nao = int(np.sqrt(len(c.get_keys_by_group("dm"))))
    pred_dm = np.zeros((nao, nao))
    df = pd.read_csv(f"{filename}.dm.csv")
    s = df.query("order == @order & method == 'pade'")
    if order < 2:
        s = df.query("order == @order & method == 'taylor'")
    for idx, row in s.iterrows():
        _, a, b = row["key"].split("_")
        pred_dm[int(a), int(b)] = row["value"]
    return pred_dm


# %%
import functools


@functools.lru_cache(maxsize=100)
def get_density_profiles(fn):
    c = convergence.Calculation(fn)

    mol, calc = pyscfify(c._config)
    calc.kernel()
    dm = calc.make_rdm1()
    profiles = {}
    profiles["ref"] = project_dm(mol, dm)
    for order in range(40):
        profiles[order] = project_dm(mol, get_QA_DM(c, fn, order))
    return profiles


# %%
for filename in filenames:
    print(filename)
    get_density_profiles(filename)
# %%
orders = (0, 1, 2, 4, 30)
for order in orders:
    plt.plot(*p[order])
# %%
profiles = {}
for filename in filenames:
    profiles[filename] = get_density_profiles(filename)
# %%
# import pickle
# with open("overview-profiles.pkl", "wb") as fh:
#     pickle.dump(profiles,fh)
# %%
