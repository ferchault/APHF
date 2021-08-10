"""
    Copyright (C) 2015 Rocco Meli

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import functools
import findiff
import subprocess
import hashlib
import configparser
import multiprocessing as mp
import os
import click
import basis_set_exchange as bse
import pickle
from numpy.lib.arraysetops import isin
from RHF import *

from matrices import *
from integrals import *
from basis import *

import numpy as np
import numpy.linalg as la
import mpmath

mpmath.mp.dps = 100


def NP2MP(array):
    return mpmath.mp.matrix(array.tolist())


def MP2NP(array):
    return np.array(array.tolist())


###########################
###########################
###########################
@functools.lru_cache(maxsize=1)
def get_ee(cachename):
    with open(cachename + "-ee.cache", "rb") as fh:
        results = pickle.load(fh)
    K = 0
    for result in results:
        i, j, k, l, E = result
        K = max(K, max(i, j, k, l))
    K = K + 1
    EE = mpmath.matrix(K, K, K, K)
    for result in results:
        i, j, k, l, E = result
        EE[i, j, k, l] = E

    return EE


def get_energy(config, offset, lval):
    mpmath.mp.dps = config["meta"].getint("dps")

    step = mpmath.mpf(f'1e-{config["meta"].getint("deltalambda")}')
    if lval is None:
        lval = step * offset

    mol, bs, N = build_system(config, lval)
    ee = get_ee(config["meta"]["cache"])
    K = bs.K
    S = S_overlap(bs)
    X = X_transform(S)
    Hc = H_core(bs, mol)

    maxiter = 100000  # Maximal number of iteration

    Pnew = np.array(mpmath.zeros(K, K).tolist())
    P = np.array(mpmath.zeros(K, K).tolist())

    converged = False

    iter = 1
    while not converged and iter <= maxiter:
        Pnew, F, E = RHF_step(bs, mol, N, Hc, X, P, ee, False)  # Perform an SCF step

        # Check convergence of the SCF cycle
        Pnew = (P + Pnew) / 2

        # print(Pnew)
        if iter % 100 == 0:
            print(
                f"{offset}@{iter}: e{int(mpmath.log10(delta_P(P, Pnew)))}/{mpmath.mp.dps-3}"
            )
        if delta_P(P, Pnew) < mpmath.mp.mpf(f"1e-{mpmath.mp.dps-3}"):
            converged = True

        if iter == maxiter:
            print("SCF NOT CONVERGED!", lval)
            return offset, (iter, None, None, None)

        P = Pnew

        iter += 1
    return offset, (iter, energy_el(P, F, Hc), E, P)


def init_config(infile):
    config = configparser.ConfigParser()
    with open(infile) as fh:
        config.read_file(fh)
    with open(infile, "rb") as fh:
        config["meta"]["cache"] = hashlib.sha256(fh.read()).hexdigest()
    return config


def cache_EE_integrals(config):
    cachename = config["meta"]["cache"] + "-ee.cache"
    if os.path.exists(cachename):
        return
    mol, bs, N = build_system(config, 0)
    ee = EE_list(bs)
    with open(cachename, "wb") as fh:
        pickle.dump(ee, fh)


def build_system(config, lval):
    reference_Zs = config["meta"]["reference"].strip().split()
    basis_Zs = config["meta"]["basis"].strip().split()
    target_Zs = config["meta"]["target"].strip().split()
    coords = config["meta"]["coords"].strip().split("\n")

    mol = []

    N = 0
    for ref, tar, bas, coord in zip(reference_Zs, target_Zs, basis_Zs, coords):
        N += int(ref)
        element = bse.lut.element_data_from_Z(int(bas))[0].capitalize()
        Z = mpmath.mpf(tar) * lval + (1 - lval) * mpmath.mpf(ref)
        atom = Atom(element, tuple([mpmath.mpf(_) for _ in coord.split()]), Z, bas)
        mol.append(atom)
    bs = Basis(config["meta"]["basisset"], mol)

    return mol, bs, N


def get_stencils(maxorder):
    stencils = {}
    for order in tqdm.tqdm(range(maxorder), desc="Build stencil"):
        if order == 0:
            weights = np.array([mpmath.mp.mpf("1.0")])
            offsets = np.array([1])
        else:
            lookup = findiff.coefficients(deriv=order, acc=2, symbolic=True)["center"]
            weights = [
                mpmath.mp.mpf(_.numerator()) / mpmath.mp.mpf(_.denominator())
                for _ in lookup["coefficients"]
            ]
            offsets = lookup["offsets"]
        stencils[order] = {"weights": weights, "offsets": offsets}
    return stencils


@click.command()
@click.argument("infile")
@click.argument("outfile")
def main(infile, outfile):
    config = init_config(infile)
    mp.set_start_method("spawn")
    mpmath.mp.dps = config["meta"].getint("dps")
    config["meta"]["revision"] = (
        subprocess.check_output("git rev-parse HEAD".split()).decode("ascii").strip()
    )

    # caching
    cache_EE_integrals(config)

    # find work
    maxorder = config["meta"].getint("orders")
    stencils = get_stencils(maxorder)
    offsets = set(
        sum([list(stencil["offsets"]) for order, stencil in stencils.items()], [])
    )
    step = mpmath.mpf(f'1e-{config["meta"].getint("deltalambda")}')
    tasks = [(config, _, None) for _ in offsets]
    tasks += [(config, None, mpmath.mpf("1.0"))]

    # evaluate
    with mp.Pool(os.cpu_count()) as p:
        res = p.starmap(
            get_energy,
            tqdm.tqdm(tasks, total=len(tasks), desc="Function evaluations"),
            chunksize=1,
        )

    res = dict(res)
    config.add_section("singlepoints")
    for c, item in enumerate(res.items()):
        offset, v = item
        if offset is None:
            offset = "target"
        iter, energy, mo_energy, dm = v
        config["singlepoints"][f"energy_{offset}"] = str(energy)
        config["singlepoints"][f"iter_{offset}"] = str(iter)
        for idx, mo in enumerate(mo_energy):
            config["singlepoints"][f"moenergy_{offset}_{idx}"] = str(mo)
        for idxA, dmA in enumerate(dm):
            for idxB, dmE in enumerate(dmA):
                config["singlepoints"][f"dm_{offset}_{idxA}_{idxB}"] = str(dmE)

    # store endpoints
    ref = res[0][1]
    target = res[None][1]
    config.add_section("endpoints")
    config["endpoints"]["reference"] = str(ref)
    config["endpoints"]["target"] = str(target)

    # stencil
    config.add_section("stencil")
    for order, stencil in stencils.items():
        for shift, weight in zip(stencil["offsets"], stencil["weights"]):
            config["stencil"][f"order_{order}_{shift}"] = str(weight)

    with open(outfile, "w") as fh:
        config.write(fh)


if __name__ == "__main__":
    main()