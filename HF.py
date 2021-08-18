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
from RHF import *

from matrices import *
from integrals import *
from basis import *

import numpy as np
import numpy.linalg as la
import mpmath

import warnings

warnings.filterwarnings("ignore")
#import pyscf.scf
#import pyscf.gto
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
    #EE = mpmath.matrix(K, K, K, K)
    EE = np.zeros((K, K, K, K)).astype(mpmath.mp.mpf)
    for result in results:
        i, j, k, l, E = result
        EE[i, j, k, l] = E

    return EE

def print_mat(mat):
    for i in range(mat.shape[0]):
        line = ""
        for j in range(mat.shape[1]):
            line += f"{mpmath.nstr(mat[i, j]):<15}"
        print (line)

def verify_pyscf(config, offset, lval):
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
    calc.kernel()
    S = calc.get_ovlp(mol)
    Hc = calc.get_hcore(mol)
    P = calc.make_rdm1()
    ee = mol.intor('int2e')
    F = mol.get_fock()
    return mol, calc, calc.energy_elec()[0], S, Hc, P, ee, F

def get_energy(config, offset, lval):
    mpmath.mp.dps = config["meta"].getint("dps")

    step = mpmath.mpf(f'1e-{config["meta"].getint("deltalambda")}')
    if lval is None:
        lval = step * offset

    #MOL, CALC, pyscf_e, pyscf_S, pyscf_Hc, pyscf_P, pyscf_ee, pyscf_F = verify_pyscf(config, offset, lval)
    mol, bs, N = build_system(config, lval)
    ee = get_ee(config["meta"]["cache"])
    #assert (abs(np.max(MP2NP(ee)) - np.max(pyscf_ee)) < 1e-6)
    #assert (np.allclose(MP2NP(ee).astype(float), pyscf_ee))
    K = bs.K
    Gfactor = G_ee_cache(K, ee)
    S = S_overlap(bs)
    #assert (np.allclose(S.astype(float), pyscf_S))
    X = X_transform(S)

    # check X
    #s, U = np.linalg.eigh(S.astype(float))
    #Xref = np.dot(U,np.diag(s**(-1./2.)))
    #assert (np.allclose((Xref.T @ S @ Xref).astype(float), np.identity(len(s))))
    #assert (np.allclose((X.T @ S @ X).astype(float), np.identity(len(s))))


    Hc = H_core(bs, mol)
    #assert (np.allclose(Hc.astype(float), pyscf_Hc))

    maxiter = 100000  # Maximal number of iteration

    P = np.array(mpmath.zeros(K, K).tolist())
    #P = np.array(pyscf_P).astype(mpmath.mp.mpf)

    converged = False

    iter = 1
    threshold = mpmath.mp.mpf(f"1e-{mpmath.mp.dps-3}")
    if offset is None:
        threshold = mpmath.mp.mpf(f"1e-100")

    while not converged and iter <= maxiter:
        Pnew, F, E = RHF_step(
            bs, mol, N, Hc, X, P, ee, Gfactor, False
        )  # Perform an SCF step
        #assert (np.allclose(F.astype(float), MOL.get_fock(dm=P.astype(float))))
        #mo_energy, mo_coeff = MOL.eig(F.astype(float), pyscf_S)
        
        #import scipy.linalg
        #Fx = np.dot(X.conj().T, np.dot(F, X))
        #e, Cx = scipy.linalg.eigh(Fx.astype(float))
        #idx = e.argsort()
        #e = e[idx]
        #Cx = Cx[:,idx]
        #e = np.diag(e)
        #C = np.dot(X,Cx)
        #Pnew2 = np.zeros((K,K))

        #for i in range(K):
        #    for j in range(K):
        #        for k in range(int(N/2)): #TODO Only for RHF
        #            Pnew2[i,j] += 2 * C[i,k] * C[j,k].conjugate()

        #assert(np.allclose(Pnew2, Pnew.astype(float)))

        # Check convergence of the SCF cycle
        Pnew = (P + Pnew) / 2

        # print(Pnew)
        if iter % 100 == 0:
            print(
                f"{offset}@{iter}: e{int(mpmath.log10(delta_P(P, Pnew)))}/{int(mpmath.log10(threshold))}"
            )
        if delta_P(P, Pnew) < threshold:
            converged = True

        if iter == maxiter:
            print("SCF NOT CONVERGED!", lval)
            return offset, (iter, None, None, None)

        P = Pnew

        iter += 1
    #assert np.allclose(P.astype(float), pyscf_P)
    #assert (energy_el(P, F, Hc) - pyscf_e) < 1e-6
    return offset, (iter, energy_el(P, F, Hc), E, P)


def init_config(infile):
    config = configparser.ConfigParser()
    with open(infile) as fh:
        config.read_file(fh)
    if "cache" not in config["meta"]:
        with open(infile, "rb") as fh:
            config["meta"]["cache"] = hashlib.sha256(fh.read()).hexdigest()
    return config


def cache_EE_integrals(config, single_core):
    cachename = config["meta"]["cache"] + "-ee.cache"
    if os.path.exists(cachename):
        return
    mol, bs, N = build_system(config, 0)
    ee = EE_list(bs, single_core)
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
            offsets = np.array([0])
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
    single_core = False
    cache_EE_integrals(config, single_core)

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
    if single_core:
        res = [get_energy(*_) for _ in tasks]
    else:
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
