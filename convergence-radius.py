#!/usr/bin/env python
import numpy as np
import sys

sys.path.insert(0, ".")
import convergence
import os
import tqdm

import mpmath

mpmath.mp.dps = 100


import warnings

warnings.filterwarnings("ignore")
import sys
from pyscf.gto import basis
import pyscf.scf
import pyscf.gto
import pyscf.qmmm
import configparser
import basis_set_exchange as bse

#%%
angstrom = 1 / 0.52917721067


def add_qmmm(calc, mol, deltaZ):
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords() / angstrom, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(float)
        q += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


def build_system(filename, lval):
    config = configparser.ConfigParser()
    with open(filename) as fh:
        config.read_file(fh)

    reference_Zs = config["meta"]["reference"].strip().split()
    basis_Zs = config["meta"]["basis"].strip().split()
    target_Zs = config["meta"]["target"].strip().split()
    coords = config["meta"]["coords"].strip().split("\n")

    mol = []

    N = 0
    atomspec = ""
    basisspec = {}
    idx = 0
    deltaZ = []
    for ref, tar, bas, coord in zip(reference_Zs, target_Zs, basis_Zs, coords):
        N += int(ref)
        deltaZ.append(int(tar) - int(ref))
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
    calc_target = add_qmmm(pyscf.scf.RHF(mol), mol, np.array(deltaZ) * lval)

    calc_target.kernel()
    return calc_target.energy_elec()[0]


def conv_radius(fn):
    c = convergence.Calculation(fn)
    coeffs = np.array([mpmath.mp.mpf(_) for _ in c.get_coefficients("energy_0")])

    def evaluate_taylor_at(coeffs, delta):
        results = []
        delta = mpmath.mp.mpf(delta)
        for n, coeff in enumerate(coeffs):
            if n == 0:
                results.append(coeff)
            else:
                results.append(results[-1] + coeff * delta ** n)
        return np.array(results)

    def evaluate_pade_at(coeffs, delta):
        results = []
        delta = mpmath.mp.mpf(delta)
        for leading in range(2, len(coeffs)):
            s = coeffs[:leading]
            n = int((len(s) - 1) / 2)
            try:
                p, q = mpmath.pade(s, n, n)
            except:
                results.append(np.nan)
            results.append(
                mpmath.polyval(p[::-1], delta) / mpmath.polyval(q[::-1], delta)
            )
        return np.array(results)

    def converges(ys, delta):
        chemacc = 0.04336 / 27.211
        residual = abs(ys - ys[-1])
        if max(residual[20:]) > chemacc:
            return False
        scf_energy = build_system(fn, delta)
        if abs(scf_energy - ys[-1]) > chemacc:
            return False
        return True

    deltas = np.linspace(0, 4, 4 * 20)
    conv_taylor = []
    conv_pade = []
    for delta in deltas[::-1]:
        if converges(evaluate_taylor_at(coeffs, -delta).astype(float), -delta):
            conv_taylor.append(-delta)
            break
    for delta in deltas[::-1]:
        if converges(evaluate_taylor_at(coeffs, delta).astype(float), delta):
            conv_taylor.append(delta)
            break
    for delta in deltas[::-1]:
        if converges(evaluate_pade_at(coeffs, -delta).astype(float), -delta):
            conv_pade.append(-delta)
            break
    for delta in deltas[::-1]:
        if converges(evaluate_pade_at(coeffs, delta).astype(float), delta):
            conv_pade.append(delta)
            break

    return conv_taylor, conv_pade


if __name__ == "__main__":
    outfile = f"{sys.argv[1]}.convradius"
    if not os.path.exists(outfile):
        print(sys.argv[1])

        dtaylor, dpade = conv_radius(sys.argv[1])
        with open(outfile, "w") as fh:
            fh.write("taylor " + (",".join([str(_) for _ in dtaylor])) + "\n")
            fh.write("pade " + (",".join([str(_) for _ in dpade])) + "\n")
