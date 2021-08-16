#!/usr/bin/env python
#%%
import warnings

warnings.filterwarnings("ignore")
import sys
from pyscf.gto import basis
import pyscf.scf
import pyscf.gto
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


def build_system(filename):
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

    eref = float(config["singlepoints"]["energy_0"])
    etarget = float(config["singlepoints"]["energy_target"])
    return mol, eref, etarget


mol, eref, etarget = build_system(sys.argv[1])

calc = pyscf.scf.RHF(mol)
calc.kernel()
if abs(calc.energy_elec()[0] - eref) < 1e-8:
    print("OK", sys.argv[1])
else:
    print("EE", sys.argv[1], "pyscf", calc.energy_elec()[0], "fd", eref)
