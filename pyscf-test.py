#!/usr/bin/env python
from pyscf import scf, gto, qmmm

import numpy as np

a = 1.809 * np.sin(104.52 / 180 * np.pi / 2) / 1.8897259886
b = 1.809 * np.cos(104.52 / 180 * np.pi / 2) / 1.8897259886
mol = gto.M(atom=f"H {a} 0 0; H {-a} 0 0; O 0 {b} 0", basis="sto-3g")
calc = scf.RHF(mol)
calc.kernel()
print(calc.mo_energy)

print("H2 plain simple")
mol = gto.M(atom=f"H 0 0 0; H {1.4/1.8897259886} 0 0", basis="sto-3g")
calc = scf.RHF(mol)
calc.kernel()
print(calc.mo_energy)

print("H2 alchemy")


def add_qmmm(calc, mol, deltaZ):
    mf = qmmm.mm_charge(calc, mol.atom_coords() / 1.8897259886, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(np.float)
        q += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


mol = gto.M(atom=f"H 0 0 0; H {1.4/1.8897259886} 0 0", basis="sto-3g")
calc = add_qmmm(scf.RHF(mol), mol, np.array((1, -1)))
calc.kernel()
print(calc.mo_energy)
