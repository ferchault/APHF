#!/usr/bin/env python
#%%
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
print(calc.energy_elec()[0])
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


# %%
def get_energy(lval):
    def add_qmmm(calc, mol, deltaZ):
        mf = qmmm.mm_charge(calc, mol.atom_coords() / 1.8897259886, deltaZ)

        def energy_nuc(self):
            q = mol.atom_charges().astype(float)
            q += deltaZ
            return mol.energy_nuc(q)

        mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

        return mf

    mol = gto.M(atom=f"H 0 0 0; H {1.4/1.8897259886} 0 0", basis="sto-3g", verbose=0)
    calc = add_qmmm(scf.RHF(mol), mol, float(lval) * np.array((1, -1)))
    calc.kernel()
    return calc.energy_elec()[0]


#%%
import findiff
import functools


def taylor(func, around, at, orders, delta):
    @functools.lru_cache(maxsize=100)
    def callfunc(lval):
        return func(lval)

    total = 0
    final = callfunc(at)
    for order in range(orders):
        if order == 0:
            weights = np.array([1.0])
            offsets = np.array([0.0])
        else:
            stencil = findiff.coefficients(deriv=order, acc=2)["center"]
            weights = stencil["coefficients"]
            offsets = stencil["offsets"]
        coefficient = (
            sum(
                [
                    callfunc(around + delta * shift) * weight
                    for shift, weight in zip(offsets, weights)
                ]
            )
            / delta ** order
        )
        coefficient *= (at - around) ** order / np.math.factorial(order)
        total += coefficient
        print(order, coefficient, total, abs(total - final))


taylor(get_energy, 0, 1, 8, 0.001)


# %%
# %%
get_energy(1)


# %%
