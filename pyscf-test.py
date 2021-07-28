#!/usr/bin/env python
from pyscf import scf, gto

import numpy as np

a = 1.809 * np.sin(104.52 / 180 * np.pi / 2) / 1.8897259886
b = 1.809 * np.cos(104.52 / 180 * np.pi / 2) / 1.8897259886
mol = gto.M(atom=f"H {a} 0 0; H {-a} 0 0; O 0 {b} 0", basis="sto-3g")
calc = scf.RHF(mol)
calc.kernel()
print(calc.mo_energy)
