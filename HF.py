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

from numpy.lib.arraysetops import isin
from RHF import *

from matrices import *
from integrals import *
from basis import *

import numpy as np
import numpy.linalg as la
import mpmath

mpmath.mp.dps = 50


class CarefulFloat(mpmath.mpf):
    @staticmethod
    def _is_careful(other):
        return type(other).__name__ in ("CarefulFloat", "mpf")

    @staticmethod
    def _accept_operator(other):
        if CarefulFloat._is_careful(other):
            return True

        if isinstance(other, np.ndarray) and CarefulFloat._is_careful(
            other.reshape(-1)[0]
        ):
            return True

        return False

    def __add__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__add__(self, other)

    def __radd__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__radd__(self, other)

    def __sub__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__sub__(self, other)

    def __rsub__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__rsub__(mpmath.mpf(self), other)

    def __mul__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__mul__(self, other)

    def __rmul__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__rmul__(self, other)

    def __div__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__div__(self, other)

    def __rdiv__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__rdiv__(self, other)

    def __mod__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__mod__(self, other)

    def __rmod__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__rmod__(self, other)

    def __pow__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__pow__(self, other)

    def __rpow__(self, other):
        if not CarefulFloat._accept_operator(other):
            raise ValueError("Invalid op")

        return mpmath.mpf.__rpow__(self, other)


TO_PREC = mpmath.mp.mpf


def NP2MP(array):
    return mpmath.mp.matrix(array.tolist())


def MP2NP(array):
    return np.array(array.tolist())


###########################
###########################
###########################


def get_energy(lval):
    mol = [
        Atom(
            "H",
            (TO_PREC("0"), TO_PREC("0"), TO_PREC("0")),
            TO_PREC("1"),
            ["1s"],
            TO_PREC("1") + lval,
        ),
        Atom(
            "H",
            (TO_PREC("0"), TO_PREC("0"), TO_PREC("1.4")),
            TO_PREC("1"),
            ["1s"],
            TO_PREC("1") - lval,
        ),
    ]
    bs = STO3G(mol)
    N = 2

    maxiter = 100  # Maximal number of iteration

    verbose = False  # Print each SCF step

    ###########################
    ###########################
    ###########################

    # Basis set size
    K = bs.K

    if verbose:
        print("Computing overlap matrix S...")
    S = S_overlap(bs)

    if verbose:
        print(S)

    if verbose:
        print("Computing orthogonalization matrix X...")
    X = X_transform(S)

    if verbose:
        print(X)

    if verbose:
        print("Computing core Hamiltonian...")
    Hc = H_core(bs, mol)

    if verbose:
        print(Hc)

    if verbose:
        print("Computing two-electron integrals...")
    ee = EE_list(bs)

    if verbose:
        print_EE_list(ee)

    Pnew = np.zeros((K, K))
    P = np.zeros((K, K))

    converged = False

    if verbose:
        print("   ##################")
        print("   Starting SCF cycle")
        print("   ##################")

    iter = 1
    while not converged and iter <= maxiter:
        if verbose:
            print("\n\n\n#####\nSCF cycle " + str(iter) + ":")
            print("#####")

        Pnew, F, E = RHF_step(bs, mol, N, Hc, X, P, ee, verbose)  # Perform an SCF step

        # Print results of the SCF step
        if verbose:
            print("\nTotal energy:", energy_tot(P, F, Hc, mol), "\n")
            print("   Orbital energies:")
            print("   ", np.diag(E))

        # Check convergence of the SCF cycle
        if delta_P(P, Pnew) < 1e-12:
            converged = True

            if verbose:
                print(
                    "\n\n\nTOTAL ENERGY:", energy_tot(P, F, Hc, mol)
                )  # Print final, total energy

        if iter == maxiter:
            print("SCF NOT CONVERGED!")

        P = Pnew

        iter += 1
    return energy_el(P, F, Hc)


if __name__ == "__main__":
    import sys

    print(get_energy(TO_PREC("0")))
    print(get_energy(TO_PREC("1")))
