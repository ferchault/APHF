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


def test_for_accurate_types(func):
    def wrapper(*args, **kwargs):
        def is_accurate(variable):
            return True
            if type(variable) == tuple:
                for e in variable:
                    if not is_accurate(e):
                        return False
                return True
            elif type(variable) == np.ndarray:
                if len(variable.shape) == 0:
                    return True
                return is_accurate(variable[0])
            else:
                return type(variable).__name__ == "mpf" or type(variable) == int

        for variable in args:
            assert is_accurate(variable)
        for k, v in kwargs:
            assert is_accurate(v)
        retval = func(*args, **kwargs)
        if not is_accurate(retval):
            print(type(retval))
        assert is_accurate(retval)
        return retval

    return wrapper


from numpy.lib.arraysetops import isin
from RHF import *

from matrices import *
from integrals import *
from basis import *

import numpy as np
import numpy.linalg as la
import mpmath

mpmath.mp.dps = 100


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

    maxiter = 100000  # Maximal number of iteration

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
        Pnew, F, E = RHF_step(bs, mol, N, Hc, X, P, ee, verbose)  # Perform an SCF step

        # Check convergence of the SCF cycle
        Pnew = (P + Pnew) / 2

        # print(Pnew)
        # print(iter, delta_P(P, Pnew))
        if delta_P(P, Pnew) < TO_PREC(f"1e-{mpmath.mp.dps-3}"):
            converged = True

        if iter == maxiter:
            print("SCF NOT CONVERGED!", lval)

        P = Pnew

        iter += 1
    return energy_el(P, F, Hc)


if __name__ == "__main__":
    import sys
    import functools
    import findiff
    import subprocess

    print(
        "REVISION",
        subprocess.check_output("git rev-parse HEAD".split()).decode("ascii").strip(),
    )
    print("DPS", mpmath.mp.dps)

    initial = get_energy(TO_PREC("0"))
    final = get_energy(TO_PREC("1"))
    print("INITIAL", initial)
    print("FINAL", final)

    print("order, total, coefficient, error")

    def taylor(func, around, at, orders, delta):
        @functools.lru_cache(maxsize=100)
        def callfunc(lval):
            return func(lval)

        def format(val):
            return mpmath.nstr(val, 10, strip_zeros=False)

        total = TO_PREC("0.0")
        final = callfunc(at)
        for order in range(orders):
            if order == 0:
                weights = np.array([TO_PREC("1.0")])
                offsets = np.array([TO_PREC("0.0")])
            else:
                stencil = findiff.coefficients(deriv=order, acc=2, symbolic=True)[
                    "center"
                ]
                weights = [
                    TO_PREC(_.numerator()) / TO_PREC(_.denominator())
                    for _ in stencil["coefficients"]
                ]
                offsets = stencil["offsets"]
            coefficient = sum(
                [
                    callfunc(around + delta * shift) * weight
                    for shift, weight in zip(offsets, weights)
                ]
            ) / delta ** TO_PREC(order)
            coefficient *= (at - around) ** TO_PREC(order) / mpmath.factorial(order)
            total += coefficient
            print(order, format(total), format(coefficient), format(total - final))

    # taylor(get_energy, TO_PREC("0."), TO_PREC("1."), 20, TO_PREC("1e-10"))
    print(mpmath.taylor(get_energy, TO_PREC("0.0"), 15, method="step", direction=0))
