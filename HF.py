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

mpmath.mp.dps = 100


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
            (mpmath.mp.mpf("0"), mpmath.mp.mpf("0"), mpmath.mp.mpf("0")),
            mpmath.mp.mpf("1"),
            ["1s"],
            mpmath.mp.mpf("1") + lval,
        ),
        Atom(
            "H",
            (mpmath.mp.mpf("0"), mpmath.mp.mpf("0"), mpmath.mp.mpf("1.4")),
            mpmath.mp.mpf("1"),
            ["1s"],
            mpmath.mp.mpf("1") - lval,
        ),
    ]
    bs = STO3G(mol)
    N = 2

    maxiter = 100000  # Maximal number of iteration

    # Basis set size
    K = bs.K

    S = S_overlap(bs)

    X = X_transform(S)

    Hc = H_core(bs, mol)
    ee = EE_list(bs)

    Pnew = np.zeros((K, K))
    P = np.zeros((K, K))

    converged = False

    iter = 1
    while not converged and iter <= maxiter:
        Pnew, F, E = RHF_step(bs, mol, N, Hc, X, P, ee, False)  # Perform an SCF step

        # Check convergence of the SCF cycle
        Pnew = (P + Pnew) / 2

        # print(Pnew)
        # print(iter, delta_P(P, Pnew))
        if delta_P(P, Pnew) < mpmath.mp.mpf(f"1e-{mpmath.mp.dps-3}"):
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

    initial = get_energy(mpmath.mp.mpf("0"))
    final = get_energy(mpmath.mp.mpf("1"))
    print("INITIAL", initial)
    print("FINAL", final)

    print("order, total, coefficient, error")

    def taylor(func, around, at, orders, delta):
        @functools.lru_cache(maxsize=100)
        def callfunc(lval):
            return func(lval)

        def format(val):
            return mpmath.nstr(val, 10, strip_zeros=False)

        total = mpmath.mp.mpf("0.0")
        final = callfunc(at)
        for order in range(orders):
            if order == 0:
                weights = np.array([mpmath.mp.mpf("1.0")])
                offsets = np.array([mpmath.mp.mpf("0.0")])
            else:
                stencil = findiff.coefficients(deriv=order, acc=2, symbolic=True)[
                    "center"
                ]
                weights = [
                    mpmath.mp.mpf(_.numerator()) / mpmath.mp.mpf(_.denominator())
                    for _ in stencil["coefficients"]
                ]
                offsets = stencil["offsets"]
            coefficient = sum(
                [
                    callfunc(around + delta * shift) * weight
                    for shift, weight in zip(offsets, weights)
                ]
            ) / delta ** mpmath.mp.mpf(order)
            coefficient *= (at - around) ** mpmath.mp.mpf(order) / mpmath.factorial(
                order
            )
            total += coefficient
            print(order, format(total), format(coefficient), format(total - final))

    # taylor(get_energy, mpmath.mp.mpf("0."), mpmath.mp.mpf("1."), 20, mpmath.mp.mpf("1e-10"))
    coeffs = mpmath.taylor(
        get_energy, mpmath.mp.mpf("0.0"), 15, method="step", direction=0
    )
    total = mpmath.mp.mpf("0.0")

    def format(val):
        return mpmath.nstr(val, 10, strip_zeros=False)

    for order, coeff in enumerate(coeffs):
        total += coeff
        print(order, format(total), format(coeff), format(total - final))
