"""
    Copyright (C) 2015 Rocco Meli, 2021 Guido Falk von Rudorff

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

from basis import *
from integrals import *

import numpy.linalg as la


def S_overlap(basis):
    """
    Compute overlap matrix S.

    INPUT:
        BASIS: basis set
    OUTPUT:
        S: Overlap matrix
    """

    # Size of the basis set
    K = basis.K

    # List of basis functions
    B = basis.basis()

    S = np.array(mpmath.zeros(K, K).tolist())

    for i, b1 in enumerate(B):
        for j, b2 in enumerate(B):
            for a1, d1 in zip(b1["a"], b1["d"]):
                for a2, d2 in zip(b2["a"], b2["d"]):
                    R1 = b1["R"]
                    R2 = b2["R"]

                    tmp = d1.conjugate() * d2
                    tmp *= overlap(
                        b1["lx"],
                        b1["ly"],
                        b1["lz"],
                        b2["lx"],
                        b2["ly"],
                        b2["lz"],
                        a1,
                        a2,
                        R1,
                        R2,
                    )

                    S[i, j] = tmp + S[i, j]
    return S


def X_transform(S):
    """
    Compute the transformation matrix X using canonical orthogonalization.

    INPUT:
        S: Overlap matrix
    OUTPUT:
        X: Transformation matrix

    Source:
        Modern Quantum Chemistry
        Szabo and Ostlund
        Dover
        1989
    """

    s, U = mpmath.mp.eighe(NP2MP(S))
    s = np.array(s)
    U = MP2NP(U)

    s = np.diag(s ** (-mpmath.mp.mpf("1.0") / mpmath.mp.mpf("2.0")))

    X = np.dot(U, s)

    return X


def T_kinetic(basis):
    """
    Compute kinetic matrix T.

    INPUT:
        BASIS: basis set
    OUTPUT:
        T: Kinetic matrix
    """
    # Size of the basis set
    K = basis.K

    # List of basis functions
    B = basis.basis()

    T = np.array(mpmath.zeros(K, K).tolist())

    for i, b1 in enumerate(B):
        for j, b2 in enumerate(B):
            for a1, d1 in zip(b1["a"], b1["d"]):
                for a2, d2 in zip(b2["a"], b2["d"]):
                    R1 = b1["R"]
                    R2 = b2["R"]

                    tmp = d1.conjugate() * d2
                    tmp *= kinetic(
                        b1["lx"],
                        b1["ly"],
                        b1["lz"],
                        b2["lx"],
                        b2["ly"],
                        b2["lz"],
                        a1,
                        a2,
                        R1,
                        R2,
                    )

                    T[i, j] += tmp

    return T


def V_nuclear(basis, atom):
    """
    Compute nuclear-electron potential energy matrix Vn.

    INPUT:
        BASIS: basis set
        ATOM: atom specifications (position and charge)
    OUTPUT:
        VN: Nuclear-attraction matrix for atom ATOM
    """
    # Size of the basis set
    K = basis.K

    # List of basis functions
    B = basis.basis()

    # Nuclear coordinates
    Rn = atom.R

    # Nuclear charge
    Zn = atom.Z

    Vn = np.array(mpmath.zeros(K, K).tolist())

    for i, b1 in enumerate(B):
        for j, b2 in enumerate(B):
            for a1, d1 in zip(b1["a"], b1["d"]):
                for a2, d2 in zip(b2["a"], b2["d"]):
                    R1 = b1["R"]
                    R2 = b2["R"]

                    tmp = d1.conjugate() * d2
                    tmp *= nuclear(
                        b1["lx"],
                        b1["ly"],
                        b1["lz"],
                        b2["lx"],
                        b2["ly"],
                        b2["lz"],
                        a1,
                        a2,
                        R1,
                        R2,
                        Rn,
                        Zn,
                    )

                    Vn[i, j] += tmp

    return Vn


def H_core(basis, molecule):
    """
    Compute core Hamiltonian (sum of T and all the VN)

    INPUT:
        BASIS: basis set
        MOLECULE: molecule, collection of atom objects
    OUTPUT:
        (T + VN): Core Hamitlonian
    """
    T = T_kinetic(basis)

    # print("Kinetic energy")
    # print(T)

    # Size of the basis set
    K = basis.K

    Vn = np.array(mpmath.zeros(K, K).tolist())

    Vnn = np.array(mpmath.zeros(K, K).tolist())

    for atom in molecule:
        Vnn = V_nuclear(basis, atom)

        # print("Nuclear attraction Vn")
        # print(Vnn)

        Vn += Vnn

    # print("Total nuclear attraction matrix")
    # print(Vn)

    return T + Vn


def P_density(C, N):
    """
    Compute dansity matrix.

    INPUT:
        C: Matrix of coefficients
        N: Number of electrons
    OUTPUT:
        P: density matrix

    Source:
        Modern Quantum Chemistry
        Szabo and Ostlund
        Dover
        1989
    """

    # Size of the basis set
    K = C.shape[0]

    P = np.array(mpmath.zeros(K, K).tolist())

    for i in range(K):
        for j in range(K):
            for k in range(int(N / 2)):  # TODO Only for RHF
                P[i, j] += 2 * C[i, k] * C[j, k].conjugate()

    return P


def G_ee_cache(K, ee):
    Gfactor = np.zeros((K, K, K, K)).astype(mpmath.mp.mpf)
    q = mpmath.mp.mpf("0.5")
    for i in range(K):
        for j in range(K):
            for k in range(K):
                for l in range(K):
                    Gfactor[i, j, k, l] = ee[i, j, k, l] - q * ee[i, l, k, j]
    return Gfactor


def G_ee(basis, Gfactor, P):
    """
    Compute core Hamiltonian matrix.

    INPUT:
        BASIS: Basis set.
        P: Density matrix
        EE: Two-electron integrals
    OUTPUT:
        G: Electron-electron interaction matrix
    """

    # Size of the basis set
    K = basis.K

    G = np.array(mpmath.zeros(K, K).tolist())

    q = mpmath.mp.mpf("0.5")
    for i, j, k, l in it.product(range(K), repeat=4):
        G[i, j] += P[k, l] * Gfactor[i, j, k, l]

    return G


if __name__ == "__main__":

    """
    Results compared with

        Modern Quantum Chemistry
        Szabo and Ostlund
        Dover
        1989

    and

        The Mathematica Journal
        Evaluation of Gaussian Molecular Integrals
        I. Overlap Integrals
        Minhhuy H?? and Julio Manuel Hern??ndez-P??rez
        2012

    and

        The Mathematica Journal
        Evaluation of Gaussian Molecular Integrals
        II. Kinetic-Energy Integrals
        Minhhuy H?? and Julio Manuel Hern??ndez-P??rez
        2013

    and

        The Mathematica Journal
        Evaluation of Gaussian Molecular Integrals
        III. Nuclear-Electron attraction Integrals
        Minhhuy H?? and Julio Manuel Hern??ndez-P??rez
        2014
    """

    # H2
    H2 = [Atom("H", (0, 0, 0), 1, ["1s"]), Atom("H", (0, 0, 1.4), 1, ["1s"])]

    # Create the basis set
    sto3g_H2 = STO3G(H2)

    # Compute matrices
    S_H2 = S_overlap(sto3g_H2)
    T_H2 = T_kinetic(sto3g_H2)
    Vn1_H2 = V_nuclear(sto3g_H2, H2[0])
    Vn2_H2 = V_nuclear(sto3g_H2, H2[1])
    H_core_H2 = H_core(sto3g_H2, H2)

    print("###########")
    print("H2 molecule")
    print("###########")

    print("\nOverlap matrix S:")
    print(S_H2)

    print("\nKinetic matrix T:")
    print(T_H2)

    print("\nElectron-nucleus interaction " + H2[0].name + " :")
    print(Vn1_H2)

    print("\nElectron-nucleus interaction " + H2[1].name + " :")
    print(Vn2_H2)

    print("\nCore Hamiltonian:")
    print(H_core_H2)

    # HeH+
    HeH = [Atom("H", (0, 0, 0), 1, ["1s"]), Atom("He", (0, 0, 1.4632), 2, ["1s"])]

    # Create the basis set
    sto3g_HeH = STO3G(HeH)

    # Compute matrices
    S_HeH = S_overlap(sto3g_HeH)
    T_HeH = T_kinetic(sto3g_HeH)
    Vn1_HeH = V_nuclear(sto3g_HeH, HeH[0])
    Vn2_HeH = V_nuclear(sto3g_HeH, HeH[1])
    H_core_HeH = H_core(sto3g_HeH, HeH)

    print("\n\n\n")
    print("############")
    print("HeH molecule")
    print("############")

    print("\nOverlap matrix S:")
    print(S_HeH)

    print("\nKinetic matrix T:")
    print(T_HeH)

    print("\nElectron-nucleus interaction " + HeH[0].name + " :")
    print(Vn1_HeH)

    print("\nElectron-nucleus interaction " + HeH[1].name + " :")
    print(Vn2_HeH)

    print("\nCore Hamiltonian:")
    print(H_core_HeH)

    # H2O
    H2O = [
        Atom("H", (0, +1.43233673, -0.96104039), 1, ["1s"]),
        Atom("H", (0, -1.43233673, -0.96104039), 1, ["1s"]),
        Atom("O", (0, 0, 0.24026010), 8, ["1s", "2s", "2p"]),
    ]

    sto3g_H2O = STO3G(H2O)

    # Overlap matrix
    S_H2O = S_overlap(sto3g_H2O)
    T_H2O = T_kinetic(sto3g_H2O)
    Vn1_H2O = V_nuclear(sto3g_H2O, H2O[0])
    Vn2_H2O = V_nuclear(sto3g_H2O, H2O[1])
    Vn3_H2O = V_nuclear(sto3g_H2O, H2O[2])

    Vn_H2O = Vn1_H2O + Vn2_H2O + Vn3_H2O

    H_core_H2O = H_core(sto3g_H2O, H2O)

    print("\n\n\n")
    print("############")
    print("H2O molecule")
    print("############")

    print("\nOverlap matrix S:")
    print(S_H2O)

    print("\nKinetic matrix T:")
    print(T_H2O)

    print("\nTotal electron-nucleus interaction:")
    print(Vn_H2O)

    print("\nCore hamiltonian:")
    print(H_core_H2O)
