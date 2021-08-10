#%%
import matplotlib.pyplot as plt
import numpy as np
import configparser
import scipy.interpolate as sci
import mpmath

#%%
def padesplit(coeffs):
    orders = []
    estimates = []
    for order in range(0, len(coeffs)):
        try:
            p, q = sci.pade(coeffs[:order], int(order / 2))
        except:
            continue

        orders.append(order)
        estimates.append(p(1) / q(1))
    return orders, np.array(estimates)


class Calculation:
    def __init__(self, filename):
        config = configparser.ConfigParser()
        with open(filename) as fh:
            config.read_file(fh)
        self._config = config
        self._maxorder = self._config["meta"].getint("orders")
        self._read_stencil()
        self._read_data()

    def _update_accuracy(self):
        mpmath.mp.dps = self._config["meta"].getint("dps")

    def _read_stencil(self):
        self._update_accuracy()

        stencils = {}
        for order in range(self._maxorder):
            stencils[order] = {}

        for label, value in self._config["stencil"].items():
            _, order, offset = label.split("_")
            stencils[int(order)][int(offset)] = value

        self._stencils = stencils

    def _read_data(self):
        self._update_accuracy()

        self._data = {"energy": {}}
        for label, value in self._config["singlepoints"].items():
            parts = label.split("_")
            if parts[0] == "energy":
                self._data["energy"][parts[1]] = value

    def get_target(self, key):
        return self._data[key]["target"]

    def get_electronic_energy_coefficients(self):
        self._update_accuracy()

        coefficients = []
        for order in range(self._maxorder):
            coeff = float(self._config["coefficients"][f"order-{order}"])
            coefficients.append(coeff)
        return np.array(coefficients)

    def get_coefficients(self, key):
        self._update_accuracy()

        step = mpmath.mpf(f'1e-{self._config["meta"].getint("deltalambda")}')

        coefficients = []
        for order in range(self._maxorder):
            stencil = self._stencils[order]
            coefficient = sum(
                [
                    mpmath.mp.mpf(self._data[key][str(shift)]) * mpmath.mp.mpf(weight)
                    for shift, weight in stencil.items()
                ]
            ) / step ** mpmath.mp.mpf(order)
            coefficient /= mpmath.factorial(order)
            coefficients.append(str(coefficient))

        return coefficients

    def get_electronic_energy_target(self):
        self._update_accuracy()

        return float(self._config["singlepoints"]["energy_target"])


for bs in "STO3G STO6G def2TZVP ccpvdz def2SVP 631G".split():
    c = Calculation(f"PROD/H2/dps-1000-{bs}.out")

    target = float(c.get_target("energy"))
    coeffs = np.array([float(_) for _ in c.get_coefficients("energy")])
    xs, ys = padesplit(coeffs)
    plt.semilogy(xs, abs(ys - target), label=bs)
plt.legend()


# %%
