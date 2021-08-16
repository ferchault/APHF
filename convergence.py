#%%
import numpy as np
import configparser
import scipy.interpolate as sci
import mpmath
import pandas as pd
import warnings
import glob

warnings.filterwarnings("ignore")

#%%
def padesplit(coeffs, target):
    orders = []
    estimates = []
    for order in range(0, len(coeffs)):
        best = None
        for n in range(0, order):
            for m in range(0, order):
                try:
                    p, q = sci.pade(coeffs[:order], m, n)
                except:
                    continue

                estimate = p(1) / q(1)
                if best is None or abs(estimate - target) < abs(best - target):
                    best = estimate
        if best is not None:
            orders.append(order - 1)
            estimates.append(best)

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

        # work around bug in old versions of calculator
        stencils[0] = {0: "1.0"}
        self._stencils = stencils

    def _read_data(self):
        self._update_accuracy()

        self._data = {"energy_0": {}}
        for label, value in self._config["singlepoints"].items():
            parts = label.split("_")
            if parts[0] == "energy":
                self._data["energy_0"][parts[1]] = value

            if parts[0] == "moenergy":
                _, offset, moid = parts
                key = f"moenergy_{moid}"
                if key not in self._data:
                    self._data[key] = {}
                self._data[key][offset] = value

            if parts[0] == "dm":
                _, offset, i, j = parts
                key = f"dm_{i}_{j}"
                if key not in self._data:
                    self._data[key] = {}
                self._data[key][offset] = value

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

    def get_keys_by_group(self, group):
        return [_ for _ in self._data.keys() if _.startswith(f"{group}_")]


dfs = []
for filename in glob.glob("PROD/*/*.out"):
    for group in "energy dm moenergy".split():
        print(filename, group)
        try:
            c = Calculation(filename)
        except:
            continue

        rows = []
        for key in c.get_keys_by_group(group):
            target = float(c.get_target(key))

            coeffs = np.array([float(_) for _ in c.get_coefficients(key)])
            xs, ys = padesplit(coeffs, target)
            for x, y in zip(xs, abs(ys - target)):
                rows.append(
                    {
                        "order": x,
                        "error": y,
                        "method": "pade",
                        "fn": filename,
                        "group": group,
                        "key": key,
                    }
                )

            for order, val in enumerate(abs(np.cumsum(coeffs) - target)):
                rows.append(
                    {
                        "order": order,
                        "error": val,
                        "method": "taylor",
                        "fn": filename,
                        "group": group,
                        "key": key,
                    }
                )

        dfs.append(pd.DataFrame(rows))
df = pd.concat(dfs)
df.to_csv("convergence.csv")
# %%
