#%%
from HF import get_energy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASEDIR = "/home/ferchault/tmp"
# %%
def get_data(dps, delta, fn=None, columnname="error"):
    columns = "order total coefficient error".split()
    column = columns.index(columnname)
    if fn is None:
        fn = f"{BASEDIR}/dps-{dps}-delta-{delta}"
    with open(fn) as fh:
        lines = fh.readlines()[5:]
    vals = []
    for line in lines:
        vals.append(float(line.strip().split()[column]))
    return np.array(vals)


# %%
plt.ylim(1e-5, 2)
plt.semilogy(get_data(100, 10))
plt.semilogy(get_data(1000, 10))
plt.semilogy(get_data(10000, 10))

# %%
plt.ylim(1e-5, 2)
plt.semilogy(get_data(1000, 10))
plt.semilogy(get_data(1000, 20))
plt.semilogy(np.abs(get_data(0, 0, fn=f"{BASEDIR}/dps-1000-delta-10-acc-8")))
plt.semilogy(np.abs(get_data(0, 0, fn=f"{BASEDIR}/dps-1000-delta-10-conv-100")))


# %%
plt.ylim(1e-5, 2)
plt.semilogy(get_data(1000, 20))
plt.semilogy(np.abs(get_data(0, 0, fn=f"debug2")))

# %%
get_data(0, 0, fn=f"debug2")
# %%
get_data(0, 0, fn=f"debug")
# %%
def get_diffiqult(fn):
    with open(fn) as fh:
        lines = fh.readlines()

    started = False
    derivs = []
    for line in lines:
        if "derivatives" in line:
            started = True
        if "EnergyTarget" in line:
            started = False
        if started:
            value = float(line.replace("[", " ").replace("]", " ").strip().split()[-1])
            derivs.append(value)
    return derivs


plt.plot(get_diffiqult("diffiqult-h2tohe.log")[:10])
# %%
derivs = get_diffiqult("diffiqult-h2tohe.log")
# %%
import math

total = 0
for order in range(len(derivs)):
    coefficient = derivs[order]
    total += coefficient
    print(order, total)
# %%
plt.plot(get_data(0, 0, fn=f"debug2", columnname="coefficient")[:-3])
plt.plot(derivs[:10])
# %%
get_data(0, 0, fn=f"debug2", columnname="coefficient")[:6] - derivs[:6]
# %%
get_energy(0.0) - -1.830999989123392302842, get_energy(1.0) - -2.6301867144052982315080
# %%
plt.plot(get_diffiqult("diffiqult-h2tohe-sto3g.log")[:10])
plt.plot(get_diffiqult("diffiqult-h2tohe-def2tzvp.log")[:10])
# %%
np.array(get_diffiqult("diffiqult-h2tohe-sto3g.log")[:10]) - np.array(
    get_diffiqult("diffiqult-h2tohe-def2tzvp.log")[:10]
)
# %%
plt.plot(np.cumsum(get_diffiqult("diffiqult-h2tohe-def2tzvp.log")[:10]))
# %%
plt.semilogy(np.abs(get_data(0, 0, fn=f"{BASEDIR}/dps-10000-delta-10")))
# %%
plt.plot(
    np.cumsum(
        get_data(0, 0, fn=f"{BASEDIR}/dps-10000-delta-10", columnname="coefficient")
    )
)

plt.axhline(-2.63018671440529)
# %%
plt.semilogy(np.abs(get_data(0, 0, fn=f"{BASEDIR}/dps-10000-delta-10")))
plt.semilogy(np.abs(get_data(0, 0, fn=f"dps-100-mpmathtaylor")))

# %%
plt.plot(
    np.cumsum(
        get_data(0, 0, fn=f"{BASEDIR}/dps-10000-delta-10", columnname="coefficient")
    )
)
plt.axhline(-2.63018671440529)
# %%
coeffs = get_data(0, 0, fn=f"{BASEDIR}/dps-10000-delta-10", columnname="coefficient")
# %%
plt.plot(coeffs[coeffs > 0])
plt.plot(-coeffs[coeffs < 0][2:])
# %%
xs = coeffs[abs(coeffs) > 1e-5]
plt.plot(abs(xs[:-1]) / abs(xs[1:]))
plt.ylim(0, 5)
# %%
from scipy.interpolate import pade

# %%
o = []
e = []
for orderhalf in range(2, 20):
    p, q = pade(coeffs[: orderhalf * 2], orderhalf)
    o.append(orderhalf * 2)
    e.append(p(1) / q(1) - -2.63018671440529)
# %%
plt.semilogy(o, np.abs(e))
# %%
