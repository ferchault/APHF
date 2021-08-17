#!/usr/bin/env python
#%%
import basis_set_exchange as bse
import click

# %%
@click.command()
@click.argument("element")
def find_compatible_basissets(element):
    found = {}
    Z = bse.lut.element_Z_from_sym("N")
    for basis in bse.get_all_basis_names():
        try:
            db = bse.get_basis(basis, element)
        except:
            continue
        try:
            db = db["elements"][str(Z)]["electron_shells"]
        except:
            continue

        works = True
        count = 0
        for shell in db:
            if shell["function_type"] != "gto":
                works = False

            for angmom, coeffs in zip(shell["angular_momentum"], shell["coefficients"]):
                if angmom > 1:
                    works = False
                if angmom == 0:
                    count += 1
                if angmom == 1:
                    count += 3
        if count * 2 < Z:
            works = False
        if works:
            found[basis] = count

    for k, v in sorted(found.items(), key=lambda x: x[1]):
        print(k)


if __name__ == "__main__":
    find_compatible_basissets()
