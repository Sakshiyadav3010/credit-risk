"""Convert credit_risk.py (percent format) into notebooks/credit_risk.ipynb without extra dependencies."""
import json, re
src = open("credit_risk.py").read()
parts = re.split(r"^# %%", src, flags=re.M)[1:]
cells = []
for p in parts:
    md = p.startswith(" [markdown]")
    body = p.split("\n", 1)[1].strip("\n") if "\n" in p else ""
    if md:
        text = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.splitlines(True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("notebooks/credit_risk.ipynb", "w"), indent=1)
print(len(cells), "cells written")
