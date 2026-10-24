#!/usr/bin/env python3
"""Populate cached_datasets/ for the 9 SOURCE tables.

The downstream pipeline caches only the 5 targets. The TransTab baseline's
Stage 0 needs the sources too, but lattle.load_data's CSV fallback only knows
'stroke' and 'seismic_bumps' and looks in the CWD rather than ./data.

Rather than edit dataset.py (shared with the running LATTLE grid, which reloads
it for every fold), this monkey-patches load_dataset for the duration of the
caching run only. Everything downstream of load_dataset - imputation, MinMax
scaling fit on train, ordinal/binary encoding, the 70/10/20 stratified split -
is the pipeline's own code, so cached sources are preprocessed identically to
cached targets.

Usage: python cache_sources.py [n_folds]
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

from transfer_learning_module.lattle import dataset as _ds
from transfer_learning_module import lattle

SOURCES = ["credit-g", "Cardiovascular-Disease-dataset", "optdigits", "mfeat-fourier",
           "sick", "churn", "car", "steel-plate-fault", "seismic_bumps"]

_orig_load_dataset = _ds.load_dataset


def _generic_csv_load(dataname):
    """Read ./data/<name>.csv the way Step 1's utils.get_dataset does:
    last column is the target, categorical inferred from dtype."""
    path = os.path.join("data", f"{dataname}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    categorical_indicator = [
        bool(dt == np.dtype("O") or pd.api.types.is_string_dtype(dt))
        for dt in X.dtypes.tolist()
    ]
    return X, y, categorical_indicator, X.columns.tolist()


def patched_load_dataset(dataname):
    """Try the pipeline's own loader; fall back to ./data/<name>.csv."""
    try:
        out = _orig_load_dataset(dataname)
        if out is not None and out[0] is not None:
            return out
    except Exception as e:
        print(f"   (stock loader failed for {dataname}: {type(e).__name__}) -> ./data fallback")
    return _generic_csv_load(dataname)


def main():
    n_folds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    _ds.load_dataset = patched_load_dataset
    os.makedirs("cached_datasets", exist_ok=True)
    made = skipped = failed = 0
    for src in SOURCES:
        for fold in range(n_folds):
            out = os.path.join("cached_datasets", f"{src}_seed{fold}.pkl")
            if os.path.exists(out):
                skipped += 1
                continue
            try:
                res = lattle.load_data([src], seed=fold)
                with open(out, "wb") as f:
                    pickle.dump(res, f)
                made += 1
            except Exception as e:
                print(f"   FAILED {src} fold {fold}: {type(e).__name__}: {e}")
                failed += 1
        # report class count once per source
        p = os.path.join("cached_datasets", f"{src}_seed0.pkl")
        if os.path.exists(p):
            _, tr, _, _, _, _, _ = pickle.load(open(p, "rb"))
            n = int(pd.Series(tr[0][1]).nunique())
            print(f"  {src:<32} {n} classes, {len(tr[0][1])} train rows")
    print(f"\ncached: {made} new, {skipped} existing, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
