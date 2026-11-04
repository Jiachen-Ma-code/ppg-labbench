"""Positive control: how much of a weak signal of known strength does the increment pipeline recover?

The real PPG is destroyed by a subject-block permutation and a synthetic signal of known strength is planted in it: a new
column for the beat model, a direction inside the top principal components for PaPaGei and Pulse-PPG (a lone column would never
survive PCA). The same score and stack then run. `recovered` is the resulting increment; `oracle` is the increment from adding the
synthetic signal straight to the base. Without it, "Δ ≈ 0" cannot be read as anything but "this recipe extracted nothing".
The noise is per window, unlike real PPG, so read this as a guide to sensitivity. Train -> val only.

The CNN cannot be planted this way: a signal in its input would need it trained again. Its control is the two heads it fits beside the labs, age and PPG heart rate: how much of each it
recovers out of fold on train and from the final model on val.

Usage:  python -m ppglabbench.positive_control [--seeds 3]
Output: data/benchmark/results/mimic_val/{positive_control.csv, positive_control_cnn.csv}, summary printed
"""
import argparse
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ppglabbench.baselines import OUT, PG, PP, PPG_MODELS, base_clf, base_reg, block_perm, load_mimic, ppg_score, task_frame

LABS = ("potassium", "haemoglobin", "platelets", "inr", "lactate", "egfr")   # rare to common, large to small
STRENGTHS = {"clf": (0.60, 0.70), "reg": (0.2, 0.4)}                         # synthetic AUROC; correlation
EMBEDDINGS = {"papagei": PG, "pulseppg": PP}


def synthetic(y, task, s, rng):
    z = rng.standard_normal(len(y))
    if task == "clf":
        return z + np.sqrt(2) * norm.ppf(s) * y
    return s * (y - y.mean()) / y.std() + np.sqrt(1 - s ** 2) * z


def plant(X, x, ppg, tr, rng):
    if ppg == "beat":
        return np.column_stack([X, x])
    X = SimpleImputer().fit(X[tr]).transform(X)    # a window Pulse-PPG could not embed is missing: filled as the probe fills it
    Z = StandardScaler().fit(X[tr]).transform(X)
    u = PCA(64, random_state=0).fit(Z[tr]).components_.T @ rng.standard_normal(64)
    u /= np.linalg.norm(u)
    along = Z @ u
    return Z + np.outer((x - x.mean()) / x.std() * along[tr].std() + along.mean() - along, u)


def one(d, bcols, tiers, labs, name, task, ppg, s, seed):
    frame = task_frame(d, tiers, name, labs[name], task, "val")
    if frame is None:
        return []
    q, y, tr, ev, groups, bases, _, _ = frame
    rng = np.random.default_rng([seed, zlib.crc32(f"{name}|{task}|{ppg}|{s}".encode())])
    blocks = [[g.to_numpy() for _, g in pd.Series(i).groupby(groups[i])] for i in (tr, ev)]
    X = q[bcols if ppg == "beat" else EMBEDDINGS[ppg]].to_numpy(float)[block_perm(blocks, len(q), rng)]
    x = synthetic(y, task, s, rng)
    s_tr, s_ev = ppg_score(PPG_MODELS[ppg][task], plant(X, x, ppg, tr, rng), y, groups, tr, ev)
    model, metric, floor = (base_reg, r2_score, None) if task == "reg" else (base_clf, roc_auc_score, 0.5)
    pred = (lambda m, A: m.predict(A)) if task == "reg" else (lambda m, A: m.predict_proba(A)[:, 1])

    def score(Z):
        return metric(y[ev], pred(model().fit(Z[tr], y[tr]), Z[ev]))

    rows = []
    sc = np.zeros(len(q))
    sc[tr], sc[ev] = s_tr, s_ev
    for b, B in bases.items():
        base = score(B)
        base = base if floor is None else max(base, floor)
        rows.append(dict(lab=name, task=task, ppg=ppg, strength=s, base=b, seed=seed, base_score=base,
                         recovered=score(np.column_stack([B, sc])) - base, oracle=score(np.column_stack([B, x])) - base))
    return rows


def cnn_heads(d):
    """R² of the CNN's age and heart-rate heads, on train (out of fold) and validation. Rows without the CNN's columns give nothing. Test rows have no score before the split is opened
    and are not read here after it."""
    rows = []
    for split, x in d[d.split != "test"].groupby("split"):
        for target, col, truth in (("age", "cnn_age_pred", "age"), ("heart_rate", "cnn_hr_pred", "hr")):
            if col in x:
                ok = x[col].notna() & x[truth].notna()
                rows.append(dict(split=split, target=target, windows=int(ok.sum()), r2=r2_score(x[truth][ok], x[col][ok])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()
    d, bcols, labs, tiers = load_mimic()
    d = d[d.split.isin(["train", "val"])].reset_index(drop=True)
    models = ["beat"] + [m for m, cols in EMBEDDINGS.items() if cols[0] in d]
    jobs = [(n, t, p, s, k) for n in LABS for t in ("clf", "reg") for p in models
            for s in STRENGTHS[t] for k in range(args.seeds)]
    res = Parallel(n_jobs=args.jobs)(delayed(one)(d, bcols, tiers, labs, *j) for j in jobs)
    r = pd.DataFrame([x for rows in res for x in rows])
    out = OUT / "mimic_val"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "positive_control.csv", index=False)
    heads = cnn_heads(d)
    if len(heads):
        heads.to_csv(out / "positive_control_cnn.csv", index=False)
        print(heads.round(3).to_string(index=False))
    m = r.groupby(["task", "strength", "base", "ppg"])[["oracle", "recovered"]].mean().round(4)
    m["share"] = (m.recovered / m.oracle).round(2)
    print(m.to_string())


if __name__ == "__main__":
    main()
