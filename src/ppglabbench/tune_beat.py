"""Choose the boosting rounds of the beat model: one value for every lab, by subject-grouped CV on the train split only.

The model runs without early stopping, so it is one recipe whatever a lab's sample size. Each round count is ranked
within each lab and task by pooled out-of-fold AUROC or R², with folds centred as in baselines.ppg_score; the lowest mean
rank wins. Validation and test are not read.

Usage:  python -m ppglabbench.tune_beat [--jobs 5]
Output: printed; copy the winner to `benchmark.beat_max_iter` in config.yaml
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from ppglabbench.baselines import K, LOG_LABS, clip_logit, load_mimic

GRID = (25, 50, 75, 100, 150, 200, 300)


def one(d, bcols, name, spec, task):
    col = spec["value"] if task == "reg" else spec["abnormal"]
    q = d[d[col].notna()].reset_index(drop=True)
    y = q[col].to_numpy(float)
    if task == "reg" and name in LOG_LABS:
        q, y = q[y > 0].reset_index(drop=True), np.log(y[y > 0])
    X, groups = q[bcols].to_numpy(float), q.group.to_numpy()
    cls = HistGradientBoostingRegressor if task == "reg" else HistGradientBoostingClassifier
    s = {n: np.zeros(len(y)) for n in GRID}
    for a, b in GroupKFold(K).split(X, groups=groups):
        m = cls(max_iter=max(GRID), learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False,
                random_state=0).fit(X[a], y[a])
        stages = m.staged_predict(X[b]) if task == "reg" else m.staged_predict_proba(X[b])
        for i, p in enumerate(stages, 1):
            if i in s:
                s[i][b] = p + y.mean() - y[a].mean() if task == "reg" else clip_logit(p[:, 1]) + clip_logit(y.mean()) - clip_logit(y[a].mean())
    fn = r2_score if task == "reg" else roc_auc_score
    return dict(lab=name, task=task, n=len(y), **{str(n): fn(y, s[n]) for n in GRID})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args(argv)
    d, bcols, labs, _ = load_mimic()
    d = d[d.split == "train"].reset_index(drop=True)
    r = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(one)(d, bcols, n, s, t) for n, s in labs.items() for t in ("clf", "reg")))
    print(r.round(4).to_string(index=False))
    grid = [str(n) for n in GRID]
    print("\nmean by task:\n", r.groupby("task")[grid].mean().round(4).to_string())
    rank = r[grid].rank(axis=1, ascending=False).mean()
    print("\nmean rank:", rank.round(2).to_dict(), "\nwinner:", rank.idxmin())


if __name__ == "__main__":
    main()
