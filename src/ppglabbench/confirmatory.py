"""Confirmatory analysis of the MIMIC test split (PROTOCOL.md).

The question: could PPG replace or trigger a draw between samples? Family A asks what PPG adds beyond what is known at the bedside: 10 labs, AUROC over the `now` base, one
PPG model per lab, with a fixed sequence inside each lab (L1 partial association, L2 gain above 0, L3 gain above +0.01). Two named hypotheses ask the same of the lab value:
lactate and haemoglobin, T1, ΔR² over `now`. They share one-sided alpha = 0.025 in a weighted graphical procedure: 0.02 for Family A and 0.005 for the named pair.
The headline summary is the mean gain over Family A with a subject-bootstrap interval. The upper-bound chain tests, at one-sided 0.025, that the true mean gain over Family A is below a
margin, first over `full`, then `now`, `monitor` and `acuity`, each only if the one before passed. A mean can hide one lab: the per-lab upper bounds, descriptive, and L1 to L3 address that.
The margin comes from what routine bedside measurements add on train and val, never from PPG. The other labs are reported with their gain and SE and are not tested.

Beside each lab's pick, as descriptive rows that enter no test: the `rate` model's L1 z and gain over `now`, the pick's gain over `now` + the rate score, the gain by age of the
previous value, a residual arm, Brier skill, partial R², calibration, net benefit and subgroups with their heterogeneity. Within-subject and between-subject summaries are left to
ppglabbench.tracking, a separate step that needs this evaluation to have been done, with its own scores, permutations and output folder.

Steps:
    plan     train and val only. Model picks, r-hat per stratum and the margin -> data/benchmark/results/confirmatory_plan.json
    dry-run  the whole of `run` with val standing in for test -> data/benchmark/results/confirmatory_dryrun/
    run      the one real evaluation -> data/benchmark/results/confirmatory_test/{results.json, report.md, predictions.npz, manifest.json}
    render   the report again from results.json in that folder, reading no data

A descriptive row that raises is recorded in results.json with its error's type and place, and what it sits beside keeps its result. A lab that raises in its own tests cannot be computed: not rejected,
still counted. results.json is in its final folder before the predictions, manifest and report are written, so none of those can lose it.


Usage:
    python -m ppglabbench.confirmatory plan [--window pre] [--B 50] [--jobs 3]
    python -m ppglabbench.confirmatory dry-run [--jobs 3]
    python -m ppglabbench.confirmatory run --open-test [--jobs 3]
    python -m ppglabbench.confirmatory render [--dry-run]

`run` refuses unless baselines.check_test_allowed passes, the plan exists, its SHA-256 is `benchmark.plan_sha256` in config.yaml, the CNN score file is the one the plan used, the plan is
not provisional (all models, B = 50, a clean tree, the primary `pre` window), the code has not changed since the plan, a dry run of this plan finished with nothing failed, and `ppglabbench.baselines --eval test` has not been run: the
confirmatory evaluation comes first.
The window is read from the plan, so dry-run and run cannot use another one. `plan` and `run` refuse a missing CNN score file on `pre`. Only `run` keeps the CNN's scores on test rows.

An outcome count on eval rows never changes the family. baselines.task_frame would drop such a lab, so its eval guards are switched
off here; a lab with too few eval subjects in a class stays in the family, is not tested, and counts as not rejected.

predictions.npz stays under data/ (DUA data, not committed). It holds no subject IDs: rows are shuffled per lab and subjects are
replaced by a random integer.
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import traceback
import zlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import chi2, norm, rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml

from ppglabbench import baselines as BL
from ppglabbench.common import DATA, ROOT, git_version, write_manifest

# Tested labs, k = 10. Rule: window coverage >= 70% on non-test rows, plus eGFR. The other labs and creatinine are descriptive: creatinine
# carries the same information as eGFR.
FAMILY_LABS = ("bun", "sodium", "potassium", "bicarbonate", "calcium", "glucose", "haemoglobin", "platelets", "wbc", "egfr")
DESCRIPTIVE_LABS = ("inr", "ptt", "lactate", "albumin", "ast", "alt", "creatinine")
NAMED_LABS = ("lactate", "haemoglobin")      # T1 hypotheses: the lab value, ΔR² over `now`
PRIMARY_WINDOW = "pre"
MODELS = ("papagei", "beat", "pulseppg", "cnn")     # candidates per Family A lab
VALUE_MODELS = ("papagei", "beat", "pulseppg")       # candidates for the named hypotheses: the CNN has no head for the value
TIE_ORDER = ("beat", "pulseppg", "cnn", "papagei")   # gains equal at 3 decimals: the simplest first, the one pretrained on these cohorts last
RATE = "rate"                                # reported beside the pick, never a candidate
PRIMARY, UPPER = "now", "full"
CHAIN = ("full", "now", "monitor", "acuity") # the upper-bound chain, from most to least known at the draw
ALPHA, ALPHA_A, ALPHA_NAMED, GAIN_FLOOR, MIN_GAIN = 0.025, 0.02, 0.005, 0.5, 0.01
B_REFITS, SUB_F, N_SPLITS = 50, 0.8, 30
MARGIN_STEP, MARGIN_GROUPS = 0.001, dict(map=("map_mmhg", "map_lt70"), pressors=("vasopressor", "n_pressors", "epi_on"), nbp=("nbp_mean", "nbp_sys", "nbp_dias"))
N_MEAN_BOOT = 2000
PREVIOUS_AGE = (("le_12h", -np.inf, 12), ("12_24h", 12, 24), ("24_48h", 24, 48), ("gt_48h", 48, np.inf))     # hours; never measured is a stratum of its own
AGE_BANDS = (("lt_50", 0, 50), ("50_64", 50, 65), ("65_79", 65, 80), ("ge_80", 80, np.inf))
NET_BENEFIT_AT = (0.05, 0.10, 0.20, 0.30, 0.50)
MIN_SUBJECTS = BL.MIN_POS_EVAL               # fewer eval subjects than this in a class: not tested, not rejected, still in k
clip_logit = BL.clip_logit
OUT = BL.OUT
PLAN = OUT / "confirmatory_plan.json"
TEST_OUT = OUT / "confirmatory_test"
DRY_OUT = OUT / "confirmatory_dryrun"
ATTEMPTS = OUT / "confirmatory_test_attempts.log"
FROZEN_PATHS = ("src", "config.yaml")
AFTER_THE_PLAN = ("plan_sha256", "test_opened")     # the keys of config.yaml's benchmark block that change after the plan


# ------------------------------------------------------------------------------------------------- statistics

def auc_influence(y, p, g, G):
    """AUROC and its per-subject influence terms T_g (Obuchowski 1997): Var = G/(G-1) sum T_g^2. Ties count one half.
    y is boolean, g holds subject positions 0..G-1."""
    y, p = np.asarray(y, bool), np.asarray(p, float)
    pos, neg = p[y], p[~y]
    n1, n0 = len(pos), len(neg)
    r = rankdata(np.r_[pos, neg])
    v10 = (r[:n1] - rankdata(pos)) / n0        # share of negatives below each positive
    v01 = 1 - (r[n1:] - rankdata(neg)) / n1    # share of positives above each negative
    auc = v10.mean()
    t = np.bincount(g[y], (v10 - auc) / n1, minlength=G) + np.bincount(g[~y], (v01 - auc) / n0, minlength=G)
    return auc, t


def cluster_se(t, G):
    return float(np.sqrt(G / (G - 1) * np.sum(t * t)))


def delta_auc(y, stack, base, g, G, floor=GAIN_FLOOR):
    """Gain = AUROC(stack) - max(AUROC(base), floor) and its cluster DeLong SE. A sample base below the floor contributes no
    influence terms, because the floor is a constant."""
    a1, t1 = auc_influence(y, stack, g, G)
    a0, t0 = auc_influence(y, base, g, G)
    below = floor is not None and a0 < floor
    d, t = (a1 - floor, t1) if below else (a1 - a0, t1 - t0)
    return dict(delta=float(d), se=cluster_se(t, G), a_stack=float(a1), a_base=float(a0))


def safe_z(num, se):
    """NaN when the SE is zero or missing: such a test never rejects."""
    return float(num / se) if np.isfinite(num) and np.isfinite(se) and se > 1e-12 else float("nan")


def eval_design(Z):
    """Eval-row design for L1: median imputation, one missing indicator per distinct missingness pattern, no constant columns."""
    Z = np.asarray(Z, float)
    cols, seen = [], set()
    for j in range(Z.shape[1]):
        c = Z[:, j]
        miss = ~np.isfinite(c)
        if miss.all():
            continue
        if miss.any():
            c = np.where(miss, np.nanmedian(c), c)
            if miss.tobytes() not in seen:
                seen.add(miss.tobytes())
                cols.append(miss.astype(float))
        cols.append(c)
    out = np.column_stack(cols) if cols else np.empty((len(Z), 0))
    return out[:, out.std(0) > 1e-12] if out.shape[1] else out


def scaled_design(Z, n):
    """An intercept and the standardised columns of Z."""
    Z = np.asarray(Z, float).reshape(n, -1)
    return np.column_stack([np.ones(n), (Z - Z.mean(0)) / Z.std(0)]) if Z.shape[1] else np.ones((n, 1))


def fit_logistic(y, X, offset=0.0):
    """Logistic maximum likelihood by damped Newton steps, so that a quasi-separated fit cannot jump. Returns the coefficients, fitted probabilities and working weights."""
    k = X.shape[1]
    ridge = 1e-8 * np.eye(k)
    b = np.zeros(k)
    b[0] = np.log(y.mean() / (1 - y.mean()))
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(X @ b + offset, -30, 30)))
        W = p * (1 - p) + 1e-10
        step = np.linalg.solve((X * W[:, None]).T @ X + ridge, X.T @ (y - p))
        big = np.max(np.abs(step))
        if not np.isfinite(big):        # a design this close to separation stops at the last finite coefficients, so that one numerical failure cannot end a lab's sequence
            break
        b = b + (step * 5 / big if big > 5 else step)
        if big < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ b + offset, -30, 30)))
    return b, p, p * (1 - p) + 1e-10


def score_z(r, e, x, g, G, k):
    """The cluster-robust statistic of L1 and N1: U_g = sum r_i e_i per subject, z = sum U_g / sqrt(c sum U_g^2), c = G/(G-1) * (n-1)/(n-k). NaN if x has nothing left."""
    n = len(r)
    u = np.bincount(g, r * e, minlength=G)
    denom = G / (G - 1) * (n - 1) / (n - k) * np.sum(u * u)
    if np.sum(r * r) < 1e-10 * max(np.sum(x * x), 1e-12) or denom <= 0:
        return float("nan")
    return float(u.sum() / np.sqrt(denom))


def score_test(y, Z, x, g, G):
    """L1: cluster-robust score test of x in logistic y ~ 1 + Z, fitted on these rows without x. One-sided, positive = adds. r is x residualised on the design in the
    working-weight metric and e = y - p_hat. Returns (z, r, e); z is NaN when the fit fails or x has nothing left after the design."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = len(y)
    X = scaled_design(Z, n)
    if not 0 < y.mean() < 1 or n <= X.shape[1] or G < 2:
        return float("nan"), None, None
    try:
        _, p, W = fit_logistic(y, X)
        r = x - X @ np.linalg.solve((X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1]), (X * W[:, None]).T @ x)
    except np.linalg.LinAlgError:
        return float("nan"), None, None
    e = y - p
    return score_z(r, e, x, g, G, X.shape[1]), r, e


def score_test_linear(y, Z, x, g, G):
    """N1: the same statistic for the value of a lab, in y ~ 1 + Z by least squares. Returns (z, r, e)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = len(y)
    X = scaled_design(Z, n)
    if n <= X.shape[1] or G < 2:
        return float("nan"), None, None
    Q, _ = np.linalg.qr(X)
    r, e = x - Q @ (Q.T @ x), y - Q @ (Q.T @ y)
    return score_z(r, e, x, g, G, X.shape[1]), r, e


def delta_r2(y, stack, base, g, G):
    """Gain in R², its cluster SE (delta method on the ratio), partial R² = 1 - SSE(stack) / SSE(base) and that SE."""
    y, stack, base = (np.asarray(v, float) for v in (y, stack, base))
    dev = (y - y.mean()) ** 2
    sst = dev.sum()
    e1, e0 = (y - stack) ** 2, (y - base) ** 2
    d, part = (e0.sum() - e1.sum()) / sst, 1 - e1.sum() / e0.sum()
    t = np.bincount(g, (e0 - e1) - d * dev, minlength=G) / sst
    t2 = np.bincount(g, (e0 * (1 - part) - e1), minlength=G) / e0.sum()
    return dict(delta=float(d), se=cluster_se(t - t.mean(), G), partial_r2=float(part), partial_se=cluster_se(t2 - t2.mean(), G),
                r2_stack=float(1 - e1.sum() / sst), r2_base=float(1 - e0.sum() / sst))


def brier_skill(y, stack, base, g, G):
    """1 - Brier(stack) / Brier(base), with its cluster SE by the delta method."""
    y, stack, base = (np.asarray(v, float) for v in (y, stack, base))
    e1, e0 = (y - stack) ** 2, (y - base) ** 2
    skill = 1 - e1.sum() / e0.sum()
    t = np.bincount(g, e0 * (1 - skill) - e1, minlength=G) / e0.sum()
    return dict(skill=float(skill), se=cluster_se(t - t.mean(), G), brier_stack=float(e1.mean()), brier_base=float(e0.mean()))


def calibration(y, p, g, G):
    """Calibration slope (y on the logit of p) and calibration-in-the-large (the intercept with the logit as an offset), with cluster-robust SEs."""
    y, x = np.asarray(y, float), clip_logit(p)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    b, q, W = fit_logistic(y, X)
    bread = np.linalg.inv((X * W[:, None]).T @ X)
    u = np.column_stack([np.bincount(g, X[:, j] * (y - q), minlength=G) for j in range(2)])
    V = bread @ (u.T @ u) @ bread * G / (G - 1)
    one = np.ones((n, 1))
    a, q1, W1 = fit_logistic(y, one, offset=x)
    u1 = np.bincount(g, y - q1, minlength=G)
    return dict(slope=float(b[1]), slope_se=float(np.sqrt(V[1, 1])), intercept=float(a[0]), intercept_se=float(np.sqrt(G / (G - 1) * np.sum(u1 * u1)) / W1.sum()))


def net_benefit(y, p, t):
    """Net benefit of acting when p >= t, per row: true positive 1, false positive -t/(1-t), divided by n when summed."""
    act = np.asarray(p, float) >= t
    return act * (np.asarray(y, float) - (1 - np.asarray(y, float)) * t / (1 - t))


def net_benefit_rows(y, stack, base, g, G):
    """At thresholds fixed in advance: net benefit of stack, base and acting on everyone, and the gain over the base with its cluster SE."""
    n, rows = len(y), []
    for t in NET_BENEFIT_AT:
        a, b = net_benefit(y, stack, t), net_benefit(y, base, t)
        u = np.bincount(g, a - b, minlength=G) / n
        rows.append(dict(threshold=t, stack=float(a.mean()), base=float(b.mean()), act_on_all=float(net_benefit(y, np.ones(n), t).mean()), gain=float((a - b).mean()),
                         se=cluster_se(u - u.mean(), G)))
    return rows


def strata_gains(y, stack, base, g, G, masks):
    """Gain in each stratum, with its cluster SE, and heterogeneity over the strata that can be computed: a Wald test that the gains are equal, from the joint influence terms, so
    it holds when a subject is in more than one stratum. Descriptive. A stratum with too few subjects in a class gives counts only."""
    rows, D, T = {}, [], []
    for name, m in masks.items():
        rows[name] = eval_counts(y[m], g[m])
        if min(rows[name]["subjects_abnormal"], rows[name]["subjects_normal"]) < MIN_SUBJECTS:
            continue
        a1, t1 = auc_influence(y[m], stack[m], g[m], G)
        a0, t0 = auc_influence(y[m], base[m], g[m], G)
        d, t = (a1 - GAIN_FLOOR, t1) if a0 < GAIN_FLOOR else (a1 - a0, t1 - t0)
        rows[name].update(delta=float(d), se=cluster_se(t, rows[name]["subjects"]))
        D.append(d)
        T.append(t)
    het = dict(levels=len(D), q=None, df=None, p=None)
    if len(D) >= 2:
        C_ = np.eye(len(D))[1:] - np.eye(len(D))[0]
        S = np.array(T) @ np.array(T).T * G / (G - 1)
        c = C_ @ np.array(D)
        q = float(c @ np.linalg.pinv(C_ @ S @ C_.T) @ c)
        het.update(q=q, df=len(D) - 1, p=float(chi2.sf(q, len(D) - 1)))
    return dict(strata=rows, heterogeneity=het)


def threshold(A, alpha=ALPHA):
    """One-sided critical z with alpha split evenly among A branches."""
    return float(norm.isf(alpha / A))


def alpha_shares(k, named):
    """What each branch of the graphical procedure starts with: ALPHA_A split over the k family labs and ALPHA_NAMED over the named hypotheses."""
    return np.r_[np.full(k, ALPHA_A / k), np.full(named, ALPHA_NAMED / named) if named else []]


def graded_sequence(Z, alpha=ALPHA, shares=None, levels=None):
    """Weighted Bonferroni-based graphical procedure. Z holds one row per branch and one column per level, in order; `levels` gives each branch's number of levels and `shares`
    its starting alpha (even by default). Among the branches with a test pending, each one's current level is tested at Phi^-1(1 - alpha * share / sum of pending shares); a
    rejection moves it to its next level, it leaves after the last, and only leaving passes its alpha on, in proportion to the others' shares. NaN never rejects and its branch
    stays in. FWER <= alpha under any dependence if each test is valid."""
    Z = np.asarray(Z, float)
    L, width = Z.shape
    shares = np.full(L, alpha / L) if shares is None else np.asarray(shares, float)
    levels = np.full(L, width) if levels is None else np.asarray(levels, int)
    reject, used = np.zeros((L, width), bool), np.full((L, width), np.nan)
    head, alive, trace = np.zeros(L, int), np.ones(L, bool), []
    round_, changed = 0, True
    while changed and alive.any():
        changed, round_ = False, round_ + 1
        total = shares[alive].sum()
        A = int(alive.sum())
        thr = {i: float(norm.isf(alpha * shares[i] / total)) for i in np.flatnonzero(alive)}
        for i in np.flatnonzero(alive):
            z = Z[i, head[i]]
            if np.isfinite(z) and z >= thr[i]:
                reject[i, head[i]], used[i, head[i]] = True, thr[i]
                trace.append(dict(round=round_, lab=int(i), level=int(head[i]) + 1, z=float(z), A=A, threshold=thr[i]))
                head[i] += 1
                changed = True
                if head[i] == levels[i]:
                    alive[i] = False
    final = np.full(L, np.nan)
    if alive.any():
        final[alive] = norm.isf(alpha * shares[alive] / shares[alive].sum())
    return dict(reject=reject, threshold=used, trace=trace, final_A=int(alive.sum()), final_thresholds=final)


def training_sd(r_hats, ses):
    """The training part of the SD of a mean gain. Refits of all labs share one training split, so their errors may move together, and the SD of a mean is at most the mean of the
    SDs: this takes them as fully correlated, the conservative end. One lab's training SD is sqrt(r-hat) times its eval SE."""
    return float(np.mean(np.sqrt(np.asarray(r_hats, float)) * np.asarray(ses, float)))


def mean_upper(mean, hi, train_sd):
    """One-sided 97.5% upper bound of a mean gain: the bootstrap's upper half-width and the training part, in quadrature."""
    return float(mean + np.hypot(max(hi - mean, 0.0), norm.isf(ALPHA) * train_sd))


def upper_chain(per_base, margin):
    """Fixed sequence over the bases, in order, at one-sided alpha. H0: the true mean gain is at least the margin. A step runs only if the one before passed, and passes if the upper bound
    of the mean is below the margin, so the chain as a whole holds alpha. The hypothesis is about all the labs, so a step with a lab missing from its mean cannot pass. Each base gives
    labs, k, mean, lo, hi, training_sd and upper: all reported, tested or not."""
    steps, open_ = [], True
    for base, x in per_base.items():
        passed = bool(open_ and x.get("labs") == x.get("k") and x.get("upper") is not None and np.isfinite(x["upper"]) and x["upper"] < margin)
        steps.append(dict(x, base=base, tested=open_, passed=passed))
        open_ = passed
    return steps


def stratum(n_val, prev_val):
    """Sample size first, as it drives the SE: sparse (< 1,000 val windows), mid (1,000-2,000), then rare, then common."""
    if n_val < 1000:
        return "sparse"
    if n_val <= 2000:
        return "mid"
    return "rare" if prev_val < 0.15 or prev_val > 0.85 else "common"


def r_hat_by_stratum(rows):
    """rows: (stratum, C_l, SE_val_l). r-hat = 4 * sum max(C_l, 0) / sum SE^2, where 4 = f / (1 - f) at the 80% subsample."""
    out = {}
    for s in sorted({r[0] for r in rows}):
        c, se = zip(*[(r[1], r[2]) for r in rows if r[0] == s])
        out[s] = float(SUB_F / (1 - SUB_F) * np.sum(np.maximum(c, 0)) / np.sum(np.square(se)))
    return out


def margin_rule(group_gains):
    """The margin of the upper-bound chain, for the mean gain over the family. group_gains holds, for each family lab and each of MAP, pressors and NBP, what that group adds over the rest
    of `now` on val, fitted on train, so no PPG gain enters. The anchor is their median, each floored at 0, rounded up to 0.001. The margin is never below +0.01, the practical gain of L3: a
    claim that PPG adds less than a gain already called not practical would say nothing more. What the three add together is reported beside it (make_plan)."""
    g = np.maximum(np.asarray(list(group_gains), float), 0)
    anchor = math.ceil(float(np.median(g)) / MARGIN_STEP - 1e-9) * MARGIN_STEP
    return dict(margin=max(MIN_GAIN, anchor), anchor=anchor, median_gain=float(np.median(g)), floor=MIN_GAIN, floor_binds=bool(anchor < MIN_GAIN))


def auc_fast(y, p):
    y = np.asarray(y, bool)
    n1 = y.sum()
    return float((rankdata(p)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def weighted_auc(order, y, w):
    """AUROC with row weights, B rows of weights at a time, ties in score count one half. `order` sorts the rows by score into tie groups: (sorted index, group starts)."""
    idx, starts = order
    pos, neg = w[:, idx] * y[idx], w[:, idx] * ~y[idx]
    p, n = np.add.reduceat(pos, starts, axis=1), np.add.reduceat(neg, starts, axis=1)
    below = np.cumsum(n, axis=1) - n
    return (p * (below + n / 2)).sum(1) / (p.sum(1) * n.sum(1))


def tie_groups(p):
    idx = np.argsort(p, kind="stable")
    return idx, np.r_[0, np.flatnonzero(np.diff(p[idx])) + 1]


def mean_gain(per_lab, base, pick, seed=0, B=N_MEAN_BOOT):
    """The headline: the mean gain over the labs given, and its percentile 95% interval from one subject bootstrap shared by all labs, with models fixed. per_lab: lab -> arrays."""
    subjects = np.unique(np.concatenate([a["group"] for a in per_lab.values()]))
    counts = np.random.default_rng(seed).multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), size=B).astype(float)
    point, boots = {}, np.zeros((B, len(per_lab)))
    for j, (lab, a) in enumerate(per_lab.items()):
        y, g = a["y"].astype(bool), np.searchsorted(subjects, a["group"])
        s, b = a[f"{base}+{pick[lab]}"].astype(float), a[base].astype(float)
        point[lab] = auc_fast(y, s) - max(auc_fast(y, b), GAIN_FLOOR)
        os_, ob = tie_groups(s), tie_groups(b)
        for lo in range(0, B, 100):
            w = counts[lo:lo + 100][:, g]
            boots[lo:lo + 100, j] = weighted_auc(os_, y, w) - np.maximum(weighted_auc(ob, y, w), GAIN_FLOOR)
    m = boots.mean(1)
    return dict(base=base, labs=len(per_lab), mean=float(np.mean(list(point.values()))), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), by_lab=point)


def half_cov(y, g, G, stacks, bases, rng, n_splits=N_SPLITS, floor=GAIN_FLOOR):
    """Training-variance component free of the train x eval interaction: covariance across refits of the gain on two disjoint subject
    halves of val, averaged over random splits. Scored on one whole set, the across-refit variance overstates it several-fold."""
    def gains(mask):
        ys = y[mask]
        return [roc_auc_score(ys, s[mask]) - max(roc_auc_score(ys, b[mask]), floor) for s, b in zip(stacks, bases)]
    out = []
    for _ in range(n_splits):
        first = rng.permutation(G)[:G // 2]
        in_first = np.isin(g, first)
        if len(np.unique(y[in_first])) < 2 or len(np.unique(y[~in_first])) < 2:
            continue
        out.append(np.cov(gains(in_first), gains(~in_first))[0, 1])
    return float(np.mean(out)) if out else float("nan")


# ------------------------------------------------------------------------------------------------- the pipeline

@contextmanager
def keep_eval_rows():
    """baselines.task_frame drops a lab on eval outcome counts. Here k is fixed, so those guards are off; the train guard stays."""
    saved = BL.MIN_POS_EVAL, BL.too_small
    BL.MIN_POS_EVAL, BL.too_small = 0, lambda *_: False
    try:
        yield
    finally:
        BL.MIN_POS_EVAL, BL.too_small = saved


def fixed_frame(d, tiers, name, spec, eval_split, task="clf"):
    """The real task_frame for one lab, without the eval-count guards. For T2, None only if train has too few abnormal rows."""
    with keep_eval_rows():
        return BL.task_frame(d, tiers, name, spec, task, eval_split)


def feature_cols(model, bcols, name=None):
    """A model's input columns. The CNN's depend on the lab, so `name` is needed for it."""
    if model == "cnn":
        return BL.cnn_cols(name)
    return {"beat": bcols, "papagei": BL.PG, "pulseppg": BL.PP, RATE: BL.RATE}[model]


def available_models(columns, bcols, name=None):
    """Models with their columns in the frame: for one lab if `name` is given, otherwise those of the cohort."""
    return [m for m in MODELS if (feature_cols(m, bcols, name)[0] if name or m != "cnn" else "cnn_fold") in columns]


def dense(groups):
    u, g = np.unique(groups, return_inverse=True)
    return g, len(u)


def fit_models(y, tr, ev, groups, bases, X, task="clf"):
    """Base models, frozen PPG scores and stacks, the same recipe as baselines.run_lab. X maps model -> feature matrix on all rows;
    tr may be a subsample of train. Stacks see the out-of-fold score on train and the full-train refit on eval, on the log-odds scale in T2."""
    model, predict = (BL.base_clf, lambda m, A: m.predict_proba(A)[:, 1]) if task == "clf" else (BL.base_reg, lambda m, A: m.predict(A))
    preds = {b: predict(model().fit(Z[tr], y[tr]), Z[ev]) for b, Z in bases.items()}
    for m, x in X.items():
        s_tr, s_ev = BL.ppg_score(BL.PPG_MODELS[m][task], x, y, groups, tr, ev)
        preds[f"score:{m}"] = s_ev
        for b, Z in bases.items():
            preds[f"{b}+{m}"] = predict(model().fit(np.column_stack([Z[tr], s_tr]), y[tr]), np.column_stack([Z[ev], s_ev]))
    return preds


def residual_score(x, eta, y, groups, tr, ev, embedded):
    """The residual arm: PPG features and the base's log-odds, fitted by boosting to the base's working residual on the log-odds scale, so a model can use PPG differently at
    different base risk. A stack on one score cannot be blamed for missing that. Out of fold on train and a refit on eval, as the score is. Embeddings pass through the probe's
    imputer, scaler and PCA first, fitted on the same rows."""
    p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
    w = np.maximum(p * (1 - p), 0.01)
    z = np.clip((y - p) / w, -10, 10)

    def fitted(fit_rows, rows):
        prep = make_pipeline(SimpleImputer(), StandardScaler(), PCA(64, random_state=0)).fit(x[fit_rows]) if embedded else None
        A = (lambda r: np.column_stack([prep.transform(x[r]) if embedded else x[r], eta[r]]))
        return HistGradientBoostingRegressor(**BL.BOOST).fit(A(fit_rows), z[fit_rows], sample_weight=w[fit_rows]).predict(A(rows))
    s_tr = np.full(len(tr), np.nan)
    for a, b in GroupKFold(BL.K).split(tr, groups=groups[tr]):
        s_tr[b] = fitted(tr[a], tr[b])
    return s_tr, fitted(tr, ev)


def eval_counts(y_ev, g_ev):
    y_ev = np.asarray(y_ev, bool)
    return dict(n=int(len(y_ev)), subjects=int(len(np.unique(g_ev))), subjects_abnormal=int(len(np.unique(g_ev[y_ev]))),
                subjects_normal=int(len(np.unique(g_ev[~y_ev]))), prevalence=float(y_ev.mean()) if len(y_ev) else float("nan"))


def pick_model(gains):
    """The highest validation gain, compared at 3 decimals, since differences below that are noise on this split. A tie goes by TIE_ORDER."""
    rounded = {m: round(float(v), 3) for m, v in gains.items() if np.isfinite(v)}
    if not rounded:
        return None
    best = max(rounded.values())
    return next(m for m in TIE_ORDER if rounded.get(m) == best)


def bedside_gains(d, tiers, name, spec):
    """What MAP, pressors and NBP each add over the rest of `now`: AUROC with the group minus AUROC with its columns held constant, including their interactions and missing
    indicators, fitted on train and scored on val, and what the three add together, which is reported and not part of the margin. No PPG column is read. Only groups in the frame count."""
    def auc(frame):
        q, y, tr, ev, _, bases, _, _ = frame
        return max(roc_auc_score(y[ev], BL.base_clf().fit(bases[PRIMARY][tr], y[tr]).predict_proba(bases[PRIMARY][ev])[:, 1]), GAIN_FLOOR)
    frame = fixed_frame(d, tiers, name, spec, "val")
    if frame is None or len(np.unique(frame[1][frame[3]])) < 2:
        return dict(lab=name, gains={})
    whole = auc(frame)
    gains = {}
    for group, cols in MARGIN_GROUPS.items():
        present = [c for c in cols if c in d]
        if present:
            gains[group] = whole - auc(fixed_frame(d.assign(**{c: 0.0 for c in present}), tiers, name, spec, "val"))
    together = [c for cols in MARGIN_GROUPS.values() for c in cols if c in d]
    joint = whole - auc(fixed_frame(d.assign(**{c: 0.0 for c in together}), tiers, name, spec, "val")) if together else float("nan")
    return dict(lab=name, gains=gains, joint=joint, auroc=whole)


def plan_lab(d, bcols, tiers, name, spec, models, B):
    """Train and val only. Gain of every candidate over the bases, the pick and the training-variance component of the pick."""
    frame = fixed_frame(d, tiers, name, spec, "val")
    if frame is None:
        return dict(lab=name, pick=None, reason="too few abnormal train rows")
    q, y, tr, ev, groups, all_bases, _, _ = frame
    counts = eval_counts(y[ev], groups[ev])
    if min(counts["subjects_abnormal"], counts["subjects_normal"]) < MIN_SUBJECTS:
        return dict(lab=name, pick=None, reason=f"fewer than {MIN_SUBJECTS} val subjects in a class", **counts)
    bases = {b: all_bases[b] for b in (CHAIN if B else (PRIMARY, UPPER))}
    models = [m for m in models if feature_cols(m, bcols, name)[0] in q]
    X = {m: q[feature_cols(m, bcols, name)].to_numpy(float) for m in models}
    preds = fit_models(y, tr, ev, groups, bases, X)
    ye = y[ev].astype(bool)
    g, G = dense(groups[ev])
    table = {m: {b: delta_auc(ye, preds[f"{b}+{m}"], preds[b], g, G) for b in bases} for m in models}
    pick = pick_model({m: table[m][PRIMARY]["delta"] for m in models})
    # refits of the pick on 80% of train subjects, always scored on the same val rows. The CNN is not refitted, so its bases and stack alone would understate its training
    # variance: a lab with the CNN as its pick takes no part in r-hat, and lab_entry gives it the largest r-hat of the base
    rng = np.random.default_rng([zlib.crc32(name.encode()), B])
    B = 0 if pick == "cnn" else B
    rows = pd.Series(np.arange(len(tr))).groupby(groups[tr]).indices
    subjects = np.array(sorted(rows))
    stacks, plain = {b: [] for b in bases}, {b: [] for b in bases}
    for _ in range(B):
        chosen = rng.choice(subjects, int(SUB_F * len(subjects)), replace=False)
        p = fit_models(y, tr[np.concatenate([rows[s] for s in chosen])], ev, groups, bases, {pick: X[pick]})
        for b in bases:
            stacks[b].append(p[f"{b}+{pick}"])
            plain[b].append(p[b])
    cov = {b: half_cov(ye, g, G, stacks[b], plain[b], rng) if B > 1 else float("nan") for b in bases}   # no refits for descriptive labs, nor for a CNN pick
    return dict(lab=name, pick=pick, stratum=stratum(counts["n"], counts["prevalence"]), table=table, cov=cov, **counts)


def plan_named(d, bcols, tiers, name, spec, models):
    """A named T1 hypothesis: the gain in R² of every candidate over `now` on val, and the pick."""
    q, y, tr, ev, groups, all_bases, _, _ = fixed_frame(d, tiers, name, spec, "val", task="reg")
    g, G = dense(groups[ev])
    if G < MIN_SUBJECTS:
        return dict(lab=name, pick=None, reason=f"fewer than {MIN_SUBJECTS} val subjects")
    models = [m for m in models if m in VALUE_MODELS]
    preds = fit_models(y, tr, ev, groups, {PRIMARY: all_bases[PRIMARY]}, {m: q[feature_cols(m, bcols)].to_numpy(float) for m in models}, task="reg")
    table = {m: delta_r2(y[ev], preds[f"{PRIMARY}+{m}"], preds[PRIMARY], g, G) for m in models}
    return dict(lab=name, pick=pick_model({m: table[m]["delta"] for m in models}), n=int(len(ev)), subjects=G, table=table)


def provisional_reasons(models, B, code_version, window):
    reasons = [f"model {m} not available" for m in MODELS if m not in models]
    if window != PRIMARY_WINDOW:
        reasons.append(f"window {window} is not the primary window {PRIMARY_WINDOW}")
    if B < B_REFITS:
        reasons.append(f"B = {B} < {B_REFITS}")
    if code_version.endswith(("-dirty", "unknown")):
        reasons.append("the working tree was not clean")
    return reasons


def parallel_map(fn, args, jobs):
    with parallel_config(backend="loky", inner_max_num_threads=1), threadpool_limits(limits=1):
        return Parallel(n_jobs=jobs)(delayed(fn)(*a) for a in args)


def failure(e):
    """A failure as text: its type and where it happened. Never its message, which could quote a value."""
    frames = traceback.extract_tb(e.__traceback__)
    where = next((f for f in reversed(frames) if "ppglabbench" in f.filename), frames[-1] if frames else None)
    return f"{type(e).__name__} at {Path(where.filename).name}:{where.lineno} in {where.name}" if where else type(e).__name__


def attempt(fn, *args):
    """A row that raises becomes a record of the failure, and what it sits beside keeps its result."""
    try:
        return fn(*args)
    except (Exception, SystemExit) as e:
        return dict(failed=failure(e))


def attempt_with_arrays(fn, *args):
    """The same for a row that returns its statistics and the predictions behind them."""
    try:
        return fn(*args)
    except (Exception, SystemExit) as e:
        return dict(failed=failure(e)), {}


def whole_lab(fn, stub, *args):
    """A lab that raises is a lab that cannot be computed: not rejected and still counted, with the failure on record. The other labs keep their results."""
    try:
        return fn(*args)
    except (Exception, SystemExit) as e:
        return dict(stub, computable=False, reason=f"failed: {failure(e)}", failed=failure(e))


def protected_map(fn, args, jobs):
    """parallel_map, and one by one if a worker dies. fn catches what a lab raises, so only the death of a process comes here, and a lab's seeds do not depend on where it runs."""
    try:
        return parallel_map(fn, args, jobs)
    except Exception as e:
        print(f"parallel run failed ({type(e).__name__}): one lab at a time", file=sys.stderr, flush=True)
        return [fn(*a) for a in args]


def make_plan(d, bcols, labs, tiers, B, jobs, window, family=FAMILY_LABS, descriptive=DESCRIPTIVE_LABS, named=NAMED_LABS, cnn_sha=None):
    """The whole plan from train and val rows. Everything else in the analysis is fixed by this object and the code."""
    d = d[d.split.isin(["train", "val"])].reset_index(drop=True)
    models = available_models(d.columns, bcols)
    no_ppg = d.drop(columns=[c for c in d if c in {*BL.PG, *BL.PP, *bcols} or c.startswith("cnn_")])
    bedside = parallel_map(bedside_gains, [(no_ppg, tiers, n, labs[n]) for n in family], jobs)
    margin = margin_rule([v for r in bedside for v in r["gains"].values()])
    joint = [max(r["joint"], 0) for r in bedside if np.isfinite(r.get("joint", float("nan")))]
    margin.update(bedside_gains={r["lab"]: r["gains"] for r in bedside}, joint_gains={r["lab"]: r.get("joint") for r in bedside}, joint_mean=float(np.mean(joint)) if joint else None, joint_median=float(np.median(joint)) if joint else None)
    jobs_args = [(d, bcols, tiers, n, labs[n], tuple(models), B) for n in family] + [(d, bcols, tiers, n, labs[n], tuple(models), 0) for n in descriptive]
    res = parallel_map(plan_lab, jobs_args, jobs)
    res_named = parallel_map(plan_named, [(d, bcols, tiers, n, labs[n], tuple(models)) for n in named], jobs)
    by_lab = {r["lab"]: r for r in res}
    usable = [by_lab[n] for n in family if by_lab[n]["pick"]]
    r_hat = {b: r_hat_by_stratum([(r["stratum"], r["cov"][b], r["table"][r["pick"]][b]["se"]) for r in usable if np.isfinite(r["cov"][b])]) for b in CHAIN}
    version = git_version()
    reasons = provisional_reasons(models, B, version, window)
    return dict(stage="confirmatory_plan", created=datetime.now().isoformat(timespec="seconds"), code_version=version,
                provisional=bool(reasons), provisional_reasons=reasons, source="mimic", window=window, eval_split="val", family=list(family), named=list(named),
                descriptive=list(descriptive), models=models, cnn_scores_sha256=cnn_sha, B=B, sub_fraction=SUB_F, n_splits=N_SPLITS, alpha=ALPHA, alpha_a=ALPHA_A, alpha_named=ALPHA_NAMED, min_gain=MIN_GAIN,
                picks={n: by_lab[n]["pick"] for n in (*family, *descriptive)}, picks_t1={r["lab"]: r["pick"] for r in res_named}, strata={n: by_lab[n].get("stratum") for n in family},
                r_hat=r_hat, margin=margin, labs=by_lab, named_labs={r["lab"]: r for r in res_named}, boost=BL.BOOST)


def lab_entry(plan, name):
    """What the evaluation of one lab reads from the plan. A stratum missing from r-hat means its covariance could not be estimated:
    use the largest r-hat of that base rather than no training variance."""
    s, pick = plan["strata"].get(name), plan["picks"].get(name)

    def r_of(v):
        largest = max((x for x in v.values() if x is not None), default=0.0)
        return largest if pick == "cnn" or v.get(s) is None else v[s]
    return dict(pick=pick, r={b: r_of(v) for b, v in plan["r_hat"].items()})


def evaluate_lab(d, bcols, tiers, name, spec, entry, eval_split):
    """One lab on the eval split. Fits on train only. Aggregates go to `stats`; row-level arrays to `arrays`, without subject IDs
    being written anywhere (the parent replaces them)."""
    out = dict(lab=name, computable=False, pick=entry["pick"], r_hat=entry["r"])
    if entry["pick"] is None:
        return dict(out, reason="no pick in the plan")
    frame = fixed_frame(d, tiers, name, spec, eval_split)
    if frame is None:
        return dict(out, reason="too few abnormal train rows")
    q, y, tr, ev, groups, all_bases, _, age_h = frame
    counts = eval_counts(y[ev], groups[ev])
    out.update(counts)
    if len(ev) == 0 or min(counts["subjects_abnormal"], counts["subjects_normal"]) < MIN_SUBJECTS:
        return dict(out, reason=f"fewer than {MIN_SUBJECTS} eval subjects in a class")
    pick = entry["pick"]
    ye = y[ev].astype(bool)
    g, G = dense(groups[ev])
    x_pick = q[feature_cols(pick, bcols, name)].to_numpy(float)
    preds = fit_models(y, tr, ev, groups, all_bases, {pick: x_pick})
    design = eval_design(all_bases[PRIMARY][ev])

    def tests(model, p):
        z1, r, e = score_test(ye.astype(float), design, p[f"score:{model}"], g, G)
        res = delta_auc(ye, p[f"{PRIMARY}+{model}"], p[PRIMARY], g, G)
        return dict(res, z1=z1, z2=safe_z(res["delta"], res["se"])), r, e

    a, r, e = tests(pick, preds)
    a["se_t"] = a["se"] * np.sqrt(1 + entry["r"][PRIMARY])
    a["z3"] = safe_z(a["delta"] - MIN_GAIN, a["se_t"])

    def bound(b):       # descriptive: no test and no claim reads this lab's bound
        gain = delta_auc(ye, preds[f"{b}+{pick}"], preds[b], g, G)
        se_t = float(gain["se"] * np.sqrt(1 + entry["r"].get(b, 0.0)))
        return dict(gain, se_t=se_t, upper=gain["delta"] + norm.isf(ALPHA) * se_t)

    def rung(b):
        return delta_auc(ye, preds[f"{b}+{pick}"], preds[b], g, G)

    def boot_se(b):     # a subject bootstrap of the SE is a check, not a decision
        res = BL.subject_boot(ye.astype(int), groups[ev], {"s": preds[f"{b}+{pick}"], "b": preds[b]}, [("s", "b")], roc_auc_score, floor=GAIN_FLOOR)["s|b"]
        return (res[2] - res[1]) / (2 * 1.96)

    def compare():      # PaPaGei was pretrained on these cohorts: a confirmed lab is reported beside the others
        others = {m: q[feature_cols(m, bcols, name)].to_numpy(float) for m in available_models(q.columns, bcols, name) if m != pick}
        other = fit_models(y, tr, ev, groups, {PRIMARY: all_bases[PRIMARY]}, others)
        return {m: tests(m, other)[0] for m in others}, {k: other[k] for m in others for k in (f"{PRIMARY}+{m}", f"score:{m}")}
    chain = {b: attempt(bound, b) for b in CHAIN}
    ladder = {b: attempt(rung, b) for b in all_bases if b not in (PRIMARY, UPPER)}
    boot = {b: attempt(boot_se, b) for b in (PRIMARY, UPPER)}
    arrays = dict(y=ye.astype(np.int8), group=groups[ev], l1_r=r, l1_e=e, **{k: preds[k] for b in CHAIN for k in (b, f"{b}+{pick}")}, **{f"score:{pick}": preds[f"score:{pick}"]})
    comparators, more = attempt_with_arrays(compare) if pick == "papagei" else ({}, {})
    arrays.update(more)
    beside, more = attempt_with_arrays(rate_rows, q, x_pick, y, tr, ev, groups, all_bases[PRIMARY], pick, design, ye, g, G)
    arrays.update(more)
    second, more = attempt_with_arrays(secondaries, d, tiers, name, spec, eval_split, bcols, q, y, tr, ev, groups, all_bases, preds, pick, x_pick, age_h, ye, g, G)
    arrays.update(more)
    out.update(computable=True, family_a=a, chain=chain, ladder=ladder, boot_se=boot, comparators=comparators, beside_rate=beside, secondary=second, n_design=int(design.shape[1]) + 1)
    return dict(out, arrays=arrays)


def previous_age_masks(age_h):
    """Strata of the previous value's age, fixed in advance, and never measured."""
    masks = {name: (age_h > lo) & (age_h <= hi) for name, lo, hi in PREVIOUS_AGE}
    masks["never"] = ~np.isfinite(age_h)
    return masks


def subgroup_masks(q, ev):
    """Subgroups fixed in advance. AF is the ICD-9 code of the admission, assigned at discharge, because no rhythm label exists per window: `no_af` is not the same as sinus rhythm."""
    out = {"sex": {"female": q.female.to_numpy(float)[ev] == 1, "male": q.female.to_numpy(float)[ev] == 0}}
    age = q.age.to_numpy(float)[ev]
    out["age_band"] = {name: (age >= lo) & (age < hi) for name, lo, hi in AGE_BANDS}
    for name, col, on, off in (("af", "dx_af", "af_coded", "no_af"), ("vasopressor", "vasopressor", "on", "off"), ("era", "era_mv", "metavision", "carevue")):
        if col in q:
            v = q[col].to_numpy(float)[ev]
            out[name] = {on: v == 1, off: v == 0}
    return out


def secondaries(d, tiers, name, spec, eval_split, bcols, q, y, tr, ev, groups, all_bases, preds, pick, x_pick, age_h, ye, g, G):
    """Descriptive rows over `now`, not tests. Each fails on its own, as a record in the results. Returns the statistics and the predictions behind them."""
    stack, base = preds[f"{PRIMARY}+{pick}"], preds[PRIMARY]

    def subgroups():
        return {k: attempt(strata_gains, ye, stack, base, g, G, masks) for k, masks in subgroup_masks(q, ev).items()}

    def by_age():
        masks = previous_age_masks(age_h[ev])
        return {b: attempt(strata_gains, ye, preds[f"{b}+{pick}"], preds[b], g, G, masks) for b in (PRIMARY, "locf") if b in preds}

    def residual():
        Z = all_bases[PRIMARY]
        eta = BL.base_clf().fit(Z[tr], y[tr]).decision_function(Z)
        r_tr, r_ev = residual_score(x_pick[:, :1] if pick == "cnn" else x_pick, eta, y.astype(float), groups, tr, ev, pick in BL.EMBEDDINGS)
        arm = BL.base_clf().fit(np.column_stack([Z[tr], r_tr]), y[tr]).predict_proba(np.column_stack([Z[ev], r_ev]))[:, 1]
        return delta_auc(ye, arm, base, g, G), {f"{PRIMARY}+residual": arm}

    def value():
        q2, y2, tr2, ev2, groups2, bases2, _, _ = fixed_frame(d, tiers, name, spec, eval_split, task="reg")
        p = fit_models(y2, tr2, ev2, groups2, {PRIMARY: bases2[PRIMARY]}, {pick: q2[feature_cols(pick, bcols, name)].to_numpy(float)}, task="reg")
        g2, G2 = dense(groups2[ev2])
        return delta_r2(y2[ev2], p[f"{PRIMARY}+{pick}"], p[PRIMARY], g2, G2)
    out = dict(brier=attempt(brier_skill, ye, stack, base, g, G), calibration=dict(stack=attempt(calibration, ye, stack, g, G), base=attempt(calibration, ye, base, g, G)),
               net_benefit=attempt(net_benefit_rows, ye, stack, base, g, G), subgroups=attempt(subgroups))
    if age_h is not None:
        out["by_previous_age"] = attempt(by_age)
    out["residual_arm"], more = attempt_with_arrays(residual)
    out["value"] = attempt(value)
    return out, more


def evaluate_named(d, bcols, tiers, name, spec, pick, eval_split):
    """A named T1 hypothesis on the eval split: N1 partial association on the value and N2 gain in R² over `now`, with the partial R²."""
    key = f"t1_{name}"
    out = dict(lab=key, name=name, computable=False, pick=pick)
    if pick is None:
        return dict(out, reason="no pick in the plan")
    q, y, tr, ev, groups, all_bases, _, _ = fixed_frame(d, tiers, name, spec, eval_split, task="reg")
    g, G = dense(groups[ev])
    out.update(n=int(len(ev)), subjects=G)
    if G < MIN_SUBJECTS:
        return dict(out, reason=f"fewer than {MIN_SUBJECTS} eval subjects")
    base = all_bases[PRIMARY]
    preds = fit_models(y, tr, ev, groups, {PRIMARY: base}, {pick: q[feature_cols(pick, bcols)].to_numpy(float)}, task="reg")
    design = eval_design(base[ev])
    z1, r, e = score_test_linear(y[ev], design, preds[f"score:{pick}"], g, G)
    res = delta_r2(y[ev], preds[f"{PRIMARY}+{pick}"], preds[PRIMARY], g, G)
    arrays = dict(y=y[ev], group=groups[ev], n1_r=r, n1_e=e, **{k: preds[k] for k in (PRIMARY, f"{PRIMARY}+{pick}", f"score:{pick}")})
    out.update(computable=True, named=dict(res, z1=z1, z2=safe_z(res["delta"], res["se"])), n_design=int(design.shape[1]) + 1)
    return dict(out, arrays=arrays)


def rate_rows(q, x_pick, y, tr, ev, groups, base, pick, design, ye, g, G):
    """Descriptive, never a test: the rate model's L1 z and gain over the base, and the pick's gain over the base + the rate score. The score enters as any
    PPG score does: out of fold on train, a full refit on eval. Returns the statistics and the predictions behind them; None without the rate columns."""
    if not set(BL.RATE) <= set(q.columns):
        return None, {}
    x = q[BL.RATE].to_numpy(float)
    s_tr, s_ev = BL.ppg_score(BL.PPG_MODELS[RATE]["clf"], x, y, groups, tr, ev)
    s = np.zeros(len(q))
    s[tr], s[ev] = s_tr, s_ev
    plain = BL.base_clf().fit(base[tr], y[tr]).predict_proba(base[ev])[:, 1]
    p = fit_models(y, tr, ev, groups, {"rate_base": np.column_stack([base, s])}, {pick: x_pick})
    alone = p["rate_base"]
    z1 = score_test(ye.astype(float), design, s_ev, g, G)[0]
    rows = dict(rate=dict(delta_auc(ye, alone, plain, g, G), z1=z1), pick_over_rate=delta_auc(ye, p[f"rate_base+{pick}"], alone, g, G))
    return rows, {f"score:{RATE}": s_ev, f"{PRIMARY}+{RATE}": alone, f"{PRIMARY}+{RATE}+{pick}": p[f"rate_base+{pick}"]}


def describe_lab(d, bcols, tiers, name, spec, pick, eval_split):
    """A lab outside the family: the gain of its pick over `now` and `full`, with its eval SE and an unadjusted 95% interval, and by age of the previous value. Not tested."""
    out = dict(lab=name, computable=False, pick=pick)
    if pick is None:
        return dict(out, reason="no pick in the plan")
    frame = fixed_frame(d, tiers, name, spec, eval_split)
    if frame is None:
        return dict(out, reason="too few abnormal train rows")
    q, y, tr, ev, groups, all_bases, _, age_h = frame
    counts = eval_counts(y[ev], groups[ev])
    out.update(counts)
    if len(ev) == 0 or min(counts["subjects_abnormal"], counts["subjects_normal"]) < MIN_SUBJECTS:
        return dict(out, reason=f"fewer than {MIN_SUBJECTS} eval subjects in a class")
    bases = {b: all_bases[b] for b in (PRIMARY, UPPER, "locf") if b in all_bases}
    x_pick = q[feature_cols(pick, bcols, name)].to_numpy(float)
    preds = fit_models(y, tr, ev, groups, bases, {pick: x_pick})
    ye = y[ev].astype(bool)
    g, G = dense(groups[ev])
    gains = {b: delta_auc(ye, preds[f"{b}+{pick}"], preds[b], g, G) for b in (PRIMARY, UPPER)}
    arrays = dict(y=ye.astype(np.int8), group=groups[ev], **{k: preds[k] for k in (*bases, *[f"{b}+{pick}" for b in bases], f"score:{pick}")})
    beside, more = attempt_with_arrays(lambda: rate_rows(q, x_pick, y, tr, ev, groups, bases[PRIMARY], pick, eval_design(bases[PRIMARY][ev]), ye, g, G))
    arrays.update(more)
    by_age = {b: attempt(lambda b=b: strata_gains(ye, preds[f"{b}+{pick}"], preds[b], g, G, previous_age_masks(age_h[ev]))) for b in (PRIMARY, "locf") if b in preds} if age_h is not None else {}
    return dict(out, computable=True, gains=gains, beside_rate=beside, by_previous_age=by_age, arrays=arrays)


def anonymise_arrays(per_lab, rng):
    """Flat arrays for predictions.npz. Subjects become a random integer shared across labs, and rows are shuffled per lab, so the
    file cannot be joined back to a table by position or by ID."""
    subjects = np.unique(np.concatenate([a["group"] for a in per_lab.values()])) if per_lab else np.empty(0)
    code = rng.permutation(len(subjects))
    flat = {}
    for lab, arrays in per_lab.items():
        order = rng.permutation(len(arrays["y"]))
        for k, v in arrays.items():
            if k == "group":
                flat[f"{lab}__group"] = code[np.searchsorted(subjects, v[order])].astype(np.int32)
            elif k == "y":
                flat[f"{lab}__y"] = v[order]
            elif np.shape(v) == order.shape:      # L1's residuals are None if L1 could not be computed
                flat[f"{lab}__{k}"] = np.asarray(v, np.float32)[order]
    return flat


def clean(x, keep_nan=False):
    """JSON without NaN or numpy types. keep_nan: for a file read back by assemble, which reads NaN as not computable."""
    if isinstance(x, dict):
        return {str(k): clean(v, keep_nan) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v, keep_nan) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if keep_nan or np.isfinite(x) else None
    return x if x is None or isinstance(x, str) else str(x)


def list_failures(x, path=""):
    """Every row that failed, by where it sits in the results."""
    if isinstance(x, dict):
        if isinstance(x.get("failed"), str):
            return [f"{path}: {x['failed']}"]
        return [f for k, v in x.items() for f in list_failures(v, f"{path}.{k}" if path else str(k))]
    return [f for i, v in enumerate(x) for f in list_failures(v, f"{path}.{i}")] if isinstance(x, list) else []


def assemble(res, described, named_res, plan, plan_sha, eval_split, dry, headline):
    """Decisions from the per-lab statistics. k is the length of the family, whatever could be computed."""
    family, named = plan["family"], plan["named"]
    k, m = len(family), len(named)
    by_lab, by_named = {r["lab"]: r for r in res}, {r["name"]: r for r in named_res}
    Z = np.full((k + m, 3), np.nan)
    for i, n in enumerate(family):
        if by_lab[n]["computable"]:
            a = by_lab[n]["family_a"]
            Z[i] = a["z1"], a["z2"], a["z3"]
    for j, n in enumerate(named):
        if by_named[n]["computable"]:
            Z[k + j, :2] = by_named[n]["named"]["z1"], by_named[n]["named"]["z2"]
    seq = graded_sequence(Z, shares=alpha_shares(k, m), levels=[3] * k + [2] * m)
    first_a, first_named = threshold(k, ALPHA_A), threshold(m, ALPHA_NAMED) if m else None
    def from_chain(n, b, key):      # a lab or a base that failed has no number here, and fails that step
        v = dig(by_lab[n], "chain", b, key) if by_lab[n]["computable"] else None
        return float("nan") if v is None else v
    def step_of(b):
        """The mean of a base and its upper bound, and the largest per-lab bound, which is descriptive."""
        h, uppers = headline.get(b) or {}, [from_chain(n, b, "upper") for n in family]
        row = dict(k=k, labs=0, mean=float("nan"), lo=float("nan"), hi=float("nan"), training_sd=float("nan"), upper=float("nan"), largest_upper=float(max(uppers)) if uppers and np.all(np.isfinite(uppers)) else float("nan"))
        if isinstance(h.get("mean"), float):
            inside = [n for n in h.get("by_lab", {})]
            ses = [from_chain(n, b, "se") for n in inside]
            if np.all(np.isfinite(ses)):     # a lab's training SD needs its eval SE; a base that failed for a lab leaves the mean without a bound
                row["training_sd"] = training_sd([by_lab[n]["r_hat"].get(b, 0.0) for n in inside], ses)
                row["upper"] = mean_upper(h["mean"], h["hi"], row["training_sd"])
            row.update(labs=len(inside), mean=h["mean"], lo=h["lo"], hi=h["hi"])
        return row
    steps = upper_chain({b: step_of(b) for b in CHAIN}, plan["margin"]["margin"])
    labs = {}
    for i, n in enumerate(family):
        r = {key: v for key, v in by_lab[n].items() if key != "arrays"}
        r["levels_cleared"] = int(seq["reject"][i].sum())
        stops = seq["final_thresholds"][i]
        r["stops_at_threshold"] = None if not np.isfinite(stops) else float(stops)
        r["flag_gain_without_association"] = bool(not seq["reject"][i, 0] and np.isfinite(Z[i, 1]) and np.isfinite(stops) and Z[i, 1] >= stops)
        if r["computable"]:
            a = r["family_a"]
            a["simultaneous"] = [a["delta"] - first_a * a["se_t"], a["delta"] + first_a * a["se_t"]]
        labs[n] = r
    t1 = {}
    for j, n in enumerate(named):
        r = {key: v for key, v in by_named[n].items() if key != "arrays"}
        r["levels_cleared"] = int(seq["reject"][k + j].sum())
        if r["computable"]:
            b = r["named"]
            b["simultaneous"] = [b["delta"] - first_named * b["se"], b["delta"] + first_named * b["se"]]
        t1[n] = r
    branch = [*family, *[f"{n} (value)" for n in named]]
    trace = [dict(t, lab=branch[t["lab"]]) for t in seq["trace"]]
    described = {r["lab"]: {key: v for key, v in r.items() if key != "arrays"} for r in described}
    failures = list_failures(clean(dict(labs=labs, named=t1, descriptive=described, headline=headline)))
    return clean(dict(dry_run=dry, eval_split=eval_split, window=plan["window"], plan_sha256=plan_sha, plan_code_version=plan["code_version"], plan_provisional=plan["provisional"],
                      models=plan["models"], alpha=ALPHA, alpha_a=ALPHA_A, alpha_named=ALPHA_NAMED, k=k, min_gain=MIN_GAIN, margin=plan["margin"]["margin"], margin_rule=plan["margin"],
                      first_threshold=first_a, first_threshold_named=first_named, headline=headline, chain=steps,
                      sequence=dict(trace=trace, final_A=seq["final_A"]),
                      counts=dict(l1=int(seq["reject"][:k, 0].sum()), l2=int(seq["reject"][:k, 1].sum()), l3=int(seq["reject"][:k, 2].sum()),
                                  named_n1=int(seq["reject"][k:, 0].sum()), named_n2=int(seq["reject"][k:, 1].sum()), chain_passed=[s["base"] for s in steps if s["passed"]],
                                  not_computable=[n for n in family if not by_lab[n]["computable"]] + [f"{n} (value)" for n in named if not by_named[n]["computable"]]),
                      labs=labs, named=t1, descriptive=described, failures=failures))


DASH, FAILED = "—", "failed"


def is_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) and math.isfinite(v)


def dig(x, *path):
    """x[path], or None as soon as a step is missing. results.json can hold None for any number and a failure record for any row."""
    for key in path:
        if isinstance(x, dict) and isinstance(key, (str, int)):
            x = x.get(key)
        elif isinstance(x, list) and isinstance(key, int) and -len(x) <= key < len(x):
            x = x[key]
        else:
            return None
    return x


def has_failed(x):
    return isinstance(x, dict) and x.get("failed") is not None


def num(v, spec="+.4f"):
    """A number as text. Anything else is a cell without a number: a failed row says so, the rest is a dash."""
    return format(v, spec) if is_number(v) else FAILED if has_failed(v) else DASH


def interval(x, est="delta", se="se"):
    """An estimate with its unadjusted 95% interval, as far as they exist."""
    d, s = dig(x, est), dig(x, se)
    if not is_number(d):
        return FAILED if has_failed(x) else DASH
    return f"{d:+.4f} ({d - 1.96 * s:+.4f}, {d + 1.96 * s:+.4f})" if is_number(s) else f"{d:+.4f}"


def rows_of(x):
    """The entries of a table that may be absent, or be a failure record instead of rows."""
    return {k: v for k, v in x.items() if k != "failed"} if isinstance(x, dict) else {}


def report_md(R):
    """The report from results.json alone. Total: any number may be None and any row a failure record, and a section that still raises says so and leaves the others."""
    labs = {n: r if isinstance(r, dict) else {} for n, r in rows_of(dig(R, "labs")).items()}
    named = {n: r if isinstance(r, dict) else {} for n, r in rows_of(dig(R, "named")).items()}
    descriptive = {n: r if isinstance(r, dict) else {} for n, r in rows_of(dig(R, "descriptive")).items()}
    computed = {n: r for n, r in labs.items() if r.get("computable")}
    c = dig(R, "counts") or {}
    margin = num(dig(R, "margin"), ".3f")

    def why(r):
        return f"not computable: {r.get('reason') or 'no reason recorded'}"

    def head():
        L = [f"# Confirmatory analysis, MIMIC, train → {dig(R, 'eval_split')}, `{dig(R, 'window')}` window", ""]
        if dig(R, "dry_run"):
            L += ["**DRY RUN. Val stands in for test and the model picks were made on the same rows, so every number is optimistic. Nothing "
                  "here is a result.**", ""]
        if dig(R, "plan_provisional"):
            L += ["The plan is provisional, so this could not be the real evaluation.", ""]
        h = dig(R, "headline", PRIMARY)
        if is_number(dig(h, "mean")):
            L += [f"**Headline.** Mean gain in AUROC over `{PRIMARY}` across {dig(h, 'labs')} labs: {num(dig(h, 'mean'))} (subject-bootstrap 95% interval {num(dig(h, 'lo'))}, {num(dig(h, 'hi'))}).", ""]
        elif has_failed(h):
            L += [f"**Headline.** The mean gain over `{PRIMARY}` failed: {h['failed']}.", ""]
        passed = dig(c, "chain_passed") or []
        L += [f"k = {dig(R, 'k')}, one-sided alpha = {dig(R, 'alpha')}: {dig(R, 'alpha_a')} for Family A, first-step z = {num(dig(R, 'first_threshold'), '.3f')}; {dig(R, 'alpha_named')} for the named hypotheses, first-step z = "
              f"{num(dig(R, 'first_threshold_named'), '.3f')}. Models offered: {', '.join(map(str, dig(R, 'models') or [])) or DASH}. Plan SHA-256 {dig(R, 'plan_sha256')}.", "",
              f"Family A over `{PRIMARY}`: L1 partial association in {dig(c, 'l1')} labs, L2 gain above 0 in {dig(c, 'l2')}, L3 gain above +{dig(R, 'min_gain')} in {dig(c, 'l3')}. "
              f"Named hypotheses: N1 in {dig(c, 'named_n1')}, N2 in {dig(c, 'named_n2')}. Upper-bound chain at +{margin}: passed over {', '.join(f'`{b}`' for b in passed) or 'no base'}. "
              "Family A with the named hypotheses and the chain together have error ≤ 0.05.", ""]
        if dig(c, "not_computable"):
            L += ["Not computable, so not rejected and still counted: " + ", ".join(map(str, c["not_computable"])) + ".", ""]
        if dig(R, "failures"):
            L += ["Failed and recorded, with no effect on any test above unless the row is a lab's own test: " + "; ".join(map(str, R["failures"])) + ".", ""]
        return L

    def family():
        L = [f"## Family A, gain over `{PRIMARY}`", "",
             "| Lab | Model | n | Subjects | Prevalence | Base AUROC | Δ | SE (eval) | SE (with training) | Simultaneous interval | z1 | z2 | z3 | Levels cleared |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, r in labs.items():
            if not r.get("computable"):
                L.append(f"| {n} | {r.get('pick') or DASH} | {r.get('n', DASH)} | {r.get('subjects', DASH)} | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} | {why(r)} | {DASH} | {DASH} | {DASH} | {r.get('levels_cleared') or 0} |")
                continue
            a = dig(r, "family_a")
            L.append(f"| {n} | {r.get('pick') or DASH} | {r.get('n', DASH)} | {r.get('subjects', DASH)} | {num(r.get('prevalence'), '.1%')} | {num(dig(a, 'a_base'), '.3f')} | {num(dig(a, 'delta'))} | {num(dig(a, 'se'), '.4f')} | {num(dig(a, 'se_t'), '.4f')} | "
                     f"{num(dig(a, 'simultaneous', 0))}, {num(dig(a, 'simultaneous', 1))} | {num(dig(a, 'z1'), '.2f')} | {num(dig(a, 'z2'), '.2f')} | {num(dig(a, 'z3'), '.2f')} | {r.get('levels_cleared', DASH)} |")
        flags = [n for n, r in labs.items() if r.get("flag_gain_without_association")]
        L += ["", "L1: the PPG score carries information beyond the base. L2: the stack gains discrimination. L3: practical gain."]
        if flags:
            L += ["", "Flag, not a claim: z2 past the threshold without L1 in " + ", ".join(flags) + "."]
        return L

    def named_hypotheses():
        L = ["## Named hypotheses, the lab value over `now`", "", "| Lab | Model | n | Subjects | R² base | ΔR² | SE | Partial R² | Simultaneous interval | z1 | z2 | Levels cleared |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, r in named.items():
            if r.get("computable"):
                b = dig(r, "named")
                L.append(f"| {n} | {r.get('pick') or DASH} | {r.get('n', DASH)} | {r.get('subjects', DASH)} | {num(dig(b, 'r2_base'), '.3f')} | {num(dig(b, 'delta'))} | {num(dig(b, 'se'), '.4f')} | {num(dig(b, 'partial_r2'))} | "
                         f"{num(dig(b, 'simultaneous', 0))}, {num(dig(b, 'simultaneous', 1))} | {num(dig(b, 'z1'), '.2f')} | {num(dig(b, 'z2'), '.2f')} | {r.get('levels_cleared', DASH)} |")
            else:
                L.append(f"| {n} | {r.get('pick') or DASH} | {r.get('n', DASH)} | {r.get('subjects', DASH)} | {DASH} | {DASH} | {DASH} | {DASH} | {why(r)} | {DASH} | {DASH} | {r.get('levels_cleared') or 0} |")
        return L

    def order():
        L = ["### Order of rejections", "", "| Round | Branch | Level | z | Pending | Threshold |", "|---|---|---|---|---|---|"]
        L += [f"| {dig(t, 'round')} | {dig(t, 'lab')} | {dig(t, 'level')} | {num(dig(t, 'z'), '.2f')} | {dig(t, 'A')} | {num(dig(t, 'threshold'), '.3f')} |" for t in dig(R, "sequence", "trace") or []]
        return L + ["", f"Branches with a test pending at the end: {dig(R, 'sequence', 'final_A')}."]

    def chain():
        L = ["## Upper-bound chain on the mean gain over Family A", "",
             f"H0 at each step: the true mean gain over the {dig(R, 'k')} labs is at least +{margin}. A step runs only if the one before passed, and passes if the one-sided 97.5% upper bound of the mean is below the margin. "
             "The mean and its bound are reported whether or not the step ran. A step with a lab missing from its mean cannot pass.", ""]
        mr = dig(R, "margin_rule")
        L += [f"Margin: median of what MAP, pressors and NBP each add over `{PRIMARY}` across the labs, {num(dig(mr, 'median_gain'), '.4f')}, rounded up, floor +{dig(mr, 'floor')}: +{num(dig(mr, 'margin'), '.3f')}. "
              f"What the three add together: mean {num(dig(mr, 'joint_mean'), '.4f')}, median {num(dig(mr, 'joint_median'), '.4f')} across the labs.", "",
              "| Base | Tested | Passed | Labs in the mean | Mean gain (95% interval) | Training SD | Upper bound |", "|---|---|---|---|---|---|---|"]
        steps = [s for s in dig(R, "chain") or [] if isinstance(s, dict)]
        for s in steps:
            L.append(f"| `{s.get('base')}` | {'yes' if s.get('tested') else 'no'} | {'yes' if s.get('passed') else 'no'} | {s.get('labs', DASH)} of {s.get('k', DASH)} | {num(s.get('mean'))} ({num(s.get('lo'))}, {num(s.get('hi'))}) | "
                     f"{num(s.get('training_sd'), '.4f')} | {num(s.get('upper'))} |")
        L += ["", "### Per-lab upper bounds, descriptive", "", "No test or claim reads these. A mean can hide one lab: this table and L1 to L3 address that."]
        for s in steps:
            L += ["", f"#### Over `{s.get('base')}`, largest {num(s.get('largest_upper'))}", "", "| Lab | Δ | SE (eval) | SE (with training) | Upper bound |", "|---|---|---|---|---|"]
            for n, r in labs.items():
                x = dig(r, "chain", s.get("base"))
                if not r.get("computable"):
                    L.append(f"| {n} | {why(r)} | {DASH} | {DASH} | {DASH} |")
                elif has_failed(x):
                    L.append(f"| {n} | failed: {x['failed']} | {DASH} | {DASH} | {DASH} |")
                else:
                    L.append(f"| {n} | {num(dig(x, 'delta'))} | {num(dig(x, 'se'), '.4f')} | {num(dig(x, 'se_t'), '.4f')} | {num(dig(x, 'upper'))} |")
        return L

    def papagei():
        shown = {n: r for n, r in computed.items() if r.get("pick") == "papagei" and is_number(r.get("levels_cleared")) and r["levels_cleared"] >= 1}
        if not shown:
            return []
        L = ["## Confirmed PaPaGei labs beside the other models", "", "PaPaGei was pretrained on these cohorts. Not tested.", "", "| Lab | Model | Δ | SE (eval) | z1 | z2 |", "|---|---|---|---|---|---|"]
        for n, r in shown.items():
            a = dig(r, "family_a")
            L.append(f"| {n} | papagei | {num(dig(a, 'delta'))} | {num(dig(a, 'se'), '.4f')} | {num(dig(a, 'z1'), '.2f')} | {num(dig(a, 'z2'), '.2f')} |")
            if has_failed(r.get("comparators")):
                L.append(f"| {n} | the others | failed | {DASH} | {DASH} | {DASH} |")
            L += [f"| {n} | {m} | {num(dig(x, 'delta'))} | {num(dig(x, 'se'), '.4f')} | {num(dig(x, 'z1'), '.2f')} | {num(dig(x, 'z2'), '.2f')} |" for m, x in rows_of(r.get("comparators")).items()]
        return L

    def rate():
        rows = [(n, r.get("pick"), dig(r, "family_a", "delta"), dig(r, "family_a", "z1"), r.get("beside_rate")) for n, r in computed.items() if r.get("beside_rate")]
        rows += [(n, r.get("pick"), dig(r, "gains", PRIMARY, "delta"), None, r.get("beside_rate")) for n, r in descriptive.items() if r.get("computable") and r.get("beside_rate")]
        if not rows:
            return []
        L = ["## Beside the pick: pulse rate and its variability, not tested", "",
             f"The `{RATE}` model over `{PRIMARY}`, and what the pick adds over `{PRIMARY}` + the rate score. Descriptive rows, with unadjusted intervals. They enter no decision above.", "",
             f"| Lab | Model | Δ pick | z1 pick | Δ {RATE} | z1 {RATE} | Δ pick over {PRIMARY} + {RATE} |", "|---|---|---|---|---|---|---|"]
        for n, pick, delta, z1, b in rows:
            L.append(f"| {n} | {pick} | {num(delta)} | {num(z1, '.2f')} | {FAILED if has_failed(b) else num(dig(b, 'rate', 'delta'))} | {num(dig(b, 'rate', 'z1'), '.2f')} | {interval(dig(b, 'pick_over_rate'))} |")
        return L

    def ladder():
        bases = [b for b in BL.LADDER if any(b in rows_of(r.get("ladder")) for r in computed.values())]
        L = ["## Other rungs of the ladder, same model, not tested", "", "| Lab | " + " | ".join(f"Δ {b}" for b in bases) + " |", "|---|" + "---|" * len(bases)]
        L += ["| " + n + " | " + " | ".join((FAILED if has_failed(dig(r, "ladder", b)) else num(dig(r, "ladder", b, "delta"))) for b in bases) + " |" for n, r in computed.items()]
        return L

    def by_age():
        L = ["## Gain by age of the previous value, descriptive", "", "Could PPG stand in where the last value is old or absent? Unadjusted 95% intervals; a stratum with too few subjects in a class gives no gain.", ""]
        ages = [a[0] for a in PREVIOUS_AGE] + ["never"]
        for base in (PRIMARY, "locf"):
            L += [f"### Over `{base}`", "", "| Lab | " + " | ".join(ages) + " | Heterogeneity p |", "|---|" + "---|" * (len(ages) + 1)]
            for n, sec in [(n, dig(r, "secondary", "by_previous_age")) for n, r in computed.items()] + [(n, r.get("by_previous_age")) for n, r in descriptive.items() if r.get("computable")]:
                x = dig(sec, base)
                if has_failed(sec) or has_failed(x):
                    L.append(f"| {n} | " + " | ".join([FAILED] * len(ages)) + f" | {DASH} |")
                elif isinstance(x, dict):
                    L.append(f"| {n} | " + " | ".join(interval(dig(x, "strata", a)) for a in ages) + f" | {num(dig(x, 'heterogeneity', 'p'), '.3f')} |")
            L.append("")
        return L

    def other_rows():
        L = ["## Other descriptive rows over `now`", "", "| Lab | Residual arm Δ | Brier skill | Calibration slope, stack | Calibration intercept, stack | Calibration slope, base | ΔR² on the value | Partial R² |", "|---|---|---|---|---|---|---|---|"]
        for n, r in computed.items():
            s = r.get("secondary")
            if has_failed(s):
                L.append(f"| {n} | " + " | ".join([FAILED] * 7) + " |")
                continue
            L.append(f"| {n} | {interval(dig(s, 'residual_arm'))} | {interval(dig(s, 'brier'), 'skill')} | {num(dig(s, 'calibration', 'stack', 'slope'), '.3f') if not has_failed(dig(s, 'calibration', 'stack')) else FAILED} | "
                     f"{num(dig(s, 'calibration', 'stack', 'intercept'), '+.3f') if not has_failed(dig(s, 'calibration', 'stack')) else FAILED} | {num(dig(s, 'calibration', 'base', 'slope'), '.3f') if not has_failed(dig(s, 'calibration', 'base')) else FAILED} | "
                     f"{interval(dig(s, 'value'))} | {FAILED if has_failed(dig(s, 'value')) else num(dig(s, 'value', 'partial_r2'))} |")
        return L

    def net():
        L = ["### Net benefit at thresholds fixed in advance: gain over the base (95% interval)", "", "| Lab | " + " | ".join(f"{t:.2f}" for t in NET_BENEFIT_AT) + " |", "|---|" + "---|" * len(NET_BENEFIT_AT)]
        for n, r in computed.items():
            x = dig(r, "secondary", "net_benefit")
            cells = [FAILED if has_failed(x) else interval(dig(x, i), "gain") for i in range(len(NET_BENEFIT_AT))]
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        return L

    def subgroups():
        L = ["### Subgroups: gain over `now`, unadjusted 95% interval, heterogeneity reported and not a test of any claim", "", "AF is the ICD-9 code of the admission, assigned at discharge: `no_af` is not the same as sinus rhythm.", "",
             "| Lab | Subgroup | Levels | Heterogeneity p |", "|---|---|---|---|"]
        for n, r in computed.items():
            sub = dig(r, "secondary", "subgroups")
            if has_failed(sub):
                L.append(f"| {n} | all | failed | {DASH} |")
            for name, x in rows_of(sub).items():
                levels = FAILED if has_failed(x) else "; ".join(f"{lv}: {interval(v)}" for lv, v in rows_of(dig(x, "strata")).items()) or DASH
                L.append(f"| {n} | {name} | {levels} | {num(dig(x, 'heterogeneity', 'p'), '.3f')} |")
        return L

    def check():
        def ratio(a, b):
            return f"{a / b:.2f}" if is_number(a) and is_number(b) and b else DASH
        L = ["## Check, not a decision: subject bootstrap SE against cluster DeLong SE", "", f"| Lab | {PRIMARY} | {UPPER} |", "|---|---|---|"]
        L += [f"| {n} | {ratio(dig(r, 'boot_se', PRIMARY), dig(r, 'family_a', 'se'))} | {ratio(dig(r, 'boot_se', UPPER), dig(r, 'chain', UPPER, 'se'))} |" for n, r in computed.items()]
        return L

    def outside():
        L = ["## Labs outside the family, not tested", "", f"Gain over `{PRIMARY}` and `{UPPER}`, unadjusted 95% interval from the eval SE.", "",
             f"| Lab | Model | n | Subjects | Δ {PRIMARY} | Δ {UPPER} |", "|---|---|---|---|---|---|"]
        for n, r in descriptive.items():
            if r.get("computable"):
                L.append(f"| {n} | {r.get('pick') or DASH} | {r.get('n', DASH)} | {r.get('subjects', DASH)} | {interval(dig(r, 'gains', PRIMARY))} | {interval(dig(r, 'gains', UPPER))} |")
            else:
                L.append(f"| {n} | {r.get('pick') or DASH} | {r.get('n', DASH)} | {r.get('subjects', DASH)} | {why(r)} | {DASH} |")
        return L

    L = []
    for name, section in (("Summary", head), ("Family A", family), ("Named hypotheses", named_hypotheses), ("Order of rejections", order), ("Upper-bound chain", chain), ("PaPaGei labs", papagei), ("Pulse rate", rate),
                          ("Ladder", ladder), ("Gain by age of the previous value", by_age), ("Other descriptive rows", other_rows), ("Net benefit", net), ("Subgroups", subgroups), ("Bootstrap check", check), ("Labs outside the family", outside)):
        try:
            rows = section()
        except Exception as e:
            rows = [f"## {name}", "", f"This section could not be rendered ({failure(e)}). results.json holds its numbers."]
        L += [*rows, ""] if rows else []
    return "\n".join(L).rstrip("\n") + "\n"


def write_report(R, folder):
    """report.md in a folder that already holds results.json. A failure leaves results.json as it is, a note beside it, and the command that renders it again. Returns the text, or None."""
    note = folder / "report_failed.txt"
    try:
        text = report_md(R)
    except Exception as e:
        message = f"report.md could not be rendered ({failure(e)}), and {folder / 'results.json'} is complete. After fixing the code, this reads no data: python -m ppglabbench.confirmatory render" + (" --dry-run" if dig(R, "dry_run") else "")
        note.write_text(message + "\n")
        print(message, file=sys.stderr, flush=True)
        return None
    (folder / "report.md").write_text(text)
    if note.exists():
        note.unlink()
    return text


def execute(d, bcols, labs, tiers, plan, plan_sha, eval_split, out, jobs, dry, seed=None):
    """The evaluation. Real run and dry run differ only in the eval split and the output folder. Nothing is written or printed about the eval split until everything has been computed. The
    statistics then go to a folder that appears in one move, so that the split counts as opened and results.json is safe before anything that could still fail: predictions, manifest, report."""
    if eval_split != ("val" if dry else "test"):
        raise ValueError("val stands in for test only in a dry run")
    d = d[d.split.isin(["train", eval_split])].reset_index(drop=True)
    missing = model_problems(plan, d.columns, bcols)
    if missing:
        sys.exit(f"picked models without features: {missing}")
    res = protected_map(whole_lab, [(evaluate_lab, dict(lab=n, pick=plan["picks"].get(n)), d, bcols, tiers, n, labs[n], lab_entry(plan, n), eval_split) for n in plan["family"]], jobs)
    described = protected_map(whole_lab, [(describe_lab, dict(lab=n, pick=plan["picks"].get(n)), d, bcols, tiers, n, labs[n], plan["picks"].get(n), eval_split) for n in plan["descriptive"]], jobs)
    named = protected_map(whole_lab, [(evaluate_named, dict(lab=f"t1_{n}", name=n, pick=plan["picks_t1"].get(n)), d, bcols, tiers, n, labs[n], plan["picks_t1"].get(n), eval_split) for n in plan["named"]], jobs)
    computed = {r["lab"]: r["arrays"] for r in res if r["computable"]}
    headline = {b: attempt(mean_gain, computed, b, plan["picks"]) for b in CHAIN} if computed else {}
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    bare = lambda rows: [{key: v for key, v in r.items() if key != "arrays"} for r in rows]
    (tmp / "per_lab.json").write_text(json.dumps(clean(dict(res=bare(res), described=bare(described), named=bare(named), headline=headline), keep_nan=True)))
    R = attempt(assemble, res, described, named, plan, plan_sha, eval_split, dry, headline)
    assembled = not has_failed(R)
    if assembled:
        (tmp / "results.json").write_text(json.dumps(clean(R), indent=1))
        (tmp / "per_lab.json").unlink()
    if dry:
        shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)

    def predictions():
        flat = anonymise_arrays({r["lab"]: r["arrays"] for r in (*res, *described, *named) if r["computable"]}, np.random.default_rng(seed))
        np.savez_compressed(out / "predictions_partial.npz", **flat)
        (out / "predictions_partial.npz").rename(out / "predictions.npz")

    def manifest():
        write_manifest(out, "confirmatory_dryrun" if dry else "confirmatory_test",
                       dict(eval_split=eval_split, window=plan["window"], dry_run=dry, plan_sha256=plan_sha, plan_code_version=plan["code_version"], alpha=ALPHA, alpha_a=ALPHA_A, alpha_named=ALPHA_NAMED,
                            k=len(plan["family"]), named=plan["named"], margin=plan["margin"]["margin"], boost=BL.BOOST, min_subjects=MIN_SUBJECTS, models=plan["models"]),
                       dict(dig(R, "counts") or {}, labs_computable=sum(r["computable"] for r in res), failures=dig(R, "failures") or []))
    problems = [f"{name} was not written: {x['failed']}" for name, x in (("predictions.npz", attempt(predictions)), ("manifest.json", attempt(manifest))) if has_failed(x)]
    if problems:
        (out / "steps_failed.txt").write_text("\n".join(problems) + "\n")
        print("; ".join(problems), file=sys.stderr, flush=True)
    if not assembled:
        sys.exit(f"the decisions could not be assembled ({R['failed']}). The statistics per lab are safe in {out / 'per_lab.json'}. After fixing the code, this reads no data: python -m ppglabbench.confirmatory render" + (" --dry-run" if dry else ""))
    write_report(R, out)
    return R


def model_problems(plan, columns, bcols):
    """Picked models whose features are not among the columns."""
    used = {(n, p) for picks in (plan["picks"], plan["picks_t1"]) for n, p in picks.items() if p}
    return sorted({p for n, p in used if feature_cols(p, bcols, n)[0] not in columns})


def dry_run_problems(sha):
    """`run` follows a dry run of this very plan that went end to end on validation rows with nothing failed: the same code, so that a crash is found before the test split is opened."""
    f = DRY_OUT / "results.json"
    if not f.exists():
        return [f"{f} does not exist: run the dry run of this plan first"]
    R = json.loads(f.read_text())
    problems = [] if R.get("plan_sha256") == sha else ["the dry run used another plan"]
    if R.get("failures"):
        problems.append(f"the dry run recorded failures: {'; '.join(R['failures'])}")
    if not (DRY_OUT / "report.md").exists() or (DRY_OUT / "report_failed.txt").exists() or (DRY_OUT / "steps_failed.txt").exists():
        problems.append("the dry run did not finish its report, predictions and manifest")
    return problems


def plan_problems(plan, changed_files, sha=None, bound=None, cnn_sha=None):
    """Reasons a plan cannot be used for the real evaluation. changed_files: code changed between the plan's commit and HEAD. sha: the plan file's SHA-256; bound: the
    SHA-256 committed in config.yaml, so that the plan a third party can check is the plan that runs. cnn_sha: the CNN score file's SHA-256 now, which has to be the one the plan used."""
    problems = []
    if plan.get("cnn_scores_sha256") != cnn_sha or cnn_sha is None:
        problems.append(f"the CNN score file's SHA-256 {cnn_sha} is not the plan's {plan.get('cnn_scores_sha256')}")
    if not bound:
        problems.append("benchmark.plan_sha256 is not set in config.yaml")
    elif sha != bound:
        problems.append(f"the plan's SHA-256 {sha} differs from benchmark.plan_sha256 {bound}")
    if plan.get("provisional", True):
        problems.append("the plan is provisional: " + "; ".join(plan.get("provisional_reasons", [])))
    if changed_files:
        problems.append("code changed since the plan: " + ", ".join(changed_files))
    return problems


def config_settings(text):
    """config.yaml without the keys that change after the plan."""
    c = yaml.safe_load(text)
    for key in AFTER_THE_PLAN:
        c.get("benchmark", {}).pop(key, None)
    return c


def code_changed_since(commit):
    """Analysis code and settings that differ between the plan's commit and HEAD. Tests, text and the keys written after the plan may change once it is frozen."""
    if commit.endswith(("-dirty", "unknown")):
        return [f"plan commit {commit} is not a clean commit"]
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", commit, "HEAD", "--", *FROZEN_PATHS], capture_output=True, text=True)
    if r.returncode:
        return [f"cannot compare with {commit}"]
    changed = [f for f in r.stdout.split() if f.endswith((".py", ".yaml")) and not Path(f).name.startswith("test_")]
    if "config.yaml" in changed:
        old, new = (subprocess.run(["git", "-C", str(ROOT), "show", f"{c}:config.yaml"], capture_output=True, text=True).stdout for c in (commit, "HEAD"))
        if config_settings(old) == config_settings(new):
            changed.remove("config.yaml")
    return changed


def read_plan():
    if not PLAN.exists():
        sys.exit(f"{PLAN} does not exist: run the plan step first")
    text = PLAN.read_bytes()
    return json.loads(text), hashlib.sha256(text).hexdigest()


def cmd_plan(args):
    if TEST_OUT.exists():
        sys.exit(f"{TEST_OUT} exists: the test split was opened, so the plan is closed")
    d, bcols, labs, tiers = BL.load_mimic(args.window, need_cnn=args.window == PRIMARY_WINDOW)
    d = d[d.split.isin(["train", "val"])].reset_index(drop=True)      # test rows are gone before anything is computed
    plan = make_plan(d, bcols, labs, tiers, args.B, args.jobs, args.window, cnn_sha=BL.cnn_sha256(args.window))
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(clean(plan), indent=1))
    print(f"{PLAN}\nSHA-256 for benchmark.plan_sha256 in config.yaml: {hashlib.sha256(PLAN.read_bytes()).hexdigest()}\nprovisional: {plan['provisional']} {plan['provisional_reasons']}\n"
          f"picks: {plan['picks']}\nT1 picks: {plan['picks_t1']}\nr-hat: {clean(plan['r_hat'])}\nmargin: {plan['margin']['margin']} (anchor {plan['margin']['anchor']}, floor binds: {plan['margin']['floor_binds']})")


def cmd_dry_run(args):
    plan, sha = read_plan()
    if plan.get("cnn_scores_sha256") != BL.cnn_sha256(plan["window"]):
        sys.exit(f"the CNN score file is not the one the plan used: run the plan step again")
    d, bcols, labs, tiers = BL.load_mimic(plan["window"], need_cnn="cnn" in plan["models"])
    d = d[d.split.isin(["train", "val"])].reset_index(drop=True)      # test rows are gone before anything is computed
    execute(d, bcols, labs, tiers, plan, sha, "val", DRY_OUT, args.jobs, dry=True)
    show(DRY_OUT)


def cmd_run(args):
    BL.check_test_allowed(TEST_OUT.name, args.open_test)
    earlier = sorted(p.name for p in OUT.glob("mimic_test*"))
    if earlier:
        sys.exit(f"test split not opened: {earlier} exist, and the confirmatory evaluation has to come first")
    plan, sha = read_plan()
    problems = plan_problems(plan, code_changed_since(plan["code_version"]), sha, BL.C.get("plan_sha256"), BL.cnn_sha256(plan.get("window", PRIMARY_WINDOW))) or dry_run_problems(sha)
    if problems:
        sys.exit("test split not opened: " + "; ".join(problems))
    # whatever can refuse, refuses before the attempt is logged. No test score or statistic is read here: the CNN's test rows are checked by their IDs and folds
    d, bcols, labs, tiers = BL.load_mimic(plan["window"], need_cnn=True)
    problems = model_problems(plan, d.columns, bcols) + [f"{n} is not among the labs" for n in (*plan["family"], *plan["descriptive"], *plan["named"]) if n not in labs]
    if problems:
        sys.exit("test split not opened: " + "; ".join(map(str, problems)))
    BL.check_cnn_rows(d, BL.cnn_rows(plan["window"]), open_test=True)
    with open(ATTEMPTS, "a") as f:     # disclosed if a crash forces a second attempt; no result is shown until all of it succeeds
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {git_version()} plan {sha}\n")
    d, bcols, labs, tiers = BL.load_mimic(plan["window"], need_cnn=True, open_test=True)
    d = d[d.split.isin(["train", "test"])].reset_index(drop=True)
    execute(d, bcols, labs, tiers, plan, sha, "test", TEST_OUT, args.jobs, dry=False)
    show(TEST_OUT)


def show(folder):
    """The report as written, or what happened to it. Reading a file cannot lose the results."""
    report = folder / "report.md"
    print(report.read_text() if report.exists() else f"{folder} holds results.json and no report.md: see report_failed.txt there")


def cmd_render(args):
    """The report again from the results in the folder. Reads no data, no plan unless the decisions were never assembled, and does not count as an attempt."""
    folder, dry = (DRY_OUT, True) if args.dry_run else (TEST_OUT, False)
    results, raw = folder / "results.json", folder / "per_lab.json"
    if results.exists():
        R = json.loads(results.read_text())
    elif raw.exists():
        plan, sha = read_plan()
        x = json.loads(raw.read_text())
        R = clean(assemble(x["res"], x["described"], x["named"], plan, sha, "val" if dry else "test", dry, x["headline"]))
        results.write_text(json.dumps(R, indent=1))
        raw.unlink()
    else:
        sys.exit(f"{folder} holds neither results.json nor per_lab.json")
    if write_report(R, folder) is None:
        sys.exit(1)
    show(folder)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--window", default=PRIMARY_WINDOW, choices=["pre", "centered"])
    p.add_argument("--B", type=int, default=B_REFITS)
    p.add_argument("--jobs", type=int, default=3)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("dry-run")
    p.add_argument("--jobs", type=int, default=3)
    p.set_defaults(fn=cmd_dry_run)
    p = sub.add_parser("run")
    p.add_argument("--open-test", action="store_true", help="required: the one evaluation of the test split")
    p.add_argument("--jobs", type=int, default=3)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("render", help="the report again from results.json, reading no data")
    p.add_argument("--dry-run", action="store_true", help="the dry run's folder instead of the test folder")
    p.set_defaults(fn=cmd_render)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
