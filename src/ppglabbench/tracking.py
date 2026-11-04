"""Within-patient tracking, a secondary and unadjusted descriptive result: does a PPG score follow a patient's lab changes, or only tell patients apart?

Pooled AUROC and R² mix the two: a score that knows nothing about the moment can still rank windows if it knows who the patient is. This
step separates the parts, train -> eval, for the 10 family labs and the beat, PaPaGei, Pulse-PPG and CNN models. Models are fitted on
train subjects only; eval subjects only ever receive fixed models. The CNN's score is its abnormal-value logit, with its sign turned to follow the value, and its scores are fixed, so it has no
change_trained arm, no plasmode and no control rows.

Scores:
  pooled          baselines.ppg_score on the lab value (log scale for skewed labs): out of fold on train, full-train refit on eval.
                  It is the score behind the reported benchmark numbers. The classification score is used in auroc.csv only.
  change_trained  the same model family fitted on differences: PPG features at this draw minus the previous draw -> change in lab.
                  A pooled model is tuned to between-subject differences (PCA keeps those directions too), so a null for the pooled
                  score alone could be a failure of the training target rather than an absence of signal.

Tables (all aggregates; a cell with < MIN_CELL pairs or windows is left out):
  variance.csv    One-way random-effects ANOVA (unbalanced) of each target into between and within subject, on eval; `gamma_<gap>`
                  is half the mean squared change at that gap, so gamma / within variance shows how much of the within-subject
                  variance is already there at short gaps. At short gaps it is an upper bound on analytical plus fast noise.
  level.csv       within_r: Pearson r of score and target after subject-mean centring, subjects with >= 2 windows. between_r: r of
                  subject means, same subjects. pooled_r, pooled_r2: as reported elsewhere, all eval windows. pooled_r_between_only:
                  pooled r with each window's score replaced by its subject's mean. cov_share_between: share of the pooled
                  covariance that lies between subjects (unstable when pooled r is near 0). within_r_now: within_r after removing
                  the centred `now` covariates from both, coefficients from train, so a shared acuity swing is taken out.
                  between_r_age_sex: the same idea on subject means. within_rank_r: within_r on ranks of the centred values.
                  between_minus_within: paired difference. Subject means average out noise, so between_r is favoured over within_r even
                  if the association were equal at both levels; pooled_r is the single-window comparison.
  change.csv      Consecutive windows of a subject. Pairs are dropped if both windows use one lab measurement or their PPG overlaps.
                  r and four-quadrant concordance (sign agreement, pairs where either change is exactly 0 left out; `n` counts the
                  rest) of predicted and true change; r_now: r after removing the change in the `now` covariates from both, coefficients
                  from train pairs; MAE; mae_gain and skill (1 - SSE / SSE of no change) against predicting no change. The
                  reference for concordance is the null, not 0.5: a change_trained model has an intercept, so where a lab drifts (haemoglobin
                  falls in hospital) a score with no alignment still agrees more than half the time. Methods: no_change; ppg_raw
                  (score difference as it is); ppg_cal (slope from train pairs);
                  change_trained; hist (baselines `hist` prediction minus the previous window's value: it sees labs charted between
                  the two windows, and shares y_prev with the true change, so it gains from regression to the mean); hist_windows
                  (the same from this target's own earlier windows only); personal_mean (the patient's mean of all earlier values, copied, nothing fitted, minus the previous
                  window's value: `hist` holds that mean as a fitted covariate, so it covers it as a covariate and not as a plain comparator; not for eGFR, whose history is
                  creatinine); chart_trend (the `<lab>_trend` column, sign and r only;
                  creatinine's trend with the sign flipped for eGFR). Strata: all, gap bands, size of the true change by the train
                  quantiles SIZE_Q, same / other admission, both draws routine, no lab charted between the windows, not_dev, outside_zone (the true change is outside
                  the lab's exclusion zone, set in advance from analytical variation: ANALYTICAL_CV) and after_intervention (see INTERVENTIONS: a treatment in the 24 h before the
                  later draw and none in the 24 h before the earlier one, so it fell between them, up to the timing of its documentation).
                  Selecting on the true change is descriptive: read `all` first, r inside a size band is range-restricted.
  stack.csv       Current value from previous value, log1p(gap) and `now` (prev_gap_now), or the same with `full`, everything charted
                  (prev_gap_full); then + the change in score (ppg_change), + the current score (ppg_level), or + change_trained.
                  A gain from ppg_level alone with none from ppg_change means the score refines who the patient is, not what changed.
  auroc.csv       auroc_shuffled: pooled AUROC with scores shuffled within subject, so with no alignment to the patient's course. This is
                  the test of whether a pooled AUROC needs tracking: if it does, the shuffled value falls (the HR control shows how far).
                  auroc_within_pairs: P(score higher at the abnormal window) over windows of one subject, 0.5 under
                  the null. auroc_between_only: each window scored by its subject's mean score (it uses later windows: descriptive; it
                  can exceed the pooled AUROC, because averaging removes window noise). Exact split: pooled = share_within_pairs *
                  auroc_within_pairs + (1 - share) * auroc_between_pairs. The share is arithmetic, about 0.1% whatever the score does,
                  and between-subject pairs still carry each subject's state at that moment, so neither says the AUROC is a trait.
                  pop both_labels: subjects with both label classes, the only ones a shuffle within subject can move, so a small drop over all subjects alone
                  proves little: share_subjects_both_labels and shuffle_drop among them. per_patient_auroc_mean, _sd and _above_half: one AUROC per such subject, each subject weighing the same.
  family.csv      Means over the 10 labs per model, CI from one subject bootstrap shared by all labs.
  plasmode.csv    Positive control of known size: a synthetic target keeps a lab's subject means and within-subject variance, with a
                  within-subject part of correlation rho with monitor HR, which PPG can read. It runs through the same steps, with HR out of
                  `now`, and shows how much of a within-subject signal this pipeline would report. Best case: HR is the easiest thing to read.
                  rho is the planted within_r; the change-scale counterpart is the row of method oracle_hr, the change in true HR.

Controls: monitor HR (ECG; abnormal > 100 bpm) and arterial-line MAP (abnormal < 70), both taken over the PPG window from other sensors, go
through every step with themselves removed from the bases. Null: scores shuffled within subject, N_PERM times (change_trained and the stack
N_SLOW); `p` is a normal approximation from the null SD, `q_bh` adjusts the lab x model cells of each of within_r, ppg_raw r, change_trained r,
auroc_pooled and auroc_within_pairs. CIs resample eval
subjects with fitted models fixed (B = N_BOOT), so they leave out training variance. All results are exploratory on val.

Usage:
    python -m ppglabbench.tracking [--window pre] [--eval val] [--labs bun sodium] [--out DIR]
    python -m ppglabbench.tracking --eval test --open-test       # guarded as baselines, and only after the confirmatory evaluation
    python -m ppglabbench.tracking --render FOLDER               # the report again from the tables there, reading no data
Output: data/benchmark/results/tracking_<eval>[_centered]/{*.csv, report.md, manifest.json}
"""
import argparse
import json
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ppglabbench.baselines import LOG_LABS, N_BOOT, OUT, PG, PP, PPG_MODELS, base_reg, check_test_allowed, cnn_cols, load_mimic, ppg_score, task_frame
from ppglabbench.common import CFG, write_manifest
from ppglabbench.confirmatory import FAMILY_LABS, TEST_OUT, attempt, failure
from ppglabbench.labels import C

MODELS = ("beat", "papagei", "pulseppg", "cnn")
GAPS = dict(gap_le12h=(0, 12), gap_12_24h=(12, 24), gap_24_72h=(24, 72), gap_gt72h=(72, np.inf))
SIZE_Q = (0.5, 0.8)
PLASMODE_LABS, PLASMODE_RHO = ("potassium", "platelets"), (0.1, 0.2, 0.3, 0.5)
N_PERM, N_SLOW, MIN_CELL = 500, 200, 20
WINDOW_H, HALF_H, TOL_H = CFG["s4_sqi"]["window_min"] / 60, C["window_half_min"] / 60, 1e-6
PPG_START = dict(pre="pre_start_s", centered="best_start_s")
FAMILY_METRICS = {"pooled_r", "between_r", "within_r", "within_r_now", "between_minus_within", "r", "r_now", "concordance", "auroc_pooled",
                  "auroc_within_pairs"}
SKILL = ("mae", "mae_gain", "skill")
# Set before the test split is opened: a change inside 1.96 x sqrt(2) x CV could be the assay. CV as a fraction, the desirable imprecision (half the within-subject CV) of the biological variation
# database, 2014 edition, as reproduced by Westgard QC (www.westgard.com/clia-and-quality-regulation-requirements/quality-requirements/biodatabase1.html): serum rows, blood for haemoglobin, platelets and
# WBC, "sodium bicarbonate" for bicarbonate. eGFR has no row: creatinine's 2.98%, times the equation's slope of 1.2 above its knot
ANALYTICAL_CV = dict(bun=.0605, sodium=.003, potassium=.023, bicarbonate=.024, calcium=.0105, glucose=.028, haemoglobin=.0143, platelets=.046, wbc=.0573, egfr=.0358)
RCV = 1.96 * np.sqrt(2)
# Lab -> the 24 h treatment flag and the change it should cause. Insulin has only an at-draw flag, so it cannot be placed between two draws. Dialysis flags remove windows from eGFR and creatinine
INTERVENTIONS = dict(haemoglobin=("prbc_24h", 1.0), potassium=("dialysis_fluid_24h", -1.0))
ASSOCIATION = ("r", "concordance")
FAMILY_COLUMNS = ["ppg", "metric", "method", "stratum", "labs", "mean_est", "lo", "hi", "min_lab", "max_lab", "null_mean"]


def tracking_stem(eval_split, window):
    """Folder name under data/benchmark/results. It never matches `mimic_test*`: `confirmatory run` takes such a folder for an earlier look at the test split."""
    return f"tracking_{eval_split}" + ("_centered" if window == "centered" else "")


# ------------------------------------------------------------------------------------------------- statistics
def demean(x, code):
    """x (1-D or 2-D) minus its mean within each code."""
    code = np.unique(code, return_inverse=True)[1]
    n = np.bincount(code)
    if x.ndim == 1:
        return x - (np.bincount(code, x) / n)[code]
    return x - np.column_stack([np.bincount(code, col) / n for col in x.T])[code]


def sums(g, x, y, k):
    """Per-subject n, Σx, Σy, Σx², Σy², Σxy (k x 6): the sufficient statistics of a Pearson r under subject resampling."""
    return np.column_stack([np.bincount(g, w, k) for w in (np.ones(len(g)), x, y, x * x, y * y, x * y)])


def corr(s):
    n, a, b, aa, bb, ab = np.moveaxis(s, -1, 0)
    return (ab - a * b / n) / np.sqrt((aa - a * a / n) * (bb - b * b / n))


def slope(s):
    n, a, b, aa, _, ab = np.moveaxis(s, -1, 0)
    return (ab - a * b / n) / (aa - a * a / n)


def ratio(s):
    return s[..., 0] / s[..., 1]


def var_sums(g, y, k):
    """Per-subject n, Σy, (Σy)²/n, Σy², n², 1 for the variance components."""
    n, a = np.bincount(g, minlength=k).astype(float), np.bincount(g, y, k)
    return np.column_stack([n, a, np.divide(a * a, n, out=np.zeros(k), where=n > 0), np.bincount(g, y * y, k), n * n, (n > 0).astype(float)])


def var_parts(s):
    """Between and within variance: one-way random-effects ANOVA estimators for unbalanced groups, between floored at 0."""
    n, a, b, yy, n2, k = np.moveaxis(s, -1, 0)
    msw, msb = (yy - b) / (n - k), (b - a * a / n) / (k - 1)
    return np.maximum((msb - msw) / ((n - n2 / n) / (k - 1)), 0), msw


def icc(s):
    between, within = var_parts(s)
    return between / (between + within)


def interval(stat, s, c):
    """Point estimate, percentile 95% CI and bootstrap values of stat over subject-level sums s; c holds subject counts (B x k)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = stat(c @ s)
        return stat(s.sum(0)), *np.nanpercentile(boots, [2.5, 97.5]), boots


def cell(metric, stat, s, c, **tags):
    est, lo, hi, boots = interval(stat, s, c)
    keep = metric in FAMILY_METRICS and tags.get("method", "ppg_raw") in ("ppg_raw", "change_trained")
    return dict(metric=metric, est=float(est), lo=float(lo), hi=float(hi), _boot=boots if keep else None, **tags)


def with_null(row, null):
    """Null mean and 2.5-97.5% range for a row, and a two-sided normal-approximation p from the null SD."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if len(null) < 2:
        return row
    sd = max(null.std(ddof=1), 1e-12)
    return dict(row, null_mean=null.mean(), null_lo=np.percentile(null, 2.5), null_hi=np.percentile(null, 97.5),
                p=2 * norm.sf(abs(row["est"] - null.mean()) / sd))


def bh(p):
    """Benjamini-Hochberg q values."""
    p = np.asarray(p, float)
    o, m = np.argsort(p), len(p)
    q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1)
    return q


def within_shuffle(rows, code, rng):
    """Row map that permutes `rows` within subject and leaves every other row in place."""
    pm = np.arange(len(code))
    pm[rows[np.argsort(code[rows], kind="stable")]] = rows[np.lexsort((rng.random(len(rows)), code[rows]))]
    return pm


def auc_bins(s):
    u, inv = np.unique(s, return_inverse=True)
    return inv, len(u)


def auc_parts(inv, m, y, w):
    """Mann-Whitney numerator and abnormal x normal pair weight, with row weights; ties count one half."""
    pos, neg = np.bincount(inv, w * y, m), np.bincount(inv, w * (1 - y), m)
    return (pos * (np.cumsum(neg) - neg / 2)).sum(), pos.sum() * neg.sum()


def within_pairs(g, y):
    """Row indices (abnormal, normal) of every pair of windows of one subject with different labels."""
    a, b = [], []
    order = np.argsort(g, kind="stable")
    for rows in np.split(order, np.flatnonzero(np.diff(g[order])) + 1):
        p, n = rows[y[rows] == 1], rows[y[rows] == 0]
        if len(p) and len(n):
            a.append(np.repeat(p, len(n)))
            b.append(np.tile(n, len(p)))
    return (np.concatenate(a), np.concatenate(b)) if a else (np.empty(0, int), np.empty(0, int))


def planted(y, hr, code, rho):
    """Target with the same subject means and within-subject variance as y, whose within-subject part has correlation exactly rho with hr."""
    yc, z = demean(y, code), demean(hr, code)
    e = yc - (yc @ z) / (z @ z) * z
    unit = lambda v: v / np.sqrt((v @ v) / len(v))
    return y - yc + np.sqrt((yc @ yc) / len(yc)) * (rho * unit(z) + np.sqrt(1 - rho ** 2) * unit(e))


# ------------------------------------------------------------------------------------------------- frames and pairs
def controls(d, tiers):
    """HR and MAP as targets: known within-patient signals read from other sensors over the PPG window, removed from the bases.
    MAP also leaves the frame under its own name, because baselines.flexible builds hinges on `map_mmhg`."""
    without = lambda drop: {k: [c for c in v if c not in drop] for k, v in tiers.items()}
    hr = d.assign(hr_abn=(d.hr_mon > 100).astype(float).where(d.hr_mon.notna()))
    map_ = d.assign(map_value=d.map_mmhg, map_abn=d.map_lt70).drop(columns=["map_mmhg", "map_lt70"])
    return dict(hr_monitor=(hr, without({"hr_mon"}), dict(value="hr_mon", abnormal="hr_abn", locf=None)),
                map_artline=(map_, without({"map_mmhg", "map_lt70"}), dict(value="map_value", abnormal="map_abn", locf=None)))


def frame_of(d, tiers, name, spec, eval_split, window):
    """baselines.task_frame for the value of `name`, with times in hours, or None if the target is skipped."""
    f = task_frame(d, tiers, name, spec, "reg", eval_split)
    if f is None:
        return None
    q, y, tr, ev, groups, bases, _, age_h = f
    ts = pd.to_datetime(q.charttime)
    t = ((ts - ts.min()) / pd.Timedelta(hours=1)).to_numpy(float)
    off = q[f"{name}_offset_min"].to_numpy(float) / 60 if f"{name}_offset_min" in q else np.zeros(len(q))
    return SimpleNamespace(d=d, tiers=tiers, name=name, spec=spec, eval_split=eval_split, q=q, y=y, tr=tr, ev=ev, groups=groups,
                           code=np.unique(groups, return_inverse=True)[1], bases=bases, age_h=age_h, t=t, lab_t=t + off,
                           ppg_t=t + q[PPG_START[window]].to_numpy(float) / 3600)


def add_pairs(fr):
    """Consecutive windows of a subject (train pairs first, then eval), without pairs that share a lab measurement or overlap in PPG."""
    prev, cur, fr.dropped = [], [], 0
    for rows in (fr.tr, fr.ev):
        o = rows[np.lexsort((fr.t[rows], fr.groups[rows]))]
        same = fr.groups[o[1:]] == fr.groups[o[:-1]]
        a, b = o[:-1][same], o[1:][same]
        ok = (np.abs(fr.lab_t[b] - fr.lab_t[a]) > TOL_H) & (np.abs(fr.ppg_t[b] - fr.ppg_t[a]) >= WINDOW_H)
        fr.dropped += int((~ok).sum())
        prev.append(a[ok])
        cur.append(b[ok])
    fr.ptr, fr.pev = np.arange(len(prev[0])), np.arange(len(prev[1])) + len(prev[0])
    fr.prev, fr.cur = np.concatenate(prev), np.concatenate(cur)
    fr.dy, fr.gap = fr.y[fr.cur] - fr.y[fr.prev], fr.t[fr.cur] - fr.t[fr.prev]
    return fr


def plant_frame(fr, rho):
    """Copy of a lab frame restricted to windows with monitor HR, with the lab replaced by planted(.., rho)."""
    hr = fr.q.hr_mon.to_numpy(float)
    tr, ev = fr.tr[np.isfinite(hr[fr.tr])], fr.ev[np.isfinite(hr[fr.ev])]
    rows, y = np.concatenate([tr, ev]), fr.y.copy()
    y[rows] = planted(fr.y[rows], hr[rows], fr.code[rows], rho)
    return add_pairs(SimpleNamespace(**{**vars(fr), "tr": tr, "ev": ev, "y": y}))


def abnormal_direction(name):
    """+1 if an abnormal value of the lab is a high one, -1 if a low one. The CNN's logit points at the abnormal value, and tracking compares a score with the value."""
    rule = C["labs"][name][1] if name in C["labs"] else {"egfr": "<60"}.get(name, ">")
    return -1.0 if rule.startswith("<") or rule == "anaemia" else 1.0


def exclusion_zone(name, y_prev):
    """The change that analytical variation could produce, on the frame's scale: the log scale for skewed labs, the value's scale otherwise. None for a target with no zone set."""
    if name not in ANALYTICAL_CV:
        return None
    rcv = RCV * ANALYTICAL_CV[name]
    return np.full(len(y_prev), np.log1p(rcv)) if name in LOG_LABS else rcv * np.abs(y_prev)


def strata(fr):
    """Boolean masks over eval pairs, and the train quantiles of |change| behind the size bands."""
    prev, cur, dy, gap = fr.prev[fr.pev], fr.cur[fr.pev], fr.dy[fr.pev], fr.gap[fr.pev]
    lo, hi = np.quantile(np.abs(fr.dy[fr.ptr]), SIZE_Q)
    m = {"all": np.ones(len(dy), bool), **{k: (gap > a) & (gap <= b) for k, (a, b) in GAPS.items()},
         "change_small": np.abs(dy) < lo, "change_mid": (np.abs(dy) >= lo) & (np.abs(dy) < hi), "change_large": np.abs(dy) >= hi}
    adm = fr.q.hadm_id.to_numpy(float)
    m["same_admission"] = adm[cur] == adm[prev]
    m["other_admission"] = ~m["same_admission"]
    routine = fr.q.routine_draw.to_numpy(float)
    m["both_routine"] = (routine[cur] == 1) & (routine[prev] == 1)
    if fr.age_h is not None:      # the LOCF of the current window is no later than the previous window's measurement
        m["no_lab_between"] = np.minimum(fr.lab_t, fr.t - HALF_H)[cur] - fr.age_h[cur] <= fr.lab_t[prev] + TOL_H
    dev = fr.q.dev_set.to_numpy(bool)[cur]
    if dev.any():
        m["not_dev"] = ~dev
    zone = exclusion_zone(fr.name, fr.y[prev])
    if zone is not None:
        m["outside_zone"] = np.abs(dy) >= zone
    if fr.name in INTERVENTIONS and INTERVENTIONS[fr.name][0] in fr.q:
        flag = fr.q[INTERVENTIONS[fr.name][0]].to_numpy(float)
        m["after_intervention"] = (flag[prev] == 0) & (flag[cur] == 1)
    return m, (float(lo), float(hi))


# ------------------------------------------------------------------------------------------------- the five steps
def variance_rows(fr, masks, c, k):
    """Step 1: components on all eval windows and on subjects with >= 2 windows; semivariance per gap band over the within variance."""
    g, y = fr.g, fr.y[fr.ev]
    multi = np.bincount(g, minlength=k)[g] >= 2
    rows = []
    for pop, m in (("all", np.ones(len(g), bool)), ("2plus", multi)):
        s = var_sums(g[m], y[m], k)
        tags = dict(pop=pop, n=int(m.sum()), subjects=len(np.unique(g[m])))
        rows += [cell("icc", icc, s, c, **tags),
                 cell("within_share", lambda v: 1 - icc(v), s, c, **tags),
                 cell("var_between", lambda v: var_parts(v)[0], s, c, **tags),
                 cell("var_within", lambda v: var_parts(v)[1], s, c, **tags)]
    s = var_sums(g[multi], y[multi], k)
    dy, pg = fr.dy[fr.pev], fr.pg
    for name in GAPS:
        m = masks[name]
        if m.sum() < MIN_CELL:
            continue
        band = np.column_stack([np.bincount(pg[m], dy[m] ** 2, k) / 2, np.bincount(pg[m], minlength=k).astype(float)])
        rows.append(cell(f"gamma_{name}", lambda v, w=6: v[..., w] / v[..., w + 1], np.hstack([s, band]), c, pop="pairs", n=int(m.sum()),
                         subjects=len(np.unique(pg[m]))))
        rows.append(cell(f"gamma_{name}_over_within", lambda v, w=6: (v[..., w] / v[..., w + 1]) / var_parts(v[..., :6])[1],
                         np.hstack([s, band]), c, pop="pairs", n=int(m.sum()), subjects=len(np.unique(pg[m]))))
    return rows


def now_residuals(fr, s):
    """Lab and score after subject centring and after removing the centred `now` covariates, coefficients from train; eval rows."""
    if not hasattr(fr, "now_c"):
        rows = np.concatenate([fr.tr, fr.ev])
        z = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()).fit(fr.bases["now"][fr.tr])
        fr.now_c = np.zeros((len(fr.y), z.transform(fr.bases["now"][:1]).shape[1]))
        fr.now_c[rows] = demean(z.transform(fr.bases["now"][rows]), fr.code[rows])
    rows = np.concatenate([fr.tr, fr.ev])
    yx = np.zeros((len(fr.y), 2))
    yx[rows] = demean(np.column_stack([fr.y[rows], s[rows]]), fr.code[rows])
    res = yx - Ridge(1.0, fit_intercept=False).fit(fr.now_c[fr.tr], yx[fr.tr]).predict(fr.now_c)
    return res[fr.ev, 0], res[fr.ev, 1]


def age_sex_residuals(fr, s, sid):
    """Subject means of lab and score minus a train fit on age and sex: (eval subject codes, lab, score)."""
    rows = np.concatenate([fr.tr, fr.ev])
    code = fr.code[rows]
    n = np.bincount(code, minlength=fr.code.max() + 1)
    cols = np.column_stack([fr.bases["age_sex"][rows], fr.y[rows], s[rows]])
    m = np.column_stack([np.bincount(code, col, len(n)) / np.maximum(n, 1) for col in cols.T])
    is_tr = np.bincount(fr.code[fr.tr], minlength=len(n)) > 0
    res = m[:, 2:] - LinearRegression().fit(m[is_tr, :2], m[is_tr, 2:]).predict(m[:, :2])
    ev_codes = np.unique(fr.code[fr.ev])
    groups = np.empty(len(n), dtype=fr.groups.dtype)
    groups[fr.code] = fr.groups
    return np.searchsorted(sid, groups[ev_codes]), res[ev_codes, 0], res[ev_codes, 1]


def level_rows(fr, s, c, k, sid, pms):
    """Step 2: pooled, between and within association of score and target."""
    g, x, y = fr.g, s[fr.ev], fr.y[fr.ev]
    n = np.bincount(g, minlength=k)
    multi, two = n[g] >= 2, np.flatnonzero(n >= 2)
    den = np.maximum(n, 1)
    xbar, ybar = np.bincount(g, x, k) / den, np.bincount(g, y, k) / den
    xc, yc = x - xbar[g], y - ybar[g]
    tags_all = dict(n=len(g), subjects=int((n > 0).sum()))
    tags_two = dict(n=int(multi.sum()), subjects=len(two))
    total, within = sums(g, x, y, k), sums(g[multi], xc[multi], yc[multi], k)
    between = sums(two, xbar[two], ybar[two], k)
    err = np.column_stack([np.bincount(g, minlength=k).astype(float), np.bincount(g, y, k), np.bincount(g, y * y, k), np.bincount(g, (y - x) ** 2, k)])
    r2 = lambda v: 1 - v[..., 3] / (v[..., 2] - v[..., 1] ** 2 / v[..., 0])
    share = lambda v: 1 - v[..., 6] / (v[..., 5] - v[..., 1] * v[..., 2] / v[..., 0])
    yr, xr = now_residuals(fr, s)
    ag, ya, xa = age_sex_residuals(fr, s, sid)
    keep = n[ag] >= 2
    rows = [cell("pooled_r", corr, total, c, **tags_all), cell("pooled_r2", r2, err, c, **tags_all),
            cell("pooled_r_between_only", corr, sums(g, xbar[g], y, k), c, **tags_all),
            cell("cov_share_between", share, np.hstack([total, within[:, 5:6]]), c, **tags_all),
            cell("within_r", corr, within, c, **tags_two),
            cell("within_rank_r", corr, sums(g[multi], rankdata(xc[multi]), rankdata(yc[multi]), k), c, **tags_two),
            cell("within_r_now", corr, sums(g[multi], xr[multi], yr[multi], k), c, **tags_two),
            cell("between_r", corr, between, c, **tags_two),
            cell("between_minus_within", lambda v: corr(v[..., 6:]) - corr(v[..., :6]), np.hstack([within, between]), c, **tags_two),
            cell("between_r_age_sex", corr, sums(ag[keep], xa[keep], ya[keep], k), c, **tags_two),
            cell("between_r_all_subjects", corr, sums(np.flatnonzero(n > 0), xbar[n > 0], ybar[n > 0], k), c, **tags_all),
            cell("slope_within", slope, within, c, **tags_two), cell("slope_between", slope, between, c, **tags_two),
            cell("score_icc", icc, var_sums(g[multi], x[multi], k), c, **tags_two),
            cell("score_sd_within_over_lab", lambda v: np.sqrt(v[..., 3] / v[..., 4]), within, c, **tags_two)]
    if "dev_set" in fr.q and fr.q.dev_set.to_numpy(bool)[fr.ev].any():
        nd = multi & ~fr.q.dev_set.to_numpy(bool)[fr.ev]
        rows.append(cell("within_r_not_dev", corr, sums(g[nd], xc[nd], yc[nd], k), c, n=int(nd.sum()), subjects=len(np.unique(g[nd]))))
    if pms:
        sxx, syy = (xc[multi] ** 2).sum(), (yc[multi] ** 2).sum()
        null_w, null_p = [], []
        for pm in pms:
            xp = s[pm[fr.ev]]
            null_w.append(((xp - xbar[g])[multi] * yc[multi]).sum() / np.sqrt(sxx * syy))
            null_p.append(np.corrcoef(xp, y)[0, 1])
        for i, r in enumerate(rows):
            if r["metric"] == "within_r":
                rows[i] = with_null(r, null_w)
            if r["metric"] == "pooled_r":
                rows[i] = with_null(r, null_p)
    return rows


def null_change(M, dy, preds):
    """Null r and concordance per stratum; M is strata x pairs (0/1), preds yields null predicted changes."""
    n, b, bb = M.sum(1), M @ dy, M @ (dy * dy)
    rs, cs = [], []
    with np.errstate(divide="ignore", invalid="ignore"):
        for x in preds:
            a, aa, ab = M @ x, M @ (x * x), M @ (x * dy)
            rs.append((ab - a * b / n) / np.sqrt((aa - a * a / n) * (bb - b * b / n)))
            cs.append((M @ (x * dy > 0)) / (M @ (x * dy != 0)))
    return np.array(rs), np.array(cs)


def change_rows(fr, method, pred, masks, c, k, metrics, null=None, **tags):
    """Step 3: association of predicted and true change and its error against no change, per stratum. NaN predictions are left out."""
    dy, g = fr.dy[fr.pev], fr.pg
    finite = np.isfinite(pred)
    rows = []
    for j, (name, m) in enumerate(masks.items()):
        m = m & finite
        if m.sum() < MIN_CELL:
            continue
        d, p, gm = dy[m], pred[m], g[m]
        base = dict(method=method, stratum=name, n=int(m.sum()), subjects=len(np.unique(gm)), **tags)
        if "r" in metrics:
            row = cell("r", corr, sums(gm, p, d, k), c, **base)
            rows.append(with_null(row, null[0][:, j]) if null else row)
        if "concordance" in metrics:
            hit = np.column_stack([np.bincount(gm, p * d > 0, k), np.bincount(gm, p * d != 0, k)])
            row = cell("concordance", ratio, hit, c, **{**base, "n": int(hit[:, 1].sum())})
            rows.append(with_null(row, null[1][:, j]) if null else row)
        if "mae" in metrics:
            e = d - p
            err = np.column_stack([np.bincount(gm, minlength=k).astype(float), np.bincount(gm, np.abs(e), k), np.bincount(gm, e * e, k),
                                   np.bincount(gm, np.abs(d), k), np.bincount(gm, d * d, k)])
            rows.append(cell("mae", lambda v: v[..., 1] / v[..., 0], err, c, **base))
            if "skill" in metrics:
                rows.append(cell("mae_gain", lambda v: (v[..., 3] - v[..., 1]) / v[..., 0], err, c, **base))
                rows.append(cell("skill", lambda v: 1 - v[..., 2] / v[..., 4], err, c, **base))
    return rows


def change_now_row(fr, pred, c, k, **tags):
    """r of predicted and true change over eval pairs, after removing the change in the `now` covariates from both; coefficients from train pairs."""
    if not hasattr(fr, "dnow"):
        z = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()).fit(fr.bases["now"][fr.tr])
        z = z.transform(fr.bases["now"])
        fr.dnow = z[fr.cur] - z[fr.prev]
    yx = np.column_stack([fr.dy, pred])
    res = (yx - Ridge(1.0, fit_intercept=False).fit(fr.dnow[fr.ptr], yx[fr.ptr]).predict(fr.dnow))[fr.pev]
    return cell("r_now", corr, sums(fr.pg, res[:, 1], res[:, 0], k), c, stratum="all", n=len(res), subjects=len(np.unique(fr.pg)), **tags)


def window_history(fr):
    """Per frame row: mean and log1p count of this target's earlier windows, and the slope over the last two (per hour, gap at least 1 h)."""
    o = np.lexsort((fr.t, fr.groups))
    s = pd.DataFrame(dict(g=fr.groups[o], y=fr.y[o], t=fr.t[o]))
    by = s.groupby("g")
    n = by.cumcount()
    out = np.empty((len(o), 3))
    out[o] = np.column_stack([(by.y.cumsum() - s.y) / n.replace(0, np.nan), np.log1p(n),
                              (by.y.shift(1) - by.y.shift(2)) / (by.t.shift(1) - by.t.shift(2)).clip(lower=1)])
    return out


def history_changes(fr):
    """Changes predicted without PPG, over eval pairs (see the module docstring)."""
    y, prev, cur = fr.y, fr.prev, fr.cur
    out = {}
    if "hist" in fr.bases:
        out["hist"] = base_reg().fit(fr.bases["hist"][fr.tr], y[fr.tr]).predict(fr.bases["hist"][cur]) - y[prev]
    A = np.column_stack([fr.bases["age_sex"][cur], y[prev], np.log1p(fr.gap), window_history(fr)[cur]])
    out["hist_windows"] = base_reg().fit(A[fr.ptr], y[cur][fr.ptr]).predict(A) - y[prev]
    if fr.spec.get("hist") and not fr.spec.get("log_locf"):
        mean = fr.q[fr.spec["hist"][0]].to_numpy(float)[cur]
        out["personal_mean"] = (np.log(np.where(mean > 0, mean, np.nan)) if fr.name in LOG_LABS else mean) - y[prev]
    return {m: v[fr.pev] for m, v in out.items()}


def stack_rows(fr, extras, c, k, n_slow, seed, score=None):
    """Step 4: gain from adding each extra to the previous value, log1p(gap) and `now` (or `full`). The null shuffles the pooled score
    within subject in train and eval and refits the stacker, for ppg_change over prev_gap_now only."""
    head = np.column_stack([fr.y[fr.prev], np.log1p(fr.gap)])
    bases = dict(prev_gap_now=np.column_stack([head, fr.bases["now"][fr.cur]]))
    if "full" in fr.bases:
        bases["prev_gap_full"] = np.column_stack([head, fr.bases["full"][fr.cur]])
    y, dy, g, ptr, pev = fr.y[fr.cur], fr.dy, fr.pg, fr.ptr, fr.pev
    ye, de = y[pev], dy[pev]

    def fit(Z, e=None):
        A = Z if e is None else np.column_stack([Z, (e - e[ptr].mean()) / e[ptr].std()])
        return Ridge(1.0).fit(A[ptr], y[ptr]).predict(A[pev])

    rows = []
    tag = dict(n=len(ye), subjects=len(np.unique(g)))
    for name, B in bases.items():
        Z = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()).fit(B[ptr]).transform(B)
        p0 = fit(Z)
        for extra, e in extras.items():
            p1 = fit(Z, e)
            s = np.column_stack([np.bincount(g, minlength=k).astype(float), np.bincount(g, ye, k), np.bincount(g, ye * ye, k),
                                 np.bincount(g, (ye - p0) ** 2, k), np.bincount(g, (ye - p1) ** 2, k), np.bincount(g, np.abs(ye - p0), k),
                                 np.bincount(g, np.abs(ye - p1), k), np.bincount(g, de * de, k)])
            t = dict(base=name, extra=extra, **tag)
            gain = cell("delta_r2", lambda v: (v[..., 3] - v[..., 4]) / (v[..., 2] - v[..., 1] ** 2 / v[..., 0]), s, c, **t)
            if name == "prev_gap_now" and extra == "ppg_change" and score is not None:
                rng = np.random.default_rng([1, seed])
                sst = (ye * ye).sum() - ye.sum() ** 2 / len(ye)
                rows_all = np.concatenate([fr.tr, fr.ev])
                null = []
                for _ in range(n_slow):
                    pm = within_shuffle(rows_all, fr.code, rng)
                    null.append((((ye - p0) ** 2).sum() - ((ye - fit(Z, score[pm[fr.cur]] - score[pm[fr.prev]])) ** 2).sum()) / sst)
                gain = with_null(gain, null)
            rows += [gain,
                     cell("partial_r2", lambda v: 1 - v[..., 4] / v[..., 3], s, c, **t),
                     cell("delta_mae", lambda v: (v[..., 6] - v[..., 5]) / v[..., 0], s, c, **t),
                     cell("base_r2", lambda v: 1 - v[..., 3] / (v[..., 2] - v[..., 1] ** 2 / v[..., 0]), s, c, **t),
                     cell("base_skill_vs_no_change", lambda v: 1 - v[..., 3] / v[..., 7], s, c, **t)]
    return rows


def auroc_rows(fr, cols, family, sid, c, k, n_perm, seed):
    """Pooled AUROC of the classification score, split into within- and between-subject pairs, and with the within part removed."""
    f = task_frame(fr.d, fr.tiers, fr.name, fr.spec, "clf", fr.eval_split)
    if f is None:
        return []
    q, y, tr, ev, groups, *_ = f
    _, x = ppg_score(family["clf"], q[cols].to_numpy(float), y, groups, tr, ev)
    y, g = y[ev].astype(float), np.searchsorted(sid, groups[ev])
    n = np.bincount(g, minlength=k)
    xbar = np.bincount(g, x, k) / np.maximum(n, 1)
    rng = np.random.default_rng([2, seed])
    rows, B = [], c.shape[0]
    pos = np.bincount(g, y, k)
    both = (pos > 0) & (pos < n)
    if (n > 0).any():
        s = np.column_stack([both.astype(float), (n > 0).astype(float)])
        rows.append(cell("share_subjects_both_labels", ratio, s, c, pop="all", n=len(y), subjects=int((n > 0).sum()), prev=float(y.mean())))
    for pop, m in (("all", np.ones(len(y), bool)), ("2plus", n[g] >= 2), ("both_labels", both[g])):
        ym, gm, xm = y[m], g[m], x[m]
        tags = dict(pop=pop, n=int(m.sum()), subjects=len(np.unique(gm)), prev=float(ym.mean()))
        if min(ym.sum(), (1 - ym).sum()) < MIN_CELL:
            continue
        inv, u = auc_bins(xm)
        boots = np.array([auc_parts(inv, u, ym, c[b, gm].astype(float)) for b in range(B)])
        total = auc_parts(inv, u, ym, np.ones(len(ym)))
        pooled = total[0] / total[1]
        lo, hi = np.percentile(boots[:, 0] / boots[:, 1], [2.5, 97.5])
        row = dict(metric="auroc_pooled", est=pooled, lo=lo, hi=hi, _boot=(boots[:, 0] / boots[:, 1]) if pop == "all" else None, **tags)
        null = []
        idx = np.arange(len(ym))
        for _ in range(n_perm):
            a, b = auc_parts(inv[within_shuffle(idx, gm, rng)], u, ym, np.ones(len(ym)))
            null.append(a / b)
        rows.append(with_null(row, null))
        rows.append(dict(metric="auroc_shuffled", est=float(np.mean(null)), lo=float(np.percentile(null, 2.5)),
                         hi=float(np.percentile(null, 97.5)), _boot=None, **tags))
        rows.append(dict(metric="shuffle_drop", est=pooled - float(np.mean(null)), lo=pooled - float(np.percentile(null, 97.5)),
                         hi=pooled - float(np.percentile(null, 2.5)), _boot=None, **tags))
        inv_m, u_m = auc_bins(xbar[gm])
        a, b = auc_parts(inv_m, u_m, ym, np.ones(len(ym)))
        bo = np.array([auc_parts(inv_m, u_m, ym, c[r, gm].astype(float)) for r in range(B)])
        rows.append(dict(metric="auroc_between_only", est=a / b, lo=np.percentile(bo[:, 0] / bo[:, 1], 2.5),
                         hi=np.percentile(bo[:, 0] / bo[:, 1], 97.5), _boot=None, **tags))
        if pop == "all":
            i, j = within_pairs(g, y)
            if len(i) >= MIN_CELL:
                win = (x[i] > x[j]) + 0.5 * (x[i] == x[j])
                s = np.column_stack([np.bincount(g[i], win, k), np.bincount(g[i], minlength=k).astype(float)])
                t = dict(tags, n=len(i), subjects=len(np.unique(g[i])))
                perm = [((x[pm][i] > x[pm][j]) + 0.5 * (x[pm][i] == x[pm][j])).mean() for pm in
                        (within_shuffle(idx, g, rng) for _ in range(n_perm))]
                rows.append(with_null(cell("auroc_within_pairs", ratio, s, c, **t), perm))
                w = c @ s
                between = (boots[:, 0] - w[:, 0]) / (boots[:, 1] - w[:, 1])
                rows.append(dict(metric="share_within_pairs", est=len(i) / total[1], lo=np.percentile(w[:, 1] / boots[:, 1], 2.5),
                                 hi=np.percentile(w[:, 1] / boots[:, 1], 97.5), _boot=None, **t))
                rows.append(dict(metric="auroc_between_pairs", est=(total[0] - win.sum()) / (total[1] - len(i)), lo=np.percentile(between, 2.5),
                                 hi=np.percentile(between, 97.5), _boot=None, **t))
                rows += per_patient_rows(g, i, j, x, idx, rng, c, k, n_perm, t)
    return rows


def per_patient_auc(g, i, j, x, k):
    """One AUROC per subject with both labels, from the pairs (abnormal row, normal row): (subjects used as 0/1, their AUROC)."""
    win = (x[i] > x[j]) + 0.5 * (x[i] == x[j])
    pairs = np.bincount(g[i], minlength=k).astype(float)
    return pairs > 0, np.divide(np.bincount(g[i], win, k), pairs, out=np.zeros(k), where=pairs > 0)


def per_patient_rows(g, i, j, x, idx, rng, c, k, n_perm, tags):
    """Mean and SD of the per-patient AUROC and the share above 0.5, with subjects weighing the same; the null shuffles scores within subject."""
    used, a = per_patient_auc(g, i, j, x, k)
    s = np.column_stack([used.astype(float), a, a * a, (used & (a > .5)).astype(float)])
    mean = lambda v: v[..., 1] / v[..., 0]
    above = lambda v: v[..., 3] / v[..., 0]
    tags = dict(tags, subjects=int(used.sum()))
    nulls = [per_patient_auc(g, i, j, x[within_shuffle(idx, g, rng)], k)[1] for _ in range(n_perm)]
    return [with_null(cell("per_patient_auroc_mean", mean, s, c, **tags), [v[used].mean() for v in nulls]),
            cell("per_patient_auroc_sd", lambda v: np.sqrt(np.maximum(v[..., 2] / v[..., 0] - mean(v) ** 2, 0)), s, c, **tags),
            with_null(cell("per_patient_auroc_above_half", above, s, c, **tags), [(v[used] > .5).mean() for v in nulls])]


# ------------------------------------------------------------------------------------------------- one target
def analyse(fr, target, kind, bcols, sid, c, n_perm, rho=None):
    """All tables for one target: variance and comparators once, level, change, stack and AUROC per PPG model."""
    k = c.shape[1]
    fr.g, fr.pg = np.searchsorted(sid, fr.groups[fr.ev]), np.searchsorted(sid, fr.groups[fr.cur[fr.pev]])
    masks, cuts = strata(fr)
    M, de, prev_e, cur_e = np.array(list(masks.values()), float), fr.dy[fr.pev], fr.prev[fr.pev], fr.cur[fr.pev]
    n_slow = min(n_perm, N_SLOW)
    planted_kind = kind == "plasmode"
    tag = dict(target=target, kind=kind, rho=rho)
    T = {t: [] for t in ("variance", "level", "change", "stack", "auroc")}
    failed = {}

    def comparators():
        T["variance"] = variance_rows(fr, masks, c, k)
        T["change"] = change_rows(fr, "no_change", np.zeros(len(de)), masks, c, k, ("mae",), ppg="-")
        for method, pred in history_changes(fr).items():
            T["change"] += change_rows(fr, method, pred, masks, c, k, ASSOCIATION + SKILL, ppg="-")
        if fr.spec["locf"]:
            sign = -1.0 if fr.spec.get("log_locf") else 1.0
            trend = sign * fr.q[fr.spec["hist"][2]].to_numpy(float)[cur_e]
            T["change"] += change_rows(fr, "chart_trend", trend, masks, c, k, ASSOCIATION, ppg="-")

    def oracle():
        hr = fr.q.hr_mon.to_numpy(float)
        T["change"] = change_rows(fr, "oracle_hr", (hr[fr.cur] - hr[fr.prev])[fr.pev], masks, c, k, ASSOCIATION, ppg="-")

    def one_model(ppg):
        cols = dict(beat=bcols, papagei=PG, pulseppg=PP, cnn=cnn_cols(target))[ppg]
        fixed_score = ppg == "cnn"
        if cols[0] not in fr.q or (fixed_score and kind != "lab"):
            return
        family, X = PPG_MODELS[ppg], fr.q[cols].to_numpy(float)
        seed = zlib.crc32(f"{target}|{ppg}|{rho}".encode())
        rng = np.random.default_rng([0, seed])
        s = np.zeros(len(X))
        s[fr.tr], s[fr.ev] = ppg_score(family["reg"], X, fr.y, fr.groups, fr.tr, fr.ev)
        if fixed_score:
            s *= abnormal_direction(target)
        pms = [within_shuffle(fr.ev, fr.code, rng) for _ in range(n_perm)]
        T["level"] += [dict(r, ppg=ppg) for r in level_rows(fr, s, c, k, sid, pms)]
        ds = s[fr.cur] - s[fr.prev]
        beta = (ds[fr.ptr] @ fr.dy[fr.ptr]) / (ds[fr.ptr] @ ds[fr.ptr])
        null = null_change(M, de, (s[pm[cur_e]] - s[pm[prev_e]] for pm in pms)) if pms else None
        T["change"] += change_rows(fr, "ppg_raw", ds[fr.pev], masks, c, k, ASSOCIATION + SKILL, null, ppg=ppg)
        T["change"].append(change_now_row(fr, ds, c, k, method="ppg_raw", ppg=ppg))
        T["change"] += change_rows(fr, "ppg_cal", beta * ds[fr.pev], masks, c, k, SKILL, ppg=ppg, train_slope=float(beta))
        extras = dict(ppg_change=ds, ppg_level=s[fr.cur])
        if not fixed_score:
            dX, pred = X[fr.cur] - X[fr.prev], np.zeros(len(fr.prev))
            pred[fr.ptr], pred[fr.pev] = ppg_score(family["reg"], dX, fr.dy, fr.groups[fr.cur], fr.ptr, fr.pev)
            null = None
            if pms:
                fitted = family["reg"]().fit(dX[fr.ptr], fr.dy[fr.ptr])
                null = null_change(M, de, (fitted.predict(X[pm[cur_e]] - X[pm[prev_e]]) for pm in pms[:n_slow]))
            T["change"] += change_rows(fr, "change_trained", pred[fr.pev], masks, c, k, ASSOCIATION + SKILL, null, ppg=ppg)
            T["change"].append(change_now_row(fr, pred, c, k, method="change_trained", ppg=ppg))
            extras["change_trained"] = pred
        T["stack"] += [dict(r, ppg=ppg) for r in stack_rows(fr, extras, c, k, n_slow if pms else 0, seed, s if pms else None)]
        if not planted_kind:
            T["auroc"] += [dict(r, ppg=ppg) for r in auroc_rows(fr, cols, family, sid, c, k, n_perm, seed)]
    def after_treatment():
        m = masks["after_intervention"]
        pg = fr.pg[m]
        s_ = np.column_stack([np.bincount(pg, minlength=k).astype(float), np.bincount(pg, de[m], k)])
        T["change"].append(cell("mean_true_change", lambda v: v[..., 1] / v[..., 0], s_, c, method="no_change", ppg="-", stratum="after_intervention", n=int(m.sum()), subjects=len(np.unique(pg)),
                                expected_sign=INTERVENTIONS[fr.name][1]))
    # each part fails on its own, on record: the rows it already gave stay
    for part, fn, args in (("comparators", oracle if planted_kind else comparators, ()), *[(ppg, one_model, (ppg,)) for ppg in MODELS]):
        x = attempt(fn, *args)
        if isinstance(x, dict) and "failed" in x:
            failed[part] = x["failed"]
    if "after_intervention" in masks and masks["after_intervention"].sum() >= MIN_CELL:
        x = attempt(after_treatment)
        if isinstance(x, dict) and "failed" in x:
            failed["after_intervention"] = x["failed"]
    counts = dict(windows=len(fr.ev), subjects=len(np.unique(fr.g)), subjects_2plus=int((np.bincount(fr.g, minlength=k) >= 2).sum()),
                  pairs=len(fr.pev), train_pairs=len(fr.ptr), pairs_dropped=fr.dropped, change_q50=cuts[0], change_q80=cuts[1], failed=failed)
    return {t: [dict(r, **tag) for r in rows] for t, rows in T.items()}, counts


# ------------------------------------------------------------------------------------------------- report
def family_table(frames):
    """Means over the family labs per model and stratum, on bootstrap values shared by all labs; null_mean is the mean of the labs' null means."""
    rows = []
    for table in ("level", "change", "auroc"):
        x = frames[table]
        if x.empty:
            continue
        x = x[(x.kind == "lab") & x._boot.notna()]
        x = x.assign(**{col: "-" for col in ("method", "stratum", "null_mean") if col not in x})
        for (ppg, metric, method, stratum), g in x.groupby(["ppg", "metric", "method", "stratum"], sort=False):
            boots = np.nanmean(np.vstack(g._boot.tolist()), axis=0)
            rows.append(dict(ppg=ppg, metric=metric, method=method, stratum=stratum, labs=len(g), mean_est=g.est.mean(),
                             lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5), min_lab=g.est.min(), max_lab=g.est.max(),
                             null_mean=pd.to_numeric(g.null_mean, errors="coerce").mean()))
    return pd.DataFrame(rows, columns=FAMILY_COLUMNS)


def report(frames, fam, targets, plasmode, stem):
    """report.md: the headline cells of each table. The CSVs hold every stratum and metric."""
    def ci(x):
        if x is None or not np.isfinite(x.est):
            return "—"
        return f"{x.est:+.3f} ({x.lo:+.3f}, {x.hi:+.3f})"

    def pick(table, **eq):
        x = frames[table]
        if x.empty:
            return None
        x = x[x.kind == eq.pop("kind", "lab")] if eq.get("kind") == "plasmode" else x[x.kind != "plasmode"]
        for key, v in eq.items():
            x = x[x[key] == v]
        return x.iloc[0] if len(x) else None

    def null(x):
        return "—" if x is None or "null_lo" not in x or not np.isfinite(x.null_lo) else f"({x.null_lo:+.3f}, {x.null_hi:+.3f})"

    def q(x):
        return "—" if x is None or "q_bh" not in x or not np.isfinite(x.q_bh) else f"{x.q_bh:.3f}"

    def table(head, lines):
        return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *["| " + " | ".join(map(str, r)) + " |" for r in lines], ""]

    L = [f"# Within-patient tracking: {stem}", "",
         "Exploratory, on a split the project has looked at before. CIs: subject bootstrap with fitted models fixed. Controls are not labs, and "
         "scales differ (log for skewed labs). Every stratum is in the CSVs.", ""]
    v = [[t, pick("variance", target=t, metric="icc", pop="2plus").n, pick("variance", target=t, metric="icc", pop="2plus").subjects,
          ci(pick("variance", target=t, metric="icc", pop="2plus")), ci(pick("variance", target=t, metric="icc", pop="all")),
          ci(pick("variance", target=t, metric="gamma_gap_le12h_over_within"))] for t in targets]
    L += ["## 1. Variance between and within subject", ""] + table(["Target", "Windows", "Subjects ≥ 2", "ICC, subjects ≥ 2", "ICC, all", "γ(≤ 12 h) / within"], v)
    L += ["## 2. Association of the pooled score with the target", ""]
    for ppg in MODELS:
        rows = [[t, ci(pick("level", target=t, ppg=ppg, metric="pooled_r")), ci(pick("level", target=t, ppg=ppg, metric="between_r")),
                 ci(pick("level", target=t, ppg=ppg, metric="within_r")), null(pick("level", target=t, ppg=ppg, metric="within_r")),
                 q(pick("level", target=t, ppg=ppg, metric="within_r")), ci(pick("level", target=t, ppg=ppg, metric="within_r_now")),
                 ci(pick("level", target=t, ppg=ppg, metric="between_minus_within")), ci(pick("level", target=t, ppg=ppg, metric="score_icc"))]
                for t in targets]
        L += [f"### {ppg}", ""] + table(["Target", "pooled r", "between r", "within r", "null range", "q", "within r, now removed", "between − within", "score ICC"], rows)
    L += ["## 3. Change on change", ""]
    for method in ("ppg_raw", "change_trained"):
        for ppg in MODELS:
            rows = []
            for t in targets:
                g = lambda m, s: pick("change", target=t, ppg=ppg, method=method, metric=m, stratum=s)
                a = g("r", "all")
                rows.append([t, a.n if a is not None else "—", ci(a), null(a), q(a), ci(g("r_now", "all")), ci(g("concordance", "all")), null(g("concordance", "all")),
                             ci(g("concordance", "gap_le12h")), ci(g("concordance", "change_large")), ci(g("r", "change_large"))])
            L += [f"### {ppg}, {method}", ""] + table(["Target", "Pairs", "r, all", "null", "q", "r, change in now removed", "concordance, all", "null", "conc., ≤ 12 h",
                                                       "conc., large change", "r, large change"], rows)
    rows = []
    for t in targets:
        g = lambda m, **e: pick("change", target=t, stratum="all", metric=m, **e)
        rows.append([t, ci(g("mae", method="no_change")), ci(g("skill", method="hist")), ci(g("concordance", method="hist")),
                     ci(g("skill", method="hist_windows")), ci(g("concordance", method="hist_windows")), ci(g("skill", method="personal_mean")), ci(g("concordance", method="personal_mean")),
                     ci(g("concordance", method="chart_trend"))]
                    + [ci(g("skill", method="ppg_cal", ppg=p)) for p in MODELS])
    L += ["### Against no change and history (all pairs; skill = 1 − SSE / SSE of no change)", ""]
    L += table(["Target", "MAE no change", "skill hist", "conc. hist", "skill hist_windows", "conc. hist_windows", "skill personal mean", "conc. personal mean", "conc. chart_trend"]
               + [f"skill {p} cal" for p in MODELS], rows)
    L += ["### Change outside the exclusion zone, and pairs around a treatment", "", "Outside the zone: the true change exceeds what analytical variation could produce, a limit set in advance. After a treatment: given between the two draws; insulin has only an at-draw flag, so it has no such pairs.", ""]
    rows = []
    for t in targets:
        for ppg in MODELS:
            g = lambda m, s, method="ppg_raw": pick("change", target=t, ppg=ppg, method=method, metric=m, stratum=s)
            z, a = g("concordance", "outside_zone"), g("concordance", "after_intervention")
            if z is not None or a is not None:
                truth = pick("change", target=t, ppg="-", method="no_change", metric="mean_true_change", stratum="after_intervention")
                rows.append([t, ppg, z.n if z is not None else "—", ci(z), null(z), ci(g("r", "outside_zone")), a.n if a is not None else "—", ci(a), null(a), ci(truth)])
    L += table(["Target", "Model", "Pairs outside", "conc., outside", "null", "r, outside", "Pairs after treatment", "conc., after treatment", "null", "mean true change, after treatment"], rows)
    L += ["## 4. Adding PPG to previous value, gap and `now`", ""]
    for ppg in MODELS:
        rows = []
        for t in targets:
            g = lambda m, b, e: pick("stack", target=t, ppg=ppg, metric=m, base=b, extra=e)
            d = g("delta_r2", "prev_gap_now", "ppg_change")
            rows.append([t, ci(g("base_r2", "prev_gap_now", "ppg_change")), ci(d), null(d), ci(g("partial_r2", "prev_gap_now", "ppg_change")),
                         ci(g("delta_r2", "prev_gap_now", "ppg_level")), ci(g("delta_r2", "prev_gap_now", "change_trained")),
                         ci(g("delta_r2", "prev_gap_full", "ppg_change")), ci(g("delta_r2", "prev_gap_full", "change_trained"))])
        L += [f"### {ppg}", ""] + table(["Target", "base R²", "ΔR² change in score", "null", "partial R²", "ΔR² current score", "ΔR² change_trained",
                                         "ΔR² change in score, over full", "ΔR² change_trained, over full"], rows)
    L += ["## 5. What the pooled AUROC is made of (classification score)", ""]
    for ppg in MODELS:
        rows = []
        for t in targets:
            g = lambda m: pick("auroc", target=t, ppg=ppg, metric=m, pop="all")
            w = g("auroc_within_pairs")
            a, sh = g("auroc_pooled"), g("auroc_shuffled")
            rows.append([t, ci(a), ci(sh), f"{a.est - sh.est:+.3f}" if a is not None else "—", q(a), ci(g("auroc_between_only")), ci(w), q(w),
                         w.subjects if w is not None else "—", f"{g('share_within_pairs').est:.2%}" if w is not None else "—"])
        L += [f"### {ppg}", ""] + table(["Target", "pooled AUROC", "within-shuffled (null range)", "pooled − shuffled", "q", "between only", "within-subject pairs", "q",
                                         "subjects with both labels", "within share of pairs"], rows)
        rows = []
        for t in targets:
            g = lambda m, p="all": pick("auroc", target=t, ppg=ppg, metric=m, pop=p)
            share, drop, pooled = g("share_subjects_both_labels"), g("shuffle_drop", "both_labels"), g("auroc_pooled", "both_labels")
            rows.append([t, ci(share), ci(pooled), ci(drop), ci(g("per_patient_auroc_mean", "all")), null(g("per_patient_auroc_mean", "all")), ci(g("per_patient_auroc_sd", "all")), ci(g("per_patient_auroc_above_half", "all"))])
        L += [f"### {ppg}: patients with both label classes", "", "Only they can move the pooled AUROC under the shuffle. Per-patient AUROC weighs each such patient the same.", ""]
        L += table(["Target", "share of patients", "pooled AUROC among them", "shuffle drop among them", "per-patient AUROC, mean", "null", "SD", "share above 0.5"], rows)
    mean = lambda x: f"{x.mean_est:+.3f} ({x.lo:+.3f}, {x.hi:+.3f})"
    L += ["## 6. Means over the 10 labs", ""] + table(
        ["Model", "Metric", "Method", "Mean (CI)", "Null mean", "Lab range"],
        [[x.ppg, x.metric, x.method, mean(x), f"{x.null_mean:+.3f}" if np.isfinite(x.null_mean) else "—", f"{x.min_lab:+.3f} to {x.max_lab:+.3f}"]
         for x in fam.itertuples() if x.stratum in ("all", "-")])
    for method, metric in (("ppg_raw", "r"), ("ppg_raw", "concordance"), ("change_trained", "r"), ("change_trained", "concordance")):
        x = fam[(fam.method == method) & (fam.metric == metric)]
        if len(x):
            order = list(dict.fromkeys(x.stratum))
            L += [f"### By stratum: {method}, {metric} (null mean in brackets for concordance)", ""]
            L += table(["Stratum"] + list(MODELS), [[st] + [next((mean(e) + (f" [{e.null_mean:.3f}]" if metric == "concordance" else "") for e in x[(x.stratum == st) & (x.ppg == p)].itertuples()), "—")
                                                           for p in MODELS] for st in order])
    if len(plasmode):
        rows = []
        for (t, r, ppg), _ in plasmode.groupby(["target", "rho", "ppg"], sort=False):
            g = lambda table, **e: pick(table, kind="plasmode", target=t, rho=r, ppg=ppg, **e)
            ch = lambda m, method: g("change", method=method, metric=m, stratum="all")
            oracle = pick("change", kind="plasmode", target=t, rho=r, ppg="-", method="oracle_hr", metric="r", stratum="all")
            rows.append([t, r, f"{oracle.est:+.3f}", ppg, ci(g("level", metric="within_r")), ci(ch("r", "ppg_raw")), ci(ch("concordance", "ppg_raw")),
                         ci(g("stack", metric="partial_r2", base="prev_gap_now", extra="ppg_change")), ci(ch("r", "change_trained")),
                         ci(ch("concordance", "change_trained")), ci(g("stack", metric="partial_r2", base="prev_gap_now", extra="change_trained"))])
        L += ["## 7. Positive control of known size: lab with a planted within-subject link to monitor HR", ""]
        L += table(["Lab", "Planted within r", "Oracle change r", "Model", "within r", "change r", "concordance", "partial R²", "change r, change_trained", "conc., change_trained",
                    "partial R², change_trained"], rows)
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------- entry point
def run(window="pre", eval_split="val", open_test=False, out=None, labs=FAMILY_LABS, n_boot=N_BOOT, n_perm=N_PERM):
    """Run every step and write the tables. The test split is refused unless baselines.check_test_allowed passes and the confirmatory
    evaluation has been done. With test, `out` is refused too, because the once-only rule rests on the folder under OUT."""
    if eval_split not in ("val", "test"):
        sys.exit(f"--eval must be val or test, not {eval_split}")
    if window not in PPG_START:
        sys.exit(f"--window must be one of {sorted(PPG_START)}, not {window}")
    stem = tracking_stem(eval_split, window)
    if eval_split == "test":
        if out is not None:
            sys.exit("test split not opened: --out is not allowed with --eval test")
        check_test_allowed(stem, open_test)
        if not TEST_OUT.exists():
            sys.exit(f"test split not opened: {TEST_OUT} does not exist, and the confirmatory evaluation has to come first")
    d, bcols, specs, tiers = load_mimic(window, open_test=eval_split == "test")
    d = d[d.split.isin(["train", eval_split])].reset_index(drop=True)    # the rest are gone before anything is computed
    out = Path(out) if out is not None else OUT / stem
    sid = np.unique(d.group[d.split == eval_split])
    c = np.random.default_rng(0).multinomial(len(sid), np.full(len(sid), 1 / len(sid)), size=n_boot)
    tables, counts = {t: [] for t in ("variance", "level", "change", "stack", "auroc")}, {}
    targets = [(n, "lab", d, tiers, specs[n]) for n in labs] + [(n, "control", *x) for n, x in controls(d, tiers).items()]
    for name, kind, dd, tt, spec in targets:
        def one_target():
            fr = frame_of(dd, tt, name if kind == "lab" else spec["value"], spec, eval_split, window)
            return None if fr is None else analyse(add_pairs(fr), name, kind, bcols, sid, c, n_perm)
        x = attempt(one_target)
        if x is None:
            print(f"  {name} skipped: too few eval windows", flush=True)
        elif isinstance(x, dict):
            counts[name] = dict(failed=dict(target=x["failed"]))
            print(f"  {name} failed: {x['failed']}", flush=True)
        else:
            counts[name] = x[1]
            for t, rows in x[0].items():
                tables[t] += rows
            print(f"  {name} done: {counts[name]}", flush=True)
    no_hr = {k: [c_ for c_ in v if c_ != "hr_mon"] for k, v in tiers.items()}
    plas = []
    for lab in (n for n in PLASMODE_LABS if n in labs):
        def one_plasmode():
            fr = frame_of(d, no_hr, lab, specs[lab], eval_split, window)
            for rho in PLASMODE_RHO if fr is not None else ():
                res, _ = analyse(plant_frame(fr, rho), lab, "plasmode", bcols, sid, c, 0, rho=rho)
                for t, rows in res.items():
                    tables[t] += rows
        x = attempt(one_plasmode)
        if isinstance(x, dict):
            counts[f"plasmode {lab}"] = dict(failed=dict(target=x["failed"]))
        print(f"  plasmode {lab} {'failed' if isinstance(x, dict) else 'done'}", flush=True)
    if not any(tables.values()):       # no folder, so that the once-only rule does not close on nothing
        sys.exit(f"tracking computed nothing, and no folder was written: { {n: x.get('failed') for n, x in counts.items()} }")
    frames = {t: pd.DataFrame(rows) for t, rows in tables.items()}
    frames = {t: x if "kind" in x else x.assign(kind=pd.Series(dtype=object)) for t, x in frames.items()}
    fam = attempt(family_table, frames)
    failed_steps = {}
    if isinstance(fam, dict):
        failed_steps["family"] = fam["failed"]
        fam = pd.DataFrame(columns=FAMILY_COLUMNS)
    frames = {t: x.drop(columns="_boot", errors="ignore") for t, x in frames.items()}
    for t, keep in (("level", lambda x: x.metric == "within_r"),
                    ("change", lambda x: (x.method == "ppg_raw") & (x.metric == "r") & (x.stratum == "all")),
                    ("change", lambda x: (x.method == "change_trained") & (x.metric == "r") & (x.stratum == "all")),
                    ("auroc", lambda x: (x.metric == "auroc_pooled") & (x["pop"] == "all")),
                    ("auroc", lambda x: x.metric == "auroc_within_pairs")):
        x = frames[t]
        if "p" in x:
            x = x if "q_bh" in x else x.assign(q_bh=np.nan)
            i = x.index[keep(x) & (x.kind == "lab") & x.p.notna()]
            x.loc[i, "q_bh"] = bh(x.loc[i, "p"])
            frames[t] = x
    out.mkdir(parents=True, exist_ok=True)
    for t, x in frames.items():
        x[x.kind != "plasmode"].to_csv(out / f"{t}.csv", index=False)
    plasmode = pd.concat([frames[t][frames[t].kind == "plasmode"].assign(table=t) for t in ("level", "change", "stack")])
    plasmode.to_csv(out / "plasmode.csv", index=False)
    fam.to_csv(out / "family.csv", index=False)
    failed = {name: x["failed"] for name, x in counts.items() if x.get("failed")}
    failed.update(failed_steps)
    write_manifest(out, stem, dict(window=window, eval=eval_split, labs=list(labs), models=list(MODELS), n_boot=n_boot, n_perm=n_perm, n_slow=N_SLOW,
                                   gaps=list(GAPS), size_q=SIZE_Q, plasmode_labs=PLASMODE_LABS, plasmode_rho=PLASMODE_RHO, min_cell=MIN_CELL),
                   dict(windows=len(d), subjects=int(d.group.nunique()), targets=counts, failed=failed))
    write_report(frames, fam, [n for n in [*labs, "hr_monitor", "map_artline"] if "windows" in counts.get(n, {})], plasmode, stem, out)
    if failed:
        print(f"failed, and recorded in {out / 'manifest.json'}: {failed}", file=sys.stderr, flush=True)
    return frames, fam


def write_report(frames, fam, targets, plasmode, stem, out):
    """report.md beside the tables, which are already written. A failure leaves them as they are, a note, and the command that renders it again from them."""
    try:
        text = report(frames, fam, targets, plasmode, stem)
    except Exception as e:
        message = f"report.md could not be rendered ({failure(e)}), and the tables in {out} are complete. After fixing the code, this reads no data: python -m ppglabbench.tracking --render {out}"
        (out / "report_failed.txt").write_text(message + "\n")
        print(message, file=sys.stderr, flush=True)
        return None
    (out / "report.md").write_text(text)
    if (out / "report_failed.txt").exists():
        (out / "report_failed.txt").unlink()
    return text


def render(out):
    """The report again from the tables in a folder, reading no data."""
    out = Path(out)
    frames = {t: pd.read_csv(out / f"{t}.csv") for t in ("variance", "level", "change", "stack", "auroc")}
    plasmode = pd.read_csv(out / "plasmode.csv")
    frames = {t: pd.concat([x, plasmode[plasmode.table == t].drop(columns="table")]) if t in set(plasmode.table) else x for t, x in frames.items()}
    fam = pd.read_csv(out / "family.csv")
    targets = [n for n, x in json.loads((out / "manifest.json").read_text())["counts"]["targets"].items() if "windows" in x]
    if write_report(frames, fam, targets, plasmode, out.name, out) is None:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default="pre", choices=sorted(PPG_START))
    ap.add_argument("--eval", default="val", choices=["val", "test"])
    ap.add_argument("--open-test", action="store_true", help="required with --eval test: its one evaluation")
    ap.add_argument("--labs", nargs="+", default=list(FAMILY_LABS), choices=list(FAMILY_LABS))
    ap.add_argument("--out", help="output folder, default under data/benchmark/results; not allowed with --eval test")
    ap.add_argument("--perms", type=int, default=N_PERM)
    ap.add_argument("--render", metavar="FOLDER", help="the report again from the tables in FOLDER, reading no data")
    args = ap.parse_args()
    if args.render:
        return render(args.render)
    run(args.window, args.eval, args.open_test, args.out, tuple(args.labs), n_perm=args.perms)


if __name__ == "__main__":
    main()
