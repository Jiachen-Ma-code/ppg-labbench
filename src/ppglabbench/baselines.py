"""Benchmark step 4: reference baselines, PPG increments and the permuted-PPG null (SPEC §5-6).

Per lab and source, models are fitted on the train split and evaluated on the eval split (val by default;
the test split is evaluated once, after the recipe is frozen):
  age_sex      logistic / ridge on age + sex
  acuity       + covariates that are not PPG-derived. MIMIC: vasopressor at the draw, emergency / urgent admission, MAP
               from the arterial waveform over the PPG window (features/abp_map.py; missing without a usable arterial waveform) and
               the draw hour: 64% of draws are routine 00-05 h draws, with fewer abnormal values, and PPG partly reads
               the hour. VitalDB: ASA class and emergency operation
  monitor      acuity + HR (ECG), RESP (exactly 0 counts as missing), SpO2, irregular-HR percent, PVC rate and NBP from the monitor numerics (MIMIC;
               features/numerics.py): 8 columns
  now          monitor + treatments at the draw and in the prior 24 h, pressor count, hospital hours, era, admission route,
               ethnicity and earlier admissions (MIMIC; features/covariates.py), and per lab the time since its last draw and the
               count of its earlier draws, without their values; VitalDB: acuity + department, operation type, approach, BMI,
               hypertension, diabetes. Age and MAP enter flexibly (see `flexible`), because PPG reads age: a base linear in
               age lets a score with no independent information show a gain. This is the primary base
  locf         + previous value of the same lab (log for skewed labs), its age (log1p h), missing indicator
               (MIMIC only: VitalDB has a single preoperative value per lab)
  hist         locf + mean and count of all earlier values and the recent trend: what a smarter use of history achieves
  locf_raw     the previous value copied as the prediction, nothing fitted (T1 only; not for eGFR, whose LOCF is
               creatinine). This is the "carry-forward" most papers report: the ladder locf_raw -> locf -> locf+<ppg>
               separates what fitting on the previous value gains from what PPG adds
  full         now + previous value, history and the blood-gas analyser's previous value: the covariates listed here (MIMIC). The `locf`
               base holds only age, sex and the previous value
  mean_train   the train mean as the prediction (T1), the floor for any model
  now_gb       sensitivity, same model family with and without PPG: gradient boosting on `now`, then on `now` + beat features,
               so an increment cannot come from a model class the base lacks
  beat         beat-morphology features -> gradient boosting, no early stopping, rounds from tune_beat.py
  papagei      PaPaGei-S 512-d embedding -> PCA(64) + logistic / ridge (PPG only)
  pulseppg     Pulse-PPG 512-d embedding -> the same probe. Unlike PaPaGei it was not pretrained on these cohorts. It needs PULSEPPG_MIN_S
               of continuous signal: the embedding of a shorter input depends on the input's length, so it counts as missing
  cnn          a small end-to-end 1D CNN (features/cnn.py), `pre` window only: nothing to fit here, the logit for the lab is the score, out of fold on train by the file's
               folds and from the final model elsewhere. It has heads for the abnormal value, so it enters T2 and not T1
  rate         HR, SDNN, RMSSD and inter-beat interval from the PPG -> gradient boosting: pulse rate and its variability, without morphology
  <base>+<ppg> base covariates + the PPG score, stacked; the train-split score is out-of-fold (GroupKFold by
               subject, each fold centred on the train mean), so the stacker never sees an in-sample PPG score. In T2 the
               score is on the log-odds scale, as a logistic stack needs: a probability is a poor covariate near 0 or 1
T1 regression (R², MAE; log scale for skewed labs) and T2 classification (AUROC, AUPRC). T3 increments are the
paired subject-bootstrap differences <base>+<ppg> − <base>; for AUROC the base counts as at least 0.5, because any
added term pulls a below-chance base towards 0.5. CIs resample eval subjects with the fitted models fixed, so they leave
out training variance; `p`, `q_bh` and `p_holm` adjust them over the main increments (subset `all`, AUROC and R²) and
are exploratory. Null control: the same stack with each subject given another subject's PPG block, repeated --perms
times; it shows what the pipeline makes from noise and is reported as a distribution, not a test of "PPG independent of
the lab given the base". Sensitivity: `within_30min` (lab within ± SENS_MIN), `not_dev` (val without the development
subjects). Subgroup `no_recent_locf` (MIMIC): eval windows
whose previous value of the lab is missing or older than STALE_H hours, where LOCF is least informative; subgroup
`arterial_line` (MIMIC): eval windows with a MAP from the waveform, where the acuity base is strongest; subset `has_locf`
(MIMIC, T1): eval windows with a previous value, the only windows where locf_raw is defined; subset `no_early_lab`: the lab was not
charted before the PPG window started, so its result cannot have changed the PPG through treatment. Descriptive subsets:
`female`, `male` and, where the columns exist (MIMIC), `eth_black` and `eth_white` (no other eth_* flag set). Labs or subsets with
< MIN_N eval windows are skipped, and for T2 also those with < 50 abnormal train rows or < 20 eval subjects in either class.

Window: `--window pre` (primary) is the best 5 min before the draw, on windows where one scores at least 0.8, and a lab is dropped
from a window if it was charted before the PPG ended. `--window centered` (sensitivity) is S4's best window around the draw.
Cohort: every adult window with beat features and a PaPaGei embedding, without windows whose PLETH record overlaps
another record of the subject; for eGFR the dialysis-flagged windows are dropped (their creatinine is set by dialysis).
VitalDB: cases under general anaesthesia (`ane_type == "General"`) with ≥ 50% good epochs and ASA below 6, split by the same salted hash on subjectid; eGFR leaves out kidney
failure (eGFR < 15 or a renal diagnosis), as it has no dialysis flag, and `egfr_incl_failure` keeps it. Note: all VitalDB cases were used in cross-validation while the methods were
developed, so its val and test splits are not independent of that work (MIMIC test subjects were not used).

Usage:
    python -m ppglabbench.baselines mimic [--eval val] [--perms 100] [--window pre]
    python -m ppglabbench.baselines vitaldb [--eval val] [--perms 100]
Output: data/benchmark/results/<source>_<eval>[_centered]/{results.csv, null.csv, report.md, manifest.json}
"""
import argparse
import hashlib
import json
import sys
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ppglabbench.common import CFG, DATA, git_version, write_manifest
from ppglabbench.features.beat import MIN_NOTCH, SEARCH_END
from ppglabbench.features.covariates import PRIOR_DX
from ppglabbench.labels import C, LAB_CACHE, abnormal, development_subjects, split_of
from ppglabbench.pipeline.s6_build_dataset import egfr_ckdepi2021

OUT = DATA / "benchmark" / "results"
FEAT = DATA / "benchmark" / "features"
VDB = DATA / "external" / "vitaldb"
K, N_BOOT, MIN_POS_TRAIN, MIN_POS_EVAL, MIN_N, STALE_H, SENS_MIN = 5, 1000, 50, 20, 100, 24, 30
PULSEPPG_MIN_S = 240      # the input length Pulse-PPG was trained on: features/pulseppg.py
LOGIT_CLIP = 1e-4
PG = [f"pg{i}" for i in range(512)]
PP = [f"pp{i}" for i in range(512)]
LOG_LABS = {"bun", "glucose", "platelets", "wbc", "inr", "ptt", "lactate", "ast", "alt", "creatinine"}
RATE = ["hr", "sdnn", "rmssd", "ibi_med", "ibi_iqr"]
MONITOR = ["hr_mon", "resp_mon", "spo2_mon", "irregular_mon", "pvc_mon", "nbp_mean", "nbp_sys", "nbp_dias"]
# MIMIC lab name -> VitalDB preoperative column (WBC, INR, lactate are not in VitalDB; PT is % activity there)
VITALDB_COL = dict(bun="preop_bun", sodium="preop_na", potassium="preop_k", bicarbonate="preop_hco3",
                   glucose="preop_gluc", haemoglobin="preop_hb", platelets="preop_plt", ptt="preop_aptt",
                   albumin="preop_alb", ast="preop_ast", alt="preop_alt")
RENAL_DX = "ESRD|end.stage (renal|kidney)|renal failure|kidney failure"
BOOST = dict(max_iter=C["beat_max_iter"], learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
             early_stopping=False, random_state=0)


def base_clf():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(max_iter=2000))


def base_reg():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(1.0))


def probe_clf(k=64):
    return make_pipeline(SimpleImputer(), StandardScaler(), PCA(k, random_state=0), LogisticRegression(C=0.05, max_iter=3000))


def probe_reg(k=64):
    return make_pipeline(SimpleImputer(), StandardScaler(), PCA(k, random_state=0), Ridge(10.0))


def boost_clf():
    return HistGradientBoostingClassifier(**BOOST)


def boost_reg():
    return HistGradientBoostingRegressor(**BOOST)


EMBEDDINGS = ("papagei", "pulseppg")


def cnn_clf():
    """Stands in for a model factory: the CNN's logits are in the data (columns from cnn_cols), with nothing to fit. See ppg_score."""
    raise TypeError("the CNN has no model to fit here")


def cnn_reg():
    """As cnn_clf, for a stack on a lab value: the abnormal-value logit is the covariate, with no centring, as it is not on the value's scale."""
    raise TypeError("the CNN has no model to fit here")


PPG_MODELS = {"beat": {"clf": boost_clf, "reg": boost_reg}, "papagei": {"clf": probe_clf, "reg": probe_reg},
              "pulseppg": {"clf": probe_clf, "reg": probe_reg}, "rate": {"clf": boost_clf, "reg": boost_reg}, "cnn": {"clf": cnn_clf, "reg": cnn_reg}}
MODELS = ("rate", "beat", "papagei", "pulseppg", "cnn")
LADDER = ("age_sex", "acuity", "monitor", "now", "locf", "hist", "full")


def cnn_cols(name):
    """The CNN's columns for one lab: its logit, and the fold of the file's out-of-fold scheme."""
    return [f"cnn_{name}", "cnn_fold"]


def reference_models(bcols, name=None):
    """The PPG models of the benchmark: name, feature columns, and the model factory per task. The CNN enters T2 only."""
    return (("beat", bcols, PPG_MODELS["beat"]), ("papagei", PG, PPG_MODELS["papagei"]), ("rate", RATE, PPG_MODELS["rate"]),
            ("pulseppg", PP, PPG_MODELS["pulseppg"]), ("cnn", cnn_cols(name), dict(clf=cnn_clf)))


def results_stem(source, eval_split, window):
    """Folder name of a run under data/benchmark/results."""
    return f"{source}_{eval_split}" + ("_centered" if source == "mimic" and window == "centered" else "")


def add_pulseppg(d, folder, key):
    """Pulse-PPG columns, where the embeddings exist. A window with less than PULSEPPG_MIN_S of continuous signal counts as missing, as does one too
    short to embed, and is imputed by the probe."""
    f = folder / "pulseppg.npz"
    if not f.exists():
        return d
    z = np.load(f, allow_pickle=True)
    ids = z["ids"].astype(int) if key == "caseid" else z["ids"]
    emb = z["emb"]
    emb[z["seconds"] < PULSEPPG_MIN_S] = np.nan
    return d.merge(pd.DataFrame(emb, columns=PP).assign(**{key: ids}), on=key, how="left")


def cnn_file(window):
    return FEAT / window / "mimic_cnn" / "cnn_scores.npz"


def cnn_sha256(window):
    """SHA-256 of the CNN's score file, None without one. A hash of the file, not an aggregate of its rows."""
    f = cnn_file(window)
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None


def check_cnn_rows(d, t, open_test):
    """The file against the cohort; its other rows are not read. Every window read is in the file (a window with no usable signal is there, without a score). A train row has a fold, since
    a final-model score on it would be in-sample, and other rows none. K folds in all, and a subject's train rows share one, so no train score comes from a fit that saw its subject. ppg_score
    draws its folds per lab on that lab's rows, and one model with one trunk cannot match all of them, so these are the file's own. Counts only, never a window."""
    rows = d if open_test else d[d.split != "test"]
    rows = rows[["draw_id", "group" if "group" in d else "subject_id", "split"]].merge(t[["draw_id", "cnn_fold"]], on="draw_id", how="left")
    who = rows.iloc[:, 1]
    problems = []
    if rows.cnn_fold.isna().any():
        problems.append(f"{int(rows.cnn_fold.isna().sum())} of {len(rows)} windows are not in the file")
    train = rows.split == "train"
    if (rows.cnn_fold[train] < 0).any():
        problems.append(f"{int((rows.cnn_fold[train] < 0).sum())} train windows have no fold")
    if (rows.cnn_fold[~train] >= 0).any():
        problems.append(f"{int((rows.cnn_fold[~train] >= 0).sum())} windows outside train have a fold")
    if rows.cnn_fold[train].nunique() != K:
        problems.append(f"{rows.cnn_fold[train].nunique()} folds on train, not {K}")
    split_up = int((rows[train].groupby(who[train]).cnn_fold.nunique() > 1).sum())
    if split_up:
        problems.append(f"{split_up} subjects are in more than one fold")
    if problems:
        sys.exit(f"{cnn_file('pre')}: " + "; ".join(problems))


def cnn_rows(window):
    """The file's IDs and folds, without a score: what check_cnn_rows reads."""
    z = np.load(cnn_file(window), allow_pickle=False)
    return pd.DataFrame(dict(draw_id=z["draw_id"], cnn_fold=z["fold"]))


def add_cnn(d, window, open_test=False):
    """The CNN's logits, fold and age and HR heads, as columns. The `pre` window only: any other has no CNN and says so. Rows of the test split hold scores too. They are dropped here
    unless open_test is true, before anything is kept or computed, so a test row has no CNN score until the split is opened; the file itself can only be read whole."""
    if window != "pre":
        print(f"window {window} has no CNN: the cnn model is left out", flush=True)
        return d
    f = cnn_file(window)
    if not f.exists():
        print(f"{f} does not exist: the cnn model is left out", flush=True)
        return d
    z = np.load(f, allow_pickle=False)
    ids = z["draw_id"]
    keep = pd.Series(ids).isin(d.draw_id if open_test else d.draw_id[d.split != "test"]).to_numpy()
    t = pd.DataFrame(z["score"][keep], columns=[f"cnn_{str(x)}" for x in z["targets"]]).assign(draw_id=ids[keep], cnn_fold=z["fold"][keep], cnn_age_pred=z["age_pred"][keep], cnn_hr_pred=z["hr_pred"][keep])
    check_cnn_rows(d, t, open_test)
    return d.merge(t, on="draw_id", how="left")


def complete_table(path, windows, module, window):
    """A table of the waveform files, once it is known to be whole: its manifest says complete, and every window the labels expect has one clean row. The
    manifest alone is not enough, because a later fetch that stopped halfway leaves the older manifest in place. Messages give counts, never a window."""
    run = f"run python -m ppglabbench.features.{module} --window {window}"
    if not path.exists() or not (path.parent / "manifest.json").exists() or not json.loads((path.parent / "manifest.json").read_text())["counts"]["complete"]:
        sys.exit(f"{path} missing or incomplete: {run}")
    t = pd.read_csv(path)
    clean = t if "error" not in t else t[t.error.isna()]
    lost = len(windows - set(clean.draw_id))
    if lost:
        sys.exit(f"{path}: {lost} of {len(windows)} windows the labels expect have no clean row: {run}")
    if clean.draw_id.duplicated().any():
        sys.exit(f"{path}: {int(clean.draw_id.duplicated().sum())} windows have more than one row: {run}")
    return clean


def check_fresh(window, need_cnn=False):
    """Refuse feature files older than the code that reads them: nothing else would notice, because the columns they hold still load. The checks: the beat
    manifest holds the code's search bound and notch depth, the covariates hold the history columns, the labels are not older than the lab cache, and the CNN's score file is the one its
    manifest names and not older than the labels it was trained on. need_cnn: a missing score file is refused too, on the `pre` window; other windows have no CNN. A data folder without clinical tables, as a reader of the release has,
    has no lab cache to compare. Messages name files and settings, never a row."""
    problems = []
    f = cnn_file(window)
    if f.exists():
        m = f.parent / "manifest.json"
        if not m.exists() or cnn_sha256(window) not in m.read_text():
            problems.append(f"{f} is not the file its manifest names: run python -m ppglabbench.features.cnn")
        if (DATA / "benchmark" / "mimic_labels.csv").stat().st_mtime > f.stat().st_mtime:
            problems.append(f"{f} is older than mimic_labels.csv, which holds its training labels: run python -m ppglabbench.features.cnn")
    elif need_cnn and window == "pre":
        problems.append(f"{f} does not exist: run python -m ppglabbench.features.cnn")
    manifest = FEAT / window / "mimic" / "manifest.json"
    params = json.loads(manifest.read_text()).get("params", {}) if manifest.exists() else {}
    if params.get("search_end") != SEARCH_END or params.get("min_notch") != MIN_NOTCH:
        problems.append(f"{manifest} does not record search_end = {SEARCH_END} and min_notch = {MIN_NOTCH}: run python -m ppglabbench.features.beat mimic {window}")
    cov = FEAT / "mimic_cov" / "covariates.csv"
    lacking = sorted({"prior_admissions", "days_since_discharge", *PRIOR_DX} - set(pd.read_csv(cov, nrows=0).columns)) if cov.exists() else ["all columns"]
    if lacking:
        problems.append(f"{cov} lacks the history columns {lacking}: run python -m ppglabbench.features.covariates")
    labels, clinical = DATA / "benchmark" / "mimic_labels.csv", DATA / "clinical"
    cache = clinical / f"{LAB_CACHE}.csv"
    if cache.exists() and labels.exists() and labels.stat().st_mtime < cache.stat().st_mtime:
        problems.append(f"{labels} is older than the lab cache {cache.name}: run python -m ppglabbench.labels")
    elif not cache.exists() and any(clinical.glob("labs_bench_v*.csv")):
        problems.append(f"{cache.name} does not exist, so {labels} was built from an older lab cache: run python -m ppglabbench.labels")
    if problems:
        sys.exit("stale files: " + "; ".join(problems))


def load_mimic(window="pre", need_cnn=False, open_test=False):
    """need_cnn: refuse a missing CNN score file. open_test: keep the CNN's scores on test rows; only the steps that may read the test split pass it, after their guards."""
    check_fresh(window, need_cnn)
    win = FEAT / window
    d = pd.read_csv(DATA / "benchmark" / "mimic_labels.csv")
    if window == "pre":
        d = d[d.pre_score >= CFG["s5_select"]["min_window_score"]]
        d = d.assign(ppg_end_s=d.pre_start_s + CFG["s4_sqi"]["window_min"] * 60)
    expected = set(d.draw_id)
    abp = complete_table(win / "mimic_abp" / "abp_map.csv", expected, "abp_map", window)
    num = complete_table(win / "mimic_num" / "numerics.csv", expected, "numerics", window)
    bf = pd.read_csv(win / "mimic" / "beat_features.csv")
    z = np.load(win / "mimic" / "papagei.npz", allow_pickle=True)
    d = d.merge(bf, on="draw_id").merge(pd.DataFrame(z["emb"], columns=PG).assign(draw_id=z["ids"]), on="draw_id")
    d = add_pulseppg(d[d.n_beats.notna() & d.pg0.notna() & ~d.record_overlap], win / "mimic", "draw_id").reset_index(drop=True)
    d = add_cnn(d, window, open_test)
    if d.vasopressor.isna().all():
        sys.exit("vasopressor is all NaN: download INPUTEVENTS_MV / INPUTEVENTS_CV, rerun S6 and ppglabbench.labels")
    d = d.merge(abp[["draw_id", "map_mmhg"]], on="draw_id", how="left")
    d["map_lt70"] = (d.map_mmhg < 70).astype(float).where(d.map_mmhg.notna())   # SOFA cardiovascular ≥ 1
    cov = pd.read_csv(FEAT / "mimic_cov" / "covariates.csv")
    d = d.merge(num[["draw_id"] + MONITOR], on="draw_id", how="left").merge(cov, on="draw_id", how="left")
    d["resp_mon"] = d.resp_mon.where(d.resp_mon != 0)     # a rate of exactly 0 counts as missing
    now = [c for c in cov.columns if c != "draw_id"]
    t = pd.to_datetime(d.charttime)
    hour = t.dt.hour + t.dt.minute / 60
    d["routine_draw"] = (t.dt.hour < 6).astype(float)
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    d["dev_set"] = d.subject_id.isin(development_subjects())
    history = lambda p: [f"{p}_hist_mean", f"{p}_hist_n", f"{p}_trend"]
    labs = {name: dict(value=name, abnormal=f"{name}_abnormal", locf=[f"{name}_locf", f"{name}_locf_age_h"], hist=history(name))
            for name in C["labs"]}
    d["egfr_value"] = d.egfr.where(~d.exclude_dialysis)
    d["egfr_abn"] = d.egfr_lt60.where(~d.exclude_dialysis)
    labs["egfr"] = dict(value="egfr_value", abnormal="egfr_abn", locf=["creat_locf", "creat_locf_age_h"], log_locf=True,
                        hist=history("creat"))
    d["creat_value"] = d.creat_mgdl.where(~d.exclude_dialysis)
    d["creat_abn"] = (d.creat_mgdl > 1.5).astype(float).where(~d.exclude_dialysis)
    labs["creatinine"] = dict(value="creat_value", abnormal="creat_abn", locf=["creat_locf", "creat_locf_age_h"],
                              hist=history("creat"))
    d["stable_creat"] = (d.creat_locf_age_h <= 48) & ((d.creat_mgdl / d.creat_locf - 1).abs() < 0.2)
    acuity = ["vasopressor", "emergency_admit", "map_mmhg", "map_lt70", "routine_draw", "hour_sin", "hour_cos"]
    tiers = dict(acuity=acuity, monitor=acuity + MONITOR, now=acuity + MONITOR + now)
    return d.rename(columns={"subject_id": "group"}), [c for c in bf.columns if c != "draw_id"], labs, tiers


def load_vitaldb():
    c = pd.read_csv(VDB / "cases.csv.gz")
    c = c[(c.age >= 18) & (c.asa != 6) & (c.ane_type == "General")].copy()    # ASA 6: brain-dead organ donors
    c["female"] = (c.sex == "F").astype(float)
    bf = pd.read_csv(FEAT / "vitaldb" / "beat_features.csv")
    z = np.load(FEAT / "vitaldb" / "papagei.npz", allow_pickle=True)
    d = c.merge(bf, on="caseid").merge(pd.DataFrame(z["emb"], columns=PG).assign(caseid=z["ids"].astype(int)), on="caseid")
    d = add_pulseppg(d[(d.good_frac >= 0.5) & d.n_beats.notna() & d.pg0.notna()], FEAT / "vitaldb", "caseid").reset_index(drop=True)
    d["split"] = [split_of(f"vitaldb:{s}", set()) for s in d.subjectid]
    labs = {}
    for name, col in VITALDB_COL.items():
        v = d[col].where(d[col] > 0).to_numpy(float)
        d[name], d[f"{name}_abnormal"] = v, abnormal(C["labs"][name][1], v, d.female.to_numpy())
        labs[name] = dict(value=name, abnormal=f"{name}_abnormal", locf=None)
    cr = d.preop_cr.where(d.preop_cr.between(0.1, 20)).to_numpy(float)
    d["egfr"] = egfr_ckdepi2021(cr, d.age.to_numpy(), d.female.to_numpy() == 1)
    d["egfr_all_abn"] = np.where(np.isnan(d.egfr), np.nan, d.egfr < 60)
    d["egfr_value"] = d.egfr.where(~((d.egfr < 15) | d.dx.astype(str).str.contains(RENAL_DX, case=False)))
    d["egfr_abn"] = np.where(d.egfr_value.isna(), np.nan, d.egfr_value < 60)
    labs["egfr"] = dict(value="egfr_value", abnormal="egfr_abn", locf=None)
    labs["egfr_incl_failure"] = dict(value="egfr", abnormal="egfr_all_abn", locf=None)
    d["pt_pct"] = d.preop_pt.where(d.preop_pt > 0)
    d["pt_pct_abn"] = np.where(d.pt_pct.isna(), np.nan, d.pt_pct < 80)
    labs["pt_pct"] = dict(value="pt_pct", abnormal="pt_pct_abn", locf=None)
    bcols = [k for k in bf.columns if k not in ("caseid", "good_frac", "n_good", "best_epoch")]
    d["emop"] = d.emop.astype(float)
    dummies = pd.get_dummies(d[["department", "optype", "approach"]].astype(str), dtype=float)
    d = pd.concat([d, dummies], axis=1)
    now = ["asa", "emop", "bmi", "preop_htn", "preop_dm"] + list(dummies.columns)
    return d.rename(columns={"subjectid": "group"}), bcols, labs, dict(acuity=["asa", "emop"], now=now)


def clip_logit(p):
    """Log-odds of a probability clipped to [LOGIT_CLIP, 1 - LOGIT_CLIP]."""
    p = np.clip(np.asarray(p, float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def ppg_score(model, X, y, groups, tr, ev):
    """Out-of-fold score on train (GroupKFold by subject) and a full-train refit scored on eval: log-odds for a classifier, values for a regressor. A fold's
    model predicts near its own training mean, which is anti-correlated with the held-out labels, so each fold is centred on the train mean, on the score's scale.
    The CNN (X: logit, fold) needs no fit: its logits are already out of fold on train and from its final model elsewhere, and its folds are centred the same way.
    A window with no usable signal has no logit and takes the train median, as a missing feature goes to a probe's imputer."""
    if model in (cnn_clf, cnn_reg):
        logit, fold = X[:, 0].copy(), X[:, 1]
        if (fold[tr] < 0).any() or (fold[ev] >= 0).any():
            sys.exit("CNN scores: train rows need a fold, and other rows must not have one")
        logit[np.isnan(logit)] = np.nanmedian(logit[tr])
        if model is cnn_clf:
            logit = np.clip(logit, clip_logit(0.0), clip_logit(1.0))
        s_tr = logit[tr].copy()
        if model is cnn_clf:
            for k in np.unique(fold[tr]):
                held = fold[tr] == k
                s_tr[held] += clip_logit(y[tr].mean()) - clip_logit(y[tr][~held].mean())
        return s_tr, logit[ev]
    if hasattr(model(), "predict_proba"):
        pred, scale = (lambda m, A: clip_logit(m.predict_proba(A)[:, 1])), clip_logit
    else:
        pred, scale = (lambda m, A: m.predict(A)), (lambda v: v)
    s_tr = np.full(len(tr), np.nan)
    for a, b in GroupKFold(K).split(tr, groups=groups[tr]):
        s_tr[b] = pred(model().fit(X[tr[a]], y[tr[a]]), X[tr[b]]) + scale(y[tr].mean()) - scale(y[tr[a]].mean())
    return s_tr, pred(model().fit(X[tr], y[tr]), X[ev])


def block_perm(blocks, n, rng):
    """Row map giving each subject the PPG rows of another subject of the same split, recycled to its window count."""
    perm = np.arange(n)
    for rows in blocks:
        order = rng.permutation(len(rows))
        for i, j in zip(order, np.roll(order, 1)):
            perm[rows[i]] = np.resize(rows[j], len(rows[i]))
    return perm


def too_small(task, y, groups):
    if len(y) < MIN_N:
        return True
    return task == "clf" and min(len(np.unique(groups[y == 1])), len(np.unique(groups[y == 0]))) < MIN_POS_EVAL


def subject_boot(y, groups, preds, pairs, metric, floor=None, seed=0):
    """Point estimate and 95% CI per prediction and per (full, base) difference, resampling subjects."""
    gain = lambda m, a, b: m[a] - (m[b] if floor is None else max(m[b], floor))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = {k: [] for k in list(preds) + [f"{a}|{b}" for a, b in pairs]}
    for _ in range(N_BOOT):
        ix = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[ix])) < 2:
            continue
        m = {k: metric(y[ix], p[ix]) for k, p in preds.items()}
        for k in preds:
            stats[k].append(m[k])
        for a, b in pairs:
            stats[f"{a}|{b}"].append(gain(m, a, b))
    point = {k: metric(y, p) for k, p in preds.items()}
    point.update({f"{a}|{b}": gain(point, a, b) for a, b in pairs})
    return {k: (point[k], *np.percentile(v, [2.5, 97.5])) for k, v in stats.items()}


def flexible(q, tr):
    """Hinges at the train quartiles of age and MAP, female x age, and pressor interactions, from train rows only."""
    def hinges(x):
        return [np.nan_to_num(np.maximum(0, x - c)) for c in np.nanquantile(x[tr], [0.25, 0.5, 0.75])]
    age = q.age.to_numpy(float)
    cols = hinges(age) + [q.female.to_numpy(float) * age]
    if "map_mmhg" in q:
        m, v = q.map_mmhg.to_numpy(float), q.vasopressor.to_numpy(float)
        cols += hinges(m) + [v * q.routine_draw.to_numpy(float), v * np.nan_to_num(m - np.nanmean(m[tr]))]
    return np.column_stack(cols)


def task_frame(d, tiers, name, spec, task, eval_split):
    """Rows, target, split indices, subjects, bases and previous value (log for skewed labs) and its age, for one lab and
    task; None if the task is skipped."""
    col = spec["value"] if task == "reg" else spec["abnormal"]
    q = d[d[col].notna() & d.split.isin(["train", eval_split])]
    if "ppg_end_s" in q and f"{name}_offset_min" in q:    # the PPG must end before the lab
        q = q[~(q[f"{name}_offset_min"] * 60 < q.ppg_end_s)]
    q = q.reset_index(drop=True)
    y = q[col].to_numpy(float)
    if task == "reg" and name in LOG_LABS:
        q, y = q[y > 0].reset_index(drop=True), np.log(y[y > 0])
    tr, ev = np.flatnonzero(q.split == "train"), np.flatnonzero(q.split == eval_split)
    if task == "clf":
        y = y.astype(int)
        if y[tr].sum() < MIN_POS_TRAIN or y[ev].sum() < MIN_POS_EVAL or (1 - y[ev]).sum() < MIN_POS_EVAL:
            return None
    groups = q.group.to_numpy()
    if too_small(task, y[ev], groups[ev]):
        return None
    bases = {"age_sex": q[["age", "female"]].to_numpy(float)}
    for tier, cols in tiers.items():
        bases[tier] = np.column_stack([bases["age_sex"], q[cols].astype(float).to_numpy()])
    bases["now"] = np.column_stack([bases["now"], flexible(q, tr)])
    v, age_h = None, None
    if spec["locf"]:
        v, age_h = q[spec["locf"][0]].to_numpy(float), q[spec["locf"][1]].to_numpy(float)
        if name in LOG_LABS or spec.get("log_locf"):
            v = np.log(np.where(v > 0, v, np.nan))
        bases["locf"] = np.column_stack([bases["age_sex"], v, np.log1p(age_h)])
        mean, n, trend = q[spec["hist"]].to_numpy(float).T
        log_mean = np.log(np.where(mean > 0, mean, np.nan)) if name in LOG_LABS or spec.get("log_locf") else mean
        bases["hist"] = np.column_stack([bases["locf"], log_mean, np.log1p(n), trend])
        # When and how often the lab was drawn, not what it showed. No earlier draw: the time is missing, which the imputer's indicator carries, and the count is 0
        bases["now"] = np.column_stack([bases["now"], np.log1p(age_h), np.log1p(np.nan_to_num(n))])
        bg = [c for c in (f"{name}_bg_locf", f"{name}_bg_locf_age_h") if c in q]
        bases["full"] = np.column_stack([bases["now"], v, log_mean, trend, q[bg].to_numpy(float)])
    return q, y, tr, ev, groups, bases, v, age_h


def run_lab(d, bcols, tiers, name, spec, eval_split, n_perm, models=None, only=None):
    """Rows and null deltas for one lab. `models` replaces the reference models (name, columns, model factory per task), and drops
    the boosting sensitivity, which is about the beat model; `only` keeps the named subsets."""
    rows, nulls = [], []
    for task in ("reg", "clf"):
        frame = task_frame(d, tiers, name, spec, task, eval_split)
        if frame is None:
            continue
        q, y, tr, ev, groups, bases, v, age_h = frame
        raw_locf = None
        subsets = {"all": np.ones(len(ev), bool)}
        if "map_mmhg" in q:   # MIMIC: windows where the acuity base has a measured MAP
            subsets["arterial_line"] = q.map_mmhg.notna().to_numpy()[ev]
        if f"{name}_offset_min" in q and "ppg_end_s" not in q:
            subsets["no_early_lab"] = ~(q[f"{name}_offset_min"].to_numpy() * 60 < q.best_start_s.to_numpy())[ev]
        if name == "lactate":    # every other lab shares the anchor's timestamp in 97.5-100% of windows
            subsets["within_30min"] = q[f"{name}_offset_min"].abs().to_numpy()[ev] <= SENS_MIN
        if name == "ptt" and "on_heparin" in q:    # abnormal PTT mostly marks a heparin infusion
            subsets["no_heparin"] = q.on_heparin.to_numpy()[ev] == 0
        if "stable_creat" in q and name in ("egfr", "creatinine"):
            subsets["stable_creat"] = q.stable_creat.to_numpy()[ev]
        elif "exclude_dialysis" in q:
            subsets["no_dialysis"] = ~q.exclude_dialysis.to_numpy(bool)[ev]
        if "dev_set" in q and eval_split == "val":
            subsets["not_dev"] = ~q.dev_set.to_numpy()[ev]
        if spec["locf"]:
            subsets["no_recent_locf"] = ~(age_h[ev] <= STALE_H)    # missing or older than STALE_H
            subsets["never_prior"] = ~np.isfinite(v[ev])    # truly no earlier value, unlike hiding it from a model
            if task == "reg" and not spec.get("log_locf"):
                raw_locf = v
                subsets["has_locf"] = np.isfinite(v[ev])
        female = q.female.to_numpy(float)[ev]
        subsets["female"], subsets["male"] = female == 1, female == 0
        if "eth_black" in q:    # white is the group without a flag of its own
            subsets["eth_black"] = q.eth_black.to_numpy(float)[ev] == 1
            subsets["eth_white"] = (q[[c for c in q if c.startswith("eth_")]].to_numpy(float)[ev] == 0).all(1)
        if only is not None:
            subsets = {k: v for k, v in subsets.items() if k in only}
        base_model = base_reg if task == "reg" else base_clf
        fit = (lambda m, A, B: m.fit(A[tr], y[tr]).predict_proba(B)[:, 1]) if task == "clf" else \
              (lambda m, A, B: m.fit(A[tr], y[tr]).predict(B))
        preds = {b: fit(base_model(), X, X[ev]) for b, X in bases.items()}
        pairs = []
        if task == "reg":
            preds["mean_train"] = np.full(len(ev), y[tr].mean())
        if "now" in bases and models is None:
            boost = boost_reg if task == "reg" else boost_clf
            Xb = np.column_stack([bases["now"], q[bcols].to_numpy(float)])
            preds["now_gb"], preds["now_gb+beat"] = fit(boost(), bases["now"], bases["now"][ev]), fit(boost(), Xb, Xb[ev])
            pairs.append(("now_gb+beat", "now_gb"))
        primary, floor = (r2_score, None) if task == "reg" else (roc_auc_score, 0.5)
        keep = lambda v: v if floor is None else max(v, floor)
        blocks = [[g.to_numpy() for _, g in pd.Series(i).groupby(groups[i])] for i in (tr, ev)]

        def stack(s_tr, s_ev, b):
            s = np.zeros(len(q))
            s[tr], s[ev] = s_tr, s_ev
            Z = np.column_stack([bases[b], s])
            return fit(base_model(), Z, Z[ev])

        for ppg, cols, family in models or reference_models(bcols, name):
            if cols[0] not in q or task not in family:
                continue
            X = q[cols].to_numpy(float)
            s_tr, preds[ppg] = ppg_score(family[task], X, y, groups, tr, ev)
            for b in bases:    # preds also holds other rows, so loop over the bases alone
                preds[f"{b}+{ppg}"] = stack(s_tr, preds[ppg], b)
                pairs.append((f"{b}+{ppg}", b))
            for p in range(0 if ppg == "rate" else n_perm):
                rng = np.random.default_rng([p, zlib.crc32(f"{name}|{task}|{ppg}".encode())])
                s_tr, s_ev = ppg_score(family[task], X[block_perm(blocks, len(q), rng)], y, groups, tr, ev)
                for b in bases:
                    nulls.append(dict(lab=name, task=task, ppg=ppg, base=b, perm=p,
                                      delta=primary(y[ev], stack(s_tr, s_ev, b)) - keep(primary(y[ev], preds[b]))))
        metrics = {"reg": {"r2": r2_score, "mae": mean_absolute_error},
                   "clf": {"auroc": roc_auc_score, "auprc": average_precision_score}}[task]
        for subset, m in subsets.items():
            ye, ge = y[ev][m], groups[ev][m]
            if too_small(task, ye, ge):
                continue
            pe, pr = {k: p[m] for k, p in preds.items()}, pairs
            if subset == "has_locf":
                pe["locf_raw"], pr = raw_locf[ev][m], pairs + [("locf", "locf_raw")]
            for mname, fn in metrics.items():
                res = subject_boot(ye, ge, pe, pr, fn, floor=0.5 if mname == "auroc" else None)
                for k, (est, lo, hi) in res.items():
                    method, _, vs = k.partition("|")
                    rows.append(dict(lab=name, task=task, subset=subset, metric=mname, method=method, vs=vs or None,
                                     est=est, lo=lo, hi=hi, n_train=len(tr), n_eval=len(ye),
                                     subjects_eval=len(np.unique(ge)),
                                     prev_eval=float(ye.mean()) if task == "clf" else np.nan))
    print(f"  {name} done", flush=True)
    return rows, nulls


def adjust(r):
    """Normal-approximation p from the CI width, BH and Holm, over the main increments (subset all, AUROC and R²)."""
    r = r.assign(p=np.nan, q_bh=np.nan, p_holm=np.nan)
    i = r.index[(r.subset == "all") & r.metric.isin(["auroc", "r2"]) & r.vs.notna() & ~r.vs.isin(["locf_raw", "now_gb"])
                & (r.lab != "egfr_incl_failure")]
    if len(i):
        se = ((r.loc[i, "hi"] - r.loc[i, "lo"]) / (2 * 1.96)).clip(lower=1e-12)
        p = 2 * norm.sf(np.abs(r.loc[i, "est"] / se)).clip(1e-300)
        o, m = np.argsort(p), len(p)
        bh, holm = np.empty(m), np.empty(m)
        bh[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1)
        holm[o] = np.maximum.accumulate(p[o] * (m - np.arange(m))).clip(max=1)
        r.loc[i, "p"], r.loc[i, "q_bh"], r.loc[i, "p_holm"] = p, bh, holm
    return r


def null_text(x):
    """Mean and 2.5-97.5% range of the null deltas in x."""
    return f"{x.delta.mean():+.3f} ({x.delta.quantile(.025):+.3f}, {x.delta.quantile(.975):+.3f})" if len(x) else "—"


def report(r, src, eval_split, n, n_sub, nulls):
    def cell(x, m, metric, vs=None):
        x = x[(x.metric == metric) & (x.method == m) & (x.vs.isna() if vs is None else x.vs == vs)]
        if x.empty:
            return "—"
        e = x.iloc[0]
        if not vs:
            return f"{e.est:.3f} ({e.lo:.3f}, {e.hi:.3f})"
        mark = "‡" if e.p_holm < 0.05 else "†" if e.q_bh < 0.05 else ""
        return f"{e.est:+.3f} ({e.lo:+.3f}, {e.hi:+.3f}){mark}"

    def null_cell(lab, task, b, p):
        x = nulls[(nulls.lab == lab) & (nulls.task == task) & (nulls.base == b) & (nulls.ppg == p)] if len(nulls) else nulls
        return null_text(x)

    def table(title, subset, task, metric, cols):
        """cols: (header, fn(x) -> str) applied to the rows of one lab."""
        out = [f"### {title}", "", "| Lab | n eval | " + ("prev | " if task == "clf" else "") + " | ".join(h for h, _ in cols)
               + " |", "|---|---|" + ("---|" if task == "clf" else "") + "---|" * len(cols)]
        for lab in dict.fromkeys(r.lab):
            x = r[(r.lab == lab) & (r.task == task) & (r.subset == subset)]
            if x.empty:
                continue
            row = [lab, str(x.n_eval.iloc[0])] + ([f"{x.prev_eval.iloc[0]:.1%}"] if task == "clf" else [])
            out.append("| " + " | ".join(row + [fn(x) for _, fn in cols]) + " |")
        return out + [""]

    bases = [b for b in LADDER if (r.method == b).any()]
    models = [p for p in MODELS if (r.method == p).any()]
    ppgs = [p for p in models if p != "rate"]
    L = [f"# PPG-LabBench baselines — {src}, train → {eval_split}", "",
         f"{n} windows, {n_sub} subjects. Δ = stacked (base + PPG score) − base (AUROC: base counted as at least 0.5), paired "
         "subject bootstrap 95% CI with the fitted models fixed. † BH q < 0.05, ‡ Holm p < 0.05 over the main increments; "
         "both ignore training variance, so read them as exploratory. Null = mean and 2.5-97.5% range of Δ over "
         f"{nulls.perm.nunique() if len(nulls) else 0} subject-block permutations of PPG.", ""]
    for task, metric, title, full in (("clf", "auroc", "T2/T3 abnormal value — AUROC", True),
                                      ("clf", "auprc", "T2/T3 abnormal value — AUPRC", False),
                                      ("reg", "r2", "T1/T3 value — R²", True),
                                      ("reg", "mae", "T1/T3 value — MAE (lower is better)", False)):
        L += [f"## {title}", ""]
        L += table("Baselines and PPG alone", "all", task, metric,
                   [(m, lambda x, m=m: cell(x, m, metric)) for m in ["mean_train"] * (task == "reg") + bases + models])
        L += table("Increment of PPG over each base", "all", task, metric,
                   [(f"Δ {b}+{p}", lambda x, b=b, p=p: cell(x, f"{b}+{p}", metric, b))
                    for b in bases for p in models])
        if not full:
            continue
        if (r.method == "now_gb").any():
            L += table("Same model family: boosting on `now`, then on `now` + beat features", "all", task, metric,
                       [("now_gb", lambda x: cell(x, "now_gb", metric)), ("Δ", lambda x: cell(x, "now_gb+beat", metric, "now_gb"))])
        L += table("Null control: subject-block permuted PPG", "all", task, metric,
                   [(f"Δ null {b}+{p}", lambda x, b=b, p=p: null_cell(x.lab.iloc[0], task, b, p))
                    for b in bases for p in ppgs])
        if (r.subset == "no_recent_locf").any():
            L += table(f"Subgroup: no LOCF within {STALE_H} h", "no_recent_locf", task, metric,
                       [("locf", lambda x: cell(x, "locf", metric))]
                       + [(f"Δ {b}+{p}", lambda x, b=b, p=p: cell(x, f"{b}+{p}", metric, b))
                          for b in ("locf", "acuity") for p in ppgs])
        if task == "reg" and (r.subset == "has_locf").any():
            L += table("LOCF ladder: previous value copied → fitted on it → + PPG (windows with a previous value)",
                       "has_locf", task, metric,
                       [("locf_raw", lambda x: cell(x, "locf_raw", metric)), ("locf", lambda x: cell(x, "locf", metric)),
                        ("Δ locf vs raw", lambda x: cell(x, "locf", metric, "locf_raw"))]
                       + [(f"Δ locf+{p}", lambda x, p=p: cell(x, f"locf+{p}", metric, "locf")) for p in ppgs])
        if (r.subset == "arterial_line").any():
            L += table("Subgroup: arterial line (MAP measured)", "arterial_line", task, metric,
                       [("acuity", lambda x: cell(x, "acuity", metric))]
                       + [(f"Δ {b}+{p}", lambda x, b=b, p=p: cell(x, f"{b}+{p}", metric, b))
                          for b in ("acuity", "locf") for p in ppgs])
        for subset, label in (("no_early_lab", "Sensitivity: lab not charted before the PPG window started"),
                              ("never_prior", "Subgroup: no earlier value of this lab at all"),
                              ("no_dialysis", "Sensitivity: without dialysis-flagged windows"),
                              ("no_heparin", "Sensitivity: PTT without a heparin infusion"),
                              ("stable_creat", "Sensitivity: creatinine stable (previous value within 48 h, change < 20%)"),
                              ("within_30min", f"Sensitivity: lactate within ± {SENS_MIN} min"),
                              ("not_dev", "Sensitivity: without the development subjects"),
                              ("female", "Descriptive: female"), ("male", "Descriptive: male"),
                              ("eth_black", "Descriptive: ethnicity group Black"),
                              ("eth_white", "Descriptive: ethnicity group White (no other group flag set)")):
            if (r.subset == subset).any():
                L += table(label, subset, task, metric,
                           [(f"Δ {b}+{p}", lambda x, b=b, p=p: cell(x, f"{b}+{p}", metric, b))
                            for b in bases[1:] for p in ppgs])
    return "\n".join(L) + "\n"


def check_test_allowed(stem, confirmed):
    """The test split is evaluated once per source, on purpose, from a clean tree and the committed split file."""
    sha = hashlib.sha256((DATA / "benchmark" / "mimic_splits.csv").read_bytes()).hexdigest()
    problems = [m for bad, m in ((not confirmed, "--open-test not given"),
                                 (git_version().endswith(("-dirty", "unknown")), "the working tree is not clean"),
                                 (sha != C["splits_sha256"], "mimic_splits.csv differs from splits_sha256 in config.yaml"),
                                 ((OUT / stem).exists(), f"{OUT / stem} exists")) if bad]
    if problems:
        sys.exit("test split not opened: " + "; ".join(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["mimic", "vitaldb"])
    ap.add_argument("--eval", default="val", choices=["val", "test"])
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--open-test", action="store_true", help="required with --eval test: its one evaluation")
    ap.add_argument("--window", default="pre", choices=["pre", "centered"])
    args = ap.parse_args()
    stem = results_stem(args.source, args.eval, args.window)
    if args.eval == "test":
        check_test_allowed(stem, args.open_test)
    d, bcols, labs, tiers = load_mimic(args.window, open_test=args.eval == "test") if args.source == "mimic" else load_vitaldb()
    d = d[d.split.isin(["train", args.eval])].reset_index(drop=True)
    print(f"{args.source}: {len(d)} windows, {d.group.nunique()} subjects, {len(bcols)} beat features, {len(labs)} labs",
          flush=True)
    res = Parallel(n_jobs=args.jobs)(delayed(run_lab)(d, bcols, tiers, n, s, args.eval, args.perms)
                                     for n, s in labs.items())
    r = adjust(pd.DataFrame([x for rs, _ in res for x in rs]))
    nulls = pd.DataFrame([x for _, ns in res for x in ns], columns=["lab", "task", "ppg", "base", "perm", "delta"])
    out = OUT / stem
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "results.csv", index=False)
    nulls.to_csv(out / "null.csv", index=False)
    md = report(r, args.source, args.eval, len(d), d.group.nunique(), nulls)
    (out / "report.md").write_text(md)
    print(md)
    write_manifest(out, f"baselines_{stem}", dict(source=args.source, eval=args.eval, window=args.window, k=K, n_boot=N_BOOT, perms=args.perms,
                                                  boost=BOOST, min_n=MIN_N, tiers=tiers),
                   dict(windows=len(d), subjects=int(d.group.nunique()), labs=list(labs)))


if __name__ == "__main__":
    main()
