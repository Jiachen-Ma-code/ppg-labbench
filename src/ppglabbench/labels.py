"""Benchmark step 2-3: multi-lab labels, LOCF covariates and frozen subject splits for every S6 window.

Input : data/s6_dataset[_<tag>]/dataset.csv (S6: one row per group B window, with age, sex, eGFR,
        diagnoses, dialysis flag) and LABEVENTS.
Labels: for each lab in config `benchmark.labs`, the measurement nearest to the window centre within
        ± lab_window_min; its offset (min) is kept so users can apply a stricter window.
LOCF  : the previous value of the same lab, strictly before both the target measurement and the window
        start (so it never is the target itself, and never overlaps the PPG); its age in hours. The blood-gas analyser's
        value of the same analyte is kept apart (`<lab>_bg_locf`), because a clinician sees it but the two differ in bias.
        Censored results (">150", ">GREATER THAN 150", "150 IS HIGHEST MEASURED PTT") count as their limit.
History: mean and count of all earlier values of the lab and its recent trend, before the same reference time.
Window: the primary PPG window is the best contiguous window in the half of the S3 window before the draw (`pre_start_s`,
        `pre_score`; ties go to the window nearest the draw, up to floating-point rounding). `best_start_s` from S4 is the centred window, kept for sensitivity.
Splits: train / val / test by a salted SHA-256 hash of subject_id, except that subjects in the development set (`dev_exclusion`) are never placed in
        test, so exact reproduction needs the split file. Where that folder is absent, as outside this project, the development set comes from the
        `dev_set` column of an existing mimic_splits.csv.

Usage:  python -m ppglabbench.labels [--s6 s6_dataset]
Output: data/benchmark/{mimic_labels.csv, mimic_splits.csv, manifest.json}  (DUA data: not committed)
"""
import argparse
import hashlib
import sys

import numpy as np
import pandas as pd

from ppglabbench.common import CFG, DATA, S1, S4, extract_rows, load_creatinine, write_manifest

C = CFG["benchmark"]
OUT = DATA / "benchmark"
LAB_CACHE = "labs_bench_v4"     # the cache of extract_rows: a new name whenever CENSORED or the rows kept change
CENSORED = r"^\s*(?:[<>]=?\s*(?:GREATER THAN|LESS THAN)?|GREATER THAN|LESS THAN)\s*(\d+\.?\d*)|^\s*(\d+\.?\d*)\s+IS HIGHEST"


def abnormal(rule, v, female):
    if rule == "anaemia":
        return np.where(np.isnan(v), np.nan, v < np.where(female == 1, 12, 13))
    op, x = (rule[:2], float(rule[2:])) if rule[:2] in (">=", "<=") else (rule[0], float(rule[1:]))
    f = {"<": np.less, ">": np.greater, ">=": np.greater_equal, "<=": np.less_equal}[op]
    return np.where(np.isnan(v), np.nan, f(v, x))


def align(d, lab, name):
    """Nearest value within ± lab_window_min, and LOCF strictly before target and window start."""
    lab = lab.sort_values("charttime").rename(columns={"charttime": "lab_time", "valuenum": name})
    tol = pd.Timedelta(minutes=C["lab_window_min"])
    near = pd.merge_asof(d.sort_values("charttime")[["draw_id", "subject_id", "charttime"]], lab,
                         left_on="charttime", right_on="lab_time", by="subject_id", direction="nearest",
                         tolerance=tol)
    near[f"{name}_offset_min"] = (near.lab_time - near.charttime).dt.total_seconds() / 60
    start = near.charttime - pd.Timedelta(minutes=C["window_half_min"])
    near["t_ref"] = near.lab_time.where(near.lab_time < start, start).fillna(start)
    g = lab.groupby("subject_id")
    hist = lab.assign(hist_n=g.cumcount() + 1, prev2=g[name].shift(1), prev2_time=g.lab_time.shift(1))
    hist["hist_mean"] = g[name].cumsum() / hist.hist_n
    prev = pd.merge_asof(near.sort_values("t_ref")[["draw_id", "subject_id", "t_ref"]],
                         hist.rename(columns={name: f"{name}_locf", "lab_time": "locf_time"}),
                         left_on="t_ref", right_on="locf_time", by="subject_id", direction="backward",
                         allow_exact_matches=False)
    prev[f"{name}_locf_age_h"] = (prev.t_ref - prev.locf_time).dt.total_seconds() / 3600
    gap = ((prev.locf_time - prev.prev2_time).dt.total_seconds() / 3600).clip(lower=1)
    prev[f"{name}_hist_mean"], prev[f"{name}_hist_n"] = prev.hist_mean, prev.hist_n
    prev[f"{name}_trend"] = (prev[f"{name}_locf"] - prev.prev2) / gap
    return near[["draw_id", name, f"{name}_offset_min"]].merge(
        prev[["draw_id", f"{name}_locf", f"{name}_locf_age_h", f"{name}_hist_mean", f"{name}_hist_n", f"{name}_trend"]], on="draw_id")


def locf_at(out, lab, name):
    """The previous value in `lab` before the same reference time as the central LOCF, and its age in hours."""
    start = out.charttime - pd.Timedelta(minutes=C["window_half_min"])
    target = out.charttime + pd.to_timedelta(out[f"{name}_offset_min"], unit="m")
    ref = out[["draw_id", "subject_id"]].assign(t_ref=target.where(target < start, start).fillna(start)).sort_values("t_ref")
    p = pd.merge_asof(ref, lab.sort_values("charttime").rename(columns={"charttime": "locf_time", "valuenum": f"{name}_bg_locf"}),
                      left_on="t_ref", right_on="locf_time", by="subject_id", direction="backward", allow_exact_matches=False)
    p[f"{name}_bg_locf_age_h"] = (p.t_ref - p.locf_time).dt.total_seconds() / 3600
    return p[["draw_id", f"{name}_bg_locf", f"{name}_bg_locf_age_h"]]


def pre_draw_window(draw_ids):
    """Start (s from the draw) and good-epoch fraction of the best window wholly before the draw, as S4 scores windows. Ties go to the window
    nearest the draw up to floating-point rounding: the scores are running sums of a float, so two windows with the same number of good epochs can
    differ in the last bit, and in about 0.3% of windows the tie goes to an earlier one (about 1 in 16,000 of those scoring at least 0.8). The
    score of the chosen window is not affected."""
    good = np.load(S4 / "epoch_good.npz")
    half, ep = CFG["s3_fetch"]["half_window_min"] * 60, CFG["s4_sqi"]["epoch_s"]
    w = CFG["s4_sqi"]["window_min"] * 60 // ep
    start, score = [], []
    for i in draw_ids:
        s = np.convolve(good[i][: half // ep].astype(float), np.ones(w) / w, "valid")
        k = len(s) - 1 - int(np.argmax(s[::-1]))
        start.append(-half + k * ep)
        score.append(s[k])
    return start, score


def overlapping_records():
    """PLETH records that overlap another PLETH record of the same subject. The two streams disagree (different heart
    rates, no correlation), so at least one may be another patient's: such windows are flagged, then left out."""
    c = pd.read_csv(S1 / "waveform_catalog.csv", parse_dates=["start", "end"])
    c = c[c.error.isna() & (c.has_pleth == True)].sort_values(["subject_id", "start"])
    later = c.end > c.groupby("subject_id").start.shift(-1)
    earlier = c.start < c.groupby("subject_id").end.cummax().groupby(c.subject_id).shift()
    return set(c.record[later | earlier])


def split_of(subject_id, dev):
    tr, va, _ = C["split"]
    u = int(hashlib.sha256(f"{C['split_salt']}:{subject_id}".encode()).hexdigest()[:12], 16) / 16 ** 12
    s = "train" if u < tr else "val" if u < tr + va else "test"
    return "train" if (s == "test" and subject_id in dev) else s


def development_subjects():
    """Subjects used for method development: from the `dev_exclusion` folder, or, where it is absent, from an existing split file."""
    folder = DATA / C["dev_exclusion"]
    if folder.exists():
        return set(pd.read_csv(folder / "groupB_draws.csv").subject_id)
    splits = OUT / "mimic_splits.csv"
    if not splits.exists():
        sys.exit(f"the development set is needed to keep those subjects out of test: neither {folder} nor {splits} exists")
    sp = pd.read_csv(splits, usecols=["subject_id", "dev_set"])
    return set(sp.subject_id[sp.dev_set])


def keep_split_file(path, new):
    """The frozen split file is never replaced by a different one: it is written only if absent, and an existing file must hold the same bytes."""
    if not path.exists():
        path.write_bytes(new)
    elif path.read_bytes() != new:
        sys.exit(f"{path} holds other splits than this run computed, and it is not replaced: nothing was written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s6", default="s6_dataset")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(DATA / args.s6 / "dataset.csv", parse_dates=["charttime"])
    items = {name: v[0] for name, v in C["labs"].items()}
    bg_items = C["blood_gas"]
    labs = extract_rows("LABEVENTS", LAB_CACHE,
                        lambda c: c.itemid.isin(list(items.values()) + list(bg_items.values())) &
                        ((c.valuenum > 0) | c.value.astype(str).str.contains(CENSORED, regex=True)),
                        {"subject_id", "itemid", "charttime", "value", "valuenum"}, dates=("charttime",))
    limit = labs.value.astype(str).str.extract(CENSORED).bfill(axis=1).iloc[:, 0].astype(float)
    labs["valuenum"] = labs.valuenum.where(labs.valuenum > 0, limit)
    labs = labs[labs.valuenum > 0]
    out = d.copy()
    for name, item in items.items():
        out = out.merge(align(d, labs[labs.itemid == item][["subject_id", "charttime", "valuenum"]], name),
                        on="draw_id", how="left")
        out[f"{name}_abnormal"] = abnormal(C["labs"][name][1], out[name].to_numpy(float), out.female.to_numpy())
    for name, item in bg_items.items():
        out = out.merge(locf_at(out, labs[labs.itemid == item][["subject_id", "charttime", "valuenum"]], name), on="draw_id", how="left")
    cr = load_creatinine()[["subject_id", "charttime", "creat_mgdl"]].rename(columns={"creat_mgdl": "valuenum"})
    loc = align(d, cr, "creat_check")
    cols = {f"creat_check_{k}": f"creat_{k}" for k in ("locf", "locf_age_h", "hist_mean", "hist_n", "trend")}
    out = out.merge(loc[["draw_id", *cols]].rename(columns=cols), on="draw_id")
    out["pre_start_s"], out["pre_score"] = pre_draw_window(out.draw_id)
    out["egfr_lt60"] = (out.egfr < 60).astype(float)
    ov = overlapping_records()
    out["record_overlap"] = out.records.str.split("|").map(lambda r: bool(ov.intersection(r)))

    dev = development_subjects()
    sp = pd.DataFrame({"subject_id": sorted(out.subject_id.unique())})
    sp["split"] = [split_of(s, dev) for s in sp.subject_id]
    sp["dev_set"] = sp.subject_id.isin(dev)
    out = out.merge(sp[["subject_id", "split"]], on="subject_id")
    keep_split_file(OUT / "mimic_splits.csv", sp.to_csv(index=False).encode())
    out.to_csv(OUT / "mimic_labels.csv", index=False)
    sha = hashlib.sha256((OUT / "mimic_splits.csv").read_bytes()).hexdigest()

    nt = out[out.split != "test"]   # the sealed test split stays out of these summaries
    cov = {n: dict(frac=float(nt[n].notna().mean()), abnormal=float(np.nanmean(nt[f"{n}_abnormal"])),
                   locf=float(nt[f"{n}_locf"].notna().mean())) for n in items}
    counts = dict(windows=len(out), subjects=int(out.subject_id.nunique()),
                  split_subjects=sp.split.value_counts().to_dict(),
                  split_windows=out.split.value_counts().to_dict(),
                  dev_subjects_in_test=int((sp.dev_set & (sp.split == "test")).sum()),
                  windows_record_overlap=int(out.record_overlap.sum()),
                  windows_pre_draw_ok=int((out.pre_score >= CFG["s5_select"]["min_window_score"]).sum()),
                  splits_sha256=sha, coverage=cov)
    write_manifest(OUT, "benchmark_labels", dict(C, s6=args.s6), counts)
    print(pd.DataFrame(cov).T.round(3).to_string())
    print({k: v for k, v in counts.items() if k != "coverage"})


if __name__ == "__main__":
    main()
