"""Benchmark covariate: bedside monitor values at the PPG window, from the open waveform numerics records (MIMIC only).

The `<record>n` numerics hold what the monitor displayed, at 1/s or 1/min. HR (ECG), RESP (impedance), SpO2, irregular-HR percent and
PVC rate are the median over the 5 min PPG window; NBP is intermittent, so it is the last reading in the hour before the window end. PULSE comes from
PLETH itself, so it is stored for checking only and never enters a base. These let the base hold what a clinician already
sees without deriving it from the PPG under test (SPEC §3). Channels are absent for some records and left missing.
Resumable: draws already in numerics.csv are skipped and failed rows are retried; rows are appended per subject.

Usage:
    python -m ppglabbench.features.numerics --limit-subjects 20   # test batch
    python -m ppglabbench.features.numerics                       # all benchmark windows
Output: data/benchmark/features/<window>/mimic_num/{numerics.csv, manifest.json}  (DUA data: not committed)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import pandas as pd
import wfdb

from ppglabbench.common import CFG, DATA, subject_dir, write_manifest
from ppglabbench.pipeline.s3_fetch_windows import Progress

WIN_S = CFG["s4_sqi"]["window_min"] * 60
NBP_LOOK_S = 3600
FEAT = DATA / "benchmark" / "features"
WINDOW = {"HR": "hr_mon", "RESP": "resp_mon", "SPO2": "spo2_mon", "PULSE": "pulse_mon",
          "IRREGULARHRPERCENT": "irregular_mon", "PVCRATEPERMINUTE": "pvc_mon"}
LAST = {"NBPMEAN": "nbp_mean", "NBPSYS": "nbp_sys", "NBPDIAS": "nbp_dias"}
COLS = list(WINDOW.values()) + list(LAST.values())


def canon(name):
    return name.upper().replace(" ", "").replace("-", "").replace("%", "")   # some records spell it %SpO2


def draw_values(d):
    pn_dir = subject_dir(d.subject_id)
    t0 = pd.Timestamp(d.charttime) + timedelta(seconds=d.start_s)
    t1 = t0 + timedelta(seconds=WIN_S)
    out = dict(draw_id=d.draw_id, subject_id=d.subject_id, has_numerics=False, error="")
    out.update({c: np.nan for c in COLS})
    for rec in d.records.split("|"):
        name = rec + "n"
        try:
            h = wfdb.rdheader(name, pn_dir=pn_dir)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                continue
            raise
        start, n = pd.Timestamp(h.base_datetime), h.sig_len
        a = max(0, int((t1 - start).total_seconds() * h.fs) - int((WIN_S + NBP_LOOK_S) * h.fs))
        b = min(n, int(np.ceil((t1 - start).total_seconds() * h.fs)))
        if b <= a:
            continue
        r = wfdb.rdrecord(name, pn_dir=pn_dir, sampfrom=a, sampto=b)
        out["has_numerics"] = True
        i0 = int((t0 - start).total_seconds() * h.fs) - a
        for j, ch in enumerate(r.sig_name):
            c, x = canon(ch), r.p_signal[:, j]
            if c in WINDOW and np.isfinite(x[max(i0, 0):]).any():
                out[WINDOW[c]] = float(np.nanmedian(x[max(i0, 0):]))
            if c in LAST and np.isfinite(x).any():
                out[LAST[c]] = float(x[np.isfinite(x)][-1])
    return out


def fetch_subject(draws, progress):
    rows = []
    for d in draws.itertuples():
        try:
            rows.append(draw_values(d))
        except Exception as e:
            rows.append(dict(draw_id=d.draw_id, subject_id=d.subject_id, has_numerics=False, error=str(e)[:200],
                             **{c: np.nan for c in COLS}))
        progress.tick()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-subjects", type=int, default=None)
    ap.add_argument("--workers", type=int, default=CFG["s3_fetch"]["workers"])
    ap.add_argument("--window", default="pre", choices=["pre", "centered"])
    args = ap.parse_args()
    OUT = FEAT / args.window / "mimic_num"
    CSV = OUT / "numerics.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    draws = pd.read_csv(DATA / "benchmark" / "mimic_labels.csv",
                        usecols=["draw_id", "subject_id", "charttime", "records", "best_start_s", "pre_start_s", "pre_score"])
    if args.window == "pre":
        draws = draws[draws.pre_score >= CFG["s5_select"]["min_window_score"]]
    draws["start_s"] = draws.pre_start_s if args.window == "pre" else draws.best_start_s
    if CSV.exists():
        prev = pd.read_csv(CSV)
        prev[prev.error.isna()].to_csv(CSV, index=False)
    done = set(pd.read_csv(CSV).draw_id) if CSV.exists() else set()
    subjects = list(draws[~draws.draw_id.isin(done)].groupby("subject_id"))[: args.limit_subjects]
    n_todo = sum(len(g) for _, g in subjects)
    print(f"benchmark windows: {len(draws)}; {len(done)} done; fetching {n_todo} from {len(subjects)} subjects")
    progress = Progress(n_todo)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_subject, g, progress) for _, g in subjects]
        for fut in as_completed(futs):
            pd.DataFrame(fut.result()).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
    f = pd.read_csv(CSV)
    write_manifest(OUT, "numerics", dict(window_s=WIN_S, nbp_look_s=NBP_LOOK_S), dict(
        windows=len(draws), attempted=len(f), complete=bool(len(f) >= len(draws) and f.error.isna().all()),
        errors=int(f.error.notna().sum()), with_numerics=int(f.has_numerics.sum()),
        **{f"with_{c}": int(f[c].notna().sum()) for c in COLS}))


if __name__ == "__main__":
    main()
