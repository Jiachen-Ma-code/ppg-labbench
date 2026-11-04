"""Benchmark covariate: mean arterial pressure (MAP) from the invasive ABP waveform, MIMIC only.

For every benchmark window, ABP is streamed from the same waveform segments as the PLETH (S3) over the same
5 min analysis window (best before the draw by default; `--window centered` is S4's window around it). ABP is a separate sensor (arterial line), so
MAP carries illness severity into the acuity base without carrying any PPG information. Per 10 s epoch an epoch
is valid if ≥ 90% of samples are finite, its mean is within 20-200 mmHg and its pulse pressure (p95 - p5) is
10-150 mmHg (rules out flush, zeroing and damped lines). MAP = median epoch mean over valid epochs; NaN with
fewer than 6 valid epochs (1 min) or no arterial waveform. Missing MAP is left to the baseline's imputer indicator,
so "no usable arterial waveform" also enters the base. It does not strictly mean no arterial line: the numerics may hold an arterial mean.
Resumable: draws already in abp_map.csv are skipped and failed rows are retried; rows are appended per subject.

Usage:
    python -m ppglabbench.features.abp_map --limit-subjects 20   # test batch
    python -m ppglabbench.features.abp_map                       # all benchmark windows
Output: data/benchmark/features/<window>/mimic_abp/{abp_map.csv, manifest.json}  (DUA data: not committed)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import pandas as pd
import wfdb

from ppglabbench.common import CFG, DATA, S3, segment_signals, segment_spans, subject_dir, write_manifest
from ppglabbench.pipeline.s3_fetch_windows import Progress

FS = CFG["fs"]
WIN_S = CFG["s4_sqi"]["window_min"] * 60
EP = CFG["s4_sqi"]["epoch_s"] * FS
ABP_NAMES = ("ABP", "ART")
MIN_EPOCHS = 6
FEAT = DATA / "benchmark" / "features"


def epoch_means(x):
    """Mean of each valid 10 s epoch."""
    out = []
    for i in range(len(x) // EP):
        e = x[i * EP:(i + 1) * EP]
        f = e[np.isfinite(e)]
        if len(f) < 0.9 * EP:
            continue
        m, pp = f.mean(), np.percentile(f, 95) - np.percentile(f, 5)
        if 20 <= m <= 200 and 10 <= pp <= 150:
            out.append(m)
    return np.array(out)


def draw_map(d):
    pn_dir = subject_dir(d.subject_id)
    t0 = pd.Timestamp(d.charttime) + timedelta(seconds=d.start_s)
    t1 = t0 + timedelta(seconds=WIN_S)
    n = WIN_S * FS
    x = np.full(n, np.nan, dtype=np.float32)
    channel = ""
    for rec in d.records.split("|"):
        for name, start, nseg, fs in segment_spans(rec, pn_dir):
            if start + timedelta(seconds=nseg / fs) <= t0 or start >= t1:
                continue
            sig = segment_signals(name, pn_dir)
            ch = next((c for c in ABP_NAMES if c in sig), None)
            if ch is None:
                continue
            a = max(0, round((t0 - start).total_seconds() * fs))
            b = min(nseg, round((t1 - start).total_seconds() * fs))
            if b <= a:
                continue
            r = wfdb.rdrecord(name, pn_dir=pn_dir, sampfrom=a, sampto=b, channel_names=[ch])
            off = round((start - t0).total_seconds() * fs) + a
            lo, hi = max(0, off), min(n, off + (b - a))
            x[lo:hi] = r.p_signal[lo - off:hi - off, 0]
            channel = ch
    m = epoch_means(x)
    return dict(draw_id=d.draw_id, subject_id=d.subject_id, abp_channel=channel, abp_epochs_valid=len(m),
                map_mmhg=round(float(np.median(m)), 1) if len(m) >= MIN_EPOCHS else np.nan, error="")


def fetch_subject(draws, progress):
    rows = []
    for d in draws.itertuples():
        try:
            rows.append(draw_map(d))
        except Exception as e:
            rows.append(dict(draw_id=d.draw_id, subject_id=d.subject_id, abp_channel="", abp_epochs_valid=0,
                             map_mmhg=np.nan, error=str(e)[:200]))
        progress.tick()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-subjects", type=int, default=None)
    ap.add_argument("--workers", type=int, default=CFG["s3_fetch"]["workers"])
    ap.add_argument("--window", default="pre", choices=["pre", "centered"])
    args = ap.parse_args()
    OUT = FEAT / args.window / "mimic_abp"
    CSV = OUT / "abp_map.csv"
    OUT.mkdir(parents=True, exist_ok=True)

    draws = pd.read_csv(DATA / "benchmark" / "mimic_labels.csv",
                        usecols=["draw_id", "subject_id", "charttime", "records", "best_start_s", "pre_start_s", "pre_score"])
    if args.window == "pre":
        draws = draws[draws.pre_score >= CFG["s5_select"]["min_window_score"]]
    draws["start_s"] = draws.pre_start_s if args.window == "pre" else draws.best_start_s
    if CSV.exists():   # rows that failed (e.g. PhysioNet 502) are dropped and fetched again
        prev = pd.read_csv(CSV)
        prev[prev.error.isna()].to_csv(CSV, index=False)
    done = set(pd.read_csv(CSV).draw_id) if CSV.exists() else set()
    todo = draws[~draws.draw_id.isin(done)]
    subjects = list(todo.groupby("subject_id"))[: args.limit_subjects]
    n_todo = sum(len(g) for _, g in subjects)
    print(f"benchmark windows: {len(draws)}; {len(done)} done; fetching {n_todo} from {len(subjects)} subjects")

    progress = Progress(n_todo)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_subject, g, progress) for _, g in subjects]
        for fut in as_completed(futs):
            pd.DataFrame(fut.result()).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)

    f = pd.read_csv(CSV)
    write_manifest(OUT, "abp_map", dict(window_s=WIN_S, epoch_s=CFG["s4_sqi"]["epoch_s"], channels=ABP_NAMES,
                                        min_epochs=MIN_EPOCHS), dict(
        windows=len(draws), attempted=len(f), complete=bool(len(f) >= len(draws) and f.error.isna().all()), errors=int(f.error.notna().sum()),
        with_abp_channel=int(f.abp_channel.notna().sum()), with_map=int(f.map_mmhg.notna().sum()),
        map_median=float(f.map_mmhg.median())))


if __name__ == "__main__":
    main()
