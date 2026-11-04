"""S3: download PLETH around every selected group A draw (s3_selected, see S2).

For each draw, stores PLETH over [t - H, t + H] as one fixed-length array aligned to the
draw time (sample H*fs = draw time), NaN where no PLETH segment covers. Only the needed
sample ranges are streamed from PhysioNet.
Resumable: draws already in s3_index.csv are skipped; the index is appended per subject.

Usage:
    python -m ppglabbench.pipeline.s3_fetch_windows --limit-subjects 5   # test batch
    python -m ppglabbench.pipeline.s3_fetch_windows                      # all of group A
Output:
    data/s3_windows/windows/<draw_id>.npz   pleth (float16), fs, t0_s (= -H)
    data/s3_windows/s3_index.csv            draw_id, subject_id, file, coverage, n_segments, error
"""
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import pandas as pd
import wfdb

from ppglabbench.common import CFG, S2, S3, segment_signals, segment_spans, subject_dir, write_manifest

WINDOWS = S3 / "windows"
INDEX = S3 / "s3_index.csv"
FS = CFG["fs"]
HALF_S = CFG["s3_fetch"]["half_window_min"] * 60


def fetch_draw(d):
    pn_dir = subject_dir(d.subject_id)
    t0, t1 = d.charttime - timedelta(seconds=HALF_S), d.charttime + timedelta(seconds=HALF_S)
    n = 2 * HALF_S * FS
    x = np.full(n, np.nan, dtype=np.float32)
    used = []
    for rec in d.records.split("|"):
        for name, start, nseg, fs in segment_spans(rec, pn_dir):
            assert fs == FS, f"{rec}: fs={fs}"
            if start + timedelta(seconds=nseg / fs) <= t0 or start >= t1:
                continue
            if "PLETH" not in segment_signals(name, pn_dir):
                continue
            a = max(0, round((t0 - start).total_seconds() * fs))
            b = min(nseg, round((t1 - start).total_seconds() * fs))
            if b <= a:
                continue
            r = wfdb.rdrecord(name, pn_dir=pn_dir, sampfrom=a, sampto=b, channel_names=["PLETH"])
            off = round((start - t0).total_seconds() * fs) + a
            lo, hi = max(0, off), min(n, off + (b - a))
            x[lo:hi] = r.p_signal[lo - off:hi - off, 0]
            used.append(name)
    coverage = float(np.isfinite(x).mean())
    row = dict(draw_id=d.draw_id, subject_id=d.subject_id, file="", coverage=round(coverage, 4),
               n_segments=len(used), segments="|".join(used), error="")
    if coverage > 0:
        np.savez_compressed(WINDOWS / f"{d.draw_id}.npz", pleth=x.astype(np.float16), fs=FS, t0_s=-HALF_S)
        row["file"] = f"{d.draw_id}.npz"
    return row


class Progress:
    """Thread-safe draw counter that prints rate and ETA every `every` draws."""

    def __init__(self, total, every=50):
        self.total, self.every, self.n, self.t0 = total, every, 0, time.time()
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.n += 1
            if self.n % self.every == 0 or self.n == self.total:
                dt = time.time() - self.t0
                eta_h = (self.total - self.n) * dt / self.n / 3600
                print(f"  {self.n}/{self.total} draws  {dt / 60:.1f} min  "
                      f"{dt / self.n:.2f} s/draw  ETA {eta_h:.1f} h", flush=True)


def fetch_subject(draws, progress):
    rows = []
    for d in draws.itertuples():
        try:
            rows.append(fetch_draw(d))
        except Exception as e:
            rows.append(dict(draw_id=d.draw_id, subject_id=d.subject_id, file="", coverage=0.0,
                             n_segments=0, segments="", error=str(e)[:200]))
        progress.tick()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-subjects", type=int, default=None)
    args = ap.parse_args()
    WINDOWS.mkdir(parents=True, exist_ok=True)

    draws = pd.read_csv(S2 / "groupA_draws.csv", parse_dates=["charttime"])
    draws = draws[draws.s3_selected]
    done = set(pd.read_csv(INDEX).draw_id) if INDEX.exists() else set()
    todo = draws[~draws.draw_id.isin(done)]
    subjects = list(todo.groupby("subject_id"))[: args.limit_subjects]
    n_todo = sum(len(g) for _, g in subjects)
    print(f"group A: {len(draws)} draws; {len(done)} done; fetching {n_todo} draws from {len(subjects)} subjects")

    progress = Progress(n_todo)
    with ThreadPoolExecutor(CFG["s3_fetch"]["workers"]) as ex:
        futs = [ex.submit(fetch_subject, g, progress) for _, g in subjects]
        for fut in as_completed(futs):
            pd.DataFrame(fut.result()).to_csv(INDEX, mode="a", header=not INDEX.exists(), index=False)

    idx = pd.read_csv(INDEX)
    have = idx[idx.coverage > 0]
    write_manifest(S3, "s3_fetch", CFG["s3_fetch"], dict(
        s3_selected_draws=len(draws), draws_attempted=len(idx), fetch_complete=len(idx) >= len(draws),
        draws_error=int(idx.error.notna().sum()), draws_no_pleth_at_time=int((idx.coverage == 0).sum()),
        draws_with_pleth=len(have), subjects_with_pleth=int(have.subject_id.nunique()),
        median_coverage=float(have.coverage.median()) if len(have) else 0.0,
        disk_mb=round(sum(f.stat().st_size for f in WINDOWS.glob("*.npz")) / 1e6, 1)))


if __name__ == "__main__":
    main()
