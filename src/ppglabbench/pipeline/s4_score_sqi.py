"""S4: score PLETH quality for every S3 draw and pick its best 5 min window.

Per 10 s epoch (all thresholds in config.yaml, s4_sqi):
  valid      no missing samples
  flat_frac  fraction of samples in constant runs >= flat_run_s (saturation / probe off)
  hr         60 / median inter-beat interval, must lie in hr_range_bpm
  tcorr      mean correlation of each beat with the epoch's average beat (template matching,
             Li & Clifford 2012; Orphanidou et al. 2015)
  skewness   skewness of the filtered epoch (Elgendi 2016) - reported, not thresholded
  ibi_cv     inter-beat interval CV - reported, not thresholded, so arrhythmic (e.g. AF)
             patients are not silently removed; use it as a flag / covariate instead
  good       valid & flat_frac <= max_flat_frac & hr in range & tcorr >= min_template_corr

Window score = fraction of good epochs in a contiguous window_min window; the best window
is the highest score, ties broken by closeness to the draw time.

Usage:
    python -m ppglabbench.pipeline.s4_score_sqi --plot 12     # also saves a review sample
Output:
    data/s4_sqi/sqi_scores.csv        one row per draw
    data/s4_sqi/epoch_good.npz        per-draw boolean epoch array (for figures / re-thresholding)
    data/s4_sqi/review_sample.csv/png random draws for manual SQI validation
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import skew

from ppglabbench.common import CFG, S3, S4, write_manifest

C = CFG["s4_sqi"]
FS = CFG["fs"]
EP = C["epoch_s"] * FS
SOS = butter(4, C["bandpass_hz"], btype="band", fs=FS, output="sos")


def finite_runs(mask):
    """(start, stop) of contiguous True runs."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def flat_fraction(x):
    same = np.r_[False, np.diff(x) == 0]
    run_min = int(C["flat_run_s"] * FS)
    flat = sum(b - a + 1 for a, b in finite_runs(same) if b - a + 1 >= run_min)
    return flat / len(x)


def bandpass(x):
    """Filter each finite run separately so NaN gaps do not leak transients."""
    y = np.full_like(x, np.nan)
    for a, b in finite_runs(np.isfinite(x)):
        if b - a >= 3 * EP:
            y[a:b] = sosfiltfilt(SOS, x[a:b])
    return y


def epoch_metrics(raw, filt):
    m = dict(valid=False, flat_frac=np.nan, hr=np.nan, tcorr=np.nan, skewness=np.nan, ibi_cv=np.nan, good=False)
    if not (np.isfinite(raw).all() and np.isfinite(filt).all()):
        return m
    m.update(valid=True, flat_frac=flat_fraction(raw), skewness=float(skew(filt)))
    lo_bpm, hi_bpm = C["hr_range_bpm"]
    p5, p95 = np.percentile(filt, [5, 95])
    peaks, _ = find_peaks(filt, distance=int(FS * 60 / hi_bpm), prominence=C["peak_prominence_frac"] * (p95 - p5))
    if len(peaks) < 3:
        return m
    ibi = np.diff(peaks) / FS
    m.update(hr=60 / np.median(ibi), ibi_cv=float(ibi.std() / ibi.mean()))
    half = int(np.median(ibi) * FS / 2)
    beats = np.array([filt[p - half:p + half] for p in peaks if p - half >= 0 and p + half <= len(filt)])
    if len(beats) >= 2:
        tpl = beats.mean(0)
        m["tcorr"] = float(np.mean([np.corrcoef(b, tpl)[0, 1] for b in beats]))
    m["good"] = bool(m["flat_frac"] <= C["max_flat_frac"] and lo_bpm <= m["hr"] <= hi_bpm
                     and m["tcorr"] >= C["min_template_corr"])
    return m


def score_draw(args):
    draw_id, file = args
    z = np.load(S3 / "windows" / file)
    raw = z["pleth"].astype(np.float32)
    t0_s = float(z["t0_s"])
    filt = bandpass(raw)
    n_ep = len(raw) // EP
    eps = pd.DataFrame([epoch_metrics(raw[i * EP:(i + 1) * EP], filt[i * EP:(i + 1) * EP]) for i in range(n_ep)])
    good = eps.good.to_numpy()

    w = C["window_min"] * 60 // C["epoch_s"]
    score = np.convolve(good, np.ones(w), "valid") / w
    centre_s = t0_s + (np.arange(len(score)) + w / 2) * C["epoch_s"]   # window centre rel. to draw
    best = np.lexsort((np.abs(centre_s), -score))[0]
    win = eps.iloc[best:best + w]
    row = dict(draw_id=draw_id, file=file,
               best_score=round(float(score[best]), 4),
               best_start_s=t0_s + best * C["epoch_s"],          # relative to draw time
               best_hr=round(float(win.hr.median()), 1),
               best_tcorr=round(float(win.tcorr.median()), 3),
               best_skew=round(float(win.skewness.median()), 3),
               best_ibi_cv=round(float(win.ibi_cv.median()), 3),
               valid_epoch_frac=round(float(eps.valid.mean()), 4),
               good_epoch_frac=round(float(good.mean()), 4))
    return row, good


def plot_review(scores, n, seed=0):
    """Random draws across the score range, best window's first 10 s, for manual labelling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = scores.sample(min(n, len(scores)), random_state=seed).sort_values("best_score")
    s.assign(manual_label="").to_csv(S4 / "review_sample.csv", index=False)
    fig, axes = plt.subplots(len(s), 1, figsize=(10, 1.3 * len(s)), squeeze=False)
    for ax, r in zip(axes[:, 0], s.itertuples()):
        z = np.load(S3 / "windows" / r.file)
        a = int((r.best_start_s - float(z["t0_s"])) * FS)
        ax.plot(np.arange(10 * FS) / FS, z["pleth"][a:a + 10 * FS].astype(float), lw=0.8)
        ax.set_ylabel(f"{r.draw_id}\nscore {r.best_score:.2f}", fontsize=6, rotation=0, ha="right")
        ax.set_yticks([])
    axes[-1, 0].set_xlabel("s (first 10 s of best window)")
    plt.tight_layout()
    plt.savefig(S4 / "review_sample.png", dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--plot", type=int, default=0, help="save a random review sample of N draws")
    args = ap.parse_args()
    S4.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(S3 / "s3_index.csv")
    idx = idx[idx.coverage > 0].head(args.limit)
    with ProcessPoolExecutor(C["workers"]) as ex:
        results = list(ex.map(score_draw, zip(idx.draw_id, idx.file), chunksize=8))
    scores = pd.DataFrame([r for r, _ in results])
    scores.to_csv(S4 / "sqi_scores.csv", index=False)
    np.savez_compressed(S4 / "epoch_good.npz", **{r["draw_id"]: g for r, g in results})
    if args.plot:
        plot_review(scores, args.plot)

    thr = CFG["s5_select"]["min_window_score"]
    write_manifest(S4, "s4_sqi", C, dict(
        draws_scored=len(scores), subjects_scored=int(idx.subject_id.nunique()),
        median_best_score=float(scores.best_score.median()),
        draws_best_score_ge_thr=int((scores.best_score >= thr).sum()), thr_used_for_count=thr,
        score_quantiles={q: float(scores.best_score.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}))


if __name__ == "__main__":
    main()
