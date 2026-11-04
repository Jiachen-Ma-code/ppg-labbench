"""Benchmark features: beat-level PPG morphology per window (MIMIC) or per case (VitalDB).

Beats are taken from the good epochs of the best 5 min window (S4; the window PaPaGei and MAP use), normalised to [0, 1] over NB points and kept if they
correlate with the record's median beat. Per beat: inter-beat interval, systolic time, systolic-diastolic
dT (stiffness), reflection index, notch visibility, APG b/a, width at 50%, rise slope. A record is
summarised by the median and IQR of each, plus HR, SDNN and RMSSD (from the kept beats).

The diastolic point behind dT and the reflection index is looked for after the systolic peak and before index SEARCH_END of the NB points. It is the top
of the rise after a notch, if that rise is at least MIN_NOTCH of the beat's amplitude; a beat without such a notch has its diastolic point at the first
local maximum of the first derivative after the peak, the inflection of the falling limb; a beat with neither has no dT or reflection index.

Usage:
    python -m ppglabbench.features.beat mimic pre        # primary: best window before the draw
    python -m ppglabbench.features.beat mimic centered   # sensitivity: S4's best window around the draw
    python -m ppglabbench.features.beat vitaldb          # all fetched VitalDB cases
Output: data/benchmark/features/<window>/mimic/ or features/vitaldb/ {beat_features.csv, manifest.json}  (not committed)
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ppglabbench.common import CFG, DATA, S3, S4, S5, write_manifest
from ppglabbench.pipeline.s4_score_sqi import EP, bandpass, epoch_metrics, finite_runs

FS = CFG["fs"]
OUT = DATA / "benchmark" / "features"
VDB = DATA / "external" / "vitaldb"
NB = 100   # points per resampled beat
HALF_S = CFG["s3_fetch"]["half_window_min"] * 60
WIN_EPOCHS = CFG["s4_sqi"]["window_min"] * 60 // CFG["s4_sqi"]["epoch_s"]
SEARCH_END = 80   # the last fifth of a beat is run-off to the next foot, so a diastolic point is not looked for there
MIN_NOTCH = 0.01  # smallest rise after a notch, as a share of the beat's amplitude: beats are scaled to [0, 1]
BEAT_FEATS = ["ibi", "t_sys", "t_sys_frac", "dT", "dT_frac", "ri", "notch", "ba", "width50", "slope"]


def first_local_max(s):
    """Index of the first point of s higher than what comes before and after it, or None. A flat stretch counts as one point, because beats
    shorter than NB samples are interpolated linearly and their derivatives rise in steps."""
    d = np.diff(s)
    moves = np.flatnonzero(np.abs(d) > 1e-9)
    turns = moves[:-1][(d[moves[:-1]] > 0) & (d[moves[1:]] < 0)]
    return int(turns[0]) + 1 if len(turns) else None


def diastolic_point(bt, g1, i_sys):
    """(index of the diastolic point or None, notch visible 1.0 or 0.0) of a beat scaled to [0, 1], with first derivative g1 and systolic peak i_sys."""
    start = i_sys + 2
    s = g1[start:SEARCH_END]
    for j in np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0)):     # the slope turns upwards: a candidate notch
        top = np.flatnonzero((s[j:-1] > 0) & (s[j + 1:] <= 0))
        if not len(top):                                      # still rising where the search stops
            break
        i_dia = start + j + top[0]
        if bt[i_dia] - bt[start + j:i_dia + 1].min() >= MIN_NOTCH:
            return i_dia, 1.0
    k = first_local_max(s)
    return (None if k is None else start + k), 0.0


def beat_table(x, good):
    """Beat-level morphology for all beats inside good epochs."""
    filt = bandpass(x.astype(np.float32))
    mask = np.zeros(len(x), bool)
    for i in np.flatnonzero(good):
        mask[i * EP:(i + 1) * EP] = True
    mask &= np.isfinite(filt)
    rows, beats = [], []
    for a, b in finite_runs(mask):
        seg = filt[a:b]
        if len(seg) < 3 * FS:
            continue
        p5, p95 = np.percentile(seg, [5, 95])
        pk, _ = find_peaks(seg, distance=int(FS * 0.3), prominence=0.3 * (p95 - p5))
        feet = [p0 + np.argmin(seg[p0:p1]) for p0, p1 in zip(pk[:-1], pk[1:])]
        for f0, f1 in zip(feet[:-1], feet[1:]):
            if 0.3 * FS <= f1 - f0 <= 2.0 * FS:
                bt = seg[f0:f1 + 1]
                beats.append(np.interp(np.linspace(0, len(bt) - 1, NB), np.arange(len(bt)), bt))
                rows.append(((f1 - f0) / FS, a + f0, a + f1))
    if len(beats) < 20:
        return None
    B = np.array(beats)
    B = (B - B.min(1, keepdims=True)) / (np.ptp(B, 1, keepdims=True) + 1e-9)
    ibi, t0, t1 = np.array(rows).T
    tpl = np.median(B, 0)
    keep = np.array([np.corrcoef(b, tpl)[0, 1] for b in B]) >= CFG["s4_sqi"]["min_template_corr"]
    B, ibi, t0, t1 = B[keep], ibi[keep], t0[keep], t1[keep]
    if len(B) < 20:
        return None
    d1 = np.gradient(B, axis=1)
    d2 = np.gradient(d1, axis=1)
    out = []
    for bt, g1, g2, T, s0, s1 in zip(B, d1, d2, ibi, t0, t1):
        i_sys = int(np.argmax(bt[:70]))
        if i_sys < 3:
            continue
        i_dia, notch = diastolic_point(bt, g1, i_sys)
        gap, ri = (np.nan, np.nan) if i_dia is None else ((i_dia - i_sys) / NB, bt[i_dia])
        ia = int(np.argmax(g2[:i_sys + 1]))
        b_val = g2[ia:i_sys + 1].min()
        out.append(dict(t0=s0, t1=s1, ibi=T, t_sys=i_sys / NB * T, t_sys_frac=i_sys / NB, dT=gap * T, dT_frac=gap, ri=ri, notch=notch,
                        ba=b_val / (g2[ia] + 1e-9), width50=(bt >= 0.5).mean(), slope=g1[:i_sys + 1].max() * NB / T))
    return pd.DataFrame(out) if len(out) >= 20 else None


def rmssd(bt):
    """Root mean square of successive differences, only between beats that really are adjacent."""
    adjacent = bt.t0.to_numpy()[1:] == bt.t1.to_numpy()[:-1]
    d = np.diff(bt.ibi.to_numpy())[adjacent]
    return np.sqrt(np.mean(d ** 2)) if len(d) else np.nan


def best_epochs(good, start):
    """The good epochs inside the best window, which starts at epoch `start`."""
    keep = np.zeros(len(good), bool)
    keep[start:start + WIN_EPOCHS] = True
    return good & keep


def draw_features(args):
    """args: draw_id, file, good epochs and optionally the epoch to stop at: filtering must not read samples after the window."""
    draw_id, file, good, *stop = args
    z = np.load(S3 / "windows" / file)
    x = z["pleth"]
    if stop:
        x, good = x[: stop[0] * EP], good[: stop[0]]
    bt = beat_table(x, good)
    if bt is None:
        return dict(draw_id=draw_id)
    f = dict(draw_id=draw_id, n_beats=len(bt), hr=60 / bt.ibi.median(), sdnn=bt.ibi.std(), rmssd=rmssd(bt))
    for c in BEAT_FEATS:
        q25, q50, q75 = bt[c].quantile([0.25, 0.5, 0.75])
        f[f"{c}_med"], f[f"{c}_iqr"] = q50, q75 - q25
    return f


def case_features(caseid):
    """VitalDB: S4 epoch rules are applied here, since VitalDB records do not go through S4."""
    x = np.load(VDB / "pleth" / f"{caseid}.npz")["pleth"].astype(np.float32)
    filt = bandpass(x)
    good = np.array([epoch_metrics(x[i * EP:(i + 1) * EP], filt[i * EP:(i + 1) * EP])["good"]
                     for i in range(len(x) // EP)], dtype=bool)
    f = dict(caseid=caseid, good_frac=good.mean() if len(good) else 0.0, n_good=int(good.sum()), best_epoch=0)
    if len(good) > WIN_EPOCHS:    # the same rule as S4: the contiguous 5 min with the most good epochs
        f["best_epoch"] = int(np.argmax(np.convolve(good.astype(int), np.ones(WIN_EPOCHS, int), "valid")))
    good = best_epochs(good, f["best_epoch"])
    bt = beat_table(x, good) if good.sum() >= 6 else None
    if bt is not None:
        f.update(n_beats=len(bt), hr=60 / bt.ibi.median(), sdnn=bt.ibi.std(), rmssd=rmssd(bt))
        for c in BEAT_FEATS:
            q25, q50, q75 = bt[c].quantile([0.25, 0.5, 0.75])
            f[f"{c}_med"], f[f"{c}_iqr"] = q50, q75 - q25
    return f


def main():
    src = sys.argv[1]
    window = sys.argv[2] if src == "mimic" else None
    out = OUT / window / src if window else OUT / src
    out.mkdir(parents=True, exist_ok=True)
    workers = CFG["s4_sqi"]["workers"]
    with ProcessPoolExecutor(workers) as ex:
        if src == "mimic":
            gb = pd.read_csv(S5 / "groupB_draws.csv")
            eg = np.load(S4 / "epoch_good.npz")
            if window == "pre":
                pre = pd.read_csv(DATA / "benchmark" / "mimic_labels.csv", usecols=["draw_id", "pre_start_s", "pre_score"])
                gb = gb.merge(pre[pre.pre_score >= CFG["s5_select"]["min_window_score"]], on="draw_id")
            col = "pre_start_s" if window == "pre" else "best_start_s"
            first = lambda r: int((getattr(r, col) + HALF_S) // CFG["s4_sqi"]["epoch_s"])
            stop = (lambda r: (first(r) + WIN_EPOCHS,)) if window == "pre" else (lambda r: ())
            f = pd.DataFrame(list(ex.map(draw_features, ((r.draw_id, r.file, best_epochs(eg[r.draw_id], first(r)), *stop(r))
                                                         for r in gb.itertuples()), chunksize=8)))
        else:
            ids = pd.read_csv(VDB / "index.csv").query("n > 0").caseid
            f = pd.DataFrame(list(ex.map(case_features, ids, chunksize=4)))
    f.to_csv(out / "beat_features.csv", index=False)
    n_ok = int(f.n_beats.notna().sum())
    write_manifest(out, f"beat_features_{src}", dict(nb=NB, feats=BEAT_FEATS, search_end=SEARCH_END, min_notch=MIN_NOTCH), dict(records=len(f), with_features=n_ok))
    print(f"{src}: {len(f)} records, {n_ok} with ≥ 20 beats")


if __name__ == "__main__":
    main()
