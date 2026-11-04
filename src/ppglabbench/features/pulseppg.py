"""Pulse-PPG embeddings (Saha et al., IMWUT 2025; weights: doi.org/10.5281/zenodo.17270930, MIT) per PPG record.

This is the comparator with no pretraining overlap: it was trained on the private MOODS field study, not on MIMIC-III or VitalDB
(PaPaGei was pretrained on both). It is a wrist model at 50 Hz, so a poor result here may reflect the domain gap.
The ResNet1D, a modified version of resnet1d (Shenda Hong, Apache-2.0), is re-implemented here (same module names, so the checkpoint loads) and the checkpoint is read with
torch.load(weights_only=True): no third-party code or pickle is executed.

Preprocessing: Chebyshev-II band-pass 0.5-12 Hz (order 4) and 50 ms moving average, as in PaPaGei, then resampled from 125 to 50 Hz.
Input: the longest finite run in the best 5 min window, up to the 4 min the model was trained on, at least 30 s. Embedding: 512-d.

The embedding depends on the input's length: the network downsamples by 64 in all, and inputs whose length modulo 64 is 33-63 land in an outlier tail. Every
input of the full 240 s has 12,000 samples, residue 32, so those are comparable. `seconds` in the output keeps each input's length, and
baselines.add_pulseppg treats every embedding with `seconds` below 240 as missing, so the probe imputes it. The benchmark therefore uses Pulse-PPG on windows
with 240 s of continuous signal.

Runs in .venv-torch.

Usage:
    python -m ppglabbench.features.pulseppg mimic pre
    python -m ppglabbench.features.pulseppg mimic centered
    python -m ppglabbench.features.pulseppg vitaldb
Output: data/benchmark/features/<window>/mimic/pulseppg.npz, data/benchmark/features/vitaldb/pulseppg.npz (ids, emb [n, 512], seconds)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import cheby2, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "data" / "external" / "pulseppg" / "weights" / "pulseppg" / "experiments" / "out" / "pulseppg" / "checkpoint_best.pkl"
FEAT = ROOT / "data" / "benchmark" / "features"
FS, FS_OUT, EPOCH_S = 125, 50, 10
MAX_S, MIN_S = 240, 30
SOS = cheby2(4, 20, [0.5, 12], btype="bandpass", fs=FS, output="sos")


def pad_same(x, kernel, stride):
    n = x.shape[-1]
    p = max(0, (((n + stride - 1) // stride) - 1) * stride + kernel - n)
    return F.pad(x, (p // 2, p - p // 2))


class Conv(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride):
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.conv = nn.Conv1d(c_in, c_out, kernel, stride)

    def forward(self, x):
        return self.conv(pad_same(x, self.kernel, self.stride))


class Block(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride, downsample, first):
        super().__init__()
        self.stride = stride if downsample else 1
        self.downsample, self.first, self.c_in, self.c_out = downsample, first, c_in, c_out
        self.bn1, self.conv1 = nn.BatchNorm1d(c_in), Conv(c_in, c_out, kernel, self.stride)
        self.bn2, self.conv2 = nn.BatchNorm1d(c_out), Conv(c_out, c_out, kernel, 1)

    def forward(self, x):
        out = x if self.first else F.relu(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(self.conv1(out))))
        short = F.max_pool1d(pad_same(x, self.stride, 1), self.stride) if self.downsample else x
        if self.c_out != self.c_in:
            a = (self.c_out - self.c_in) // 2
            short = F.pad(short.transpose(-1, -2), (a, self.c_out - self.c_in - a)).transpose(-1, -2)
        return out + short


class Net(nn.Module):
    def __init__(self, base=128, kernel=11, stride=2, n_block=12):
        super().__init__()
        self.instnorm = nn.InstanceNorm1d(1)
        self.first_block_conv, self.first_block_bn = Conv(1, base, kernel, 1), nn.BatchNorm1d(base)
        blocks = []
        for i in range(n_block):
            c_in = base if i == 0 else base * 2 ** ((i - 1) // 4)
            c_out = c_in * 2 if i % 4 == 0 and i != 0 else c_in
            blocks.append(Block(c_in, c_out, kernel, stride, i % 2 == 1, i == 0))
        self.basicblock_list = nn.ModuleList(blocks)

    def forward(self, x):
        out = F.relu(self.first_block_bn(self.first_block_conv(self.instnorm(x))))
        for b in self.basicblock_list:
            out = b(out)
        return out.max(-1)[0]


def load_model():
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["net"]
    net = Net()
    net.load_state_dict(state, strict=True)
    return net.eval()


def prepare(x):
    """The longest finite run, filtered and resampled to 50 Hz and z-scored, or None if shorter than MIN_S."""
    x = np.asarray(x, np.float64)
    edges = np.flatnonzero(np.diff(np.r_[0, np.isfinite(x).astype(int), 0]))
    if not len(edges):
        return None
    runs = list(zip(edges[::2], edges[1::2]))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if b - a < MIN_S * FS or np.ptp(x[a:b]) == 0:
        return None
    s = np.convolve(sosfiltfilt(SOS, x[a:b]), np.ones(6) / 6, mode="same")[: MAX_S * FS]
    s = resample_poly(s, 2, 5)
    return ((s - s.mean()) / (s.std() + 1e-8)).astype(np.float32)


def main():
    src = sys.argv[1]
    window = sys.argv[2] if src == "mimic" else None
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model().to(dev)
    if window == "pre":
        labels = pd.read_csv(ROOT / "data" / "benchmark" / "mimic_labels.csv", usecols=["draw_id", "file", "pre_start_s", "pre_score"])
        labels = labels[labels.pre_score >= 0.8].reset_index(drop=True)
        ids = labels.draw_id.to_numpy()

        def get(i):
            r = labels.iloc[i]
            a = int((r.pre_start_s + 600) * FS)
            return np.load(ROOT / "data" / "s3_windows" / "windows" / r.file)["pleth"][a:a + 300 * FS]
    elif window == "centered":
        gb = ROOT / "data" / "s5_groupB"
        ids = pd.read_csv(gb / "groupB_draws.csv").draw_id.to_numpy()
        X = np.load(gb / "groupB_X.npy", mmap_mode="r")
        get = lambda i: np.asarray(X[i])
    else:
        vdb = ROOT / "data" / "external" / "vitaldb"
        idx = pd.read_csv(vdb / "index.csv")
        ids = idx[idx.n > 0].caseid.to_numpy()
        first = pd.read_csv(FEAT / "vitaldb" / "beat_features.csv").set_index("caseid").best_epoch
        ep = EPOCH_S * FS
        get = lambda i: np.load(vdb / "pleth" / f"{ids[i]}.npz")["pleth"][first[ids[i]] * ep:(first[ids[i]] + 30) * ep]
    if limit:
        ids = ids[:limit]
    emb, sec = np.full((len(ids), 512), np.nan, np.float32), np.zeros(len(ids))
    batch, where = [], []

    def flush():
        with torch.inference_mode():
            emb[where] = model(torch.from_numpy(np.stack(batch))[:, None, :].to(dev)).cpu().numpy()
        batch.clear()
        where.clear()

    for i in range(len(ids)):
        s = prepare(get(i))
        if s is None:
            continue
        sec[i] = len(s) / FS_OUT
        if batch and (len(s) != len(batch[0]) or len(batch) == 16):
            flush()
        batch.append(s)
        where.append(i)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(ids)}", flush=True)
    if batch:
        flush()
    out = FEAT / window / src if window else FEAT / src
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / ("pulseppg_sample.npz" if limit else "pulseppg.npz"), ids=ids, emb=emb, seconds=sec)
    print(f"{src}: {len(ids)} records, {(sec > 0).sum()} embedded, median {np.median(sec):.0f} s")


if __name__ == "__main__":
    main()
