"""PaPaGei-S embeddings (Pillai et al., ICLR 2025; weights: zenodo.org/records/13983110) per PPG record.

The ResNet1D backbone is re-implemented here (same module names, so the checkpoint loads), and the
checkpoint is read with torch.load(weights_only=True): no third-party code or pickle is executed.
The two MoE regression heads are not needed for embeddings and are skipped.
This is a modified version, for inference only, of PaPaGei's models/resnet.py, itself adapted from Shenda Hong's resnet1d (Apache-2.0): see THIRD_PARTY_NOTICES.md.

Runs in .venv-torch (torch needs numpy<2; the pipeline venv stays on numpy 2).
Preprocessing follows the PaPaGei README: Chebyshev-II band-pass 0.5-12 Hz (order 4) + 50 ms moving
average (pyPPG defaults) over the whole signal, then 10 s segments at 125 Hz, z-score per segment. NaN gaps cannot be
filtered across, so each finite run is filtered on its own; segments touching a gap are skipped. A record's embedding is
the mean over its segments (512-d), from the best 5 min window (MIMIC: S4; VitalDB: `best_epoch` of features/beat.py).

Usage:
    python -m ppglabbench.features.papagei mimic pre              # primary: best window before the draw
    python -m ppglabbench.features.papagei mimic centered         # sensitivity: S4's best window, group B
    python -m ppglabbench.features.papagei vitaldb               # after ppglabbench.features.beat vitaldb
Output: data/benchmark/features/<window>/mimic/papagei.npz, data/benchmark/features/vitaldb/papagei.npz (ids, emb [n, 512], n_seg)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import cheby2, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "data" / "external" / "papagei" / "weights" / "papagei_s.pt"
OUT = ROOT / "data" / "experiments" / "papagei"
FS, SEG = 125, 1250
SOS = cheby2(4, 20, [0.5, 12], btype="bandpass", fs=FS, output="sos")


def pad_same(x, k, s):
    n = x.shape[-1]
    p = max(0, ((n + s - 1) // s - 1) * s + k - n)
    return F.pad(x, (p // 2, p - p // 2))


class ConvSame(nn.Module):
    def __init__(self, cin, cout, k, s):
        super().__init__()
        self.k, self.s = k, s
        self.conv = nn.Conv1d(cin, cout, k, s)

    def forward(self, x):
        return self.conv(pad_same(x, self.k, self.s))


class Block(nn.Module):
    def __init__(self, cin, cout, k, s, downsample, first):
        super().__init__()
        self.cin, self.cout, self.first, self.downsample = cin, cout, first, downsample
        self.s = s if downsample else 1
        self.bn1, self.bn2 = nn.BatchNorm1d(cin), nn.BatchNorm1d(cout)
        self.conv1, self.conv2 = ConvSame(cin, cout, k, self.s), ConvSame(cout, cout, k, 1)

    def forward(self, x):
        out = x if self.first else F.relu(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(self.conv1(out))))   # dropout is identity in eval
        idt = F.max_pool1d(pad_same(x, self.s, 1), self.s) if self.downsample else x
        if self.cout != self.cin:
            c1 = (self.cout - self.cin) // 2
            idt = F.pad(idt.transpose(-1, -2), (c1, self.cout - self.cin - c1)).transpose(-1, -2)
        return out + idt


class PaPaGeiS(nn.Module):
    def __init__(self, base=32, k=3, s=2, n_block=18, n_out=512):
        super().__init__()
        self.first_block_conv = ConvSame(1, base, k, 1)
        self.first_block_bn = nn.BatchNorm1d(base)
        blocks = []
        for i in range(n_block):
            cin = base if i == 0 else int(base * 2 ** ((i - 1) // 4))
            cout = cin * 2 if (i % 4 == 0 and i != 0) else cin
            blocks.append(Block(cin, cout, k, s, downsample=(i % 2 == 1), first=(i == 0)))
        self.basicblock_list = nn.ModuleList(blocks)
        self.final_bn = nn.BatchNorm1d(cout)
        self.dense = nn.Linear(cout, n_out)

    def forward(self, x):
        out = F.relu(self.first_block_bn(self.first_block_conv(x)))
        for b in self.basicblock_list:
            out = b(out)
        return self.dense(F.relu(self.final_bn(out)).mean(-1))


def load_model():
    m = PaPaGeiS()
    sd = {k.removeprefix("module."): v for k, v in torch.load(WEIGHTS, map_location="cpu", weights_only=True).items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, f"backbone weights missing: {missing[:5]}"
    assert all(k.startswith(("expert_layers", "gating_network")) for k in unexpected), unexpected[:5]
    return m.eval()


def segments(x, max_seg=None):
    x = x.astype(np.float64)
    f = np.full(len(x), np.nan)
    edges = np.flatnonzero(np.diff(np.r_[0, np.isfinite(x).astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= SEG:
            f[a:b] = np.convolve(sosfiltfilt(SOS, x[a:b]), np.ones(6) / 6, mode="same")   # 50 ms at 125 Hz
    segs = []
    for i in range(len(x) // SEG):
        s = f[i * SEG:(i + 1) * SEG]
        if not np.isfinite(s).all() or np.ptp(x[i * SEG:(i + 1) * SEG]) == 0:
            continue
        segs.append((s - s.mean()) / (s.std() + 1e-8))
        if max_seg and len(segs) >= max_seg:
            break
    return np.array(segs, dtype=np.float32)


def pre_window(r):
    """The 5 min window before the draw, from the stored window around it (which starts 600 s before the draw)."""
    z = np.load(ROOT / "data" / "s3_windows" / "windows" / r.file)["pleth"]
    a = int((r.pre_start_s + 600) * FS)
    return z[a:a + 300 * FS]


def main():
    src = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model().to(dev)
    out = OUT / f"{src}_emb.npz"
    feat = ROOT / "data" / "benchmark" / "features"
    if src == "mimic":
        tag = sys.argv[2] if len(sys.argv) > 2 else "s5_groupB_partial"
        if tag in ("pre", "centered"):
            out = feat / tag / "mimic" / "papagei.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
        if tag == "pre":
            labels = pd.read_csv(ROOT / "data" / "benchmark" / "mimic_labels.csv", usecols=["draw_id", "file", "pre_start_s", "pre_score"])
            labels = labels[labels.pre_score >= 0.8].reset_index(drop=True)
            ids = labels.draw_id.to_numpy()
            get = lambda i: pre_window(labels.iloc[i])
        else:
            gb = ROOT / "data" / ("s5_groupB" if tag == "centered" else tag)
            ids = pd.read_csv(gb / "groupB_draws.csv").draw_id.to_numpy()
            X = np.load(gb / "groupB_X.npy", mmap_mode="r")
            get = lambda i: np.asarray(X[i])
    else:
        vdb = ROOT / "data" / "external" / "vitaldb"
        out = feat / "vitaldb" / "papagei.npz"
        idx = pd.read_csv(vdb / "index.csv")
        ids = idx[idx.n > 0].caseid.to_numpy()
        first = pd.read_csv(feat / "vitaldb" / "beat_features.csv").set_index("caseid").best_epoch
        get = lambda i: np.load(vdb / "pleth" / f"{ids[i]}.npz")["pleth"][first[ids[i]] * SEG:(first[ids[i]] + 30) * SEG]
    emb, nseg = np.full((len(ids), 512), np.nan, np.float32), np.zeros(len(ids), int)
    with torch.inference_mode():
        for i in range(len(ids)):
            s = segments(get(i), max_seg=60)
            if len(s):
                emb[i], nseg[i] = model(torch.from_numpy(s[:, None, :]).to(dev)).cpu().numpy().mean(0), len(s)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(ids)}", flush=True)
    np.savez_compressed(out, ids=ids, emb=emb, n_seg=nseg)
    print(f"{src}: {len(ids)} records, {(nseg > 0).sum()} embedded, median {np.median(nseg):.0f} segments")


if __name__ == "__main__":
    main()
