"""End-to-end CNN on PPG windows: a small 1D ResNet trained on the train split, for the MIMIC `pre` window.

The other PPG models of the benchmark are hand features and pretrained encoders with a probe. This one is trained for the task, so a gain it finds is one those
may miss. It predicts every binary lab target at once, plus age and heart rate as regression heads.

Input   : the 5 min `pre` window, 125 to 50 Hz by resample_poly, z-scored per window over its valid samples, clipped to +- CLIP_SD. Gaps are bridged for the
          filter only and then set to 0. A window with no valid samples or no variance has no score. Training reads random 30 s crops; a score is the mean logit
          over 10 evenly spaced 30 s crops.
Labels  : the binary targets from baselines.task_frame, train rows only. task_frame needs eval rows for its own size check, so validation rows are loaded and
          dropped. Age, and HR from the beat features. Missing labels are masked per head.
Fits    : 5 folds on the train split, by subject, for out-of-fold scores, and one final model on all of train, which scores validation and test. Folds are
          GroupKFold(K) on the cohort's train rows, the call baselines.ppg_score makes. ppg_score makes it per lab, on that lab's rows, so its folds differ from
          lab to lab and from these; a model with one trunk needs a single assignment. Every score is still from a model that never saw the subject. Train windows outside the
          cohort are never trained on, and take their subject's fold, or the final model if the subject has none.
Stopping: a train-only rule. 10% of a fit's training subjects form its slice, held out of its gradient steps. The checkpoint kept is the epoch with the highest mean
          AUROC over the heads on that slice, averaged over 5 epochs. No validation or test data enters any choice.
Test    : the final model scores test windows as the embeddings are computed for every window, so test waveforms are read. Test labels are not: their lines of the
          label table are skipped as it is parsed, and the window list reads metadata columns only. Test rows are in no training set and no slice, and no aggregate
          of a test score is computed.
Repeats : one seed per fit, recorded in the manifest. MPS is not deterministic, so a re-run gives different numbers: cnn_scores.npz, identified by the SHA-256 in
          the manifest, is what fixes the results.

Environments: prepare and assemble read the benchmark's tables and write its manifest, so they run in .venv. Fitting runs in .venv-torch and needs numpy, scipy and torch.

Usage:
    .venv/bin/python -m ppglabbench.features.cnn prepare [--workers 4]
    .venv-torch/bin/python -m ppglabbench.features.cnn [--fits fold0,fold1,fold2,fold3,fold4,final] [--device auto|mps|cpu] [--threads 4]
    .venv/bin/python -m ppglabbench.features.cnn assemble
A finished fit is skipped, so a killed run loses only the fit it was in. Fits can run at once in separate processes, with different --fits.

Output: data/benchmark/features/pre/mimic_cnn/
    cnn_scores.npz  draw_id (N), targets (T), score (N x T raw logits, not centred per fold: train rows out-of-fold, other rows from the final model), fold (N, the fit
                    that scored a train row; -1 for other rows), age_pred, hr_pred (N, years and beats/min, same rule). NaN where a window has no score.
    manifest.json   settings, seeds, fold sizes, stopping epochs, SHA-256 of cnn_scores.npz and of each checkpoint
    ckpt_<fit>.pt, fit_<fit>.json   the model of each fit, and its per-epoch aggregates. work/ holds the prepared arrays and can be deleted afterwards.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OUT = DATA / "benchmark" / "features" / "pre" / "mimic_cnn"
WORK = OUT / "work"

FS, FS_OUT, WINDOW_S, CROP_S, N_CROPS = 125, 50, 300, 30, 10
N_IN, N_OUT, CROP = WINDOW_S * FS, WINDOW_S * FS_OUT, CROP_S * FS_OUT
UP, DOWN = 2, 5                     # 125 -> 50 Hz
CLIP_SD = 6
MIN_WINDOW_SCORE = 0.8              # s5_select.min_window_score: the windows every embedding is computed for
WIDTHS, STRIDES, KERNEL, STEM_KERNEL, STEM_STRIDE, DROPOUT = (16, 32, 48, 64), (1, 2, 2, 2), 7, 15, 2, 0.2
EPOCHS, BATCH, LR, WEIGHT_DECAY, REG_WEIGHT = 40, 32, 1e-3, 1e-4, 0.2
SLICE_FRAC, SMOOTH, MIN_EACH = 0.10, 5, 5     # slice share of subjects; epochs averaged; a head needs this many positives and negatives on the slice to count
K = 5                               # baselines.K
SEED = 20261103
FINAL = "final"
FITS = tuple(f"fold{k}" for k in range(K)) + (FINAL,)
SEEDS = {name: SEED + i for i, name in enumerate(FITS)}
TRAIN, VAL, TEST = 0, 1, 2
SPLITS = {"train": TRAIN, "val": VAL, "test": TEST}
SETTINGS = dict(fs=FS, fs_out=FS_OUT, window_s=WINDOW_S, crop_s=CROP_S, n_crops=N_CROPS, clip_sd=CLIP_SD, min_window_score=MIN_WINDOW_SCORE, widths=WIDTHS, strides=STRIDES,
                kernel=KERNEL, stem_kernel=STEM_KERNEL, stem_stride=STEM_STRIDE, dropout=DROPOUT, epochs=EPOCHS, batch=BATCH, optimizer="AdamW", lr=LR, weight_decay=WEIGHT_DECAY,
                schedule="cosine over all epochs", reg_weight=REG_WEIGHT, slice_frac=SLICE_FRAC, smooth_epochs=SMOOTH, min_each=MIN_EACH, k=K)


def preprocess(w):
    """A float32 window at FS_OUT, z-scored and clipped, or None if it has no valid samples, no variance or the wrong length."""
    from scipy.signal import resample_poly
    w = np.asarray(w, np.float64)
    if len(w) != N_IN:
        return None
    ok = np.isfinite(w)
    if not ok.any() or np.ptp(w[ok]) == 0:
        return None
    if not ok.all():    # bridged for the filter only
        w = np.interp(np.arange(N_IN), np.flatnonzero(ok), w[ok])
    z = resample_poly(w, UP, DOWN, padtype="line")
    valid = np.repeat(ok.reshape(-1, DOWN).any(1), UP)     # DOWN source samples give UP output samples: missing if none of them is valid
    z = (z - z[valid].mean()) / (z[valid].std() + 1e-8)
    z[~valid] = 0.0
    return np.clip(z, -CLIP_SD, CLIP_SD).astype(np.float32)


def row_folds(split, subject, cohort):
    """Fold per row: GroupKFold(K) over the cohort's train rows. A train row outside the cohort takes its subject's fold, or -1 if the subject has none. Other rows: -1."""
    from sklearn.model_selection import GroupKFold
    fold = np.full(len(split), -1, np.int8)
    tr = np.flatnonzero((split == TRAIN) & cohort)
    for k, (_, b) in enumerate(GroupKFold(K).split(tr, groups=subject[tr])):
        fold[tr[b]] = k
    known = dict(zip(subject[tr].tolist(), fold[tr].tolist()))
    other = np.flatnonzero((split == TRAIN) & ~cohort)
    fold[other] = [known.get(s, -1) for s in subject[other].tolist()]
    return fold


def fit_rows(name, split, fold, cohort):
    """The rows a fit trains on and the rows it scores."""
    train = (split == TRAIN) & cohort
    if name == FINAL:
        return np.flatnonzero(train), np.flatnonzero((split != TRAIN) | (fold < 0))
    k = FITS.index(name)
    return np.flatnonzero(train & (fold != k)), np.flatnonzero((split == TRAIN) & (fold == k))


def take_slice(train_ix, subject, seed):
    """Split a fit's training rows by subject: (rows for gradient steps, slice rows)."""
    subs = np.unique(subject[train_ix])
    picked = np.random.default_rng(seed).permutation(subs)[:max(1, int(round(SLICE_FRAC * len(subs))))]
    m = np.isin(subject[train_ix], picked)
    return train_ix[~m], train_ix[m]


def stopping_epoch(slice_auc):
    """The epoch with the highest slice AUROC, averaged over SMOOTH epochs centred on it."""
    v = np.asarray(slice_auc, float)
    return int(np.convolve(np.pad(v, SMOOTH // 2, mode="edge"), np.ones(SMOOTH) / SMOOTH, mode="valid").argmax())


def mean_auc(t, o):
    """Mean AUROC over the heads with at least MIN_EACH positives and negatives, and how many heads that is. t, o: rows x heads, NaN in t where a label is missing."""
    from scipy.stats import rankdata
    v = []
    for k in range(t.shape[1]):
        m = np.isfinite(t[:, k])
        y = t[m, k]
        n1 = int(y.sum())
        n0 = len(y) - n1
        if n1 >= MIN_EACH and n0 >= MIN_EACH:
            v.append((rankdata(o[m, k])[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return (float(np.mean(v)) if v else float("nan")), len(v)


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def make_net(n_out):
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, c_in, c, stride):
            super().__init__()
            self.c1, self.b1 = nn.Conv1d(c_in, c, KERNEL, stride, KERNEL // 2, bias=False), nn.BatchNorm1d(c)
            self.c2, self.b2 = nn.Conv1d(c, c, KERNEL, 1, KERNEL // 2, bias=False), nn.BatchNorm1d(c)
            self.short = nn.Identity() if c_in == c and stride == 1 else nn.Sequential(nn.Conv1d(c_in, c, 1, stride, bias=False), nn.BatchNorm1d(c))

        def forward(self, x):
            return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.short(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(1, WIDTHS[0], STEM_KERNEL, STEM_STRIDE, STEM_KERNEL // 2, bias=False), nn.BatchNorm1d(WIDTHS[0]), nn.ReLU())
            layers, c_in = [], WIDTHS[0]
            for c, st in zip(WIDTHS, STRIDES):
                layers += [Block(c_in, c, st), Block(c, c, 1)]
                c_in = c
            self.body, self.drop, self.head = nn.Sequential(*layers), nn.Dropout(DROPOUT), nn.Linear(c_in, n_out)

        def forward(self, x):
            return self.head(self.drop(self.body(self.stem(x)).mean(-1)))

    return Net()


def predict(net, X, rows, device, batch=128):
    """Mean over the evenly spaced crops, for rows of X, an array or memory map of windows: one row of outputs per window."""
    import torch
    net.eval()
    starts = np.linspace(0, N_OUT - CROP, N_CROPS).astype(int)
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            w = torch.from_numpy(np.array(X[rows[i:i + batch]]))
            total = 0
            for st in starts:
                total = total + net(w[:, st:st + CROP][:, None].to(device)).cpu()
            out.append(total / N_CROPS)
    return torch.cat(out).numpy() if out else np.empty((0, 0))


def train_model(X, T, grad, sl, seed, device, epochs=EPOCHS, log=print):
    """Train on rows `grad` of X, an array of windows, with labels T (rows x [targets + age + HR], NaN where missing), and keep the epoch stopping_epoch picks on rows `sl`.
    Returns the net, the mean and SD of the regression targets, the epoch kept and the per-epoch aggregates."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    dev = torch.device(device)
    n_t = T.shape[1] - 2
    assert np.isfinite(T[np.r_[grad, sl], n_t:]).all(), "age and HR must be present on every training row"
    mu, sd = T[grad, n_t:].mean(0), T[grad, n_t:].std(0)
    Xg = torch.from_numpy(np.array(X[grad]))
    Tg, Ts = torch.from_numpy(T[grad]).to(dev), torch.from_numpy(T[sl]).to(dev)
    mu_t, sd_t = torch.tensor(mu, device=dev), torch.tensor(sd, device=dev)
    net = make_net(n_t + 2).to(dev)
    with torch.no_grad():    # the heads start at each target's prevalence
        prev = np.nanmean(T[grad, :n_t], 0).clip(1e-3, 1 - 1e-3)
        net.head.bias[:n_t] = torch.tensor(np.log(prev / (1 - prev)), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def losses(out, t):
        y, m = t[:, :n_t], torch.isfinite(t[:, :n_t])
        b = F.binary_cross_entropy_with_logits(out[:, :n_t], torch.nan_to_num(y), reduction="none") * m
        per = b.sum(0) / m.sum(0).clamp(min=1)
        return per[m.any(0)].mean(), ((out[:, n_t:] - (t[:, n_t:] - mu_t) / sd_t) ** 2).mean(0)

    offs, records, states = torch.arange(CROP), [], []
    for ep in range(epochs):
        t1 = time.time()
        net.train()
        perm = torch.randperm(len(Xg))
        tot, nb = 0.0, 0
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            b = perm[i:i + BATCH]
            st = torch.randint(0, N_OUT - CROP + 1, (len(b),))
            bce, reg = losses(net(Xg[b[:, None], st[:, None] + offs][:, None].to(dev)), Tg[b.to(dev)])
            opt.zero_grad(set_to_none=True)
            (bce + REG_WEIGHT * reg.sum()).backward()
            opt.step()
            tot += float(bce.detach())
            nb += 1
        sched.step()
        t2 = time.time()
        o = predict(net, X, sl, dev)
        ot = T[sl]
        with torch.no_grad():
            bce_sl = float(losses(torch.from_numpy(o).to(dev), Ts)[0])
        auc, heads = mean_auc(ot[:, :n_t], o[:, :n_t])
        rec = dict(epoch=ep, train_bce=tot / max(nb, 1), sl_auc=auc, sl_heads=heads, sl_bce=bce_sl, sl_r2_age=r2(ot[:, n_t], o[:, n_t] * sd[0] + mu[0]),
                   sl_r2_hr=r2(ot[:, n_t + 1], o[:, n_t + 1] * sd[1] + mu[1]), sec_train=t2 - t1, sec_slice=time.time() - t2)
        records.append(rec)
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        log(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()}))
    stop = stopping_epoch([r["sl_auc"] for r in records])
    net.load_state_dict(states[stop])
    return net, mu, sd, stop, records


def load_prepared():
    if not (WORK / "prepared.npz").exists():
        raise SystemExit(f"{WORK / 'prepared.npz'} missing: run .venv/bin/python -m ppglabbench.features.cnn prepare")
    with np.load(WORK / "prepared.npz") as z:
        return {k: z[k] for k in z.files}


def check_fit(name, p, train, grad, sl, scored):
    """What must hold for every fit, on the real rows."""
    split, subject, T = p["split"], p["subject"], p["T"]
    for rows in (train, grad, sl):
        assert (split[rows] == TRAIN).all(), "a fit may train only on the train split"
        assert p["cohort"][rows].all(), "a fit may train only on the cohort"
    assert not np.isin(subject[grad], subject[sl]).any(), "a subject is in both the gradient rows and the slice"
    assert not np.isin(subject[scored], subject[train]).any(), "a scored window's subject is in the fit's training rows"
    assert np.isnan(T[split != TRAIN]).all(), "a label outside the train split"


def run_fit(name, device, threads):
    import torch
    torch.set_num_threads(threads)
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    done = OUT / f"fit_{name}.json"
    if done.exists():
        print(f"{name}: already finished", flush=True)
        return
    p = load_prepared()
    train, scored = fit_rows(name, p["split"], p["fold"], p["cohort"])
    grad, sl = take_slice(train, p["subject"], SEEDS[name])
    check_fit(name, p, train, grad, sl, scored)
    scored = scored[p["usable"][scored]]
    # Validation and test windows are a separate file, which only the final model opens, after its training
    X_train = np.load(WORK / "windows_train.npy", mmap_mode="r")
    fx, order = p["file_ix"], np.r_[grad, sl]
    X = np.array(X_train[fx[order]])
    t0 = time.time()
    net, mu, sd, stop, records = train_model(X, p["T"][order], np.arange(len(grad)), np.arange(len(grad), len(order)), SEEDS[name], device, log=lambda s: print(name, s, flush=True))
    t_train = time.time() - t0
    del X
    t1 = time.time()
    n_t = len(p["targets"])
    S = np.full((len(scored), n_t + 2), np.nan, np.float32)
    in_train = p["split"][scored] == TRAIN
    if in_train.any():
        S[in_train] = predict(net, X_train, fx[scored[in_train]], device)
    if not in_train.all():
        assert name == FINAL, "a fold's fit scores train rows only"
        S[~in_train] = predict(net, np.load(WORK / "windows_other.npy", mmap_mode="r"), fx[scored[~in_train]], device)
    S[:, n_t:] = S[:, n_t:] * sd + mu
    t_infer = time.time() - t1
    ck = OUT / f"ckpt_{name}.pt"
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, mu=mu.tolist(), sd=sd.tolist(), seed=SEEDS[name], stopping_epoch=stop), str(ck) + ".tmp")
    os.replace(str(ck) + ".tmp", ck)
    np.save(WORK / f"scored_{name}.npy", scored)
    np.save(WORK / f"scores_{name}.npy", S)
    info = dict(name=name, seed=SEEDS[name], device=device, threads=threads, n_train_rows=len(grad), n_slice_rows=len(sl), n_train_subjects=len(np.unique(p["subject"][grad])),
                n_slice_subjects=len(np.unique(p["subject"][sl])), n_scored_rows=len(scored), stopping_epoch=stop, slice_auc_smoothed_at_stop=float(np.convolve(np.pad([r["sl_auc"] for r in records], SMOOTH // 2, mode="edge"), np.ones(SMOOTH) / SMOOTH, mode="valid")[stop]),
                sec_train_total=t_train, sec_infer=t_infer, epochs=records)
    (OUT / f"fit_{name}.json.tmp").write_text(json.dumps(info, indent=1))
    os.replace(OUT / f"fit_{name}.json.tmp", done)
    print(f"{name}: stopped at epoch {stop}; {t_train:.0f} s training, {t_infer:.0f} s scoring", flush=True)


def _window(path, start):
    a = int((start + 600) * FS)
    return preprocess(np.load(path)["pleth"][a:a + N_IN])


def load_without_test():
    """baselines.load_mimic with the test lines of the label table skipped as it is parsed, so no test label is ever in memory."""
    import pandas as pd

    from ppglabbench import baselines as B
    path = DATA / "benchmark" / "mimic_labels.csv"
    test = set((np.flatnonzero(pd.read_csv(path, usecols=["split"]).split == "test") + 1).tolist())    # line 0 is the header
    read = pd.read_csv

    def skipping(f, *a, **k):
        return read(f, *a, **({**k, "skiprows": lambda i: i in test} if Path(str(f)).name == path.name else k))
    pd.read_csv = skipping
    try:
        d, bcols, labs, tiers = B.load_mimic("pre")
    finally:
        pd.read_csv = read
    assert set(d.split.unique()) <= {"train", "val"}
    return d, labs, tiers


def prepare(workers):
    import pandas as pd
    from joblib import Parallel, delayed
    from sklearn.metrics import adjusted_rand_score
    from sklearn.model_selection import GroupKFold

    from ppglabbench import baselines as B
    from ppglabbench.common import git_version
    assert K == B.K, "folds differ from the benchmark's"
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = pd.read_csv(DATA / "benchmark" / "mimic_labels.csv", usecols=["draw_id", "subject_id", "file", "pre_start_s", "pre_score", "split"])
    meta = meta[meta.pre_score >= MIN_WINDOW_SCORE].reset_index(drop=True)
    assert meta.draw_id.is_unique
    d, labs, tiers = load_without_test()
    split = meta.split.map(SPLITS).to_numpy(np.int8)
    subject_id = meta.subject_id.to_numpy()
    n, targets = len(meta), list(labs)
    row = pd.Series(np.arange(n), index=meta.draw_id)
    tc = d[d.split == "train"]
    cohort = np.zeros(n, bool)
    cohort[row[tc.draw_id].to_numpy()] = True
    assert (split[cohort] == TRAIN).all()
    T = np.full((n, len(targets) + 2), np.nan, np.float32)
    agree = {}
    fold = row_folds(split, subject_id, cohort)
    for k, name in enumerate(targets):
        fr = B.task_frame(d, tiers, name, labs[name], "clf", "val")
        assert fr is not None, f"task_frame skipped {name}"
        q, y, tr, groups = fr[0], fr[1], fr[2], fr[4]
        rows = row[q.draw_id.iloc[tr]].to_numpy()
        T[rows, k] = y[tr]
        lab_fold = np.empty(len(tr), int)     # the assignment ppg_score uses for this lab, against this module's
        for f, (_, b) in enumerate(GroupKFold(K).split(tr, groups=groups[tr])):
            lab_fold[b] = f
        agree[name] = dict(rows=len(tr), same_fold_number=float((lab_fold == fold[rows]).mean()), adjusted_rand=float(adjusted_rand_score(lab_fold, fold[rows])))
    T[row[tc.draw_id].to_numpy(), len(targets)] = tc.age.to_numpy(np.float32)
    T[row[tc.draw_id].to_numpy(), len(targets) + 1] = tc.hr.to_numpy(np.float32)
    assert np.isnan(T[split != TRAIN]).all()
    # The seal, against the split file itself: no test subject has a training row
    s = pd.read_csv(DATA / "benchmark" / "mimic_splits.csv", usecols=["subject_id", "split"])
    assert not (set(s.subject_id[s.split == "test"]) & set(subject_id[cohort]))
    usable = np.zeros(n, bool)
    file_ix = np.zeros(n, np.int64)
    wave = DATA / "s3_windows" / "windows"
    t1 = time.time()
    for fname, rows in (("windows_train.npy", np.flatnonzero(split == TRAIN)), ("windows_other.npy", np.flatnonzero(split != TRAIN))):
        file_ix[rows] = np.arange(len(rows))
        X = np.lib.format.open_memmap(WORK / fname, mode="w+", dtype=np.float32, shape=(len(rows), N_OUT))
        for i in range(0, len(rows), 1024):
            batch = rows[i:i + 1024]
            res = Parallel(n_jobs=workers, batch_size=32)(delayed(_window)(wave / meta.file[j], meta.pre_start_s[j]) for j in batch)
            for j, r in enumerate(res):
                if r is not None:
                    X[i + j], usable[batch[j]] = r, True
        X.flush()
    assert usable[cohort].all(), "a cohort window has no usable signal"
    t_cache = time.time() - t1
    np.savez(WORK / "prepared.npz", draw_id=meta.draw_id.to_numpy(dtype="U"), targets=np.array(targets), split=split, subject=np.unique(subject_id, return_inverse=True)[1].astype(np.int32),
             cohort=cohort, fold=fold, usable=usable, file_ix=file_ix, T=T)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    info = dict(code_version=git_version(), head=head, sha256_baselines=sha256(Path(B.__file__)), sha256_labels=sha256(WORK / "prepared.npz"), sec_cache=t_cache, sec_total=time.time() - t0,
                windows={k: int((split == v).sum()) for k, v in SPLITS.items()}, cohort_train=int(cohort.sum()), train_outside_cohort=int(((split == TRAIN) & ~cohort).sum()),
                usable={k: int(usable[split == v].sum()) for k, v in SPLITS.items()}, fold_rows=[int(((fold == k) & cohort).sum()) for k in range(K)], fold_subjects=[int(len(np.unique(subject_id[(fold == k) & cohort]))) for k in range(K)],
                train_rows_without_fold=int(((split == TRAIN) & (fold < 0)).sum()), targets=targets, labelled_train_rows={t: int(np.isfinite(T[:, i]).sum()) for i, t in enumerate(targets)},
                positive_train_rows={t: int(np.nansum(T[:, i])) for i, t in enumerate(targets)}, per_lab_fold_agreement=agree)
    (OUT / "prepare.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ("labelled_train_rows", "positive_train_rows", "per_lab_fold_agreement", "targets")}, indent=1))
    print("seal: no test subject has a training row; no label outside the train split; validation labels dropped")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def assemble_arrays(p, results):
    """The output arrays from the prepared rows and, per fit, (rows scored, scores: rows x [targets + age + HR]). Every usable row is scored exactly once. Computes no statistic of a score."""
    n_t = len(p["targets"])
    S = np.full((len(p["split"]), n_t + 2), np.nan, np.float32)
    seen = np.zeros(len(p["split"]), int)
    for rows, s in results.values():
        S[rows] = s
        seen[rows] += 1
    assert (seen <= 1).all(), "a window scored twice"
    assert (seen[p["usable"]] == 1).all(), "a usable window has no score"
    return dict(draw_id=p["draw_id"], targets=p["targets"], score=S[:, :n_t], fold=p["fold"], age_pred=S[:, n_t], hr_pred=S[:, n_t + 1])


def assemble():
    from ppglabbench.common import write_manifest
    p = load_prepared()
    missing = [f for f in FITS if not (OUT / f"fit_{f}.json").exists()]
    if missing:
        raise SystemExit(f"fits not finished: {missing}")
    arrays = assemble_arrays(p, {f: (np.load(WORK / f"scored_{f}.npy"), np.load(WORK / f"scores_{f}.npy")) for f in FITS})
    np.savez_compressed(OUT / "cnn_scores.npz", **arrays)
    fits = {f: json.loads((OUT / f"fit_{f}.json").read_text()) for f in FITS}
    prep = json.loads((OUT / "prepare.json").read_text())
    params = dict(settings=SETTINGS, seeds=SEEDS, stopping_rule="highest slice mean AUROC, averaged over 5 epochs; slice is 10% of the fit's training subjects", window="pre",
                  label_code=dict(code_version=prep["code_version"], head=prep["head"], sha256_baselines=prep["sha256_baselines"]), determinism="MPS is not deterministic: this file, not a re-run, fixes the results")
    counts = dict(complete=True, windows=prep["windows"], usable=prep["usable"], cohort_train=prep["cohort_train"], train_outside_cohort=prep["train_outside_cohort"], train_rows_without_fold=prep["train_rows_without_fold"],
                  fold_rows=prep["fold_rows"], fold_subjects=prep["fold_subjects"], targets=prep["targets"], stopping_epochs={f: fits[f]["stopping_epoch"] for f in FITS},
                  devices={f: fits[f]["device"] for f in FITS}, sec_train={f: round(fits[f]["sec_train_total"]) for f in FITS}, sha256_scores=sha256(OUT / "cnn_scores.npz"), sha256_checkpoints={f: sha256(OUT / f"ckpt_{f}.pt") for f in FITS})
    write_manifest(OUT, "cnn_mimic_pre", params, counts)


def main():
    ap = argparse.ArgumentParser(description="End-to-end CNN on the MIMIC pre window")
    ap.add_argument("step", nargs="?", default="fit", choices=["prepare", "fit", "assemble"])
    ap.add_argument("--fits", default=",".join(FITS))
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "prepare":
        prepare(a.workers)
    elif a.step == "assemble":
        assemble()
    else:
        for name in a.fits.split(","):
            assert name in FITS, f"unknown fit {name}: {FITS}"
            run_fit(name, a.device, a.threads)


if __name__ == "__main__":
    main()
