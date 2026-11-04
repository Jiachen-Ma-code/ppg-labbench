"""Operating characteristics of the confirmatory procedure, by plasmode on train and validation rows, and the sample size simulation for TRIPOD+AI. Test rows are gone before anything is computed.

The CNN is left out: its scores are fixed and cannot follow a simulated outcome. The real covariates, PPG features, subjects and label pattern stay; the outcome is simulated. From the real outcome on train, a base (`now` or `full`) gives each window a log-odds. The
simulated outcome adds to it a subject effect, logistic noise and, with strength gamma, a signal built from the real PPG features: a direction in principal components 5-35 of the embedding
plus a nonlinear term in the beat features, so PPG holds the signal without any feature being edited. gamma = 0 is the null: PPG tells nothing about the outcome given that base.

One replicate draws train outcomes, fits the real pipeline (fit_models: score and stack), and freezes it. Eval draws then resample validation subjects to the size of the test split, with fresh
outcomes per position, and compute the statistics of confirmatory.py. Rates are over replicates x draws, and their Monte Carlo errors treat replicates as the clusters.

The upper-bound chain is on the mean over Family A, so it simulates whole families: every lab's pipeline fitted, then draws of test-sized subject resamples shared by the labs, with a subject effect
partly shared by them, through confirmatory.mean_gain. The signal is spread over the labs by a weight per lab, equal or a mix in which a few labs carry the mean.

Tables in --out:
  marginal.csv   per scenario and base: mean true gain, spread between replicates, SE, and rates of z2, z1, z3 at the thresholds of the procedure
  chain.csv      the chain per scenario and base: true mean, its SE from the bootstrap, training SD, chance of passing alone and in sequence
  boundary.csv   the rows of chain.csv with a true mean at the margin: the size of a step (nominal 0.025)
  family.csv     family-wise error of the graphical procedure under the global null, and the chance of each step passing when PPG adds nothing over `full`
  sample_size.csv  smallest gain with 80% power at the first step, by stratum (TRIPOD+AI)
  report.md, manifest.json

Usage:  python -m ppglabbench.operating [--window pre] [--reps 20] [--draws 60] [--jobs 3] [--chain-reps 20] [--chain-draws 30] [--chain-B 500] [--out DIR]
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA

from ppglabbench import baselines as BL
from ppglabbench import confirmatory as C
from ppglabbench.common import write_manifest

SIGMA = dict(acuity=1.0, monitor=1.0, now=1.0, full=0.5)     # subject effect on the log-odds: the base `full` already carries most of what is lasting about a subject
SHARED = 0.5                        # share of a subject effect common to all labs: one patient's labs move together, and the bootstrap of the mean has to carry that
EQUAL, MIX = (1,) * 10, (0, 0, 0, .5, .5, 1, 1, 1.5, 2, 2)     # signal per Family A lab. A gain goes about with its square, so in the mix a few labs carry the mean
R_ASSUMED = dict(acuity=.14, monitor=.14, now=.14, full=.70)   # r-hat when there is no complete plan to read: the largest of an earlier plan's strata, `now`'s for the two it lacks
G_RATIO_DEFAULT = 759 / 823         # test subjects / validation subjects
GAMMAS = dict(clf=(0.0, 0.3, 0.6, 0.9, 1.3, 1.8), reg=(0.0, 0.1, 0.2, 0.3, 0.45))
LEVELS = {"1.96": 1.96, "first_named": C.threshold(2, C.ALPHA_NAMED), "first_a": C.threshold(10, C.ALPHA_A)}
L1_DRAWS = 20                       # the score test refits on every draw, so it runs on fewer of them
_prepared = {}


def signal(q, bcols, tr, lab):
    """A fixed function of the real PPG features for this lab: unit variance on train."""
    world = np.random.default_rng(zlib.crc32(lab.encode()))
    parts = []
    if BL.PG[0] in q:
        x = np.nan_to_num(q[BL.PG].to_numpy(float))
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        pcs = PCA(40, random_state=0).fit((x[tr] - mu) / sd).transform((x - mu) / sd)
        pcs = (pcs - pcs[tr].mean(0)) / pcs[tr].std(0)
        a = world.standard_normal(30)
        parts.append(pcs[:, 5:35] @ (a / np.linalg.norm(a)))
    b = q[bcols].to_numpy(float)
    b = np.where(np.isfinite(b), b, np.nanmedian(b[tr], axis=0))
    b = (b - b[tr].mean(0)) / (b[tr].std(0) + 1e-9)
    j = world.choice(b.shape[1], 3, replace=False)
    parts.append(b[:, j[0]] + .5 * b[:, j[1]] * b[:, j[2]])
    parts = [(p - p[tr].mean()) / p[tr].std() for p in parts]
    s = np.sum(parts, axis=0)
    return (s - s[tr].mean()) / s[tr].std()


def prepare(d, bcols, tiers, spec, lab, task, gen_base):
    """What every replicate of one lab shares. It is cached per worker, since it does not depend on the simulated outcome."""
    key = (lab, task, gen_base)
    if key not in _prepared:
        q, y, tr, ev, groups, bases, _, _ = C.fixed_frame(d, tiers, lab, spec, "val", task=task)
        P = dict(q=q, tr=tr, ev=ev, groups=groups, bases={b: bases[b] for b in C.CHAIN}, code=np.unique(groups, return_inverse=True)[1], prev=float(np.mean(y[tr])) if task == "clf" else None)
        if task == "clf":
            kappa = np.sqrt(1 + .346 * SIGMA[gen_base] ** 2)     # keeps the slope on the log-odds after the subject effect is added
            P["eta"] = kappa * BL.base_clf().fit(bases[gen_base][tr], y[tr]).decision_function(bases[gen_base])
        else:
            P["eta"] = BL.base_reg().fit(bases[gen_base][tr], y[tr]).predict(bases[gen_base])
            res = y[tr] - P["eta"][tr]
            P["sd"] = float(res.std())
            P["icc"] = float(np.clip(_icc(res, groups[tr]), 0.05, 0.95))
        P["sig"] = signal(q, bcols, tr, lab)
        ev_groups = groups[ev]
        P["rows"] = {u: np.flatnonzero(ev_groups == u) for u in np.unique(ev_groups)}
        P["subjects"] = np.array(sorted(P["rows"]))
        P["X"] = {m: q[C.feature_cols(m, bcols)].to_numpy(float) for m in C.VALUE_MODELS if C.feature_cols(m, bcols)[0] in q}    # the CNN's scores are fixed: they cannot follow a simulated outcome
        _prepared[key] = P
    return _prepared[key]


def _icc(x, g):
    u, inv, n = np.unique(g, return_inverse=True, return_counts=True)
    m = np.bincount(inv, x) / n
    k = len(u)
    msb, msw = np.sum(n * (m - x.mean()) ** 2) / (k - 1), np.sum((x - m[inv]) ** 2) / (len(x) - k)
    n0 = (len(x) - np.sum(n ** 2) / len(x)) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def outcome(P, rows, positions, gamma, tau, sigma, rng, task):
    """A fresh simulated outcome for `rows`; positions gives each row's subject position, one subject effect per position."""
    u = rng.standard_normal(int(positions.max()) + 1)[positions]
    if task == "clf":
        return P["eta"][rows] + sigma * u + gamma * P["sig"][rows] + rng.logistic(0, 1, len(rows)) > tau
    return P["eta"][rows] + P["sd"] * (np.sqrt(P["icc"]) * u + np.sqrt(1 - P["icc"]) * rng.standard_normal(len(rows)) + gamma * P["sig"][rows])


def replicate(d, bcols, tiers, spec, lab, task, gen_base, model, gamma, seed, draws, g_ratio, bases=(C.PRIMARY, C.UPPER)):
    """One frozen pipeline and its eval draws. Returns aggregates only. g_ratio: test subjects / validation subjects, so that a lab's draws have about the size it will have on test."""
    P = prepare(d, bcols, tiers, spec, lab, task, gen_base)
    g_test = max(int(round(len(P["subjects"]) * g_ratio)), 2)
    rng = np.random.default_rng([seed, zlib.crc32(f"{lab}|{task}|{gen_base}|{model}|{gamma}".encode())])
    tr, ev, groups = P["tr"], P["ev"], P["groups"]
    n = len(P["q"])
    sigma = SIGMA[gen_base]
    tau = None
    if task == "clf":
        latent = P["eta"][tr] + sigma * rng.standard_normal(P["code"].max() + 1)[P["code"][tr]] + gamma * P["sig"][tr] + rng.logistic(0, 1, len(tr))
        tau = np.quantile(latent, 1 - P["prev"])
    y = np.zeros(n, int if task == "clf" else float)
    y[tr] = outcome(P, tr, P["code"][tr], gamma, tau, sigma, rng, task)
    preds = C.fit_models(y, tr, ev, groups, {b: P["bases"][b] for b in bases}, {model: P["X"][model]}, task=task)
    out = dict(lab=lab, task=task, gen_base=gen_base, model=model, gamma=gamma, seed=seed, rows={})
    D = {b: np.full(draws, np.nan) for b in bases}
    SE = {b: np.full(draws, np.nan) for b in bases}
    Z1 = np.full(L1_DRAWS, np.nan)
    n_design_cols = []
    ev_rows = np.arange(len(ev))
    for m in range(draws):
        ss = rng.choice(P["subjects"], g_test)
        ro = np.concatenate([P["rows"][u] for u in ss])
        gg = np.repeat(np.arange(g_test), [len(P["rows"][u]) for u in ss])
        yy = outcome(P, ev[ro], gg, gamma, tau, sigma, rng, task)
        if task == "clf":
            if yy.sum() < 5 or (~yy).sum() < 5:
                continue
        for b in bases:
            stack, base = preds[f"{b}+{model}"][ro], preds[b][ro]
            res = C.delta_auc(yy, stack, base, gg, g_test) if task == "clf" else C.delta_r2(yy, stack, base, gg, g_test)
            D[b][m], SE[b][m] = res["delta"], res["se"]
        if m < L1_DRAWS:
            Z = C.eval_design(P["bases"][C.PRIMARY][ev][ro])
            score = preds[f"score:{model}"][ro]
            Z1[m] = C.score_test(yy.astype(float), Z, score, gg, g_test)[0] if task == "clf" else C.score_test_linear(yy, Z, score, gg, g_test)[0]
            n_design_cols.append(Z.shape[1] + 1)
    for b in bases:
        out["rows"][b] = dict(D=D[b], SE=SE[b])
    out["z1"] = Z1
    out["n_design"] = int(np.mean(n_design_cols)) if n_design_cols else 0
    out["n_mean"] = float(np.mean([len(P["rows"][u]) for u in P["subjects"]]) * g_test)
    return out


def cluster_rate(by_replicate):
    """Mean of the per-replicate rates and its Monte Carlo error, replicates as the clusters."""
    r = np.array([np.nanmean(x) for x in by_replicate if np.isfinite(x).any()], float)
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan")


def summarise(results):
    """marginal.csv from the replicates of each scenario."""
    keys = sorted({(r["lab"], r["task"], r["gen_base"], r["model"], r["gamma"]) for r in results}, key=str)
    rows = []
    for key in keys:
        reps = [r for r in results if (r["lab"], r["task"], r["gen_base"], r["model"], r["gamma"]) == key]
        lab, task, gen, model, gamma = key
        for b in reps[0]["rows"]:
            D = [r["rows"][b]["D"] for r in reps]
            SE = [r["rows"][b]["SE"] for r in reps]
            truth = np.array([np.nanmean(x) for x in D])
            z2 = [x / np.maximum(s, 1e-12) for x, s in zip(D, SE)]
            row = dict(lab=lab, task=task, gen_base=gen, model=model, gamma=gamma, base=b, replicates=len(reps), draws=len(D[0]), mean_gain=truth.mean(), mc=truth.std(ddof=1) / np.sqrt(len(truth)) if len(truth) > 1 else np.nan,
                       sd_between=truth.std(ddof=1) if len(truth) > 1 else np.nan, se_eval=float(np.nanmean(SE)), n=reps[0]["n_mean"], design_columns=reps[0]["n_design"])
            cover = [np.abs(x - t) <= 1.96 * s for x, s, t in zip(D, SE, truth)]
            row["cover95"], row["cover95_mc"] = cluster_rate(cover)
            for name, c in LEVELS.items():
                row[f"z2>{name}"], row[f"z2>{name}_mc"] = cluster_rate([z > c for z in z2])
                if b == C.PRIMARY:
                    row[f"z1>{name}"], row[f"z1>{name}_mc"] = cluster_rate([r["z1"] > c for r in reps])
            if task == "clf":
                z3 = [(x - C.MIN_GAIN) / np.maximum(s, 1e-12) for x, s in zip(D, SE)]
                row["z3>first_a"], row["z3>first_a_mc"] = cluster_rate([z > LEVELS["first_a"] for z in z3])
            rows.append(row)
    return pd.DataFrame(rows)


def family_world(d, bcols, tiers, labs, family, gen, model, scale, weights, seed, draws, g_ratio, r_by_lab, bases, B):
    """One replicate of the whole family, with models frozen as fitted, and its draws. r_by_lab: base -> lab -> r-hat. With B = 0 only the point means, for tuning. Aggregates only."""
    rng = np.random.default_rng([seed, zlib.crc32(f"{gen}|{model}|{scale:.5f}|{bases}".encode())])
    sigma, fitted = SIGMA[gen], {}
    for lab, w in zip(family, weights):
        P = prepare(d, bcols, tiers, labs[lab], lab, "clf", gen)
        tr, gamma = P["tr"], scale * w
        latent = P["eta"][tr] + sigma * rng.standard_normal(P["code"].max() + 1)[P["code"][tr]] + gamma * P["sig"][tr] + rng.logistic(0, 1, len(tr))
        tau = np.quantile(latent, 1 - P["prev"])
        y = np.zeros(len(P["q"]), int)
        y[tr] = outcome(P, tr, P["code"][tr], gamma, tau, sigma, rng, "clf")
        preds = C.fit_models(y, tr, P["ev"], P["groups"], {b: P["bases"][b] for b in bases}, {model: P["X"][model]})
        fitted[lab] = (P, {k: preds[k] for b in bases for k in (b, f"{b}+{model}")}, gamma, tau)
    subjects = np.unique(np.concatenate([x[0]["subjects"] for x in fitted.values()]))
    g_test = max(int(round(len(subjects) * g_ratio)), 2)
    picks = {lab: model for lab in family}
    out = {b: {k: np.full(draws, np.nan) for k in ("mean", "hi", "train", "upper")} for b in bases}
    for m in range(draws):
        ss = rng.choice(subjects, g_test)
        common = rng.standard_normal(g_test)
        per_lab = {}
        for lab, (P, preds, gamma, tau) in fitted.items():
            here = [(i, P["rows"][u]) for i, u in enumerate(ss) if u in P["rows"]]
            ro = np.concatenate([r for _, r in here])
            gg = np.repeat([i for i, _ in here], [len(r) for _, r in here])
            u = np.sqrt(SHARED) * common + np.sqrt(1 - SHARED) * rng.standard_normal(g_test)
            rows = P["ev"][ro]
            yy = P["eta"][rows] + sigma * u[gg] + gamma * P["sig"][rows] + rng.logistic(0, 1, len(ro)) > tau
            per_lab[lab] = dict(y=yy, group=gg, **{k: v[ro] for k, v in preds.items()})
        for b in bases:
            if not B:
                out[b]["mean"][m] = np.mean([C.auc_fast(a["y"], a[f"{b}+{model}"]) - max(C.auc_fast(a["y"], a[b]), C.GAIN_FLOOR) for a in per_lab.values()])
                continue
            h = C.mean_gain(per_lab, b, picks, seed=int(rng.integers(1 << 31)), B=B)
            ses = [C.delta_auc(a["y"], a[f"{b}+{model}"], a[b], *C.dense(a["group"]))["se"] for a in per_lab.values()]
            train = C.training_sd([r_by_lab[b][lab] for lab in per_lab], ses)
            out[b]["mean"][m], out[b]["hi"][m], out[b]["train"][m], out[b]["upper"][m] = h["mean"], h["hi"], train, C.mean_upper(h["mean"], h["hi"], train)
    return out


def tune_scale(d, bcols, tiers, labs, family, gen, model, weights, target, base, g_ratio, jobs):
    """The signal scale at which the true mean gain over `base` is about `target`: a grid of whole-family worlds and interpolation, twice."""
    def mean_at(scales):
        args = [(d, bcols, tiers, labs, family, gen, model, float(x), weights, 900 + i, 8, g_ratio, None, (base,), 0) for i, x in enumerate(scales)]
        return np.array([np.nanmean(r[base]["mean"]) for r in C.parallel_map(family_world, args, jobs)])
    grid = np.array([0, .4, .8, 1.4, 2.2, 3.5])
    scale = float(np.interp(target, np.maximum.accumulate(mean_at(grid)), grid))
    grid = np.linspace(.75 * scale, 1.25 * scale, 6)
    return float(np.interp(target, np.maximum.accumulate(mean_at(grid)), grid))


def chain_rows(worlds, scenario, gen, scale, margin, bases):
    """chain.csv for one scenario: each base's true mean and the chance of its step passing, alone and in sequence. Replicates are the Monte Carlo clusters."""
    rows, open_ = [], [np.ones_like(w[bases[0]]["upper"], bool) for w in worlds]
    for b in bases:
        alone = [w[b]["upper"] < margin for w in worlds]
        sequence = [o & a for o, a in zip(open_, alone)]
        open_ = sequence
        truth = np.array([np.nanmean(w[b]["mean"]) for w in worlds])
        half = np.concatenate([w[b]["hi"] - w[b]["mean"] for w in worlds]) / norm.isf(C.ALPHA)
        rate, mc = cluster_rate(alone)
        rate_seq, mc_seq = cluster_rate(sequence)
        rows.append(dict(scenario=scenario, gen_base=gen, scale=scale, base=b, replicates=len(worlds), draws=len(worlds[0][b]["upper"]), margin=margin, true_mean=truth.mean(), sd_between=truth.std(ddof=1) if len(truth) > 1 else np.nan,
                         se_bootstrap=float(np.nanmean(half)), training_sd=float(np.nanmean(np.concatenate([w[b]["train"] for w in worlds]))), passes_alone=rate, passes_alone_mc=mc, passes_in_sequence=rate_seq, passes_in_sequence_mc=mc_seq))
    return rows


def smallest_gain(rows, target=.8):
    """Smallest gain with `target` power, by linear interpolation on the probit of the rate over the gains simulated. None when the grid does not reach it."""
    x = rows.sort_values("mean_gain")
    g, p = x.mean_gain.to_numpy(), np.clip(x.power.to_numpy(), 1e-3, 1 - 1e-3)
    zt = norm.ppf(target)
    z = norm.ppf(p)
    for i in range(1, len(g)):
        if z[i - 1] < zt <= z[i]:
            return float(g[i - 1] + (zt - z[i - 1]) / (z[i] - z[i - 1]) * (g[i] - g[i - 1]))
    return float(g[0]) if z[0] >= zt else None


def table(x, cols):
    x = x[cols]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in x.itertuples(index=False)] + [""]


def run(d, bcols, labs, tiers, out, reps=20, draws=60, jobs=3, g_ratio=G_RATIO_DEFAULT, family=C.FAMILY_LABS, named=C.NAMED_LABS, curve_labs=("bun", "potassium"), models=("beat", "papagei"), window="pre",
        r_by_lab=None, chain_reps=20, chain_draws=30, chain_B=500, near=(("near_beat", .003), ("near_with_the_cnn", .0077))):
    """Every table. Rows other than train and validation are gone first."""
    d = d[d.split.isin(["train", "val"])].reset_index(drop=True)
    assert set(d.split) <= {"train", "val"}
    draws = max(draws, L1_DRAWS)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in models if C.feature_cols(m, bcols)[0] in d]
    lean = models[0]
    no_ppg = d.drop(columns=[c for c in d if c in {*BL.PG, *BL.PP, *bcols} or c.startswith("cnn_")])
    bedside = C.parallel_map(C.bedside_gains, [(no_ppg, tiers, n, labs[n]) for n in family], jobs)
    margin_info = C.margin_rule([v for r in bedside for v in r["gains"].values()])
    margin = margin_info["margin"]
    joint = np.array([max(r.get("joint", 0.0), 0.0) for r in bedside])
    print("margin", margin_info, "joint mean", round(float(joint.mean()), 4), "median", round(float(np.median(joint)), 4), flush=True)
    pd.DataFrame([dict(lab=r["lab"], auroc_now=r.get("auroc"), joint=r.get("joint"), **r["gains"]) for r in bedside]).to_csv(out / "bedside_gains.csv", index=False)
    r_by_lab = r_by_lab or {b: {lab: R_ASSUMED[b] for lab in family} for b in C.CHAIN}

    def spec_of(lab, task, gen, model, gamma, seeds, bases=(C.PRIMARY, C.UPPER)):
        return [(d, bcols, tiers, labs[lab], lab, task, gen, model, gamma, s, draws, g_ratio, bases) for s in range(seeds)]

    # power curves over `now`, at the sizes of the test split: rates of L1, L2 and L3 by true gain
    jobs_curve = [a for lab in curve_labs for m in models for g in GAMMAS["clf"] for a in spec_of(lab, "clf", C.PRIMARY, m, g, reps)]
    jobs_curve += [a for lab in named for g in GAMMAS["reg"] for a in spec_of(lab, "reg", C.PRIMARY, models[-1], g, reps)]
    curve = C.parallel_map(replicate, jobs_curve, jobs)
    print("curves done", flush=True)
    # the global null across the whole family, one model
    jobs_null = [a for lab in family if lab not in curve_labs for a in spec_of(lab, "clf", C.PRIMARY, lean, 0.0, reps)]
    null = C.parallel_map(replicate, jobs_null, jobs) + [r for r in curve if r["gamma"] == 0.0 and r["model"] == lean and r["task"] == "clf"]
    null_named = [r for r in curve if r["gamma"] == 0.0 and r["task"] == "reg"]
    print("null done", flush=True)

    # the chain on the mean, on whole families
    def worlds(gen, scale, weights, bases):
        args = [(d, bcols, tiers, labs, family, gen, lean, scale, weights, s, chain_draws, g_ratio, r_by_lab, bases, chain_B) for s in range(chain_reps)]
        return C.parallel_map(family_world, args, jobs)
    chain = chain_rows(worlds(C.UPPER, 0.0, EQUAL, C.CHAIN), "ppg_adds_nothing_over_full", C.UPPER, 0.0, margin, C.CHAIN)
    print("chain under the null done", flush=True)
    boundary = []
    for name, weights, base_list in (("mix", MIX, C.CHAIN), ("equal", EQUAL, (C.UPPER, C.PRIMARY))):
        for b in base_list:
            scale = tune_scale(d, bcols, tiers, labs, family, b, lean, weights, margin, b, g_ratio, jobs)
            boundary += chain_rows(worlds(b, scale, weights, (b,)), f"boundary_{name}", b, scale, margin, (b,))
            print("boundary", name, b, round(scale, 3), round(boundary[-1]["true_mean"], 4), round(boundary[-1]["passes_alone"], 4), flush=True)
    pd.DataFrame(boundary).to_csv(out / "boundary.csv", index=False)
    for name, target in near:     # a true mean over `now` near what validation suggests; the other bases follow from the generator
        scale = tune_scale(d, bcols, tiers, labs, family, C.UPPER, lean, MIX, target, C.PRIMARY, g_ratio, jobs)
        chain += chain_rows(worlds(C.UPPER, scale, MIX, C.CHAIN), name, C.UPPER, scale, margin, C.CHAIN)
        print(name, round(scale, 3), flush=True)
    chain_table = pd.DataFrame(chain + boundary)
    chain_table.to_csv(out / "chain.csv", index=False)

    marginal = summarise(curve + [r for r in null if r["lab"] not in curve_labs])
    marginal.to_csv(out / "marginal.csv", index=False)

    # family-wise error under the global null, replicate by replicate
    shares, levels = C.alpha_shares(len(family), len(named)), [3] * len(family) + [2] * len(named)
    fam_rates = []
    by = {(r["lab"], r["seed"]): r for r in null}
    by_named = {(r["lab"], r["seed"]): r for r in null_named}
    for s in range(reps):
        hits = []
        for m in range(L1_DRAWS):
            Z = np.full((len(family) + len(named), 3), np.nan)
            for i, lab in enumerate(family):
                r = by[(lab, s)]
                a = r["rows"][C.PRIMARY]
                Z[i] = r["z1"][m], a["D"][m] / max(a["SE"][m], 1e-12), (a["D"][m] - C.MIN_GAIN) / max(a["SE"][m], 1e-12)
            for j, lab in enumerate(named):
                r = by_named[(lab, s)]
                a = r["rows"][C.PRIMARY]
                Z[len(family) + j, :2] = r["z1"][m], a["D"][m] / max(a["SE"][m], 1e-12)
            hits.append(C.graded_sequence(Z, shares=shares, levels=levels)["reject"].any())
        fam_rates.append(np.mean(hits))
    fwer = (float(np.mean(fam_rates)), float(np.std(fam_rates, ddof=1) / np.sqrt(len(fam_rates))) if len(fam_rates) > 1 else float("nan"))
    z1_failed = float(np.mean([np.isnan(r["z1"][:L1_DRAWS]).mean() for r in null]))
    steps = [x for x in chain if x["scenario"] == "ppg_adds_nothing_over_full"]
    pd.DataFrame([dict(what="z1_not_computable_under_null", rate=z1_failed, mc=float("nan"), replicates=reps, draws=L1_DRAWS), dict(what="fwer_global_null", rate=fwer[0], mc=fwer[1], replicates=reps, draws=L1_DRAWS)]
                 + [dict(what=f"chain_{x['base']}_alone", rate=x["passes_alone"], mc=x["passes_alone_mc"], replicates=x["replicates"], draws=x["draws"]) for x in steps]
                 + [dict(what=f"chain_{x['base']}_in_sequence", rate=x["passes_in_sequence"], mc=x["passes_in_sequence_mc"], replicates=x["replicates"], draws=x["draws"]) for x in steps]).to_csv(out / "family.csv", index=False)

    # smallest gain with 80% power at the first step, by stratum
    sizes = []
    for (lab, task, model), g in marginal[(marginal.gen_base == C.PRIMARY) & (marginal.base == C.PRIMARY)].groupby(["lab", "task", "model"]):
        if lab not in (*curve_labs, *named):
            continue
        first = "first_a" if task == "clf" else "first_named"
        row = dict(lab=lab, task=task, model=model, n=g.n.iloc[0])
        for level, col in (("L2_or_N2", f"z2>{first}"), ("L1_or_N1", f"z1>{first}")):
            row[f"smallest_gain_80pct_{level}"] = smallest_gain(g.assign(power=g[col]))
        if task == "clf":
            row["smallest_gain_80pct_L3"] = smallest_gain(g.assign(power=g["z3>first_a"]))
        sizes.append(row)
    pd.DataFrame(sizes).to_csv(out / "sample_size.csv", index=False)

    keep = ["lab", "task", "model", "gamma", "base", "replicates", "draws", "n", "mean_gain", "sd_between", "se_eval", "cover95", "z2>first_a", "z2>first_a_mc", "z1>first_a", "z1>first_a_mc", "z3>first_a"]
    sub = marginal[(marginal.gen_base == C.PRIMARY) & (marginal.base == C.PRIMARY) & (marginal.task == "clf")]
    keep_named = ["lab", "model", "gamma", "replicates", "draws", "n", "mean_gain", "sd_between", "se_eval", "cover95", "z2>first_named", "z2>first_named_mc", "z1>first_named", "z1>first_named_mc"]
    sub_named = marginal[(marginal.task == "reg") & (marginal.base == C.PRIMARY)]
    chain_cols = ["scenario", "base", "replicates", "draws", "true_mean", "sd_between", "se_bootstrap", "training_sd", "passes_alone", "passes_alone_mc", "passes_in_sequence"]
    L = ["# Operating characteristics, by plasmode on train and validation rows", "",
         f"Window `{window}`. Evaluation draws resample validation subjects to {g_ratio:.3f} of their number, the ratio of test to validation subjects, with fresh outcomes. Errors are Monte Carlo, replicates as clusters. Family A first-step z = {LEVELS['first_a']:.3f}; named first-step z = {LEVELS['first_named']:.3f}.", "",
         f"Margin +{margin:.3f}: median of bedside gains {margin_info['median_gain']:.4f}, anchor {margin_info['anchor']:.3f}, floor +{margin_info['floor']}, floor binds: {margin_info['floor_binds']}. What MAP, pressors and NBP add together: mean {joint.mean():.4f}, median {np.median(joint):.4f}.", "",
         f"Family-wise error under the global null, 10 labs and 2 named hypotheses: {fwer[0]:.4f} ± {fwer[1]:.4f} ({reps} replicates x {L1_DRAWS} draws; nominal 0.025).", "",
         f"## The chain on the mean ({chain_B} bootstrap resamples per draw; the procedure uses {C.N_MEAN_BOOT})", ""] + table(pd.DataFrame(chain), chain_cols)
    L += ["## A step at the boundary: true mean at the margin (nominal 0.025)", ""] + table(pd.DataFrame(boundary), chain_cols[:-1])
    L += ["## Family A over `now`", ""] + table(sub, keep)
    L += ["## Named hypotheses over `now`", ""] + table(sub_named, keep_named)
    L += ["## Smallest gain with 80% power at the first step", ""] + table(pd.DataFrame(sizes), list(sizes[0]))
    (out / "report.md").write_text("\n".join(L) + "\n")
    write_manifest(out, "operating", dict(window=window, reps=reps, draws=draws, l1_draws=L1_DRAWS, g_ratio=g_ratio, models=models, sigma=SIGMA, gammas=GAMMAS, margin=margin_info, shared=SHARED, mix=MIX, chain_reps=chain_reps, chain_draws=chain_draws, chain_B=chain_B, r_by_lab=r_by_lab),
                   dict(windows=len(d), replicates=len(curve) + len(jobs_null), family_worlds=len(chain_table) and int(chain_table.replicates.sum())))
    return dict(marginal=marginal, boundary=boundary, chain=chain, fwer=fwer, sizes=sizes, margin=margin_info)


def plan_r_hat(family):
    """r-hat per base and lab from the plan, if it is a plan of the chain's bases; None otherwise."""
    if not C.PLAN.exists():
        return None
    plan = json.loads(C.PLAN.read_text())
    if not set(C.CHAIN) <= set(plan.get("r_hat", {})) or not set(family) <= set(plan.get("picks", {})):
        return None
    return {b: {lab: C.lab_entry(plan, lab)["r"].get(b, R_ASSUMED[b]) for lab in family} for b in C.CHAIN}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default=C.PRIMARY_WINDOW, choices=["pre", "centered"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--chain-reps", type=int, default=20)
    ap.add_argument("--chain-draws", type=int, default=30)
    ap.add_argument("--chain-B", type=int, default=500)
    ap.add_argument("--out", default=str(BL.OUT / "operating"))
    args = ap.parse_args()
    d, bcols, labs, tiers = BL.load_mimic(args.window)
    d = d[d.split.isin(["train", "val"])].reset_index(drop=True)      # test rows are gone before anything is computed
    n = pd.read_csv(BL.DATA / "benchmark" / "mimic_splits.csv", usecols=["split"]).split.value_counts()
    r_by_lab = plan_r_hat(C.FAMILY_LABS)
    print("r-hat", "from the plan" if r_by_lab else f"assumed: {R_ASSUMED}", flush=True)
    run(d, bcols, labs, tiers, args.out, args.reps, args.draws, args.jobs, float(n["test"] / n["val"]), window=args.window, r_by_lab=r_by_lab, chain_reps=args.chain_reps, chain_draws=args.chain_draws, chain_B=args.chain_B)


if __name__ == "__main__":
    main()
