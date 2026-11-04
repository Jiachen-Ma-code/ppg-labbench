"""Evaluate your own PPG embeddings on the benchmark: the whole ladder for one model, with one command.

Input : an .npz with `ids` and `emb` [n, d]. `ids` name the windows: draw_id strings for MIMIC, integer caseid for VitalDB. Every id
        must be in the cohort of that source and window; `--drop-extra-ids` ignores the others. A row of NaN means "could not embed", and
        a cohort window absent from the file counts the same: both go to the probe's imputer, as for Pulse-PPG. The share of cohort
        windows embedded is reported, and below `--min-coverage` in train or eval the run stops.
Method: the code of ppglabbench.baselines, unchanged. `run_lab` takes the embedding through the benchmark's probe (imputer, scaler, PCA to 64
        or to d if smaller, then logistic or ridge), stacks its out-of-fold score on train and its refit score on eval on each base of the
        ladder, bootstraps subjects with the fitted models fixed, and repeats the stack with each subject given another subject's
        embedding for the null control. Rows outside train and the eval split are gone before anything is computed.
Report: per lab, the model alone, its increment over each rung with the interval, the null range, and the reference rows of beat, PaPaGei
        and Pulse-PPG if a results file for the same source, split and window exists. Estimates only.
Test  : `--eval test` refuses until `benchmark.test_opened` is true in config.yaml, a committed marker, because the benchmark's one
        confirmatory evaluation of the test split comes before any other use of it.
Demo  : `--demo` runs all of this on generated data, with one embedding carrying a planted signal and one of pure noise. No restricted
        data is read, so it checks an install.

Usage:
    python -m ppglabbench.evaluate --embeddings my_model.npz --name my_model --source {mimic,vitaldb} [--eval val] [--window pre] [--perms 20]
    python -m ppglabbench.evaluate --demo
Output (--out, default data/evaluate/<name>_<source>_<eval>[_centered]/): results.csv, null.csv, report.md, manifest.json
"""
import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
import tempfile
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ppglabbench import baselines as BL
from ppglabbench.common import CFG, DATA, write_manifest

MIN_COVERAGE = 0.5
MAIN_SUBSETS = ("all", "has_locf")    # the ladder and the copied previous value; --subsets adds the sensitivity and subgroup rows
RESERVED = {*BL.LADDER, *BL.PPG_MODELS, "mean_train", "locf_raw", "now_gb"}
KEYS = {"mimic": "draw_id", "vitaldb": "caseid"}
REFERENCE_MODELS = ("beat", "papagei", "pulseppg")
TASKS = (("clf", "auroc", "Abnormal value, AUROC"), ("reg", "r2", "Value, R²"))
DEMO_LAB, DEMO_PERMS, DEMO_BOOT = "demo_lab", 10, 200


class EvaluateError(ValueError):
    """A problem the user can fix; the message says what it is."""


# ------------------------------------------------------------------------------------------------- input

def check_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise EvaluateError(f"--name {name!r}: use letters, digits, _ and - only")
    if name in RESERVED:
        raise EvaluateError(f"--name {name!r} is taken by a row of the benchmark: choose another")


def check_test_open(eval_split):
    """The test split stays closed until config.yaml marks it open. Called before anything is read."""
    if eval_split == "test" and CFG["benchmark"].get("test_opened") is not True:
        raise EvaluateError(
            "--eval test refused: benchmark.test_opened is not true in config.yaml. The benchmark evaluates the test split once, with its "
            "confirmatory analysis (ppglabbench.confirmatory), before any other model reads the split, so that nothing seen on it can shape "
            "that analysis. The marker is set in a commit once that evaluation is done. Until then, use --eval val.")


def read_embeddings(path):
    """ids and emb [n, d] as floats from a user's .npz, or a clear error. Pickles are never loaded."""
    path = Path(path)
    if not path.is_file():
        raise EvaluateError(f"{path}: no such file")
    try:
        z = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as e:
        raise EvaluateError(f"{path}: not a readable .npz ({e})") from e
    if not hasattr(z, "files"):
        raise EvaluateError(f"{path}: an .npy array, not an .npz: save with np.savez(path, ids=ids, emb=emb)")
    with z:
        missing = [k for k in ("ids", "emb") if k not in z.files]
        if missing:
            raise EvaluateError(f"{path}: missing key(s) {missing}; found {sorted(z.files)}. Expected `ids` and `emb` [n, d]")
        try:
            ids, emb = z["ids"], z["emb"]
        except ValueError as e:
            raise EvaluateError(f"{path}: `ids` or `emb` is an object array, which is not loaded because it needs a pickle. Save plain arrays: "
                                "np.savez(path, ids=np.asarray(ids, dtype=str), emb=np.asarray(emb, dtype=float))") from e
    if ids.ndim != 1:
        raise EvaluateError(f"`ids` must be 1-D, got shape {ids.shape}")
    if emb.ndim != 2:
        raise EvaluateError(f"`emb` must be 2-D [n, d], got shape {emb.shape}")
    if emb.shape[0] != len(ids):
        raise EvaluateError(f"`ids` has {len(ids)} rows and `emb` has {emb.shape[0]}: they must match")
    if len(ids) == 0 or emb.shape[1] == 0:
        raise EvaluateError(f"`emb` is empty, shape {emb.shape}")
    if ids.dtype.kind not in "iuUS":
        raise EvaluateError(f"`ids` must hold strings or integers, got dtype {ids.dtype}")
    if emb.dtype.kind not in "fiu":
        raise EvaluateError(f"`emb` must be numeric, got dtype {emb.dtype}")
    emb = emb.astype(float)
    if np.isinf(emb).any():
        raise EvaluateError(f"`emb` holds {int(np.isinf(emb).any(1).sum())} rows with infinite values")
    nan = np.isnan(emb)
    partial_rows = int((nan.any(1) & ~nan.all(1)).sum())
    if partial_rows:
        raise EvaluateError(f"`emb` holds {partial_rows} rows partly NaN: a window is either embedded or all NaN")
    if nan.all():
        raise EvaluateError("`emb` is NaN in every row")
    duplicates = len(ids) - len(np.unique(ids))
    if duplicates:
        raise EvaluateError(f"`ids` holds {duplicates} duplicates: one row per window")
    return ids, emb


def normalise_ids(ids, source):
    """draw_id strings for MIMIC, integer caseid for VitalDB. IDs are never printed in a message: they hold subject IDs in MIMIC."""
    if source == "mimic":
        if ids.dtype.kind not in "US":
            raise EvaluateError(f"MIMIC ids must be draw_id strings, <subject_id>_<YYYYmmddHHMM>; got dtype {ids.dtype}")
        return ids.astype(str)
    if ids.dtype.kind in "iu":
        return ids.astype(int)
    try:
        return ids.astype(str).astype(int)
    except ValueError as e:
        raise EvaluateError(f"VitalDB ids must be integer caseid; got dtype {ids.dtype} with values that are not integers") from e


def match_cohort(ids, emb, cohort_ids, drop_extra):
    """The rows of the file that are in the cohort. An id outside it is an error unless the user asks to ignore it."""
    known = pd.Series(ids).isin(cohort_ids).to_numpy()
    extra = int((~known).sum())
    if extra == len(ids):
        raise EvaluateError(f"none of the {len(ids)} ids is in the cohort of {len(cohort_ids)} windows: check the id format")
    if extra and not drop_extra:
        raise EvaluateError(f"{extra} of {len(ids)} ids are not in the cohort of {len(cohort_ids)} windows, which requires beat features, a PaPaGei "
                            "embedding and the source's quality rules. Pass --drop-extra-ids to ignore them")
    return ids[known], emb[known], extra


def check_coverage(d, col, eval_split, min_coverage):
    seen = d[col].notna()
    cov = dict(windows=len(d), embedded=int(seen.sum()), share=float(seen.mean()), train=float(seen[d.split == "train"].mean()),
               eval=float(seen[d.split == eval_split].mean()))
    if min(cov["train"], cov["eval"]) < min_coverage:
        raise EvaluateError(f"coverage too low: {cov['train']:.1%} of train and {cov['eval']:.1%} of {eval_split} windows are embedded, below "
                            f"{min_coverage:.0%}. Missing rows would all go to the imputer. Check the ids, or lower --min-coverage")
    return cov


def check_out(out):
    """The project's own results stay where they are, and a folder there could block the confirmatory run."""
    if Path(out).resolve().is_relative_to(BL.OUT.resolve()):
        raise EvaluateError(f"--out {out}: this folder holds the benchmark's own results. Choose another")


# ------------------------------------------------------------------------------------------------- the report

def pick(r, lab, task, metric, method, vs=None):
    x = r[(r.lab == lab) & (r.task == task) & (r.subset == "all") & (r.metric == metric) & (r.method == method)]
    x = x[x.vs.isna()] if vs is None else x[x.vs == vs]
    return x.iloc[0] if len(x) else None


def ci(e, signed=False):
    if e is None:
        return "—"
    f = "+.3f" if signed else ".3f"
    return f"{e.est:{f}} ({e.lo:{f}}, {e.hi:{f}})"


def lab_lines(r, nulls, ref, name, lab):
    """One lab: for each task, a table with a row for the model alone and a row per rung of the ladder."""
    L = [f"## {lab}", ""]
    refs = [m for m in REFERENCE_MODELS if ref is not None and (ref.method == m).any()]
    for task, metric, title in TASKS:
        alone = pick(r, lab, task, metric, name)
        if alone is None:
            continue
        head = f"n eval {alone.n_eval}, {alone.subjects_eval} subjects" + (f", prevalence {alone.prev_eval:.1%}" if task == "clf" else "")
        other = pick(ref, lab, task, metric, refs[0]) if refs else None
        if other is not None and other.n_eval != alone.n_eval:
            head += f"; reference rows on n eval {other.n_eval}"
        columns = ["Rung", "Base alone", name, f"Null for {name}", *refs]
        L += [f"### {title} ({head})", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        L.append("| model alone | — | " + " | ".join([ci(alone), "—"] + [ci(pick(ref, lab, task, metric, m)) for m in refs]) + " |")
        for b in BL.LADDER:
            base = pick(r, lab, task, metric, b)
            if base is None:
                continue
            x = nulls[(nulls.lab == lab) & (nulls.task == task) & (nulls.base == b) & (nulls.ppg == name)]
            cells = [ci(base), ci(pick(r, lab, task, metric, f"{b}+{name}", b), True), BL.null_text(x)]
            L.append(f"| {b} | " + " | ".join(cells + [ci(pick(ref, lab, task, metric, f"{b}+{m}", b), True) for m in refs]) + " |")
        L.append("")
        if task == "reg":
            s = r[(r.lab == lab) & (r.subset == "has_locf") & (r.metric == "r2")]
            raw, fit, plus = (s[(s.method == m) & mask] for m, mask in (("locf_raw", s.vs.isna()), ("locf", s.vs.isna()), (f"locf+{name}", s.vs == "locf")))
            if len(raw) and len(fit) and len(plus):
                L += [f"Windows with a previous value (n {raw.n_eval.iloc[0]}): the value copied {ci(raw.iloc[0])}, fitted on it {ci(fit.iloc[0])}, "
                      f"fitted plus {name} {ci(plus.iloc[0], True)}.", ""]
    return L


def body_lines(r, nulls, ref, name):
    return [line for lab in dict.fromkeys(r.lab) for line in lab_lines(r, nulls, ref, name, lab)]


def report_md(r, nulls, ref, name, source, eval_split, cov, info):
    perms = nulls.perm.nunique() if len(nulls) else 0
    L = [f"# {name} on PPG-LabBench: {source}, train -> {eval_split}", "",
         f"Embeddings: {info['file']}, {info['dim']} dimensions, {info['probe']}.",
         f"Coverage: {cov['share']:.1%} of {cov['windows']} cohort windows embedded (train {cov['train']:.1%}, {eval_split} {cov['eval']:.1%}). "
         "Missing rows go to the probe's imputer.",
         *([f"Ids ignored because they are outside the cohort: {info['extra']}."] if info["extra"] else []), "",
         f"Model alone: the probe's score. Under {name} and the reference columns: the stacked model (base + score) minus the base. AUROC counts the "
         "base as at least 0.5. Skewed labs are on a log scale. In brackets, a 95% subject-bootstrap interval with the fitted models fixed, "
         f"which leaves out training variance. Null: mean and 2.5-97.5% range of the increment over {perms} subject-block permutations of the "
         "embedding. Nothing here is corrected for the number of labs or rungs: these are estimates, and this report draws no conclusion from them.", "",
         info["reference"], ""]
    return "\n".join(L + body_lines(r, nulls, ref, name))


def load_reference(path):
    """The rows of a reference results file on the whole eval split, its description for the report, and None if there is no file."""
    if path is None or not Path(path).is_file():
        return None, "Reference rows: none, as no results file was found for this source, split and window."
    ref = pd.read_csv(path)
    manifest = Path(path).parent / "manifest.json"
    version = json.loads(manifest.read_text()).get("code_version", "unknown") if manifest.exists() else "unknown"
    return ref[ref.subset == "all"], f"Reference rows: {Path(path).parent.name}/{Path(path).name}, code version {version}."


# ------------------------------------------------------------------------------------------------- the run

def run_model(cohort, name, cols, eval_split, perms, jobs, only):
    """The reference pipeline for one embedding, through baselines.run_lab. d must already hold only train and the eval split."""
    d, bcols, labs, tiers = cohort
    k = min(64, len(cols))
    models = ((name, cols, {"clf": partial(BL.probe_clf, k), "reg": partial(BL.probe_reg, k)}),)
    res = Parallel(n_jobs=jobs)(delayed(BL.run_lab)(d, bcols, tiers, n, s, eval_split, perms, models=models, only=only) for n, s in labs.items())
    r = pd.DataFrame([x for rows, _ in res for x in rows])
    nulls = pd.DataFrame([x for _, ns in res for x in ns], columns=["lab", "task", "ppg", "base", "perm", "delta"])
    return r, nulls


def evaluate(cohort, key, ids, emb, name, out, source, eval_split="val", window=None, perms=20, jobs=3, reference=None, drop_extra=False,
             min_coverage=MIN_COVERAGE, only=MAIN_SUBSETS, file_info=""):
    """Everything after the file is read and the cohort is loaded: the checks against the cohort, the run and the output."""
    d, bcols, labs, tiers = cohort
    ids, emb, extra = match_cohort(ids, emb, d[key], drop_extra)
    d = d[d.split.isin(["train", eval_split])].reset_index(drop=True)     # other splits are gone before anything is computed
    cols = [f"user_emb{i}" for i in range(emb.shape[1])]
    d = d.merge(pd.DataFrame(emb, columns=cols).assign(**{key: ids}), on=key, how="left")
    cov = check_coverage(d, cols[0], eval_split, min_coverage)
    r, nulls = run_model((d, bcols, labs, tiers), name, cols, eval_split, perms, jobs, only)
    if r.empty:
        raise EvaluateError("no lab had enough windows to evaluate")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ref, ref_text = load_reference(reference)
    k = min(64, len(cols))
    info = dict(file=file_info, dim=len(cols), extra=extra, reference=ref_text, probe=f"probe: imputer, scaler, PCA({k}), logistic or ridge")
    r.to_csv(out / "results.csv", index=False)
    nulls.to_csv(out / "null.csv", index=False)
    (out / "report.md").write_text(report_md(r, nulls, ref, name, source, eval_split, cov, info))
    write_manifest(out, "evaluate", dict(name=name, source=source, eval=eval_split, window=window, perms=perms, n_boot=BL.N_BOOT, subsets=only, embeddings=file_info,
                                         dim=len(cols), probe_components=k, min_coverage=min_coverage), dict(coverage=cov, ids_ignored=extra, labs=sorted(set(r.lab))))
    return dict(results=r, nulls=nulls, reference=ref, coverage=cov, out=out)


def load_cohort(source, window):
    try:
        return BL.load_mimic(window) if source == "mimic" else BL.load_vitaldb()
    except FileNotFoundError as e:
        raise EvaluateError(f"the cohort needs {e.filename}, which is missing: build the benchmark features first (make features, see README.md)") from e


# ------------------------------------------------------------------------------------------------- the demo

def demo_cohort(seed=0, n_subjects=600):
    """A cohort shaped like load_mimic's output, all generated, and the per-window signal that none of its covariates can see."""
    rng = np.random.default_rng(seed)
    sid = np.repeat(np.arange(n_subjects), rng.integers(1, 4, n_subjects))
    n = len(sid)
    d = pd.DataFrame(dict(draw_id=[f"demo{s}_{i}" for i, s in enumerate(sid)], group=sid, split=np.where(sid % 10 < 7, "train", "val")))
    d["age"], d["female"] = rng.normal(60, 15, n_subjects)[sid], (rng.random(n_subjects) < .45).astype(float)[sid]
    severity, subject, signal = rng.normal(0, 1, n), rng.normal(0, 1, n_subjects)[sid], rng.normal(0, 1, n)
    hour = rng.uniform(0, 24, n)
    d["vasopressor"], d["emergency_admit"] = (severity + rng.normal(0, 1, n) > 1.5).astype(float), (rng.random(n) < .8).astype(float)
    d["map_mmhg"] = np.where(rng.random(n) < .5, 80 - 6 * severity + rng.normal(0, 8, n), np.nan)
    d["map_lt70"] = (d.map_mmhg < 70).astype(float).where(d.map_mmhg.notna())
    d["routine_draw"], d["hour_sin"], d["hour_cos"] = (hour < 6).astype(float), np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    d["hr_mon"], d["spo2_mon"] = 85 + 6 * severity + rng.normal(0, 10, n), 97 - severity + rng.normal(0, 1.5, n)
    d["on_sedation"], d["n_pressors"] = (severity + rng.normal(0, 1, n) > 1).astype(float), (severity + rng.normal(0, 1, n) > 2).astype(float)
    from_age = .02 * (d.age.to_numpy() - 60)
    y = from_age + .5 * severity + .8 * subject + .8 * signal + rng.normal(0, .6, n)
    previous = from_age + .8 * subject + rng.normal(0, .4, n)
    lab = DEMO_LAB
    d[lab], d[f"{lab}_abnormal"] = y, (y > .5).astype(float)
    d[f"{lab}_locf"], d[f"{lab}_locf_age_h"] = np.where(rng.random(n) < .85, previous, np.nan), rng.exponential(24, n)
    d[f"{lab}_hist_mean"], d[f"{lab}_hist_n"], d[f"{lab}_trend"] = previous + rng.normal(0, .3, n), rng.integers(1, 9, n), rng.normal(0, .2, n)
    acuity = ["vasopressor", "emergency_admit", "map_mmhg", "map_lt70", "routine_draw", "hour_sin", "hour_cos"]
    monitor = acuity + ["hr_mon", "spo2_mon"]
    labs = {lab: dict(value=lab, abnormal=f"{lab}_abnormal", locf=[f"{lab}_locf", f"{lab}_locf_age_h"], hist=[f"{lab}_hist_mean", f"{lab}_hist_n", f"{lab}_trend"])}
    return (d, [], labs, dict(acuity=acuity, monitor=monitor, now=monitor + ["on_sedation", "n_pressors"])), signal


def demo_embeddings(signal, seed=1, dim=128):
    """Two embeddings of the demo's windows. Planted: the signal along one direction, among other structure, with 5% of windows left out
    of the file. Noise: nothing but noise, with 5% of its rows NaN."""
    rng = np.random.default_rng(seed)
    n = len(signal)
    direction = rng.normal(0, 1, dim)
    direction /= np.linalg.norm(direction)
    planted = rng.normal(0, 1, (n, 8)) @ rng.normal(0, 1, (8, dim)) + 6 * signal[:, None] * direction + rng.normal(0, 1, (n, dim))
    noise = rng.normal(0, 1, (n, dim))
    noise[rng.random(n) < .05] = np.nan
    return planted, rng.random(n) >= .05, noise


def run_demo(out=None, seed=0):
    """The whole path on generated data, in process. Returns the results of both embeddings; the printing is in `main`."""
    cohort, signal = demo_cohort(seed)
    planted, keep, noise = demo_embeddings(signal, seed + 1)
    ids = cohort[0].draw_id.to_numpy(str)
    saved, BL.N_BOOT = BL.N_BOOT, DEMO_BOOT
    results = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(out) if out else Path(tmp)
            base.mkdir(parents=True, exist_ok=True)
            for name, a, b in (("planted", ids[keep], planted[keep]), ("noise", ids, noise)):
                path = base / f"{name}.npz"
                np.savez(path, ids=a, emb=b)
                file_ids, file_emb = read_embeddings(path)
                with contextlib.redirect_stdout(io.StringIO()):
                    results[name] = evaluate(cohort, "draw_id", normalise_ids(file_ids, "mimic"), file_emb, name, base / name, "demo", perms=DEMO_PERMS, jobs=1, file_info=path.name)
    finally:
        BL.N_BOOT = saved
    return results


def demo_gain(res, name, base):
    return pick(res["results"], DEMO_LAB, "clf", "auroc", f"{base}+{name}", base)


def demo_check(results):
    """What the demo must show on any working install: a clear gain from the planted signal on every rung, and about 0 from noise."""
    problems = []
    for b in BL.LADDER:
        p, q = demo_gain(results["planted"], "planted", b), demo_gain(results["noise"], "noise", b)
        if p.lo <= 0.02:
            problems.append(f"planted over {b}: {ci(p, True)}, expected a clear gain")
        if abs(q.est) >= 0.02:
            problems.append(f"noise over {b}: {ci(q, True)}, expected about 0")
    return problems


# ------------------------------------------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", help=".npz with ids and emb [n, d]")
    ap.add_argument("--name", help="your model's name in the output")
    ap.add_argument("--source", choices=sorted(KEYS))
    ap.add_argument("--eval", default="val", choices=["val", "test"])
    ap.add_argument("--window", default="pre", choices=["pre", "centered"], help="MIMIC only")
    ap.add_argument("--perms", type=int, default=20, help="subject-block permutations for the null control")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--reference", help="results.csv to show beside your model; default: the benchmark's run for this source, split and window")
    ap.add_argument("--drop-extra-ids", action="store_true", help="ignore ids outside the cohort instead of stopping")
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE, help="smallest share of train and of eval windows embedded")
    ap.add_argument("--subsets", action="store_true", help="also compute the sensitivity and subgroup rows in results.csv; slower")
    ap.add_argument("--demo", action="store_true", help="run on generated data, no restricted data needed")
    args = ap.parse_args(argv)
    try:
        if args.demo:
            return demo(args.out)
        if not (args.embeddings and args.name and args.source):
            ap.error("--embeddings, --name and --source are required, unless --demo")
        check_name(args.name)
        check_test_open(args.eval)
        stem = BL.results_stem(args.source, args.eval, args.window)
        out = args.out or DATA / "evaluate" / f"{args.name}_{stem}"
        check_out(out)
        reference = args.reference or BL.OUT / stem / "results.csv"
        if args.reference and not Path(args.reference).is_file():
            raise EvaluateError(f"--reference {args.reference}: no such file")
        ids, emb = read_embeddings(args.embeddings)
        ids = normalise_ids(ids, args.source)
        cohort = load_cohort(args.source, args.window)
        res = evaluate(cohort, KEYS[args.source], ids, emb, args.name, out, args.source, args.eval, args.window, args.perms, args.jobs, reference, args.drop_extra_ids,
                       args.min_coverage, None if args.subsets else MAIN_SUBSETS, f"{Path(args.embeddings).name}, SHA-256 {hashlib.sha256(Path(args.embeddings).read_bytes()).hexdigest()}")
    except EvaluateError as e:
        sys.exit(f"error: {e}")
    cov = res["coverage"]
    print(f"{args.name}: {cov['share']:.1%} of {cov['windows']} cohort windows embedded. Written to {res['out']}: results.csv, null.csv, report.md, manifest.json")


def demo(out):
    results = run_demo(out)
    for name, res in results.items():
        cov = res["coverage"]
        print(f"\n===== {name}: {cov['share']:.1%} of {cov['windows']} synthetic windows embedded =====\n")
        print("\n".join(body_lines(res["results"], res["nulls"], None, name)))
    problems = demo_check(results)
    if problems:
        sys.exit("demo check failed:\n  " + "\n  ".join(problems))
    print(f"Demo check passed: on generated data, the planted signal gains on every rung ({ci(demo_gain(results['planted'], 'planted', 'now'), True)} over `now`) "
          f"and noise stays near 0 ({ci(demo_gain(results['noise'], 'noise', 'now'), True)}).")
    if out:
        print(f"Written to {out}")


if __name__ == "__main__":
    main()
