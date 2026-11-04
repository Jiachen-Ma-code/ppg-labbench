"""Hand-checkable tests of ppglabbench.tracking. Synthetic data only: no MIMIC row is read, and the test split never gets past its guards.

Usage:  python -m unittest tests.test_tracking
"""
import contextlib
import fnmatch
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ppglabbench import confirmatory as C  # noqa: E402
from ppglabbench import tracking as T  # noqa: E402


def setUpModule():
    """No test of this module can write under the real results folder, whatever it forgets to patch."""
    global _out, _patch
    _out = tempfile.TemporaryDirectory()
    _patch = mock.patch.object(T, "OUT", Path(_out.name))
    _patch.start()


def tearDownModule():
    _patch.stop()
    _out.cleanup()


class TestStatistics(unittest.TestCase):
    def test_corr_from_sums_matches_numpy(self):
        r = np.random.default_rng(0)
        x, y, g = r.normal(size=200), r.normal(size=200), r.integers(0, 30, 200)
        self.assertAlmostEqual(T.corr(T.sums(g, x, y, 30).sum(0)), np.corrcoef(x, y)[0, 1], places=12)

    def test_counts_equal_explicit_resampling(self):
        r = np.random.default_rng(1)
        g = np.repeat(np.arange(20), 3)
        x, y = r.normal(size=60), r.normal(size=60)
        c = r.multinomial(20, np.full(20, 1 / 20), size=1)
        ix = np.concatenate([np.flatnonzero(g == i).repeat(1) for i in range(20) for _ in range(c[0, i])])
        self.assertAlmostEqual(float(T.corr(c @ T.sums(g, x, y, 20))[0]), np.corrcoef(x[ix], y[ix])[0, 1], places=12)

    def test_variance_components_by_hand(self):
        # groups (1, 3) and (2, 4, 6): MSW = (2 + 8) / 3, MSB = SSB / 1, n0 = (5 - 13 / 5) / 1 = 2.4
        y, g = np.array([1., 3., 2., 4., 6.]), np.array([0, 0, 1, 1, 1])
        between, within = T.var_parts(T.var_sums(g, y, 2).sum(0))
        ssb = 2 * (2 - 3.2) ** 2 + 3 * (4 - 3.2) ** 2
        self.assertAlmostEqual(within, 10 / 3)
        self.assertAlmostEqual(between, max((ssb - 10 / 3) / 2.4, 0))

    def test_variance_components_recover_known_parts(self):
        r = np.random.default_rng(2)
        n = r.integers(2, 8, 4000)
        g = np.repeat(np.arange(4000), n)
        y = r.normal(0, 2, 4000)[g] + r.normal(0, 1, len(g))    # between 4, within 1, ICC 0.8
        s = T.var_sums(g, y, 4000).sum(0)
        self.assertAlmostEqual(float(T.icc(s)), 0.8, delta=0.02)
        self.assertAlmostEqual(float(T.var_parts(s)[1]), 1.0, delta=0.03)

    def test_within_shuffle_stays_within_subject(self):
        r = np.random.default_rng(3)
        code = r.integers(0, 15, 100)
        rows = np.flatnonzero(r.random(100) < 0.7)
        pm = T.within_shuffle(rows, code, r)
        np.testing.assert_array_equal(code[pm], code)
        np.testing.assert_array_equal(np.sort(pm[rows]), np.sort(rows))
        other = np.setdiff1d(np.arange(100), rows)
        np.testing.assert_array_equal(pm[other], other)
        self.assertGreater((pm[rows] != rows).mean(), 0.3)

    def test_auc_parts_match_sklearn_with_ties_and_weights(self):
        r = np.random.default_rng(4)
        s, y = r.integers(0, 12, 300).astype(float), r.integers(0, 2, 300).astype(float)
        inv, u = T.auc_bins(s)
        a, b = T.auc_parts(inv, u, y, np.ones(300))
        self.assertAlmostEqual(a / b, roc_auc_score(y, s), places=12)
        w = r.integers(0, 4, 300)
        ix = np.repeat(np.arange(300), w)
        a, b = T.auc_parts(inv, u, y, w.astype(float))
        self.assertAlmostEqual(a / b, roc_auc_score(y[ix], s[ix]), places=12)

    def test_pooled_auroc_is_the_weighted_sum_of_pair_types(self):
        r = np.random.default_rng(5)
        g = r.integers(0, 80, 400)
        y = (r.random(80)[g] + r.normal(0, .3, 400) > .6).astype(float)
        x = r.normal(size=400) + y
        inv, u = T.auc_bins(x)
        a, b = T.auc_parts(inv, u, y, np.ones(400))
        i, j = T.within_pairs(g, y)
        self.assertTrue(((y[i] == 1) & (y[j] == 0) & (g[i] == g[j])).all())
        self.assertEqual(len(i), sum(int((y[g == k] == 1).sum() * (y[g == k] == 0).sum()) for k in range(80)))
        win = (x[i] > x[j]) + .5 * (x[i] == x[j])
        share = len(i) / b
        within, between = win.mean(), (a - win.sum()) / (b - len(i))
        self.assertAlmostEqual(a / b, share * within + (1 - share) * between, places=12)

    def test_planted_correlation_and_means_are_exact(self):
        r = np.random.default_rng(6)
        n = r.integers(1, 7, 300)
        g = np.repeat(np.arange(300), n)
        y, hr = r.normal(0, 2, 300)[g] + r.normal(size=len(g)), r.normal(80, 10, 300)[g] + r.normal(0, 5, len(g))
        for rho in (0.1, 0.5):
            p = T.planted(y, hr, g, rho)
            a, b = T.demean(p, g), T.demean(hr, g)
            self.assertAlmostEqual((a @ b) / np.sqrt((a @ a) * (b @ b)), rho, places=10)
            self.assertAlmostEqual((a @ a), (T.demean(y, g) @ T.demean(y, g)), places=8)
            np.testing.assert_allclose(p - a, y - T.demean(y, g), atol=1e-10)

    def test_covariates_are_removed_with_train_coefficients(self):
        r = np.random.default_rng(7)
        n, g = 4000, np.repeat(np.arange(800), 5)
        z = r.normal(size=n)
        y, s = z + .5 * r.normal(size=n), z + .5 * r.normal(size=n)      # linked only through z
        split = np.arange(800)[g] < 640
        tr, ev = np.flatnonzero(split), np.flatnonzero(~split)
        now = np.column_stack([z, r.normal(size=n)])
        prev, cur = np.arange(n - 1)[g[:-1] == g[1:]], np.arange(1, n)[g[:-1] == g[1:]]
        fr = T.SimpleNamespace(y=y, tr=tr, ev=ev, code=g, bases=dict(now=now), cur=cur, prev=prev, dy=y[cur] - y[prev],
                               ptr=np.flatnonzero(split[cur]), pev=np.flatnonzero(~split[cur]))
        fr.pg = g[cur[fr.pev]] - 640
        yr, xr = T.now_residuals(fr, s)
        self.assertGreater(np.corrcoef(T.demean(y[ev], g[ev]), T.demean(s[ev], g[ev]))[0, 1], .7)
        self.assertLess(abs(np.corrcoef(yr, xr)[0, 1]), .08)
        c = r.multinomial(160, np.full(160, 1 / 160), size=50)
        raw = np.corrcoef((s[cur] - s[prev])[fr.pev], fr.dy[fr.pev])[0, 1]
        row = T.change_now_row(fr, s[cur] - s[prev], c, 160, method="ppg_raw", ppg="x")
        self.assertGreater(raw, .7)
        self.assertLess(abs(row["est"]), .08)

    def test_bh(self):
        np.testing.assert_allclose(T.bh([0.01, 0.04, 0.03, 0.2]), [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)


def fake_cohort(subjects=900, seed=0):
    """A load_mimic stand-in with known truth. `sodium` has a large within-subject part that PPG reads; `bun` differs only between
    subjects, and PPG reads only that; `potassium` is unrelated to PPG. The monitor HR is read by the beat `hr` feature. Test rows hold
    a sentinel that breaks any statistic they reach."""
    r = np.random.default_rng(seed)
    n = r.integers(1, 6, subjects)
    g = np.repeat(np.arange(subjects), n)
    N = len(g)
    split = np.array(["train", "val", "test"])[r.choice(3, subjects, p=[.7, .15, .15])][g]
    stable, moving = r.normal(size=subjects)[g], r.normal(size=N)
    lab_noise = r.normal(size=N)
    t = pd.Timestamp(2100, 1, 1) + pd.to_timedelta(np.concatenate([np.cumsum(r.uniform(8, 40, k)) for k in n]), unit="h")
    hr = 80 + 8 * r.normal(size=subjects)[g] + 12 * r.normal(size=N)
    d = pd.DataFrame(dict(group=g, split=split, charttime=t.astype(str), hadm_id=g * 10.0, age=60 + 10 * r.normal(size=subjects)[g],
                          female=(r.random(subjects) < .5)[g].astype(float), dev_set=False, pre_start_s=-400.0, best_start_s=-150.0, ppg_end_s=-100.0,
                          hr_mon=hr, map_mmhg=75 + 10 * r.normal(size=N), vasopressor=(r.random(N) < .1).astype(float),
                          routine_draw=(r.random(N) < .6).astype(float), emergency_admit=1.0, hour_sin=0.0, hour_cos=1.0))
    d["map_lt70"] = (d.map_mmhg < 70).astype(float)
    d["sodium"] = 140 + 1.5 * r.normal(size=subjects)[g] + 3 * moving + 1.0 * lab_noise
    d["bun"] = np.exp(3 + .6 * stable + .05 * r.normal(size=N))
    d["potassium"] = 4 + .4 * r.normal(size=subjects)[g] + .4 * r.normal(size=N)
    for name, rule in (("sodium", lambda v: v < 136), ("bun", lambda v: v > 20), ("potassium", lambda v: v > 4.4)):
        d[f"{name}_offset_min"] = 0.0
        d[f"{name}_abnormal"] = rule(d[name]).astype(float)
        by = d.groupby("group")[name]
        d[f"{name}_locf"], d[f"{name}_locf_age_h"] = by.shift(1), 24.0
        d[f"{name}_hist_mean"], d[f"{name}_hist_n"] = by.transform(lambda v: v.expanding().mean().shift(1)), 1.0
        d[f"{name}_trend"] = by.diff().groupby(d.group).shift(1)
        d.loc[split == "test", name] = 1e9       # sentinel
    bcols = ["hr", "beat_stable", "beat_moving", "beat_noise"]
    d["hr"], d["beat_stable"], d["beat_moving"] = hr + r.normal(0, 1, N), stable + .2 * r.normal(size=N), moving + .2 * r.normal(size=N)
    d["beat_noise"] = r.normal(size=N)
    emb = np.column_stack([stable, moving]) @ r.normal(size=(2, 512)) + r.normal(0, .5, (N, 512))
    d = pd.concat([d, pd.DataFrame(emb, columns=T.PG), pd.DataFrame(emb + r.normal(0, .5, (N, 512)), columns=T.PP)], axis=1)
    # the CNN's logits point at the abnormal value: for sodium, abnormal when low, that means a low value. Train rows have a fold by subject, the others none
    d["cnn_sodium"], d["cnn_bun"], d["cnn_potassium"] = -moving + .2 * r.normal(size=N), stable + .2 * r.normal(size=N), r.normal(size=N)
    d["cnn_fold"] = np.where(split == "train", g % 5, -1)
    d["prbc_24h"], d["dialysis_fluid_24h"] = (r.random(N) < .3).astype(float), (r.random(N) < .3).astype(float)
    specs = {n_: dict(value=n_, abnormal=f"{n_}_abnormal", locf=[f"{n_}_locf", f"{n_}_locf_age_h"], hist=[f"{n_}_hist_mean", f"{n_}_hist_n", f"{n_}_trend"])
             for n_ in ("sodium", "bun", "potassium")}
    tiers = dict(acuity=["vasopressor", "map_mmhg", "map_lt70", "routine_draw"], monitor=["vasopressor", "map_mmhg", "map_lt70", "routine_draw", "hr_mon"],
                 now=["vasopressor", "map_mmhg", "map_lt70", "routine_draw", "hr_mon", "emergency_admit"])
    return d, bcols, specs, tiers


def stand_in_split_file(folder):
    """check_test_allowed hashes data/benchmark/mimic_splits.csv first: a stand-in, so that the guards run with no data/ folder."""
    (Path(folder) / "benchmark").mkdir()
    (Path(folder) / "benchmark" / "mimic_splits.csv").write_text("subject_id,split\n")
    return mock.patch.object(C.BL, "DATA", Path(folder))


class TestGuards(unittest.TestCase):
    def setUp(self):
        self.load = mock.patch.object(T, "load_mimic", side_effect=AssertionError("data loaded"))
        self.load.start()
        self.addCleanup(self.load.stop)

    def test_test_split_refused_without_open_test(self):
        with tempfile.TemporaryDirectory() as t, stand_in_split_file(t), self.assertRaises(SystemExit) as e:
            T.run(eval_split="test")
        self.assertIn("--open-test not given", str(e.exception))

    def test_test_split_refused_with_an_output_folder(self):
        with self.assertRaises(SystemExit) as e:
            T.run(eval_split="test", open_test=True, out="elsewhere")
        self.assertIn("--out", str(e.exception))

    def test_test_split_refused_when_any_baselines_condition_fails(self):
        with mock.patch.object(T, "check_test_allowed", side_effect=SystemExit("test split not opened: x")) as chk, self.assertRaises(SystemExit):
            T.run(eval_split="test", open_test=True)
        chk.assert_called_once_with("tracking_test", True)

    def test_confirmatory_evaluation_has_to_come_first(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(T, "check_test_allowed"), mock.patch.object(T, "TEST_OUT", Path(tmp) / "absent"):
            with self.assertRaises(SystemExit) as e:
                T.run(eval_split="test", open_test=True)
        self.assertIn("confirmatory", str(e.exception))

    def test_its_test_folder_does_not_block_the_confirmatory_run(self):
        self.assertEqual(T.tracking_stem("test", "pre"), "tracking_test")
        for split in ("val", "test"):
            for window in T.PPG_START:
                self.assertFalse(fnmatch.fnmatch(T.tracking_stem(split, window), "mimic_test*"), (split, window))
        with tempfile.TemporaryDirectory() as t, mock.patch.object(C.BL, "check_test_allowed"), mock.patch.object(C, "OUT", Path(t)), mock.patch.object(C, "PLAN", Path(t) / "none.json"), \
                mock.patch.object(C.BL, "load_mimic", side_effect=AssertionError("data loaded")):
            (Path(t) / "tracking_test").mkdir()
            with self.assertRaises(SystemExit) as e:
                C.cmd_run(C.argparse.Namespace(open_test=True, jobs=1))
        self.assertIn("does not exist", str(e.exception))      # the next guard along, not "has to come first"

    def test_an_earlier_baselines_folder_still_does(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(C.BL, "check_test_allowed"), mock.patch.object(C, "OUT", Path(t)), \
                mock.patch.object(C.BL, "load_mimic", side_effect=AssertionError("data loaded")):
            (Path(t) / "mimic_test").mkdir()
            with self.assertRaisesRegex(SystemExit, "has to come first"):
                C.cmd_run(C.argparse.Namespace(open_test=True, jobs=1))

    def test_only_val_and_test(self):
        for split in ("train", "dev", ""):
            with self.assertRaises(SystemExit):
                T.run(eval_split=split)

    def test_unknown_window(self):
        with self.assertRaises(SystemExit):
            T.run(window="post")


class TestArguments(unittest.TestCase):
    def test_test_split_path_keeps_train_and_test_only(self):
        seen = {}

        def stop(d, *_):
            seen["splits"] = set(d.split)
            sys.exit("stopped")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(T, "check_test_allowed"), mock.patch.object(T, "TEST_OUT", Path(tmp)), \
                mock.patch.object(T, "load_mimic", return_value=fake_cohort(300)), mock.patch.object(T, "frame_of", side_effect=stop), self.assertRaises(SystemExit):
            T.run(eval_split="test", open_test=True, labs=("sodium",))
        self.assertEqual(seen["splits"], {"train", "test"})

    def test_centered_window_runs_and_writes_its_own_folder_name(self):
        self.assertEqual(T.tracking_stem("val", "centered"), "tracking_val_centered")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(T, "load_mimic", return_value=fake_cohort(700)) as load, mock.patch.object(T, "N_SLOW", 5):
            frames, _ = T.run("centered", "val", out=tmp, labs=("sodium",), n_boot=50, n_perm=5)
            load.assert_called_once_with("centered", open_test=False)
            self.assertTrue((Path(tmp) / "report.md").exists())
        self.assertGreater(frames["level"].query("target == 'sodium' and ppg == 'beat' and metric == 'within_r'").est.iloc[0], .5)


class TestEndToEnd(unittest.TestCase):
    """Known truth through the whole pipeline, on a fake cohort. Heavy, so run once."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with mock.patch.object(T, "load_mimic", return_value=fake_cohort()), mock.patch.object(T, "PLASMODE_LABS", ("potassium",)), \
                mock.patch.object(T, "PLASMODE_RHO", (0.5,)), mock.patch.object(T, "N_SLOW", 50):
            cls.frames, cls.fam = T.run("pre", "val", out=cls.tmp.name, labs=("sodium", "bun", "potassium"), n_boot=200, n_perm=100)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def get(self, table, **eq):
        x = self.frames[table]
        x = x[x.kind != "plasmode"] if "kind" not in eq else x
        for k, v in eq.items():
            x = x[x[k] == v]
        self.assertEqual(len(x), 1, eq)
        return x.iloc[0]

    def test_patients_with_both_labels(self):
        for ppg in ("beat", "cnn"):
            share = self.get("auroc", target="sodium", ppg=ppg, metric="share_subjects_both_labels", pop="all")
            self.assertTrue(0 < share.est < 1, ppg)
            drop = self.get("auroc", target="sodium", ppg=ppg, metric="shuffle_drop", pop="both_labels")
            pooled = self.get("auroc", target="sodium", ppg=ppg, metric="auroc_pooled", pop="both_labels")
            shuffled = self.get("auroc", target="sodium", ppg=ppg, metric="auroc_shuffled", pop="both_labels")
            self.assertAlmostEqual(drop.est, pooled.est - shuffled.est)
            self.assertGreater(drop.est, .1)      # sodium's PPG signal is within subject

    def test_a_stable_lab_has_few_patients_who_could_move_a_shuffle(self):
        stable = self.get("auroc", target="bun", ppg="beat", metric="share_subjects_both_labels", pop="all").est
        self.assertLess(stable, self.get("auroc", target="sodium", ppg="beat", metric="share_subjects_both_labels", pop="all").est)

    def test_per_patient_auroc_rows(self):
        mean = self.get("auroc", target="sodium", ppg="beat", metric="per_patient_auroc_mean")
        self.assertGreater(mean.est, .7)
        self.assertLess(mean.null_hi, mean.est)
        self.assertGreater(self.get("auroc", target="sodium", ppg="beat", metric="per_patient_auroc_above_half").est, .6)
        self.assertGreater(self.get("auroc", target="sodium", ppg="beat", metric="per_patient_auroc_sd").est, 0)
        self.assertAlmostEqual(self.get("auroc", target="potassium", ppg="beat", metric="per_patient_auroc_mean").est, .5, delta=.08)

    def test_personal_mean_comparator(self):
        r = self.get("change", target="sodium", ppg="-", method="personal_mean", metric="r", stratum="all")
        self.assertTrue(np.isfinite(r.est))
        self.assertIn("skill personal mean", (Path(self.tmp.name) / "report.md").read_text())

    def test_the_outside_zone_and_intervention_strata_reach_the_tables(self):
        for stratum in ("outside_zone", "after_intervention"):
            r = self.get("change", target="potassium", ppg="beat", method="ppg_raw", metric="r", stratum=stratum)
            self.assertLess(r.n, self.get("change", target="potassium", ppg="beat", method="ppg_raw", metric="r", stratum="all").n)
        truth = self.get("change", target="potassium", ppg="-", method="no_change", metric="mean_true_change", stratum="after_intervention")
        self.assertEqual(truth.expected_sign, -1.0)
        self.assertIn("after a treatment", (Path(self.tmp.name) / "report.md").read_text().lower())

    def test_the_cnn_has_its_rows_and_no_change_trained_or_plasmode(self):
        self.assertGreater(self.get("level", target="sodium", ppg="cnn", metric="within_r").est, .5)     # the sign follows the value although abnormal means low
        self.assertGreater(self.get("level", target="bun", ppg="cnn", metric="between_r").est, .5)
        x = self.frames["change"]
        self.assertFalse(((x.ppg == "cnn") & (x.method == "change_trained")).any())
        self.assertFalse(((x.ppg == "cnn") & (x.kind != "lab")).any())
        self.assertFalse(((self.frames["level"].ppg == "cnn") & (self.frames["level"].kind != "lab")).any())

    def test_no_sentinel_reached_any_table(self):
        for name, x in self.frames.items():
            self.assertLess(x.select_dtypes("number").drop(columns=["rho"], errors="ignore").abs().max().max(), 1e7, name)

    def test_outputs_written_with_aggregates_only(self):
        for f in ("variance", "level", "change", "stack", "auroc", "family", "plasmode"):
            x = pd.read_csv(Path(self.tmp.name) / f"{f}.csv")
            self.assertTrue(len(x))
            self.assertFalse({"group", "subject_id", "draw_id", "charttime", "hadm_id"} & set(x.columns))
        self.assertTrue((Path(self.tmp.name) / "report.md").stat().st_size > 0)

    def test_tracking_lab_shows_within_association(self):
        for ppg in ("beat", "papagei"):
            r = self.get("level", target="sodium", ppg=ppg, metric="within_r")
            self.assertGreater(r.est, 0.5, ppg)
            self.assertGreater(r.lo, r.null_hi)
            self.assertGreater(self.get("change", target="sodium", ppg=ppg, method="ppg_raw", metric="concordance", stratum="all").est, .75)
            self.assertGreater(self.get("stack", target="sodium", ppg=ppg, metric="partial_r2", base="prev_gap_now", extra="ppg_change").lo, .1)

    def test_between_only_lab_has_no_within_association_but_a_high_pooled_auroc(self):
        for ppg in ("beat", "papagei"):
            self.assertLess(abs(self.get("level", target="bun", ppg=ppg, metric="within_r").est), .12, ppg)
            self.assertGreater(self.get("level", target="bun", ppg=ppg, metric="between_r").est, .7, ppg)
            a = self.get("auroc", target="bun", ppg=ppg, metric="auroc_pooled", pop="all")
            s = self.get("auroc", target="bun", ppg=ppg, metric="auroc_shuffled", pop="all")
            self.assertGreater(a.est, .8, ppg)
            self.assertLess(abs(a.est - s.est), .02, ppg)

    def test_unrelated_lab_is_null(self):
        for ppg in ("beat", "papagei", "pulseppg"):
            r = self.get("level", target="potassium", ppg=ppg, metric="within_r")
            self.assertLess(r.lo, 0, ppg)
            self.assertGreater(r.hi, 0, ppg)
            self.assertAlmostEqual(self.get("change", target="potassium", ppg=ppg, method="ppg_raw", metric="concordance", stratum="all").est, .5, delta=.08)

    def test_control_reads_hr_and_not_map(self):
        self.assertGreater(self.get("level", target="hr_monitor", ppg="beat", metric="within_r").est, .9)
        self.assertGreater(self.get("change", target="hr_monitor", ppg="beat", method="ppg_raw", metric="concordance", stratum="all").est, .9)
        self.assertLess(abs(self.get("level", target="map_artline", ppg="beat", metric="within_r").est), .15)

    def test_hr_and_map_left_their_own_bases(self):
        d, _, specs, tiers = fake_cohort(700)
        d = d[d.split != "test"].reset_index(drop=True)
        for name, (dd, tt, spec) in T.controls(d, tiers).items():
            self.assertNotIn(spec["value"], tt["now"])
            f = T.frame_of(dd, tt, spec["value"], spec, "val", "pre")
            y = f.y[f.ev]
            for col in f.bases["now"].T:
                keep = np.isfinite(col[f.ev])
                if col[f.ev][keep].std() > 0:
                    self.assertLess(abs(np.corrcoef(col[f.ev][keep], y[keep])[0, 1]), .5, name)

    def test_plasmode_recovers_a_planted_link(self):
        for ppg in ("beat",):
            r = self.get("level", kind="plasmode", target="potassium", rho=.5, ppg=ppg, metric="within_r")
            self.assertGreater(r.est, .3)
        self.assertGreater(self.get("change", kind="plasmode", target="potassium", rho=.5, ppg="-", method="oracle_hr", metric="r", stratum="all").est, .3)

    def test_paired_difference_and_family_means(self):
        d = self.get("level", target="bun", ppg="beat", metric="between_minus_within")
        w, b = self.get("level", target="bun", ppg="beat", metric="within_r"), self.get("level", target="bun", ppg="beat", metric="between_r")
        self.assertAlmostEqual(d.est, b.est - w.est, places=10)
        x = self.fam[(self.fam.ppg == "beat") & (self.fam.metric == "within_r")].iloc[0]
        labs = self.frames["level"]
        self.assertAlmostEqual(x.mean_est, labs[(labs.kind == "lab") & (labs.ppg == "beat") & (labs.metric == "within_r")].est.mean(), places=10)
        self.assertLess(x.lo, x.mean_est)
        self.assertGreater(x.hi, x.mean_est)

    def test_the_bases_hold_the_labs_time_and_count_but_the_controls_do_not(self):
        d, _, specs, tiers = fake_cohort(700)
        d = d[d.split != "test"].reset_index(drop=True)
        flexible = 9
        f = T.frame_of(d, tiers, "sodium", specs["sodium"], "val", "pre")
        self.assertEqual(f.bases["now"].shape[1], 2 + len(tiers["now"]) + flexible + 2)
        np.testing.assert_allclose(f.bases["now"][:, -2], np.log1p(f.q.sodium_locf_age_h))
        self.assertEqual(f.bases["full"].shape[1], f.bases["now"].shape[1] + 3)
        for name, (dd, tt, spec) in T.controls(d, tiers).items():
            c = T.frame_of(dd, tt, spec["value"], spec, "val", "pre")
            extra = 0 if name == "hr_monitor" else -flexible + 4       # without MAP, flexible holds the age hinges and female x age
            self.assertEqual(c.bases["now"].shape[1], 2 + len(tt["now"]) + flexible + extra, name)

    def test_pairs_are_consecutive_and_dropped_when_they_share_a_measurement(self):
        d, _, specs, tiers = fake_cohort(700)
        d = d[d.split != "test"].reset_index(drop=True)
        f = T.add_pairs(T.frame_of(d, tiers, "sodium", specs["sodium"], "val", "pre"))
        self.assertTrue((f.groups[f.prev] == f.groups[f.cur]).all() and (f.gap > 0).all())
        n = len(f.prev)
        f.lab_t = f.lab_t.copy()
        f.lab_t[f.cur[0]] = f.lab_t[f.prev[0]]
        self.assertEqual(len(T.add_pairs(f).prev), n - 1)
        self.assertEqual(f.dropped, 1)




class TestAdditions(unittest.TestCase):
    def test_per_patient_auc_by_hand(self):
        # subject 0: abnormal .9 against normal .1 and .95: 1 of 2. Subject 1: abnormal .4 against normal .4: a tie, one half
        g, x = np.array([0, 0, 0, 1, 1]), np.array([.9, .1, .95, .4, .4])
        i, j = np.array([0, 0, 3]), np.array([1, 2, 4])
        used, a = T.per_patient_auc(g, i, j, x, 3)
        self.assertEqual(used.tolist(), [True, True, False])
        self.assertEqual(a[:2].tolist(), [.5, .5])

    def test_exclusion_zones_by_hand(self):
        self.assertEqual(set(T.ANALYTICAL_CV), set(C.FAMILY_LABS))
        self.assertAlmostEqual(T.exclusion_zone("sodium", np.array([140.]))[0], 1.96 * np.sqrt(2) * .003 * 140)      # value scale: about 1.2 mmol/L
        self.assertAlmostEqual(T.exclusion_zone("glucose", np.array([5.]))[0], np.log1p(1.96 * np.sqrt(2) * .028))   # log scale: the same whatever the level
        self.assertIsNone(T.exclusion_zone("hr_mon", np.array([80.])))

    def test_abnormal_direction(self):
        self.assertEqual(T.abnormal_direction("sodium"), -1.0)
        self.assertEqual(T.abnormal_direction("haemoglobin"), -1.0)
        self.assertEqual(T.abnormal_direction("egfr"), -1.0)
        self.assertEqual(T.abnormal_direction("potassium"), 1.0)
        self.assertEqual(T.abnormal_direction("glucose"), 1.0)

    def test_interventions_are_the_ones_the_stored_flags_allow(self):
        self.assertEqual(set(T.INTERVENTIONS), {"haemoglobin", "potassium"})
        self.assertEqual(T.INTERVENTIONS["haemoglobin"], ("prbc_24h", 1.0))
        self.assertEqual(T.INTERVENTIONS["potassium"], ("dialysis_fluid_24h", -1.0))


class TestFaults(unittest.TestCase):
    """Tracking runs on test rows after the opening. A part that raises is recorded, what it sits beside stays, and the tables are written before the report."""

    def go(self, out, *patches):
        with contextlib.ExitStack() as stack:
            for p in (mock.patch.object(T, "load_mimic", return_value=fake_cohort(subjects=300)), mock.patch.object(T, "PLASMODE_LABS", ()), *patches):
                stack.enter_context(p)
            with contextlib.redirect_stderr(io.StringIO()):
                T.run("pre", "val", out=out, labs=("sodium", "potassium"), n_boot=20, n_perm=2)
        return json.loads((Path(out) / "manifest.json").read_text())["counts"]

    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.out = t.name

    def boom(self, name):
        return mock.patch.object(T, name, side_effect=RuntimeError("boom 3.2"))

    def test_each_part_of_a_model_fails_alone(self):
        for name, kept, lost in (("auroc_rows", "level", "auroc"), ("stack_rows", "change", "stack"), ("level_rows", "variance", "level")):
            with self.subTest(name):
                c = self.go(self.out, self.boom(name))
                failed = c["targets"]["sodium"]["failed"]
                self.assertEqual(set(failed), set(T.MODELS))
                self.assertIn("RuntimeError", failed["beat"])
                self.assertNotIn("3.2", json.dumps(c["failed"]))
                self.assertGreater(len(pd.read_csv(Path(self.out) / f"{kept}.csv")), 0)
                self.assertTrue((Path(self.out) / "report.md").exists())

    def test_the_comparators_fail_alone(self):
        c = self.go(self.out, self.boom("variance_rows"))
        self.assertEqual(list(c["targets"]["sodium"]["failed"]), ["comparators"])
        self.assertGreater(len(pd.read_csv(Path(self.out) / "level.csv")), 0)

    def test_a_target_fails_alone(self):
        frame_of = T.frame_of

        def not_sodium(d, tiers, name, *args):
            if name == "sodium":
                raise RuntimeError("boom")
            return frame_of(d, tiers, name, *args)
        c = self.go(self.out, mock.patch.object(T, "frame_of", not_sodium))
        self.assertIn("target", c["failed"]["sodium"])
        self.assertIn("potassium", set(pd.read_csv(Path(self.out) / "level.csv").target))
        self.assertNotIn("sodium", set(pd.read_csv(Path(self.out) / "level.csv").target))
        self.assertTrue((Path(self.out) / "report.md").exists())

    def test_nothing_computed_writes_no_folder_so_the_once_only_rule_does_not_close(self):
        folder = Path(self.out) / "fresh"
        with self.assertRaisesRegex(SystemExit, "computed nothing") as e:
            self.go(folder, self.boom("frame_of"))
        self.assertIn("RuntimeError", str(e.exception))
        self.assertNotIn("3.2", str(e.exception))
        self.assertFalse(folder.exists())

    def test_the_tests_cannot_reach_the_real_results_folder(self):
        self.assertEqual(T.OUT, Path(_out.name))

    def test_the_family_table_fails_alone(self):
        c = self.go(self.out, self.boom("family_table"))
        self.assertIn("family", c["failed"])
        self.assertGreater(len(pd.read_csv(Path(self.out) / "level.csv")), 0)
        self.assertTrue((Path(self.out) / "report.md").exists())

    def test_a_failed_report_keeps_the_tables_and_can_be_rendered_again(self):
        self.go(self.out, self.boom("report"))
        folder = Path(self.out)
        self.assertFalse((folder / "report.md").exists())
        self.assertIn("--render", (folder / "report_failed.txt").read_text())
        self.assertNotIn("3.2", (folder / "report_failed.txt").read_text())
        for t in ("variance", "level", "change", "stack", "auroc", "plasmode", "family"):
            self.assertTrue((folder / f"{t}.csv").exists(), t)
        with mock.patch.object(T, "load_mimic", side_effect=AssertionError("data was read")):
            T.render(self.out)
        self.assertTrue((folder / "report.md").exists())
        self.assertFalse((folder / "report_failed.txt").exists())
        self.assertIn("Against no change", (folder / "report.md").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=1)
