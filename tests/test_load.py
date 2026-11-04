"""What baselines reads from the feature files, on a small generated data folder. No restricted data is needed.

Usage:  python -m unittest tests.test_load
"""
import contextlib
import hashlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ppglabbench import baselines as BL
from ppglabbench import labels, tune_beat

WINDOWS = ["w0", "w1", "w2", "w3", "w4", "w5"]
PRE_SCORE = [.9, .9, .9, .9, .9, .5]     # the last window has no usable window before the draw


def make_folder(root):
    """Every file load_mimic reads, for six windows of six subjects."""
    bench = root / "benchmark"
    (bench / "features" / "mimic_cov").mkdir(parents=True)
    t = pd.DataFrame(dict(draw_id=WINDOWS, subject_id=np.arange(6), pre_score=PRE_SCORE, pre_start_s=-300, charttime=[f"2000-01-0{i + 1} 04:00:00" for i in range(6)], record_overlap=False,
                          vasopressor=0.0, emergency_admit=1.0, split="train", exclude_dialysis=False, egfr=70.0, egfr_lt60=0.0, creat_mgdl=1.0, creat_locf=1.0, creat_locf_age_h=5.0))
    t.to_csv(bench / "mimic_labels.csv", index=False)
    pd.DataFrame(dict(subject_id=np.arange(6), split="train", dev_set=False)).to_csv(bench / "mimic_splits.csv", index=False)
    history = {c: 0.0 for c in ("prior_admissions", "days_since_discharge", *BL.PRIOR_DX)}
    pd.DataFrame(dict(draw_id=WINDOWS, on_sedation=0.0, **history)).to_csv(bench / "features" / "mimic_cov" / "covariates.csv", index=False)
    for w, ids in (("pre", WINDOWS[:5]), ("centered", WINDOWS)):
        folder = bench / "features" / w
        (folder / "mimic").mkdir(parents=True)
        pd.DataFrame(dict(draw_id=ids, n_beats=30.0, hr=70.0)).to_csv(folder / "mimic" / "beat_features.csv", index=False)
        (folder / "mimic" / "manifest.json").write_text(json.dumps(dict(params=dict(search_end=BL.SEARCH_END, min_notch=BL.MIN_NOTCH))))
        np.savez(folder / "mimic" / "papagei.npz", ids=np.array(ids, dtype=object), emb=np.ones((len(ids), 512)))
        (folder / "mimic_abp").mkdir()
        pd.DataFrame(dict(draw_id=ids, subject_id=0, map_mmhg=80.0, error=np.nan)).to_csv(folder / "mimic_abp" / "abp_map.csv", index=False)
        (folder / "mimic_num").mkdir()
        num = pd.DataFrame({c: 12.0 for c in BL.MONITOR}, index=range(len(ids))).assign(draw_id=ids, subject_id=0, error=np.nan)
        num.to_csv(folder / "mimic_num" / "numerics.csv", index=False)
        for sub in ("mimic_abp", "mimic_num"):
            (folder / sub / "manifest.json").write_text(json.dumps(dict(counts=dict(complete=True))))


class LoadCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        make_folder(self.root)
        for p in (mock.patch.object(BL, "DATA", self.root), mock.patch.object(BL, "FEAT", self.root / "benchmark" / "features"),
                  mock.patch.object(labels, "DATA", self.root), mock.patch.object(labels, "OUT", self.root / "benchmark")):
            p.start()
            self.addCleanup(p.stop)
        self.feat = self.root / "benchmark" / "features"

    def table(self, window, sub, name):
        return self.feat / window / sub / name


class TestCompleteness(LoadCase):
    TABLES = (("mimic_abp", "abp_map.csv"), ("mimic_num", "numerics.csv"))

    def test_whole_tables_load(self):
        for window, n in (("pre", 5), ("centered", 6)):
            self.assertEqual(len(BL.load_mimic(window)[0]), n)

    def test_a_missing_row_stops_it_although_the_manifest_says_complete(self):
        for sub, name in self.TABLES:
            f = self.table("pre", sub, name)
            saved = f.read_bytes()
            t = pd.read_csv(f)
            t[t.draw_id != "w2"].to_csv(f, index=False)
            with self.subTest(sub), self.assertRaises(SystemExit) as e:
                BL.load_mimic("pre")
            self.assertIn("1 of 5", str(e.exception))
            self.assertIn("--window pre", str(e.exception))
            self.assertNotIn("w2", str(e.exception).replace(str(self.root), ""))     # counts only: a window is never named
            f.write_bytes(saved)

    def test_a_row_that_failed_counts_as_missing(self):
        for sub, name in self.TABLES:
            f = self.table("centered", sub, name)
            saved = f.read_bytes()
            t = pd.read_csv(f)
            t["error"] = t.error.astype(object)
            t.loc[t.draw_id == "w5", "error"] = "502"
            t.to_csv(f, index=False)
            with self.subTest(sub), self.assertRaisesRegex(SystemExit, "1 of 6"):
                BL.load_mimic("centered")
            f.write_bytes(saved)

    def test_a_repeated_row_stops_it(self):
        f = self.table("pre", "mimic_abp", "abp_map.csv")
        t = pd.read_csv(f)
        pd.concat([t, t.iloc[:1]]).to_csv(f, index=False)
        with self.assertRaisesRegex(SystemExit, "more than one row"):
            BL.load_mimic("pre")

    def test_the_manifest_still_counts(self):
        f = self.table("pre", "mimic_num", "manifest.json")
        f.write_text(json.dumps(dict(counts=dict(complete=False))))
        with self.assertRaisesRegex(SystemExit, "missing or incomplete"):
            BL.load_mimic("pre")
        f.unlink()
        with self.assertRaisesRegex(SystemExit, "missing or incomplete"):
            BL.load_mimic("pre")

    def test_an_absent_table_stops_it(self):
        self.table("centered", "mimic_abp", "abp_map.csv").unlink()
        with self.assertRaisesRegex(SystemExit, "missing or incomplete"):
            BL.load_mimic("centered")

    def test_windows_the_labels_do_not_expect_may_be_extra(self):
        f = self.table("pre", "mimic_abp", "abp_map.csv")
        t = pd.read_csv(f)
        pd.concat([t, t.iloc[:1].assign(draw_id="other")]).to_csv(f, index=False)
        self.assertEqual(len(BL.load_mimic("pre")[0]), 5)

    def test_released_tables_have_no_error_column(self):
        for sub, name in self.TABLES:
            f = self.table("pre", sub, name)
            pd.read_csv(f).drop(columns="error").to_csv(f, index=False)
        self.assertEqual(len(BL.load_mimic("pre")[0]), 5)


class TestLoadedValues(LoadCase):
    def test_resp_of_exactly_zero_is_missing(self):
        f = self.table("pre", "mimic_num", "numerics.csv")
        t = pd.read_csv(f)
        t["resp_mon"] = [0.0, 0.2, np.nan, 18.0, 25.0]
        t.to_csv(f, index=False)
        d = BL.load_mimic("pre")[0].set_index("draw_id")
        self.assertTrue(np.isnan(d.resp_mon["w0"]) and np.isnan(d.resp_mon["w2"]))
        self.assertEqual(list(d.resp_mon[["w1", "w3", "w4"]]), [.2, 18., 25.])

    def test_map_comes_from_the_waveform_table_alone(self):
        f = self.table("pre", "mimic_abp", "abp_map.csv")
        t = pd.read_csv(f)
        t["map_mmhg"] = [65.0, np.nan, 80.0, 80.0, 80.0]
        t.to_csv(f, index=False)
        d = BL.load_mimic("pre")[0].set_index("draw_id")
        self.assertEqual(d.map_lt70["w0"], 1.0)
        self.assertTrue(np.isnan(d.map_mmhg["w1"]) and np.isnan(d.map_lt70["w1"]))
        self.assertEqual(d.map_lt70["w2"], 0.0)

    def test_monitor_has_its_eight_columns(self):
        self.assertEqual(len(BL.MONITOR), 8)
        tiers = BL.load_mimic("pre")[3]
        self.assertEqual(tiers["monitor"], tiers["acuity"] + BL.MONITOR)


class TestPulsePPG(LoadCase):
    def write(self, folder, ids, seconds):
        emb = np.arange(len(ids) * 512, dtype=np.float32).reshape(len(ids), 512) + 1
        np.savez(folder / "pulseppg.npz", ids=np.array(ids, dtype=object), emb=emb, seconds=np.array(seconds, float))
        return emb

    def test_under_240_seconds_is_missing(self):
        d = pd.DataFrame(dict(draw_id=["w0", "w1", "w2", "w3", "w4"]))
        emb = self.write(self.root, ["w0", "w1", "w2", "w3"], [240., 239.98, 30., 0.])
        got = BL.add_pulseppg(d, self.root, "draw_id").set_index("draw_id")[BL.PP]
        np.testing.assert_array_equal(got.loc["w0"].to_numpy(float), emb[0])
        self.assertTrue(got.loc[["w1", "w2", "w3", "w4"]].isna().all().all())

    def test_integer_case_ids(self):
        d = pd.DataFrame(dict(caseid=[7, 8]))
        emb = self.write(self.root, ["7", "8"], [240., 100.])
        got = BL.add_pulseppg(d, self.root, "caseid").set_index("caseid")[BL.PP]
        np.testing.assert_array_equal(got.loc[7].to_numpy(float), emb[0])
        self.assertTrue(got.loc[8].isna().all())

    def test_no_file_changes_nothing(self):
        d = pd.DataFrame(dict(draw_id=["w0"]))
        self.assertIs(BL.add_pulseppg(d, self.root, "draw_id"), d)

    def test_the_rule_reaches_the_cohort_and_the_window_stays_in_it(self):
        self.write(self.feat / "pre" / "mimic", WINDOWS[:5], [240., 100., 240., 240., 240.])
        d = BL.load_mimic("pre")[0].set_index("draw_id")
        self.assertEqual(len(d), 5)
        self.assertTrue(np.isnan(d.pp0["w1"]))
        self.assertTrue(np.isfinite(d.pp0["w0"]))

    def test_the_probe_imputes_them(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 80))
        X[:20] = np.nan
        y = (rng.random(300) < .4).astype(int)
        p = BL.probe_clf().fit(X, y).predict_proba(X)[:, 1]
        self.assertTrue(np.isfinite(p).all())

    def test_the_threshold_is_the_models_longest_input(self):
        """Read from the source: torch and this module live in different environments."""
        text = (Path(BL.__file__).parent / "features" / "pulseppg.py").read_text()
        self.assertEqual(int(re.search(r"^MAX_S, MIN_S = (\d+), \d+$", text, re.M).group(1)), BL.PULSEPPG_MIN_S)


if __name__ == "__main__":
    unittest.main()


class TestStaleFiles(LoadCase):
    """load_mimic refuses files older than the code that reads them, and says which and what to run."""

    def manifest(self, window="pre"):
        return self.table(window, "mimic", "manifest.json")

    def test_a_fresh_folder_loads(self):
        self.assertEqual(len(BL.load_mimic("pre")[0]), 5)

    def test_a_beat_manifest_without_the_settings_stops_it(self):
        self.manifest().write_text(json.dumps(dict(params=dict(nb=100))))
        with self.assertRaisesRegex(SystemExit, "features.beat mimic pre"):
            BL.load_mimic("pre")

    def test_another_search_bound_or_notch_depth_stops_it(self):
        for params in (dict(search_end=BL.SEARCH_END + 5, min_notch=BL.MIN_NOTCH), dict(search_end=BL.SEARCH_END, min_notch=BL.MIN_NOTCH * 2)):
            self.manifest().write_text(json.dumps(dict(params=params)))
            with self.assertRaisesRegex(SystemExit, "search_end"):
                BL.load_mimic("pre")

    def test_only_the_window_read_is_checked(self):
        self.manifest("centered").write_text(json.dumps(dict(params=dict())))
        self.assertEqual(len(BL.load_mimic("pre")[0]), 5)
        with self.assertRaisesRegex(SystemExit, "features.beat mimic centered"):
            BL.load_mimic("centered")

    def test_a_missing_beat_manifest_stops_it(self):
        self.manifest().unlink()
        with self.assertRaisesRegex(SystemExit, "search_end"):
            BL.load_mimic("pre")

    def test_covariates_without_history_stop_it(self):
        f = self.feat / "mimic_cov" / "covariates.csv"
        pd.read_csv(f).drop(columns=["prior_liver"]).to_csv(f, index=False)
        with self.assertRaisesRegex(SystemExit, "prior_liver.*features.covariates"):
            BL.load_mimic("pre")

    def lab_cache(self, name="labs_bench_v4"):
        (self.root / "clinical").mkdir(exist_ok=True)
        f = self.root / "clinical" / f"{name}.csv"
        f.write_text("x\n")
        return f

    def test_labels_older_than_the_lab_cache_stop_it(self):
        self.assertEqual(labels.LAB_CACHE, "labs_bench_v4")
        f = self.lab_cache()
        table = self.root / "benchmark" / "mimic_labels.csv"
        os.utime(f, (table.stat().st_mtime + 60, table.stat().st_mtime + 60))
        with self.assertRaisesRegex(SystemExit, "older than the lab cache"):
            BL.load_mimic("pre")

    def test_labels_newer_than_the_lab_cache_load(self):
        f = self.lab_cache()
        table = self.root / "benchmark" / "mimic_labels.csv"
        os.utime(f, (table.stat().st_mtime - 60, table.stat().st_mtime - 60))
        self.assertEqual(len(BL.load_mimic("pre")[0]), 5)

    def test_only_an_older_lab_cache_stops_it(self):
        self.lab_cache("labs_bench_v3")
        with self.assertRaisesRegex(SystemExit, "older lab cache"):
            BL.load_mimic("pre")

    def test_a_folder_without_clinical_tables_has_nothing_to_compare(self):
        self.assertFalse((self.root / "clinical").exists())
        self.assertEqual(len(BL.load_mimic("pre")[0]), 5)

    def test_the_messages_hold_no_identifier(self):
        self.manifest().unlink()
        with self.assertRaises(SystemExit) as e:
            BL.load_mimic("pre")
        self.assertFalse(set(WINDOWS) & set(re.findall(r"\bw\d\b", str(e.exception))))


class TestTuneBeatJobs(unittest.TestCase):
    def test_jobs_reach_the_pool(self):
        d = pd.DataFrame(dict(group=np.repeat(np.arange(30), 2), split="train", x=np.arange(60.0)))
        d["v"], d["a"] = np.arange(60.0) % 7, (np.arange(60) % 2).astype(float)
        seen = []

        class Pool:
            def __init__(self, n_jobs):
                seen.append(n_jobs)

            def __call__(self, jobs):
                return [fn(*a, **k) for fn, a, k in jobs]

        with mock.patch.object(tune_beat, "load_mimic", return_value=(d, ["x"], dict(lab=dict(value="v", abnormal="a")), None)), \
                mock.patch.object(tune_beat, "Parallel", Pool), mock.patch.object(tune_beat, "GRID", (5, 10)):
            with contextlib.redirect_stdout(io.StringIO()):
                tune_beat.main(["--jobs", "2"])
                tune_beat.main([])
        self.assertEqual(seen, [2, 5])


class TestCnnFile(LoadCase):
    """The CNN's score file: a synthetic one. Splits: w0-w2 train, w3 val, w4 test; two folds here."""

    def setUp(self):
        super().setUp()
        k = mock.patch.object(BL, "K", 2)
        k.start()
        self.addCleanup(k.stop)
        f = self.root / "benchmark" / "mimic_labels.csv"
        t = pd.read_csv(f)
        t["split"] = ["train", "train", "train", "val", "test", "train"]
        t.to_csv(f, index=False)
        self.folder = self.feat / "pre" / "mimic_cnn"
        self.folder.mkdir()

    def write(self, ids=("w0", "w1", "w2", "w3", "w4"), fold=(0, 1, 0, -1, -1), manifest=True):
        f = self.folder / "cnn_scores.npz"
        np.savez(f, draw_id=np.array(ids, dtype="U"), targets=np.array(["bun", "egfr"], dtype="U"), score=np.column_stack([np.arange(len(ids), dtype=float), -np.arange(len(ids), dtype=float)]),
                 fold=np.array(fold), age_pred=np.full(len(ids), 60.), hr_pred=np.full(len(ids), 80.))
        if manifest:
            (self.folder / "manifest.json").write_text(json.dumps(dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest())))

    def test_columns_load(self):
        self.write()
        d = BL.load_mimic("pre")[0].set_index("draw_id")
        self.assertEqual(d.loc["w2", "cnn_bun"], 2.0)
        self.assertEqual(d.loc["w2", "cnn_egfr"], -2.0)
        self.assertEqual(d.loc["w1", "cnn_fold"], 1)
        self.assertEqual(d.loc["w3", "cnn_fold"], -1)
        self.assertEqual(d.loc["w0", "cnn_age_pred"], 60.0)

    def test_test_rows_hold_no_score_until_the_split_is_opened(self):
        self.write()
        d = BL.load_mimic("pre")[0].set_index("draw_id")
        self.assertTrue(np.isnan(d.loc["w4", ["cnn_bun", "cnn_egfr", "cnn_fold", "cnn_age_pred", "cnn_hr_pred"]].to_numpy(float)).all())
        self.assertEqual(d.loc["w3", "cnn_bun"], 3.0)
        d = BL.load_mimic("pre", open_test=True)[0].set_index("draw_id")
        self.assertEqual(d.loc["w4", "cnn_bun"], 4.0)

    def test_a_missing_file_is_only_a_note_unless_it_is_needed(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            d = BL.load_mimic("pre")[0]
        self.assertFalse([c for c in d if c.startswith("cnn_")])
        self.assertIn("the cnn model is left out", out.getvalue())
        with self.assertRaisesRegex(SystemExit, "cnn_scores.npz does not exist"):
            BL.load_mimic("pre", need_cnn=True)

    def test_another_window_has_no_cnn_and_says_so(self):
        self.write()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            d = BL.load_mimic("centered", need_cnn=True)[0]
        self.assertFalse([c for c in d if c.startswith("cnn_")])
        self.assertIn("window centered has no CNN", out.getvalue())

    def test_a_file_that_is_not_the_one_its_manifest_names_stops_it(self):
        self.write()
        (self.folder / "manifest.json").write_text(json.dumps(dict(sha256="0" * 64)))
        with self.assertRaisesRegex(SystemExit, "not the file its manifest names"):
            BL.load_mimic("pre")
        (self.folder / "manifest.json").unlink()
        with self.assertRaisesRegex(SystemExit, "not the file its manifest names"):
            BL.load_mimic("pre")

    def test_a_file_older_than_the_labels_stops_it(self):
        self.write()
        later = (self.folder / "cnn_scores.npz").stat().st_mtime + 60
        os.utime(self.root / "benchmark" / "mimic_labels.csv", (later, later))
        with self.assertRaisesRegex(SystemExit, "older than mimic_labels.csv"):
            BL.load_mimic("pre")

    def test_the_sha256_is_of_the_file(self):
        self.assertIsNone(BL.cnn_sha256("pre"))
        self.write()
        self.assertEqual(BL.cnn_sha256("pre"), hashlib.sha256((self.folder / "cnn_scores.npz").read_bytes()).hexdigest())

    def test_a_window_without_a_score_stops_it(self):
        self.write(ids=("w0", "w1", "w2", "w4"), fold=(0, 1, 0, -1))
        with self.assertRaisesRegex(SystemExit, "1 of 4 windows are not in the file"):
            BL.load_mimic("pre")

    def test_a_score_missing_on_a_test_row_says_nothing_before_opening(self):
        self.write(ids=("w0", "w1", "w2", "w3"), fold=(0, 1, 0, -1))
        self.assertEqual(len(BL.load_mimic("pre")[0]), 5)
        with self.assertRaisesRegex(SystemExit, "not in the file"):
            BL.load_mimic("pre", open_test=True)

    def test_folds_are_checked(self):
        for fold, message in (((0, -1, 0, -1, -1), "train windows have no fold"), ((0, 1, 0, 1, -1), "windows outside train have a fold"), ((0, 0, 0, -1, -1), "folds on train, not 2")):
            self.write(fold=fold)
            with self.assertRaisesRegex(SystemExit, message):
                BL.load_mimic("pre")

    def test_rows_outside_the_cohort_do_not_count(self):
        """The file holds every window with a usable score: a train row outside the cohort may have fold -1 and a final-model score."""
        self.write(ids=("w0", "w1", "w2", "w3", "w4", "x8", "x9"), fold=(0, 1, 0, -1, -1, -1, 1))
        d = BL.load_mimic("pre")[0]
        self.assertEqual(len(d), 5)
        self.assertEqual(int((d.split == "train").sum()), 3)

    def test_a_final_model_score_on_a_cohort_train_row_stops_it(self):
        self.write(fold=(0, 1, -1, -1, -1))
        with self.assertRaisesRegex(SystemExit, "1 train windows have no fold"):
            BL.load_mimic("pre")

    def test_a_subject_in_two_folds_stops_it(self):
        f = self.root / "benchmark" / "mimic_labels.csv"
        t = pd.read_csv(f)
        t["subject_id"] = [0, 0, 1, 2, 3, 4]
        t.to_csv(f, index=False)
        self.write()
        with self.assertRaisesRegex(SystemExit, "1 subjects are in more than one fold"):
            BL.load_mimic("pre")

    def test_the_messages_hold_no_identifier(self):
        self.write(ids=("w0", "w1", "w2", "w4"), fold=(0, 1, 0, -1))
        with self.assertRaises(SystemExit) as e:
            BL.load_mimic("pre")
        self.assertFalse(re.findall(r"\bw\d\b", str(e.exception)))


class TestCnnScore(unittest.TestCase):
    def test_the_logit_is_the_score_and_folds_are_centred_by_hand(self):
        # train y 1,0,0,1,1,0: mean .5. Fold 0 holds rows 0-2, whose training rows are 1,1,0 with mean 2/3: shift logit(.5) - logit(2/3). Fold 1: mean 1/3, shift logit(.5) - logit(1/3)
        x = np.column_stack([np.array([.1, .2, .3, .4, .5, .6, 9., 8.]), np.array([0, 0, 0, 1, 1, 1, -1, -1])])
        y = np.array([1, 0, 0, 1, 1, 0, 0, 1])
        tr, ev = np.arange(6), np.array([6, 7])
        s_tr, s_ev = BL.ppg_score(BL.cnn_clf, x, y, np.arange(8), tr, ev)
        shift = np.log(2)
        np.testing.assert_allclose(s_tr, [.1 - shift, .2 - shift, .3 - shift, .4 + shift, .5 + shift, .6 + shift])
        np.testing.assert_allclose(s_ev, [9., 8.])

    def test_a_window_with_no_usable_signal_takes_the_train_median(self):
        x = np.column_stack([np.array([1., 2., 3., np.nan, 7., np.nan]), np.array([0, 0, 1, 1, -1, -1])])
        for model in (BL.cnn_clf, BL.cnn_reg):
            s_tr, s_ev = BL.ppg_score(model, x, np.array([0, 1, 0, 1, 0, 1]), np.arange(6), np.arange(4), np.array([4, 5]))
            self.assertTrue(np.isfinite(s_tr).all())
            self.assertEqual(s_ev.tolist(), [7., 2.])
        self.assertTrue(np.isnan(x[3, 0]))      # the caller's array is left as it was

    def test_a_logit_outside_the_clip_of_the_other_scores_is_clipped(self):
        x = np.column_stack([np.array([20., 1., 0., 1., -20.]), np.array([0, 0, 1, 1, -1])])
        s_tr, s_ev = BL.ppg_score(BL.cnn_clf, x, np.array([1, 0, 1, 0, 1]), np.arange(5), np.arange(4), np.array([4]))
        self.assertAlmostEqual(s_ev[0], BL.clip_logit(0.0))
        self.assertAlmostEqual(s_tr[0], BL.clip_logit(1.0))

    def test_a_stack_on_the_value_takes_the_logit_as_it_is(self):
        x = np.column_stack([np.arange(8.), np.array([0, 0, 0, 1, 1, 1, -1, -1])])
        s_tr, s_ev = BL.ppg_score(BL.cnn_reg, x, np.arange(8.) * 3, np.arange(8), np.arange(6), np.array([6, 7]))
        np.testing.assert_allclose(s_tr, np.arange(6.))
        np.testing.assert_allclose(s_ev, [6., 7.])

    def test_rows_on_the_wrong_side_of_the_split_stop_it(self):
        x = np.column_stack([np.zeros(4), np.array([0, -1, -1, 1])])
        for tr, ev in ((np.array([0, 1]), np.array([2])), (np.array([0]), np.array([3]))):
            with self.assertRaisesRegex(SystemExit, "CNN scores"):
                BL.ppg_score(BL.cnn_clf, x, np.array([0, 1, 0, 1]), np.arange(4), tr, ev)

    def test_the_cnn_is_in_the_reference_models_for_t2_only(self):
        models = {m: (cols, family) for m, cols, family in BL.reference_models(["b0"], "bun")}
        self.assertEqual(models["cnn"][0], ["cnn_bun", "cnn_fold"])
        self.assertEqual(set(models["cnn"][1]), {"clf"})
