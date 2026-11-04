"""Tests of ppglabbench.evaluate on generated data. No data/ folder is needed and no restricted data is read.

Usage:  python -m unittest tests.test_evaluate
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

from ppglabbench import baselines as BL
from ppglabbench import common
from ppglabbench import evaluate as E
from ppglabbench import positive_control as PC


class TmpCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def npz(self, **arrays):
        path = self.tmp / "m.npz"
        np.savez(path, **arrays)
        return path


class TestReadEmbeddings(TmpCase):
    IDS = np.array(["a_1", "a_2", "b_1"])
    EMB = np.arange(6.).reshape(3, 2)

    def refused(self, pattern, **arrays):
        with self.assertRaisesRegex(E.EvaluateError, pattern):
            E.read_embeddings(self.npz(**arrays))

    def test_good_file(self):
        ids, emb = E.read_embeddings(self.npz(ids=self.IDS, emb=self.EMB.astype(np.float32)))
        self.assertEqual(ids.tolist(), self.IDS.tolist())
        self.assertEqual(emb.dtype, float)

    def test_a_row_of_nan_means_not_embedded(self):
        emb = self.EMB.copy()
        emb[1] = np.nan
        self.assertTrue(np.isnan(E.read_embeddings(self.npz(ids=self.IDS, emb=emb))[1][1]).all())

    def test_no_such_file(self):
        with self.assertRaisesRegex(E.EvaluateError, "no such file"):
            E.read_embeddings(self.tmp / "absent.npz")

    def test_not_an_npz(self):
        (self.tmp / "x.npz").write_bytes(b"not a zip file at all")
        with self.assertRaisesRegex(E.EvaluateError, "not a readable"):
            E.read_embeddings(self.tmp / "x.npz")

    def test_an_npy_file(self):
        np.save(self.tmp / "x.npy", self.EMB)
        with self.assertRaisesRegex(E.EvaluateError, "not an .npz"):
            E.read_embeddings(self.tmp / "x.npy")

    def test_missing_keys(self):
        self.refused(r"missing key\(s\) \['emb'\]; found \['ids'\]", ids=self.IDS)
        self.refused(r"missing key\(s\) \['ids'\]", emb=self.EMB, other=self.EMB)

    def test_a_pickle_is_not_loaded(self):
        self.refused("object array", ids=np.array(["a", "b", "c"], dtype=object), emb=self.EMB)

    def test_dimensions(self):
        self.refused("`emb` must be 2-D", ids=self.IDS, emb=self.EMB[:, 0])
        self.refused("`emb` must be 2-D", ids=self.IDS, emb=np.zeros((3, 2, 2)))
        self.refused("`ids` must be 1-D", ids=self.IDS[:, None], emb=self.EMB)
        self.refused("3 rows and `emb` has 2", ids=self.IDS, emb=self.EMB[:2])
        self.refused("empty", ids=self.IDS, emb=np.zeros((3, 0)))

    def test_types(self):
        self.refused("`ids` must hold strings or integers", ids=np.array([1.5, 2.5, 3.5]), emb=self.EMB)
        self.refused("`emb` must be numeric", ids=self.IDS, emb=np.array([["x", "y"]] * 3))

    def test_duplicates(self):
        self.refused("1 duplicates", ids=np.array(["a", "a", "b"]), emb=self.EMB)

    def test_values(self):
        inf, part, none = self.EMB.copy(), self.EMB.copy(), np.full((3, 2), np.nan)
        inf[0, 0], part[2, 1] = np.inf, np.nan
        self.refused("infinite", ids=self.IDS, emb=inf)
        self.refused("1 rows partly NaN", ids=self.IDS, emb=part)
        self.refused("NaN in every row", ids=self.IDS, emb=none)

    def test_id_format_per_source(self):
        self.assertEqual(E.normalise_ids(np.array([1, 2]), "vitaldb").tolist(), [1, 2])
        self.assertEqual(E.normalise_ids(np.array(["1", "2"]), "vitaldb").tolist(), [1, 2])
        with self.assertRaisesRegex(E.EvaluateError, "integer caseid"):
            E.normalise_ids(np.array(["x1", "2"]), "vitaldb")
        with self.assertRaisesRegex(E.EvaluateError, "draw_id strings"):
            E.normalise_ids(np.array([1, 2]), "mimic")

    def test_names(self):
        E.check_name("my-model_2")
        for bad in ("a b", "x+y", "a|b", ""):
            with self.assertRaisesRegex(E.EvaluateError, "letters, digits"):
                E.check_name(bad)
        for taken in ("now", "beat", "papagei", "locf_raw"):
            with self.assertRaisesRegex(E.EvaluateError, "taken"):
                E.check_name(taken)


class TestCohort(unittest.TestCase):
    def test_ids_outside_the_cohort(self):
        ids, emb = np.array(["1234567_a", "1234567_b", "x"]), np.zeros((3, 2))
        cohort = pd.Series(["1234567_a", "1234567_b", "q"])
        with self.assertRaises(E.EvaluateError) as e:
            E.match_cohort(ids, emb, cohort, False)
        self.assertIn("1 of 3 ids are not in the cohort of 3 windows", str(e.exception))
        self.assertIn("--drop-extra-ids", str(e.exception))
        self.assertNotIn("1234567", str(e.exception))       # IDs stay out of messages
        kept, kept_emb, extra = E.match_cohort(ids, emb, cohort, True)
        self.assertEqual((kept.tolist(), kept_emb.shape, extra), (["1234567_a", "1234567_b"], (2, 2), 1))

    def test_no_id_in_the_cohort(self):
        with self.assertRaisesRegex(E.EvaluateError, "none of the 2 ids.*format"):
            E.match_cohort(np.array(["x", "y"]), np.zeros((2, 2)), pd.Series(["a"]), True)

    def test_coverage(self):
        d = pd.DataFrame(dict(split=["train"] * 4 + ["val"] * 4, e=[1, 1, 1, np.nan, 1, 1, np.nan, np.nan]))
        cov = E.check_coverage(d, "e", "val", 0.5)
        self.assertEqual((cov["embedded"], cov["train"], cov["eval"], cov["share"]), (5, .75, .5, .625))
        with self.assertRaisesRegex(E.EvaluateError, r"75.0% of train and 50.0% of val.*--min-coverage"):
            E.check_coverage(d, "e", "val", 0.6)

    def test_the_output_folder_cannot_be_the_benchmark_results(self):
        E.check_out(Path(tempfile.gettempdir()) / "mine")
        with self.assertRaisesRegex(E.EvaluateError, "own results"):
            E.check_out(BL.OUT / "mimic_val")


class TestEvaluateRun(TmpCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort, cls.signal = E.demo_cohort(n_subjects=300)
        cls.saved, BL.N_BOOT = BL.N_BOOT, 20

    @classmethod
    def tearDownClass(cls):
        BL.N_BOOT = cls.saved

    def run_on(self, ids, emb, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return E.evaluate(self.cohort, "draw_id", ids, emb, "m", self.tmp / "out", "demo", perms=1, jobs=1, file_info="m.npz", **kw)

    def test_fewer_dimensions_than_the_probe_components(self):
        ids = self.cohort[0].draw_id.to_numpy(str)
        res = self.run_on(ids, np.random.default_rng(0).normal(0, 1, (len(ids), 8)))
        self.assertTrue((res["results"].method == "now+m").any())
        self.assertIn("PCA(8)", (self.tmp / "out" / "report.md").read_text())

    def test_other_splits_are_gone_before_anything_is_computed(self):
        d = self.cohort[0].copy()
        d.loc[d.group % 10 == 0, "split"] = "test"
        ids = d.draw_id.to_numpy(str)
        seen = {}

        def spy(cohort, *a):
            seen["splits"] = set(cohort[0].split)
            raise E.EvaluateError("stop here")

        with mock.patch.object(E, "run_model", spy), self.assertRaisesRegex(E.EvaluateError, "stop here"):
            E.evaluate((d, *self.cohort[1:]), "draw_id", ids, np.zeros((len(ids), 4)), "m", self.tmp / "out", "demo")
        self.assertEqual(seen["splits"], {"train", "val"})

    def test_missing_rows_go_to_the_imputer_and_coverage_is_written(self):
        ids = self.cohort[0].draw_id.to_numpy(str)
        emb = np.random.default_rng(0).normal(0, 1, (len(ids), 8))
        keep = np.arange(len(ids)) % 4 != 0
        res = self.run_on(ids[keep], emb[keep])
        self.assertAlmostEqual(res["coverage"]["share"], keep.mean(), places=6)
        text = (self.tmp / "out" / "report.md").read_text()
        self.assertIn(f"{keep.mean():.1%} of {len(ids)} cohort windows embedded", text)
        self.assertIn("imputer", text)

    def test_low_coverage_stops(self):
        ids = self.cohort[0].draw_id.to_numpy(str)
        with self.assertRaisesRegex(E.EvaluateError, "coverage too low"):
            self.run_on(ids[:20], np.zeros((20, 4)))

    def test_same_rows_as_the_reference_run(self):
        """A reference model passed through evaluate's path gives the rows of the baselines run: same score, stack, bootstrap and null."""
        d, _, labs, tiers = self.cohort
        rng = np.random.default_rng(1)
        pg = pd.DataFrame(rng.normal(0, 1, (len(d), 512)) + self.signal[:, None] * rng.normal(0, .3, 512), columns=BL.PG)
        full = pd.concat([d.assign(b0=rng.normal(0, 1, len(d))), pg], axis=1)
        for c in BL.RATE:
            full[c] = rng.normal(0, 1, len(d))
        with contextlib.redirect_stdout(io.StringIO()):
            ref_rows, ref_nulls = BL.run_lab(full, ["b0"], tiers, E.DEMO_LAB, labs[E.DEMO_LAB], "val", 2, only=E.MAIN_SUBSETS)
            cols = [f"e{i}" for i in range(512)]
            mine, mine_nulls = E.run_model((pd.concat([d, pg.set_axis(cols, axis=1)], axis=1), [], labs, tiers), "papagei", cols, "val", 2, 1, E.MAIN_SUBSETS)
        ref_rows, ref_nulls = pd.DataFrame(ref_rows), pd.DataFrame(ref_nulls)
        select = lambda r: r[r.method.eq("papagei") | r.method.str.endswith("+papagei")].reset_index(drop=True)
        pd.testing.assert_frame_equal(select(ref_rows), select(mine))
        pd.testing.assert_frame_equal(ref_nulls[ref_nulls.ppg == "papagei"].reset_index(drop=True), mine_nulls.reset_index(drop=True))


class TestTestSplitRefusal(TmpCase):
    ARGS = ["--embeddings", "absent.npz", "--name", "m", "--source", "vitaldb", "--eval", "test"]

    @staticmethod
    def marker(value):
        return mock.patch.dict(E.CFG["benchmark"], test_opened=value)

    def test_refused_with_the_reason_before_anything_is_read(self):
        with self.marker(False), mock.patch.object(E, "read_embeddings", side_effect=AssertionError("read")), \
                mock.patch.object(BL, "load_mimic", side_effect=AssertionError("read")), mock.patch.object(BL, "load_vitaldb", side_effect=AssertionError("read")):
            with self.assertRaises(SystemExit) as e:
                E.main(self.ARGS)
        text = str(e.exception)
        self.assertIn("--eval test refused", text)
        self.assertIn("benchmark.test_opened is not true in config.yaml", text)
        self.assertIn("before any other model reads the split", text)
        self.assertIn("use --eval val", text)

    def test_refused_for_mimic_too(self):
        with self.marker(False), self.assertRaises(SystemExit) as e:
            E.main(["--embeddings", "absent.npz", "--name", "m", "--source", "mimic", "--eval", "test"])
        self.assertIn("refused", str(e.exception))

    def test_only_true_opens_it(self):
        for value in (False, None, "true", "yes", 1):
            with self.marker(value), self.assertRaises(E.EvaluateError):
                E.check_test_open("test")

    def test_a_missing_marker_is_closed(self):
        saved = E.CFG["benchmark"].pop("test_opened")
        try:
            with self.assertRaises(E.EvaluateError):
                E.check_test_open("test")
        finally:
            E.CFG["benchmark"]["test_opened"] = saved

    def test_open_once_marked(self):
        with self.marker(True):
            E.check_test_open("test")

    def test_a_results_folder_does_not_open_it(self):
        """The marker alone decides: a folder is not something an outside user has."""
        (self.tmp / "confirmatory_test").mkdir()
        with self.marker(False), mock.patch.object(BL, "OUT", self.tmp), self.assertRaises(E.EvaluateError):
            E.check_test_open("test")

    def test_val_is_always_open(self):
        with self.marker(False):
            E.check_test_open("val")

    def test_the_committed_config_is_closed(self):
        self.assertIs(yaml.safe_load((common.ROOT / "config.yaml").read_text())["benchmark"]["test_opened"], False)


class TestDemo(unittest.TestCase):
    """The whole path on generated data. About half a minute."""

    @classmethod
    def setUpClass(cls):
        no_data = AssertionError("restricted data was read")
        with tempfile.TemporaryDirectory() as t, mock.patch.object(BL, "load_mimic", side_effect=no_data), mock.patch.object(BL, "load_vitaldb", side_effect=no_data), \
                mock.patch.object(pd, "read_csv", side_effect=no_data):
            cls.results = E.run_demo(Path(t))
            cls.files = {n: sorted(p.name for p in (Path(t) / n).iterdir()) for n in cls.results}
            cls.report = (Path(t) / "planted" / "report.md").read_text()

    def test_planted_signal_gains_clearly_on_every_rung(self):
        for b in BL.LADDER:
            gain = E.demo_gain(self.results["planted"], "planted", b)
            self.assertGreater(gain.est, .05, b)
            self.assertGreater(gain.lo, .02, b)

    def test_noise_stays_near_zero_on_every_rung(self):
        for b in BL.LADDER:
            gain = E.demo_gain(self.results["noise"], "noise", b)
            self.assertLess(abs(gain.est), .02, b)
            self.assertLess(gain.lo, 0, b)

    def test_the_gain_is_outside_the_null_range(self):
        n = self.results["planted"]["nulls"]
        x = n[(n.task == "clf") & (n.base == "now")].delta
        self.assertEqual(len(x), E.DEMO_PERMS)
        self.assertGreater(E.demo_gain(self.results["planted"], "planted", "now").est, x.max())

    def test_the_check_passes(self):
        self.assertEqual(E.demo_check(self.results), [])

    def test_coverage(self):
        self.assertLess(self.results["planted"]["coverage"]["share"], 1)
        self.assertLess(self.results["noise"]["coverage"]["share"], 1)

    def test_files(self):
        self.assertEqual(self.files["planted"], ["manifest.json", "null.csv", "report.md", "results.csv"])

    def test_report(self):
        for rung in BL.LADDER:
            self.assertIn(f"| {rung} |", self.report)
        for text in ("| model alone |", "Null for planted", "AUROC", "R²", "Reference rows: none", "subject-bootstrap", "Windows with a previous value"):
            self.assertIn(text, self.report)
        for claim in ("significan", "p_holm", "q_bh", "†", "‡", "p <"):
            self.assertNotIn(claim, self.report)

    def test_reference_rows_appear_beside_the_model(self):
        res = self.results["planted"]
        ref = res["results"].copy()
        ref["method"] = ref.method.str.replace("planted", "beat")
        text = "\n".join(E.lab_lines(res["results"], res["nulls"], ref, "planted", E.DEMO_LAB))
        self.assertIn("| beat |", text)
        self.assertNotIn("| papagei |", text)
        row = E.demo_gain(res, "planted", "now")
        line = next(s for s in text.split("\n") if s.startswith("| now |"))
        self.assertEqual(line.count(E.ci(row, True)), 2)      # the model and its copy as a reference

    def test_no_reference_file(self):
        self.assertIsNone(E.load_reference(None)[0])
        self.assertIsNone(E.load_reference("/no/such/file.csv")[0])


class TestSubsets(unittest.TestCase):
    """Descriptive subsets of run_lab. Generated data."""

    @classmethod
    def setUpClass(cls):
        (d, _, labs, tiers), signal = E.demo_cohort(n_subjects=500)
        rng = np.random.default_rng(2)
        group = rng.choice(["white", "black", "hispanic"], d.group.max() + 1, p=[.45, .45, .1])[d.group]
        d["eth_black"], d["eth_hispanic"] = (group == "black").astype(float), (group == "hispanic").astype(float)
        cols = [f"e{i}" for i in range(8)]
        d = pd.concat([d, pd.DataFrame(rng.normal(0, 1, (len(d), 8)) + signal[:, None], columns=cols)], axis=1)
        cls.d, cls.labs, cls.tiers = d, labs, tiers
        cls.models = (("m", cols, {"clf": lambda: BL.probe_clf(8), "reg": lambda: BL.probe_reg(8)}),)
        cls.saved, BL.N_BOOT = BL.N_BOOT, 20

    @classmethod
    def tearDownClass(cls):
        BL.N_BOOT = cls.saved

    def rows(self, d):
        with contextlib.redirect_stdout(io.StringIO()):
            return pd.DataFrame(BL.run_lab(d, [], self.tiers, E.DEMO_LAB, self.labs[E.DEMO_LAB], "val", 0, models=self.models)[0])

    def test_counts(self):
        r = self.rows(self.d)
        val = self.d[self.d.split == "val"]
        want = dict(female=(val.female == 1).sum(), male=(val.female == 0).sum(), eth_black=(val.eth_black == 1).sum(), eth_white=((val.eth_black == 0) & (val.eth_hispanic == 0)).sum())
        got = r[(r.task == "reg")].groupby("subset").n_eval.first()
        for subset, n in want.items():
            self.assertEqual(got[subset], n, subset)

    def test_without_ethnicity_columns(self):
        r = self.rows(self.d.drop(columns=["eth_black", "eth_hispanic"]))
        self.assertTrue({"female", "male"} <= set(r.subset))
        self.assertFalse({"eth_black", "eth_white"} & set(r.subset))

    def test_rows_of_other_subsets_do_not_move(self):
        r = self.rows(self.d)
        with contextlib.redirect_stdout(io.StringIO()):
            only = pd.DataFrame(BL.run_lab(self.d, [], self.tiers, E.DEMO_LAB, self.labs[E.DEMO_LAB], "val", 0, models=self.models, only=("all",))[0])
        pd.testing.assert_frame_equal(r[r.subset == "all"].reset_index(drop=True), only.reset_index(drop=True))

    def test_report_tables(self):
        r = self.rows(self.d)
        text = BL.report(BL.adjust(r.assign(method=r.method.str.replace("+m", "+beat", regex=False).replace("m", "beat"))), "x", "val", len(self.d), 500,
                         pd.DataFrame(columns=["lab", "task", "ppg", "base", "perm", "delta"]))
        for title in ("Descriptive: female", "Descriptive: male", "Descriptive: ethnicity group Black", "Descriptive: ethnicity group White"):
            self.assertIn(title, text)


class TestPositiveControl(unittest.TestCase):
    def test_pulseppg_with_missing_rows(self):
        (d, _, labs, tiers), _ = E.demo_cohort(n_subjects=300)
        rng = np.random.default_rng(3)
        pp = pd.DataFrame(rng.normal(0, 1, (len(d), 512)), columns=BL.PP)
        pp.loc[rng.random(len(d)) < .1, BL.PP] = np.nan
        d = pd.concat([d, pp], axis=1)
        rows = PC.one(d, [], tiers, labs, E.DEMO_LAB, "clf", "pulseppg", .7, 0)
        self.assertEqual(len(rows), len(BL.LADDER))
        self.assertEqual({r["ppg"] for r in rows}, {"pulseppg"})
        self.assertTrue(np.isfinite([r["recovered"] for r in rows]).all())


if __name__ == "__main__":
    unittest.main()
