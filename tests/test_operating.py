"""Tests of ppglabbench.operating on a small synthetic cohort. No MIMIC row is read.

Usage:  python -m unittest tests.test_operating
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ppglabbench import confirmatory as C  # noqa: E402
from ppglabbench import operating as O  # noqa: E402
from ppglabbench import positive_control as PC  # noqa: E402
from tests.test_confirmatory import synthetic_cohort, with_cnn  # noqa: E402

BL = C.BL


class TestGenerator(unittest.TestCase):
    def setUp(self):
        O._prepared.clear()
        self.d, self.bcols, self.spec, self.tiers = synthetic_cohort(labs=("bun", "lactate"), n_train=200, n_val=100)

    def test_a_replicate_is_aggregates_with_the_size_scaled_to_test(self):
        r = O.replicate(self.d, self.bcols, self.tiers, self.spec["bun"], "bun", "clf", "now", "beat", 0.0, 0, 8, .5)
        self.assertEqual(set(r["rows"]), {C.PRIMARY, C.UPPER})
        self.assertEqual(r["rows"][C.PRIMARY]["D"].shape, (8,))
        self.assertFalse({"group", "draw_id"} & set(r))
        P = O._prepared["bun", "clf", "now"]
        self.assertAlmostEqual(r["n_mean"], np.mean([len(v) for v in P["rows"].values()]) * round(len(P["subjects"]) * .5))

    def test_the_value_task(self):
        r = O.replicate(self.d, self.bcols, self.tiers, self.spec["lactate"], "lactate", "reg", "now", "beat", .3, 0, 5, 1.0)
        self.assertTrue(np.isfinite(r["rows"][C.PRIMARY]["D"]).all())

    def test_the_signal_is_a_fixed_function_of_the_features(self):
        q, y, tr, *_ = C.fixed_frame(self.d, self.tiers, "bun", self.spec["bun"], "val")
        a, b = O.signal(q, self.bcols, tr, "bun"), O.signal(q, self.bcols, tr, "bun")
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(a[tr].std(), 1)
        self.assertGreater(np.abs(a - O.signal(q, self.bcols, tr, "other")).max(), 0)

    def test_smallest_gain_by_interpolation(self):
        x = pd.DataFrame(dict(mean_gain=[0., .02, .04], power=[.025, .5, .9]))
        g = O.smallest_gain(x)
        self.assertTrue(.02 < g < .04)
        self.assertIsNone(O.smallest_gain(pd.DataFrame(dict(mean_gain=[0., .02], power=[.025, .3]))))


class TestFamilyWorld(unittest.TestCase):
    LABS = ("bun", "potassium", "sodium")

    @classmethod
    def setUpClass(cls):
        O._prepared.clear()
        cls.d, cls.bcols, cls.spec, cls.tiers = synthetic_cohort(labs=cls.LABS, n_train=150, n_val=90)
        cls.r = {b: {lab: .25 for lab in cls.LABS} for b in C.CHAIN}

    def world(self, scale=0.0, weights=(1, 1, 1), B=50, draws=4, bases=C.CHAIN, seed=0, gen="full"):
        return O.family_world(self.d, self.bcols, self.tiers, self.spec, self.LABS, gen, "beat", scale, weights, seed, draws, .9, self.r, bases, B)

    def test_it_applies_the_procedures_bound(self):
        w = self.world()
        for b in C.CHAIN:
            self.assertEqual(w[b]["mean"].shape, (4,))
            for m in range(4):
                self.assertAlmostEqual(w[b]["upper"][m], C.mean_upper(w[b]["mean"][m], w[b]["hi"][m], w[b]["train"][m]))

    def test_tuning_runs_without_the_bootstrap(self):
        w = self.world(B=0, bases=("now",), gen="now", draws=3)
        self.assertTrue(np.isfinite(w["now"]["mean"]).all())
        self.assertTrue(np.isnan(w["now"]["upper"]).all())

    def test_a_signal_raises_the_mean(self):
        small, big = self.world(scale=0.0, B=0, bases=("now",), gen="now"), self.world(scale=3.0, B=0, bases=("now",), gen="now")
        self.assertGreater(big["now"]["mean"].mean(), small["now"]["mean"].mean() + .01)

    def test_a_mix_of_weights_puts_the_signal_in_some_labs(self):
        mixed = np.array(O.MIX)
        self.assertEqual(len(mixed), 10)
        self.assertEqual((mixed == 0).sum(), 3)

    def test_chain_rows_by_hand(self):
        w = lambda full, now: dict(full=dict(mean=np.zeros(2), hi=np.zeros(2), train=np.zeros(2), upper=np.array(full)), now=dict(mean=np.zeros(2), hi=np.zeros(2), train=np.zeros(2), upper=np.array(now)))
        rows = O.chain_rows([w([.005, .02], [.005, .005]), w([.005, .005], [.02, .02])], "x", "full", 0.0, .01, ("full", "now"))
        self.assertEqual(rows[0]["passes_alone"], .75)
        self.assertEqual(rows[1]["passes_alone"], .5)
        self.assertEqual(rows[1]["passes_in_sequence"], .25)      # only the draw that passed `full` and then `now`


class TestRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        O._prepared.clear()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d, cls.bcols, cls.spec, cls.tiers = synthetic_cohort(labs=("bun", "potassium", "lactate"), n_train=150, n_val=80)
        test = cls.d.iloc[:5].assign(split="test", bun=1e9, bun_abnormal=1.0)
        d = pd.concat([cls.d, test], ignore_index=True)
        with mock.patch.object(O, "GAMMAS", dict(clf=(0.0, 1.0), reg=(0.0, .3))), mock.patch.object(O, "L1_DRAWS", 3), mock.patch.object(O, "tune_scale", return_value=.5), mock.patch.object(O, "EQUAL", (1, 1)), mock.patch.object(O, "MIX", (0, 2)):
            cls.res = O.run(d, cls.bcols, cls.spec, cls.tiers, cls.tmp.name, reps=2, draws=3, jobs=1, family=("bun", "potassium"), named=("lactate",), curve_labs=("bun",), models=("beat",), chain_reps=2, chain_draws=2, chain_B=30, near=(("near", .003),))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_tables_exist_with_aggregates_only(self):
        for f in ("marginal.csv", "chain.csv", "boundary.csv", "family.csv", "sample_size.csv", "bedside_gains.csv", "report.md", "manifest.json"):
            self.assertTrue((Path(self.tmp.name) / f).exists(), f)
        text = "".join((Path(self.tmp.name) / f).read_text() for f in ("marginal.csv", "chain.csv", "boundary.csv", "family.csv", "report.md"))
        self.assertFalse(any(str(i) in text for i in np.unique(self.d.group)))
        self.assertLess(pd.read_csv(Path(self.tmp.name) / "marginal.csv").mean_gain.abs().max(), 1)      # no test row reached a statistic

    def test_the_family_rows(self):
        what = set(pd.read_csv(Path(self.tmp.name) / "family.csv").what)
        self.assertIn("z1_not_computable_under_null", what)
        self.assertIn("fwer_global_null", what)
        self.assertTrue({f"chain_{b}_alone" for b in C.CHAIN} <= what)
        self.assertTrue({f"chain_{b}_in_sequence" for b in C.CHAIN} <= what)
        self.assertTrue(0 <= self.res["fwer"][0] <= 1)

    def test_the_chain_rows(self):
        x = pd.read_csv(Path(self.tmp.name) / "chain.csv")
        self.assertEqual(set(x.scenario), {"ppg_adds_nothing_over_full", "near", "boundary_mix", "boundary_equal"})
        self.assertEqual(list(x[x.scenario == "near"].base), list(C.CHAIN))
        b = pd.read_csv(Path(self.tmp.name) / "boundary.csv")
        self.assertEqual(set(b[b.scenario == "boundary_mix"].base), set(C.CHAIN))
        self.assertEqual(set(b[b.scenario == "boundary_equal"].base), {"full", "now"})


class TestPositiveControlOfTheCnn(unittest.TestCase):
    def test_the_heads(self):
        d, _, _, _ = synthetic_cohort(labs=("bun",), n_train=200, n_val=100)
        out = PC.cnn_heads(with_cnn(d, ("bun",)))
        self.assertEqual(set(out.target), {"age", "heart_rate"})
        self.assertEqual(set(out.split), {"train", "val"})
        self.assertGreater(out[(out.target == "age")].r2.min(), 0)

    def test_test_rows_are_not_read(self):
        d, _, _, _ = synthetic_cohort(labs=("bun",), n_train=200, n_val=100)
        d = with_cnn(d, ("bun",))
        d = pd.concat([d, d.iloc[:5].assign(split="test", cnn_age_pred=np.nan, cnn_hr_pred=np.nan)], ignore_index=True)
        self.assertEqual(set(PC.cnn_heads(d).split), {"train", "val"})

    def test_nothing_without_the_columns(self):
        d, _, _, _ = synthetic_cohort(labs=("bun",), n_train=50, n_val=30)
        self.assertEqual(len(PC.cnn_heads(d)), 0)


if __name__ == "__main__":
    unittest.main()
