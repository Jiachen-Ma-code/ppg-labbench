"""Pure-function tests on synthetic inputs. No data/ folder is needed and no restricted data is read.

Usage:  python -m unittest tests.test_smoke
"""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy.stats import norm

from ppglabbench import baselines as BL
from ppglabbench import common, confirmatory, labels
from ppglabbench.features import covariates


class TestPaths(unittest.TestCase):
    def test_root_is_the_repository_root(self):
        self.assertTrue((common.ROOT / "config.yaml").is_file())
        self.assertTrue((common.ROOT / "src" / "ppglabbench" / "common.py").is_file())
        self.assertEqual(common.DATA, common.ROOT / "data")

    def test_frozen_paths_exist(self):
        self.assertEqual(confirmatory.FROZEN_PATHS, ("src", "config.yaml"))
        for p in confirmatory.FROZEN_PATHS:
            self.assertTrue((common.ROOT / p).exists(), p)

    def test_treatment_items_are_found(self):
        self.assertEqual(set(covariates.ITEMS), {"metavision", "carevue"})


class TestAbnormal(unittest.TestCase):
    def check(self, rule, v, want, female=None):
        got = labels.abnormal(rule, np.array(v, float), np.zeros(len(v)) if female is None else np.array(female))
        np.testing.assert_array_equal(got, np.array(want, float))

    def test_thresholds_at_the_boundary(self):
        self.check("<135", [134.9, 135, 135.1], [1, 0, 0])
        self.check(">5.0", [4.9, 5.0, 5.1], [0, 0, 1])
        self.check(">=180", [179.9, 180, 180.1], [0, 1, 1])
        self.check("<=8", [7.9, 8, 8.1], [1, 1, 0])

    def test_missing_stays_missing(self):
        self.check("<135", [np.nan, 130], [np.nan, 1])
        self.check("anaemia", [np.nan, 11], [np.nan, 1], female=[1, 1])

    def test_anaemia_is_sex_specific(self):
        self.check("anaemia", [11.9, 12.0, 12.9, 13.0], [1, 0, 1, 0], female=[1, 1, 0, 0])


class TestSplitOf(unittest.TestCase):
    IDS = [f"synthetic:{i}" for i in range(20000)]

    def test_matches_the_salted_hash_by_hand(self):
        tr, va, _ = labels.C["split"]
        for s in self.IDS[:200]:
            u = int(hashlib.sha256(f"{labels.C['split_salt']}:{s}".encode()).hexdigest()[:12], 16) / 16 ** 12
            self.assertEqual(labels.split_of(s, set()), "train" if u < tr else "val" if u < tr + va else "test")

    def test_proportions(self):
        counts = pd.Series([labels.split_of(s, set()) for s in self.IDS]).value_counts(normalize=True)
        for name, share in zip(("train", "val", "test"), labels.C["split"]):
            self.assertAlmostEqual(counts[name], share, delta=0.01)

    def test_a_development_subject_never_enters_test(self):
        by_split = {}
        for s in self.IDS:
            by_split.setdefault(labels.split_of(s, set()), []).append(s)
        for s in by_split["test"][:50]:
            self.assertEqual(labels.split_of(s, {s}), "train")
            self.assertEqual(labels.split_of(s, {"synthetic:other"}), "test")
        for name in ("train", "val"):
            for s in by_split[name][:50]:
                self.assertEqual(labels.split_of(s, {s}), name)


class TestDevelopmentSubjects(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name)
        (self.data / "benchmark").mkdir()
        patches = mock.patch.object(labels, "DATA", self.data), mock.patch.object(labels, "OUT", self.data / "benchmark")
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def write_folder(self, subjects):
        folder = self.data / labels.C["dev_exclusion"]
        folder.mkdir()
        pd.DataFrame(dict(subject_id=subjects, draw_id="x")).to_csv(folder / "groupB_draws.csv", index=False)

    def write_splits(self, dev, other):
        sp = pd.DataFrame(dict(subject_id=dev + other, split="train", dev_set=[True] * len(dev) + [False] * len(other)))
        sp.to_csv(self.data / "benchmark" / "mimic_splits.csv", index=False)

    def test_the_folder_comes_first(self):
        self.write_folder([1, 2])
        self.write_splits([3], [4])
        self.assertEqual(labels.development_subjects(), {1, 2})

    def test_the_split_file_stands_in_when_the_folder_is_absent(self):
        self.write_splits([1, 2], [3, 4])
        self.assertEqual(labels.development_subjects(), {1, 2})

    def test_neither_stops_with_a_message(self):
        with self.assertRaisesRegex(SystemExit, "development set"):
            labels.development_subjects()

    def test_assignments_agree(self):
        """The folder holds more subjects than the split file, which only lists those with windows. Both give the same split."""
        ids = [f"synthetic:{i}" for i in range(2000)]
        hashed = {s: labels.split_of(s, set()) for s in ids}
        dev = [s for s in ids if hashed[s] == "test"][:40] + [s for s in ids if hashed[s] == "val"][:10]
        self.write_folder(dev + ["synthetic:no_windows"])
        from_folder = labels.development_subjects()
        (self.data / labels.C["dev_exclusion"] / "groupB_draws.csv").unlink()
        (self.data / labels.C["dev_exclusion"]).rmdir()
        self.write_splits(dev, [s for s in ids if s not in dev])
        from_file = labels.development_subjects()
        self.assertEqual([labels.split_of(s, from_folder) for s in ids], [labels.split_of(s, from_file) for s in ids])


class TestBlockPerm(unittest.TestCase):
    def setUp(self):
        self.train = [np.array([0, 1, 2]), np.array([3]), np.array([4, 5])]
        self.ev = [np.array([6, 7]), np.array([8])]
        self.blocks = [self.train, self.ev]
        self.n = 10      # row 9 is in no block

    def test_each_subject_gets_rows_of_another_subject_of_its_split(self):
        for seed in range(20):
            perm = BL.block_perm(self.blocks, self.n, np.random.default_rng(seed))
            self.assertEqual(perm[9], 9)
            for split in self.blocks:
                pool = np.concatenate(split)
                for rows in split:
                    self.assertTrue(np.isin(perm[rows], pool).all())
                    self.assertFalse(np.isin(perm[rows], rows).any())
                    self.assertEqual(len({frozenset(perm[rows].tolist())}), 1)

    def test_recycled_to_the_window_count(self):
        for seed in range(20):
            perm = BL.block_perm(self.blocks, self.n, np.random.default_rng(seed))
            donors = [r for r in self.train if perm[0] in r]
            self.assertEqual(len(donors), 1)
            np.testing.assert_array_equal(perm[self.train[0]], np.resize(donors[0], 3))

    def test_seed_fixes_the_map(self):
        a = BL.block_perm(self.blocks, self.n, np.random.default_rng(7))
        b = BL.block_perm(self.blocks, self.n, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)


class TestFlexible(unittest.TestCase):
    def setUp(self):
        self.age = np.array([20., 40, 60, 80, 100])
        self.female = np.array([1., 0, 1, 0, 1])
        self.tr = np.array([0, 1, 2, 3])

    def test_without_map(self):
        X = BL.flexible(pd.DataFrame(dict(age=self.age, female=self.female)), self.tr)
        q = np.quantile(self.age[self.tr], [.25, .5, .75])      # 35, 50, 65
        self.assertEqual(X.shape, (5, 4))
        for j, c in enumerate(q):
            np.testing.assert_allclose(X[:, j], np.maximum(0, self.age - c))
        np.testing.assert_allclose(X[:, 3], self.female * self.age)

    def test_quartiles_come_from_train_rows_only(self):
        changed = self.age.copy()
        changed[4] = 1000.     # not a train row
        a = BL.flexible(pd.DataFrame(dict(age=self.age, female=self.female)), self.tr)
        b = BL.flexible(pd.DataFrame(dict(age=changed, female=self.female)), self.tr)
        np.testing.assert_allclose(a[:4], b[:4])

    def test_with_map(self):
        m = np.array([60., np.nan, 80, 100, 70])
        v = np.array([1., 1, 0, 1, 0])
        routine = np.array([1., 0, 0, 1, 1])
        X = BL.flexible(pd.DataFrame(dict(age=self.age, female=self.female, map_mmhg=m, vasopressor=v, routine_draw=routine)), self.tr)
        self.assertEqual(X.shape, (5, 9))
        for j, c in zip((4, 5, 6), np.nanquantile(m[self.tr], [.25, .5, .75])):
            np.testing.assert_allclose(X[:, j], np.nan_to_num(np.maximum(0, m - c)))
        self.assertTrue(np.isfinite(X).all())       # missing MAP becomes 0
        np.testing.assert_allclose(X[:, 7], v * routine)
        np.testing.assert_allclose(X[:, 8], v * np.nan_to_num(m - 80.))     # train mean of 60, 80, 100


class TestAdjust(unittest.TestCase):
    @staticmethod
    def row(est, half=0.0196, lab="albumin", subset="all", metric="auroc", vs="now"):
        return dict(lab=lab, subset=subset, metric=metric, vs=vs, est=est, lo=est - half, hi=est + half)

    def test_only_the_main_increments_are_adjusted(self):
        r = pd.DataFrame([self.row(.02), self.row(.02, subset="not_dev"), self.row(.02, metric="auprc"), self.row(.02, vs=None),
                          self.row(.02, vs="locf_raw"), self.row(.02, vs="now_gb"), self.row(.02, lab="egfr_incl_failure"), self.row(.02, metric="r2")])
        p = BL.adjust(r).p.notna().tolist()
        self.assertEqual(p, [True, False, False, False, False, False, False, True])

    def test_by_hand(self):
        # half = 0.0196 gives SE = 0.01, so z = est / 0.01
        r = BL.adjust(pd.DataFrame([self.row(.03), self.row(.01), self.row(.02)]))
        p = 2 * norm.sf([3., 1., 2.])
        np.testing.assert_allclose(r.p, p)
        o = np.argsort(p)
        by_order = np.array([p[o[0]] * 3, p[o[1]] * 3 / 2, p[o[2]]])
        bh = np.minimum.accumulate(by_order[::-1])[::-1]
        holm = np.maximum.accumulate([p[o[0]] * 3, p[o[1]] * 2, p[o[2]]])
        np.testing.assert_allclose(r.q_bh.to_numpy()[o], bh)
        np.testing.assert_allclose(r.p_holm.to_numpy()[o], holm)

    def test_input_is_not_changed(self):
        r = pd.DataFrame([self.row(.03)])
        BL.adjust(r)
        self.assertNotIn("p", r)


if __name__ == "__main__":
    unittest.main()
