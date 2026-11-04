"""ppglabbench.features.cnn on generated data: folds, rows, stopping rule, output schema and a short training run. Tests that need torch are skipped outside the torch
environment; the comparison with baselines.ppg_score needs the main environment, which has what baselines imports.

Usage:  .venv-torch/bin/python -m unittest tests.test_cnn
        .venv/bin/python -m unittest tests.test_cnn
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ppglabbench.features import cnn as C

try:
    from sklearn.model_selection import GroupKFold
except ImportError:
    GroupKFold = None

try:
    import torch
except ImportError:
    torch = None

try:
    from ppglabbench import baselines as B
except ImportError:
    B = None


def rows(n_subjects=60, seed=0):
    """Generated rows: split, subject, cohort. Subjects have 1 to 10 windows; a few train windows are outside the cohort, and some train subjects have only such windows."""
    rng = np.random.default_rng(seed)
    subject = np.repeat(1000 + np.arange(n_subjects) * 7, rng.integers(1, 11, n_subjects))
    which = rng.choice([C.TRAIN, C.VAL, C.TEST], n_subjects, p=[.7, .15, .15])
    split = which[np.searchsorted(1000 + np.arange(n_subjects) * 7, subject)].astype(np.int8)
    cohort = (split == C.TRAIN) & (rng.random(len(subject)) > .1)
    return split, subject, cohort


@unittest.skipIf(GroupKFold is None, "needs sklearn")
class TestFolds(unittest.TestCase):
    def test_same_call_as_the_benchmark_on_the_same_rows(self):
        split, subject, cohort = rows()
        fold = C.row_folds(split, subject, cohort)
        tr = np.flatnonzero((split == C.TRAIN) & cohort)
        for k, (_, b) in enumerate(GroupKFold(5).split(tr, groups=subject[tr])):
            self.assertEqual(sorted(np.flatnonzero(fold[tr] == k)), sorted(b))

    @unittest.skipIf(B is None, "needs what baselines imports")
    def test_matches_the_folds_ppg_score_uses(self):
        """A model that records what it is fitted on shows which rows ppg_score holds out, on a frame holding exactly the cohort's train rows."""
        split, subject, cohort = rows()
        tr = np.flatnonzero((split == C.TRAIN) & cohort)
        fold = C.row_folds(split, subject, cohort)
        self.assertEqual(C.K, B.K)
        seen = []

        class Recorder:
            def fit(self, X, y):
                seen.append(X[:, 0].astype(int))
                return self

            def predict_proba(self, A):
                return np.full((len(A), 2), .5)

        X = np.arange(len(split), dtype=float)[:, None]
        y = np.resize([0, 1], len(split))
        B.ppg_score(Recorder, X, y, subject, tr, tr[:1])
        for k, used in enumerate(seen[:C.K]):
            self.assertEqual(sorted(used), sorted(tr[fold[tr] != k]))

    def test_a_subject_has_one_fold_and_only_train_rows_have_one(self):
        split, subject, cohort = rows()
        fold = C.row_folds(split, subject, cohort)
        self.assertTrue((fold[split != C.TRAIN] == -1).all())
        self.assertTrue((fold[cohort] >= 0).all())
        for s in np.unique(subject):
            self.assertEqual(len(set(fold[subject == s])), 1)

    def test_a_train_window_outside_the_cohort_takes_its_subjects_fold(self):
        split = np.zeros(12, np.int8)
        subject = np.repeat(np.arange(6), 2)
        cohort = np.ones(12, bool)
        cohort[1] = False       # its subject still has a cohort window
        cohort[[10, 11]] = False    # this subject has none
        fold = C.row_folds(split, subject, cohort)
        self.assertEqual(fold[1], fold[0])
        self.assertTrue((fold[[10, 11]] == -1).all())


@unittest.skipIf(GroupKFold is None, "needs sklearn")
class TestRows(unittest.TestCase):
    def setUp(self):
        self.split, self.subject, self.cohort = rows(300)
        self.fold = C.row_folds(self.split, self.subject, self.cohort)

    def test_test_and_val_rows_are_never_trained_on_or_in_a_slice(self):
        for name in C.FITS:
            train, _ = C.fit_rows(name, self.split, self.fold, self.cohort)
            grad, sl = C.take_slice(train, self.subject, C.SEEDS[name])
            for r in (train, grad, sl):
                self.assertTrue((self.split[r] == C.TRAIN).all(), name)
                self.assertTrue(self.cohort[r].all(), name)

    def test_no_subject_in_both_gradient_rows_and_slice(self):
        for name in C.FITS:
            train, _ = C.fit_rows(name, self.split, self.fold, self.cohort)
            grad, sl = C.take_slice(train, self.subject, C.SEEDS[name])
            self.assertFalse(set(self.subject[grad]) & set(self.subject[sl]), name)
            self.assertEqual(len(grad) + len(sl), len(train))

    def test_slice_is_about_a_tenth_of_subjects_and_fixed_by_the_seed(self):
        train, _ = C.fit_rows(C.FINAL, self.split, self.fold, self.cohort)
        _, sl = C.take_slice(train, self.subject, 1)
        n = len(np.unique(self.subject[train]))
        self.assertEqual(len(np.unique(self.subject[sl])), round(.1 * n))
        self.assertEqual(sorted(sl), sorted(C.take_slice(train, self.subject, 1)[1]))
        self.assertNotEqual(sorted(sl), sorted(C.take_slice(train, self.subject, 2)[1]))

    def test_no_fit_scores_a_subject_it_trained_on(self):
        for name in C.FITS:
            train, scored = C.fit_rows(name, self.split, self.fold, self.cohort)
            self.assertFalse(set(self.subject[train]) & set(self.subject[scored]), name)

    def test_only_the_final_model_scores_val_and_test(self):
        for name in C.FITS:
            _, scored = C.fit_rows(name, self.split, self.fold, self.cohort)
            if name == C.FINAL:
                self.assertTrue(((self.split[scored] != C.TRAIN) | (self.fold[scored] < 0)).all())
                self.assertTrue(set(np.flatnonzero(self.split != C.TRAIN)) <= set(scored))
            else:
                self.assertTrue((self.split[scored] == C.TRAIN).all(), name)

    def test_every_row_is_scored_by_exactly_one_fit(self):
        count = np.zeros(len(self.split), int)
        for name in C.FITS:
            count[C.fit_rows(name, self.split, self.fold, self.cohort)[1]] += 1
        self.assertTrue((count == 1).all())

    def test_seeds_differ_per_fit(self):
        self.assertEqual(len(set(C.SEEDS.values())), len(C.FITS))


class TestStopping(unittest.TestCase):
    def test_a_lone_spike_does_not_beat_a_sustained_rise(self):
        v = [.5] * 40
        v[5] = .9                       # one noisy epoch: average .58
        v[30:35] = [.62] * 5            # sustained: average .62 at the middle
        self.assertEqual(C.stopping_epoch(v), 32)

    def test_edges_count(self):
        self.assertIn(C.stopping_epoch([.5] * 10 + [.7, .7, .7, .7, .7]), (12, 13, 14))

    def test_mean_auc_counts_only_heads_with_enough_of_each_class(self):
        t = np.full((40, 3), np.nan)
        t[:, 0] = np.r_[np.zeros(20), np.ones(20)]     # usable, and perfectly separated below
        t[:, 1] = np.r_[np.zeros(36), np.ones(4)]      # four positives: too few
        o = np.tile(np.arange(40.0)[:, None], (1, 3))
        auc, heads = C.mean_auc(t, o)
        self.assertEqual(heads, 1)
        self.assertEqual(auc, 1.0)


class TestPreprocess(unittest.TestCase):
    @staticmethod
    def signal():
        return 100 + 5 * np.sin(2 * np.pi * 1.2 * np.arange(C.N_IN) / C.FS)

    def test_length_and_scale(self):
        z = C.preprocess(self.signal())
        self.assertEqual(len(z), C.N_OUT)
        self.assertEqual(z.dtype, np.float32)
        self.assertAlmostEqual(float(z.mean()), 0, places=3)
        self.assertAlmostEqual(float(z.std()), 1, places=3)

    def test_amplitude_does_not_matter(self):
        np.testing.assert_allclose(C.preprocess(self.signal()), C.preprocess(10 * self.signal()), atol=1e-3)

    def test_clipped(self):
        x = self.signal()
        x[5000:5100] = 1e5
        self.assertLessEqual(float(np.abs(C.preprocess(x)).max()), C.CLIP_SD)

    def test_a_gap_is_zero_after_scaling(self):
        x = self.signal()
        x[1000:2000] = np.nan
        z = C.preprocess(x)
        self.assertTrue((z[500:750] == 0).all())
        self.assertTrue(np.isfinite(z).all())

    def test_no_signal_is_none(self):
        self.assertIsNone(C.preprocess(np.full(C.N_IN, np.nan)))
        self.assertIsNone(C.preprocess(np.ones(C.N_IN)))
        self.assertIsNone(C.preprocess(self.signal()[:-1]))


@unittest.skipIf(GroupKFold is None, "needs sklearn")
class TestSchema(unittest.TestCase):
    def setUp(self):
        split, subject, cohort = rows(120)
        n = len(split)
        self.p = dict(draw_id=np.array([f"w{i}" for i in range(n)]), targets=np.array(["a", "b", "c"]), split=split, subject=subject, cohort=cohort,
                      fold=C.row_folds(split, subject, cohort), usable=np.ones(n, bool))
        self.p["usable"][np.flatnonzero(split == C.TEST)[:1]] = False

    def results(self):
        out = {}
        for name in C.FITS:
            _, scored = C.fit_rows(name, self.p["split"], self.p["fold"], self.p["cohort"])
            scored = scored[self.p["usable"][scored]]
            out[name] = scored, np.full((len(scored), 5), float(C.FITS.index(name)), np.float32)
        return out

    def test_fields_and_shapes(self):
        a = C.assemble_arrays(self.p, self.results())
        n = len(self.p["split"])
        self.assertEqual(sorted(a), ["age_pred", "draw_id", "fold", "hr_pred", "score", "targets"])
        self.assertEqual(a["score"].shape, (n, 3))
        for k in ("draw_id", "fold", "age_pred", "hr_pred"):
            self.assertEqual(len(a[k]), n)
        self.assertEqual(a["score"].dtype, np.float32)

    def test_a_train_row_holds_its_folds_score_and_others_the_final_models(self):
        a = C.assemble_arrays(self.p, self.results())
        rows_with_fold = np.flatnonzero(a["fold"] >= 0)
        self.assertTrue((a["score"][rows_with_fold, 0] == a["fold"][rows_with_fold]).all())
        others = np.flatnonzero((a["fold"] < 0) & self.p["usable"])
        self.assertTrue((a["score"][others, 0] == C.FITS.index(C.FINAL)).all())
        self.assertTrue((a["fold"][self.p["split"] != C.TRAIN] == -1).all())

    def test_an_unusable_window_has_no_score(self):
        a = C.assemble_arrays(self.p, self.results())
        self.assertTrue(np.isnan(a["score"][~self.p["usable"]]).all())

    def test_a_window_scored_twice_or_not_at_all_stops_it(self):
        r = self.results()
        r["fold0"] = np.r_[r["fold0"][0], r["fold1"][0][:1]], np.r_[r["fold0"][1], r["fold1"][1][:1]]
        with self.assertRaises(AssertionError):
            C.assemble_arrays(self.p, r)
        r = self.results()
        r["fold0"] = r["fold0"][0][1:], r["fold0"][1][1:]
        with self.assertRaises(AssertionError):
            C.assemble_arrays(self.p, r)

    def test_npz_round_trip_needs_no_pickle(self):
        a = C.assemble_arrays(self.p, self.results())
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "cnn_scores.npz"
            np.savez_compressed(f, **a)
            with np.load(f) as z:
                self.assertEqual(sorted(z.files), sorted(a))
                self.assertEqual(z["draw_id"][0], "w0")


@unittest.skipIf(GroupKFold is None, "needs sklearn")
class TestChecks(unittest.TestCase):
    def setUp(self):
        split, subject, cohort = rows(200)
        n = len(split)
        fold = C.row_folds(split, subject, cohort)
        T = np.full((n, 5), np.nan, np.float32)
        T[(split == C.TRAIN) & cohort] = 0
        self.p = dict(split=split, subject=subject, cohort=cohort, fold=fold, T=T)
        self.train, self.scored = C.fit_rows("fold0", split, fold, cohort)
        self.grad, self.sl = C.take_slice(self.train, subject, 0)

    def test_a_clean_fit_passes(self):
        C.check_fit("fold0", self.p, self.train, self.grad, self.sl, self.scored)

    def test_a_test_row_in_training_stops_it(self):
        t = np.flatnonzero(self.p["split"] == C.TEST)[:1]
        for grad, sl in ((np.r_[self.grad, t], self.sl), (self.grad, np.r_[self.sl, t])):
            with self.assertRaises(AssertionError):
                C.check_fit("fold0", self.p, self.train, grad, sl, self.scored)

    def test_a_label_on_a_test_row_stops_it(self):
        self.p["T"][np.flatnonzero(self.p["split"] == C.TEST)[0], 0] = 1
        with self.assertRaises(AssertionError):
            C.check_fit("fold0", self.p, self.train, self.grad, self.sl, self.scored)

    def test_a_subject_in_both_gradient_rows_and_slice_stops_it(self):
        with self.assertRaises(AssertionError):
            C.check_fit("fold0", self.p, self.train, np.r_[self.grad, self.sl[:1]], self.sl, self.scored)


@unittest.skipIf(torch is None, "needs torch")
class TestNet(unittest.TestCase):
    def test_parameter_count_and_output(self):
        net = C.make_net(19)
        self.assertEqual(sum(p.numel() for p in net.parameters()), 207107)
        net.eval()
        with torch.no_grad():
            self.assertEqual(tuple(net(torch.zeros(2, 1, C.CROP)).shape), (2, 19))

    def test_scoring_covers_the_window_with_evenly_spaced_crops(self):
        starts = np.linspace(0, C.N_OUT - C.CROP, C.N_CROPS).astype(int)
        self.assertEqual((starts[0], starts[-1] + C.CROP), (0, C.N_OUT))
        net = C.make_net(3)
        X = np.random.default_rng(0).normal(size=(5, C.N_OUT)).astype(np.float32)
        out = C.predict(net, X, np.arange(5), "cpu")
        manual = sum(net(torch.from_numpy(X[:, s:s + C.CROP])[:, None]).detach() for s in starts) / C.N_CROPS
        np.testing.assert_allclose(out, manual.numpy(), rtol=1e-5, atol=1e-5)


@unittest.skipIf(torch is None, "needs torch")
class TestTraining(unittest.TestCase):
    def test_a_short_run_learns_a_planted_signal_and_keeps_the_chosen_epoch(self):
        """Two heads read the pulse rate of a generated signal; age is noise. Masked labels must not break the loss."""
        rng = np.random.default_rng(0)
        n, t = 240, np.arange(C.N_OUT) / C.FS_OUT
        fast = rng.random(n) > .5
        X = np.stack([np.sin(2 * np.pi * (2.0 if f else .8) * t + rng.random() * 6) + rng.normal(0, .1, C.N_OUT) for f in fast]).astype(np.float32)
        T = np.column_stack([fast, fast, rng.normal(60, 10, n), np.where(fast, 120, 48) + rng.normal(0, 2, n)]).astype(np.float32)
        T[::3, 1] = np.nan
        grad, sl = np.arange(200), np.arange(200, 240)
        torch.set_num_threads(2)
        logged = []
        net, mu, sd, stop, records = C.train_model(X, T, grad, sl, 0, "cpu", epochs=5, log=logged.append)
        self.assertEqual(len(records), 5)
        self.assertEqual(len(logged), 5)
        self.assertEqual(stop, C.stopping_epoch([r["sl_auc"] for r in records]))
        self.assertGreater(records[stop]["sl_auc"], .8)
        self.assertGreater(records[-1]["sl_r2_hr"], .5)     # the regression heads train on: on a signal this easy the AUROC saturates, and the rule keeps the first tied epoch
        self.assertLess(records[-1]["sl_r2_age"], .2)       # age is noise here
        o = C.predict(net, X, sl, "cpu")
        self.assertEqual(o.shape, (40, 4))
        self.assertTrue(np.isfinite(o).all())

    def test_the_same_seed_repeats_on_the_cpu(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, C.N_OUT)).astype(np.float32)
        T = np.column_stack([rng.random(40) > .5, rng.normal(size=40), rng.normal(size=40)]).astype(np.float32)
        torch.set_num_threads(1)
        run = lambda: C.train_model(X, T, np.arange(32), np.arange(32, 40), 7, "cpu", epochs=1, log=lambda s: None)[0]
        np.testing.assert_allclose(C.predict(run(), X, np.arange(4), "cpu"), C.predict(run(), X, np.arange(4), "cpu"), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
