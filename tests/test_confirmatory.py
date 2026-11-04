"""Hand-checkable tests of ppglabbench.confirmatory. Synthetic data only: no MIMIC row is read, and `run` never gets past its guards.

Usage:  python -m unittest tests.test_confirmatory
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ppglabbench import confirmatory as C  # noqa: E402

BL = C.BL


def classic_delong_var(pos, neg):
    """DeLong, DeLong & Clarke-Pearson (1988) by explicit loops: S10/m + S01/n with m-1 and n-1 divisors."""
    psi = lambda a, b: 1.0 if a > b else 0.5 if a == b else 0.0
    m, n = len(pos), len(neg)
    v10 = np.array([np.mean([psi(p, q) for q in neg]) for p in pos])
    v01 = np.array([np.mean([psi(p, q) for p in pos]) for q in neg])
    return v10.mean(), v10.var(ddof=1) / m + v01.var(ddof=1) / n


def obuchowski_var(y, p, g):
    """Obuchowski (1997) by explicit loops over subjects: S10/M + S01/N + 2 S11/(MN)."""
    psi = lambda a, b: 1.0 if a > b else 0.5 if a == b else 0.0
    pos, neg = p[y == 1], p[y == 0]
    M, N = len(pos), len(neg)
    auc = np.mean([psi(a, b) for a in pos for b in neg])
    subjects = np.unique(g)
    I = len(subjects)
    s10 = s01 = s11 = 0.0
    for u in subjects:
        v10 = sum(np.mean([psi(p[i], b) for b in neg]) for i in np.flatnonzero((g == u) & (y == 1)))
        v01 = sum(np.mean([psi(a, p[j]) for a in pos]) for j in np.flatnonzero((g == u) & (y == 0)))
        m_i, n_i = int(((g == u) & (y == 1)).sum()), int(((g == u) & (y == 0)).sum())
        s10 += (v10 - m_i * auc) ** 2
        s01 += (v01 - n_i * auc) ** 2
        s11 += (v10 - m_i * auc) * (v01 - n_i * auc)
    k = I / (I - 1)
    return auc, k * s10 / M ** 2 + k * s01 / N ** 2 + 2 * k * s11 / (M * N)


def se_of(y, p, g):
    d, G = C.dense(g)
    auc, t = C.auc_influence(np.asarray(y, bool), np.asarray(p, float), d, G)
    return auc, C.cluster_se(t, G)


class TestClusterDeLong(unittest.TestCase):
    def test_tiny_by_hand(self):
        """Positives .9, .8; negatives .7, .85. Pairs won: 3 of 4, AUROC .75."""
        y = np.array([1, 1, 0, 0])
        p = np.array([.9, .8, .7, .85])
        auc, var = classic_delong_var(p[y == 1], p[y == 0])     # V10 = 1, .5; V01 = 1, .5; S = .125 each; .125/2 + .125/2
        self.assertAlmostEqual(auc, .75)
        self.assertAlmostEqual(var, .125)
        # every window its own subject: T = +.125, -.125, +.125, -.125; sum T^2 = .0625; times 4/3
        auc, se = se_of(y, p, np.arange(4))
        self.assertAlmostEqual(auc, .75)
        self.assertAlmostEqual(se, np.sqrt(.0625 * 4 / 3))

    def test_cluster_changes_se_by_hand(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([.9, .8, .7, .85])
        # subjects {.9, .85} and {.8, .7}: T_g = .125 - .125 = 0 for both, so SE = 0
        self.assertAlmostEqual(se_of(y, p, np.array([0, 1, 1, 0]))[1], 0.0)
        # subjects {.9, .7} and {.8, .85}: T_g = +.25, -.25; sum T^2 = .125; times 2/1 = .25
        self.assertAlmostEqual(se_of(y, p, np.array([0, 1, 0, 1]))[1], .5)

    def test_hanley_mcneil_1982(self):
        """The published rating table (Radiology 143:29): Wilcoxon area .8932, SE .0320 by their exponential approximation."""
        neg = np.repeat([1, 2, 3, 4, 5], [33, 6, 6, 11, 2])
        pos = np.repeat([1, 2, 3, 4, 5], [3, 2, 2, 11, 33])
        auc, var = classic_delong_var(pos, neg)
        self.assertAlmostEqual(auc, .8932, places=4)
        self.assertAlmostEqual(np.sqrt(var), .0320, delta=.004)      # DeLong is not their formula, but must be close
        y, p = np.r_[np.ones(51, int), np.zeros(58, int)], np.r_[pos, neg].astype(float)
        got_auc, got_se = se_of(y, p, np.arange(109))
        self.assertAlmostEqual(got_auc, auc)
        m, n, G = 51, 58, 109       # the unclustered form differs from classic DeLong only in its divisors
        s10 = np.array([np.mean(np.where(a > neg, 1, np.where(a == neg, .5, 0))) for a in pos]).var(ddof=1)
        s01 = np.array([np.mean(np.where(pos > b, 1, np.where(pos == b, .5, 0))) for b in neg]).var(ddof=1)
        self.assertAlmostEqual(got_se ** 2, G / (G - 1) * (s10 * (m - 1) / m ** 2 + s01 * (n - 1) / n ** 2))

    def test_against_brute_force_clustered(self):
        rng = np.random.default_rng(1)
        g = rng.integers(0, 15, 60)
        y = (rng.random(60) < .4).astype(int)
        p = np.round(y + rng.normal(0, 1, 60), 1)       # rounding makes ties
        auc0, var0 = obuchowski_var(y, p, g)
        auc, se = se_of(y, p, g)
        self.assertAlmostEqual(auc, auc0)
        self.assertAlmostEqual(se ** 2, var0)

    def test_subject_jackknife_agrees(self):
        rng = np.random.default_rng(2)
        G = 150
        nw = rng.integers(1, 6, G)
        g = np.repeat(np.arange(G), nw)
        b = rng.normal(0, 1, G)[g]
        y = ((b + rng.normal(0, 1, len(g))) > 0.3).astype(int)
        p = .6 * b + .4 * y + rng.normal(0, 1, len(g))
        _, se = se_of(y, p, g)
        loo = [se_of(y[g != u], p[g != u], g[g != u])[0] for u in range(G)]
        jack = np.sqrt((G - 1) / G * np.sum((np.array(loo) - np.mean(loo)) ** 2))
        self.assertAlmostEqual(se / jack, 1, delta=.03)

    def test_copies_inside_a_subject_do_not_shrink_the_se(self):
        rng = np.random.default_rng(3)
        y = (rng.random(80) < .5).astype(int)
        p = y + rng.normal(0, 1, 80)
        auc, se = se_of(y, p, np.arange(80))
        y2, p2 = np.tile(y, 2), np.tile(p, 2)
        auc2, se2 = se_of(y2, p2, np.tile(np.arange(80), 2))
        self.assertAlmostEqual(auc, auc2)
        self.assertAlmostEqual(se, se2, places=10)
        self.assertLess(se_of(y2, p2, np.arange(160))[1], .8 * se)    # treated as independent, the SE is wrongly small

    def test_delta_and_floor(self):
        rng = np.random.default_rng(4)
        y = (rng.random(300) < .5)
        g = np.arange(300)
        p = y + rng.normal(0, 1, 300)
        same = C.delta_auc(y, p, p, g, 300)
        self.assertEqual((same["delta"], same["se"]), (0.0, 0.0))
        self.assertTrue(np.isnan(C.safe_z(same["delta"], same["se"])))
        below = -p                                       # a below-chance base: gain over .5, stack's own SE
        res = C.delta_auc(y, p, below, g, 300)
        a1, t1 = C.auc_influence(y, p, g, 300)
        self.assertAlmostEqual(res["delta"], a1 - .5)
        self.assertAlmostEqual(res["se"], C.cluster_se(t1, 300))


class TestScoreTest(unittest.TestCase):
    def test_no_covariates_by_hand(self):
        """p_hat = .5, so r = x - mean(x) = 5/3, 2/3, -1/3, -4/3, -1/3, -1/3 and e = +-.5.
        U = 7/6, 5/6, 0; sum 2; sum of squares 74/36; c = 3/2 * 5/5; z = 2 / sqrt(1.5 * 74/36)."""
        y = np.array([1, 1, 0, 0, 1, 0.])
        x = np.array([2, 1, 0, -1, 0, 0.])
        g = np.array([0, 0, 1, 1, 2, 2])
        z, r, e = C.score_test(y, np.empty((6, 0)), x, g, 3)
        np.testing.assert_allclose(r, [5 / 3, 2 / 3, -1 / 3, -4 / 3, -1 / 3, -1 / 3], atol=1e-6)
        np.testing.assert_allclose(e, [.5, .5, -.5, -.5, .5, -.5], atol=1e-6)
        self.assertAlmostEqual(z, 2 / np.sqrt(1.5 * 74 / 36), places=5)

    def test_one_binary_covariate_by_hand(self):
        """A saturated design: p_hat is the stratum mean, weights are constant inside a stratum, r is x minus its stratum mean."""
        y = np.array([1, 0, 0, 0, 1, 1, 1, 0.])
        s = np.array([0, 0, 0, 0, 1, 1, 1, 1.])
        x = np.array([3, 1, 0, 0, 2, 2, 0, 0.])
        g = np.arange(8)
        z, r, e = C.score_test(y, s[:, None], x, g, 8)
        p = np.where(s == 0, .25, .75)
        rh = x - np.where(s == 0, x[s == 0].mean(), x[s == 1].mean())
        u = rh * (y - p)
        self.assertAlmostEqual(z, u.sum() / np.sqrt(8 / 7 * (7 / 6) * np.sum(u * u)), places=4)

    def test_invariances(self):
        rng = np.random.default_rng(5)
        G = 200
        nw = rng.integers(1, 5, G)
        g = np.repeat(np.arange(G), nw)
        n = len(g)
        Z = rng.normal(0, 1, (n, 3))
        x = rng.normal(0, 1, n) + .3 * Z[:, 0]
        y = (rng.random(n) < 1 / (1 + np.exp(-Z[:, 0] - .5 * x))).astype(float)
        z, *_ = C.score_test(y, Z, x, g, G)
        self.assertAlmostEqual(C.score_test(y, Z, 7 + 3 * x, g, G)[0], z, places=4)     # location and scale of the score
        self.assertAlmostEqual(C.score_test(y, Z, -x, g, G)[0], -z, places=6)           # sign
        self.assertGreater(z, 2)                                                       # a real association shows
        self.assertTrue(np.isnan(C.score_test(y, Z, 2 * Z[:, 1] - Z[:, 2], g, G)[0]))   # nothing left after the design

    def test_copies_inside_a_subject(self):
        rng = np.random.default_rng(6)
        n = 120
        y = (rng.random(n) < .4).astype(float)
        x = rng.normal(0, 1, n)
        z1 = C.score_test(y, np.empty((n, 0)), x, np.arange(n), n)[0]
        z2 = C.score_test(np.tile(y, 2), np.empty((2 * n, 0)), np.tile(x, 2), np.tile(np.arange(n), 2), n)[0]
        self.assertAlmostEqual(z1, z2, places=2)      # differs only through (n-1)/(n-k)

    def test_null_size(self):
        rng = np.random.default_rng(7)
        zs = []
        for _ in range(400):
            G = 150
            nw = rng.integers(1, 5, G)
            g = np.repeat(np.arange(G), nw)
            n = len(g)
            Z = rng.normal(0, 1, (n, 4))
            b = rng.normal(0, 1, G)[g]
            y = ((Z[:, 0] + b + rng.logistic(0, 1, n)) > .5).astype(float)
            x = .5 * Z[:, 0] + rng.normal(0, 1, G)[g] + rng.normal(0, 1, n)     # independent of y given Z, clustered as y is
            zs.append(C.score_test(y, Z, x, g, G)[0])
        self.assertAlmostEqual(np.std(zs), 1, delta=.12)
        self.assertAlmostEqual(np.mean(zs), 0, delta=.15)

    def test_failure_is_nan(self):
        y = np.ones(10)
        self.assertTrue(np.isnan(C.score_test(y, np.empty((10, 0)), np.arange(10.), np.arange(10), 10)[0]))

    def test_eval_design(self):
        Z = np.array([[1, np.nan, np.nan, 5], [2, np.nan, np.nan, 5], [3, 1, 7, 5], [4, 3, 9, 5.]])
        D = C.eval_design(Z)
        # column 0, one shared indicator, the two imputed columns; the constant column goes
        self.assertEqual(D.shape, (4, 4))
        self.assertEqual(sum(set(np.unique(D[:, j])) == {0.0, 1.0} for j in range(4)), 1)
        self.assertTrue(np.isfinite(D).all())


class TestSequence(unittest.TestCase):
    def test_thresholds(self):
        self.assertAlmostEqual(C.threshold(len(C.FAMILY_LABS)), 2.807, places=3)
        self.assertAlmostEqual(C.threshold(16), 2.955, places=3)
        self.assertAlmostEqual(C.threshold(8), 2.734, places=3)
        self.assertAlmostEqual(C.threshold(1), 1.960, places=3)

    def test_nothing_clears(self):
        s = C.graded_sequence(np.zeros((10, 3)))
        self.assertFalse(s["reject"].any())
        self.assertEqual(s["final_A"], 10)
        np.testing.assert_allclose(s["final_thresholds"], 2.807, atol=5e-4)

    def test_threshold_is_exactly_at_the_line(self):
        Z = np.zeros((10, 3))
        Z[0, 0] = C.threshold(10) + 1e-9
        Z[1, 0] = C.threshold(10) - 1e-9
        s = C.graded_sequence(Z)
        self.assertTrue(s["reject"][0, 0])
        self.assertFalse(s["reject"][1, 0])

    def test_fixed_sequence_inside_a_lab(self):
        Z = np.zeros((10, 3))
        Z[0] = 1.0, 9.0, 9.0           # a large L2 or L3 without L1 is not rejected
        Z[1] = 9.0, 1.0, 9.0           # L3 waits for L2
        s = C.graded_sequence(Z)
        self.assertEqual(s["reject"][0].tolist(), [False, False, False])
        self.assertEqual(s["reject"][1].tolist(), [True, False, False])

    def test_passing_a_level_keeps_the_lab_in_A(self):
        """L1 and L2 cleared but L3 pending: the lab still holds its share, so A stays 10."""
        Z = np.zeros((10, 3))
        Z[0] = 9.0, 9.0, 0.0
        Z[1, 0] = 2.79                 # between threshold(9) = 2.773 and threshold(10) = 2.807
        s = C.graded_sequence(Z)
        self.assertEqual(s["final_A"], 10)
        self.assertFalse(s["reject"][1, 0])

    def test_alpha_recycles_only_when_a_lab_leaves(self):
        self.assertAlmostEqual(C.threshold(9), 2.773, places=3)
        Z = np.zeros((10, 3))
        Z[0] = 9.0, 9.0, 9.0           # leaves after L3, so A becomes 9
        Z[1, 0] = 2.79                 # rejected only then
        s = C.graded_sequence(Z)
        self.assertEqual(s["final_A"], 9)
        self.assertTrue(s["reject"][1, 0])
        self.assertAlmostEqual(s["threshold"][1, 0], C.threshold(9))
        order = [(t["lab"], t["level"]) for t in s["trace"]]
        self.assertEqual(order, [(0, 1), (0, 2), (0, 3), (1, 1)])
        self.assertEqual([t["A"] for t in s["trace"]], [10, 10, 10, 9])

    def test_cascade_to_the_last_lab(self):
        """Every lab clears everything at its A: the last one meets 1.96."""
        Z = np.tile(np.array([[9.0, 9.0, 9.0]]), (10, 1))
        Z[9] = 2.0, 2.0, 2.0
        s = C.graded_sequence(Z)
        self.assertTrue(s["reject"].all())
        self.assertEqual(s["final_A"], 0)
        self.assertTrue(np.isnan(s["final_thresholds"]).all())
        self.assertAlmostEqual(s["threshold"][9, 0], C.threshold(1))

    def test_nan_never_rejects_and_stays_in_A(self):
        Z = np.tile(np.array([[9.0, 9.0, 9.0]]), (10, 1))
        Z[3] = np.nan
        s = C.graded_sequence(Z)
        self.assertFalse(s["reject"][3].any())
        self.assertEqual(s["final_A"], 1)
        self.assertEqual(s["reject"].sum(), 27)

    def test_order_of_labs_does_not_matter(self):
        rng = np.random.default_rng(8)
        for _ in range(200):
            Z = rng.normal(2.2, 1.2, (10, 3))
            p = rng.permutation(10)
            self.assertTrue((C.graded_sequence(Z)["reject"][p] == C.graded_sequence(Z[p])["reject"]).all())

    def test_error_under_the_global_null(self):
        rng = np.random.default_rng(9)
        hits = np.mean([C.graded_sequence(rng.standard_normal((10, 3)))["reject"].any() for _ in range(20000)])
        self.assertLessEqual(hits, .025 + 3 * np.sqrt(.025 * .975 / 20000))     # independent tests at L1: 1 - (1 - .025/10)^10 = .0247
        self.assertGreater(hits, .02)


class TestCarvedOutAlpha(unittest.TestCase):
    """The weighted procedure: 0.02 over 10 family labs and 0.005 over the 2 named hypotheses."""

    def setUp(self):
        self.shares = C.alpha_shares(10, 2)
        self.levels = [3] * 10 + [2] * 2

    def test_shares_and_first_thresholds_by_hand(self):
        self.assertAlmostEqual(self.shares.sum(), .025)
        np.testing.assert_allclose(self.shares[:10], .002)
        np.testing.assert_allclose(self.shares[10:], .0025)
        self.assertAlmostEqual(C.threshold(10, C.ALPHA_A), 2.878, places=3)
        self.assertAlmostEqual(C.threshold(2, C.ALPHA_NAMED), 2.807, places=3)

    def test_nothing_clears_keeps_the_starting_thresholds(self):
        s = C.graded_sequence(np.zeros((12, 3)), shares=self.shares, levels=self.levels)
        np.testing.assert_allclose(s["final_thresholds"][:10], 2.878, atol=5e-4)
        np.testing.assert_allclose(s["final_thresholds"][10:], 2.807, atol=5e-4)
        self.assertEqual(s["final_A"], 12)

    def test_a_named_hypothesis_leaves_after_two_levels_and_passes_its_alpha_on(self):
        Z = np.zeros((12, 3))
        Z[10] = 9.0, 9.0, np.nan
        Z[0, 0] = 2.86          # below 2.878 at the start; after one named branch of .0025 has left, .002 * .025 / .0225 gives z = 2.845
        s = C.graded_sequence(Z, shares=self.shares, levels=self.levels)
        self.assertEqual(s["final_A"], 11)
        self.assertTrue(s["reject"][10, :2].all())
        self.assertTrue(s["reject"][0, 0])
        self.assertAlmostEqual(s["threshold"][0, 0], norm.isf(.002 * .025 / .0225))
        self.assertAlmostEqual(s["threshold"][0, 0], 2.845, places=3)
        self.assertEqual([(t["lab"], t["level"]) for t in s["trace"]], [(10, 1), (10, 2), (0, 1)])

    def test_a_family_lab_passing_on_gives_the_named_more_in_proportion(self):
        Z = np.zeros((12, 3))
        Z[0] = 9.0, 9.0, 9.0
        s = C.graded_sequence(Z, shares=self.shares, levels=self.levels)
        self.assertAlmostEqual(s["final_thresholds"][10], norm.isf(.0025 * .025 / .023))
        self.assertAlmostEqual(s["final_thresholds"][1], norm.isf(.002 * .025 / .023))

    def test_the_last_branch_gets_all_of_alpha(self):
        Z = np.tile(np.array([[9.0, 9.0, 9.0]]), (12, 1))
        Z[1] = 2.0, 2.0, 2.0
        s = C.graded_sequence(Z, shares=self.shares, levels=self.levels)
        self.assertTrue(s["reject"][1].all())
        self.assertAlmostEqual(s["threshold"][1, 0], 1.960, places=3)

    def test_nan_holds_its_share(self):
        Z = np.tile(np.array([[9.0, 9.0, 9.0]]), (12, 1))
        Z[11] = np.nan
        s = C.graded_sequence(Z, shares=self.shares, levels=self.levels)
        self.assertEqual(s["final_A"], 1)
        self.assertFalse(s["reject"][11].any())

    def test_order_of_branches_does_not_matter(self):
        rng = np.random.default_rng(8)
        for _ in range(200):
            Z = rng.normal(2.3, 1.2, (12, 3))
            Z[10:, 2] = np.nan
            p = rng.permutation(12)
            a = C.graded_sequence(Z, shares=self.shares, levels=self.levels)["reject"][p]
            b = C.graded_sequence(Z[p], shares=self.shares[p], levels=np.array(self.levels)[p])["reject"]
            self.assertTrue((a == b).all())

    def test_error_under_the_global_null(self):
        rng = np.random.default_rng(9)
        Z = rng.standard_normal((20000, 12, 3))
        hits = np.mean([C.graded_sequence(z, shares=self.shares, levels=self.levels)["reject"].any() for z in Z])
        self.assertLessEqual(hits, .025 + 3 * np.sqrt(.025 * .975 / 20000))
        self.assertGreater(hits, .02)


class TestUpperBoundChain(unittest.TestCase):
    @staticmethod
    def base(upper, labs=10):
        return dict(k=10, labs=labs, mean=upper - .004, lo=upper - .008, hi=upper - .001, training_sd=.001, upper=upper)

    def test_fixed_sequence_stops_at_the_first_failure(self):
        steps = C.upper_chain(dict(full=self.base(.005), now=self.base(.012), monitor=self.base(.005), acuity=self.base(.005)), .01)
        self.assertEqual([s["tested"] for s in steps], [True, True, False, False])
        self.assertEqual([s["passed"] for s in steps], [True, False, False, False])

    def test_every_step_reports_its_mean_and_bound_tested_or_not(self):
        steps = C.upper_chain(dict(full=self.base(.02), now=self.base(.005)), .01)
        self.assertEqual([s["tested"] for s in steps], [True, False])
        self.assertFalse(steps[1]["passed"])
        self.assertEqual(steps[1]["upper"], .005)
        self.assertEqual(steps[1]["mean"], .001)

    def test_the_bound_has_to_be_below_the_margin(self):
        self.assertFalse(C.upper_chain(dict(full=self.base(.01)), .01)[0]["passed"])
        self.assertTrue(C.upper_chain(dict(full=self.base(.00999)), .01)[0]["passed"])

    def test_a_lab_missing_from_the_mean_cannot_pass(self):
        self.assertFalse(C.upper_chain(dict(full=self.base(.005, labs=9)), .01)[0]["passed"])

    def test_no_bound_cannot_pass(self):
        for upper in (None, float("nan")):
            x = dict(self.base(.005), upper=upper)
            self.assertFalse(C.upper_chain(dict(full=x), .01)[0]["passed"])

    def test_training_sd_takes_the_labs_as_fully_correlated(self):
        self.assertAlmostEqual(C.training_sd([.25, 1.0], [.02, .01]), (.5 * .02 + 1.0 * .01) / 2)
        # above what independent labs would give, and equal when the labs are the same
        self.assertGreater(C.training_sd([.25, 1.0], [.02, .01]), np.sqrt((.5 * .02) ** 2 + (1.0 * .01) ** 2) / 2)
        self.assertAlmostEqual(C.training_sd([.25] * 4, [.02] * 4), .5 * .02)

    def test_mean_upper_by_hand(self):
        self.assertAlmostEqual(C.mean_upper(.002, .005, 0.0), .005)      # no training part: the bootstrap percentile
        self.assertAlmostEqual(C.mean_upper(.002, .005, .001), .002 + np.sqrt(.003 ** 2 + (1.959964 * .001) ** 2), places=6)
        self.assertAlmostEqual(C.mean_upper(.002, .001, 0.0), .002)      # a percentile below the mean adds nothing

    def test_a_true_mean_at_the_margin_is_rejected_about_alpha_with_a_mix_of_labs(self):
        """Independent labs, gains that differ, a normal bootstrap stand-in with its own SE: the bound falls below the margin 2.5% of the time."""
        rng = np.random.default_rng(0)
        gains, se = np.array([0, 0, 0, .002, .002, .01, .01, .02, .028, .028]), np.array([.002, .002, .002, .003, .003, .005, .005, .006, .007, .007])
        self.assertAlmostEqual(gains.mean(), .01, places=6)
        means = rng.normal(gains, se, (100000, 10)).mean(1)
        sd = np.sqrt(np.sum(se ** 2)) / 10
        hits = np.mean([C.mean_upper(m, m + 1.959964 * sd, 0.0) < .01 for m in means[:40000]])
        self.assertAlmostEqual(hits, .025, delta=.004)


class TestMeanGain(unittest.TestCase):
    def test_weighted_auc_matches_sklearn_with_ties_and_weights(self):
        rng = np.random.default_rng(0)
        y = rng.random(300) < .4
        p = np.round(y + rng.normal(0, 1, 300), 1)
        w = rng.integers(0, 4, (5, 300)).astype(float)
        got = C.weighted_auc(C.tie_groups(p), y, w)
        for i in range(5):
            self.assertAlmostEqual(got[i], roc_auc_score(y, p, sample_weight=w[i]))

    def arrays(self, seed, groups):
        rng = np.random.default_rng(seed)
        y = rng.random(len(groups)) < .4
        return dict(y=y.astype(np.int8), group=groups, now=rng.normal(0, 1, len(y)) + .5 * y, **{"now+beat": rng.normal(0, 1, len(y)) + y})

    def test_the_point_is_the_mean_of_the_lab_gains_and_the_interval_holds_it(self):
        groups = np.repeat(np.arange(9000, 9300), 3)
        labs = dict(a=self.arrays(1, groups), b=self.arrays(2, groups))
        m = C.mean_gain(labs, "now", dict(a="beat", b="beat"), B=200)
        want = [roc_auc_score(x["y"], x["now+beat"]) - roc_auc_score(x["y"], x["now"]) for x in labs.values()]
        self.assertAlmostEqual(m["mean"], np.mean(want))
        self.assertLess(m["lo"], m["mean"])
        self.assertGreater(m["hi"], m["mean"])
        self.assertEqual(m["labs"], 2)

    def test_one_bootstrap_is_shared_by_all_labs(self):
        """Two copies of one lab: the mean's interval equals that lab's, because every draw resamples the same subjects in both."""
        groups = np.repeat(np.arange(300), 3)
        a = self.arrays(3, groups)
        one = C.mean_gain(dict(a=a), "now", dict(a="beat"), B=200)
        two = C.mean_gain(dict(a=a, b=a), "now", dict(a="beat", b="beat"), B=200)
        self.assertAlmostEqual(one["lo"], two["lo"])
        self.assertAlmostEqual(one["hi"], two["hi"])


class TestPick(unittest.TestCase):
    def test_highest_gain_wins(self):
        self.assertEqual(C.pick_model(dict(papagei=.03, beat=.01, pulseppg=.02)), "papagei")

    def test_a_tie_at_three_decimals_goes_by_the_fixed_order(self):
        self.assertEqual(C.pick_model(dict(papagei=.0201, beat=.0199, pulseppg=.0202)), "beat")
        self.assertEqual(C.pick_model(dict(papagei=.0201, pulseppg=.0202, beat=.01)), "pulseppg")
        self.assertEqual(C.pick_model(dict(papagei=.0204, beat=.0196, pulseppg=.01)), "beat")
        self.assertEqual(C.TIE_ORDER[-1], "papagei")          # the model pretrained on these cohorts comes last

    def test_a_gain_that_cannot_be_computed_is_ignored(self):
        self.assertEqual(C.pick_model(dict(papagei=float("nan"), beat=.01)), "beat")
        self.assertIsNone(C.pick_model(dict(papagei=float("nan"))))

    def test_every_candidate_has_a_place(self):
        self.assertEqual(set(C.TIE_ORDER), set(C.MODELS))

    def test_the_cnn_in_a_tie(self):
        self.assertEqual(C.pick_model(dict(papagei=.02, beat=.01, pulseppg=.01, cnn=.02)), "cnn")
        self.assertEqual(C.pick_model(dict(papagei=.02, beat=.02, pulseppg=.01, cnn=.02)), "beat")
        self.assertEqual(C.pick_model(dict(papagei=.02, beat=.01, pulseppg=.02, cnn=.02)), "pulseppg")


class TestSecondaryStatistics(unittest.TestCase):
    def test_delta_r2_and_partial_r2_by_hand(self):
        y = np.array([1., 2., 3., 4.])
        res = C.delta_r2(y, y, np.full(4, 2.5), np.arange(4), 4)
        self.assertAlmostEqual(res["delta"], 1.0)
        self.assertAlmostEqual(res["partial_r2"], 1.0)
        res = C.delta_r2(y, np.array([1.5, 2., 3., 3.5]), np.array([2., 2., 3., 3.]), np.arange(4), 4)
        # SST 5; SSE stack .5, SSE base 2: gain .3; partial 1 - .5/2 = .75
        self.assertAlmostEqual(res["delta"], .3)
        self.assertAlmostEqual(res["partial_r2"], .75)

    def test_delta_r2_se_against_a_subject_bootstrap(self):
        rng = np.random.default_rng(1)
        G = 400
        g = np.repeat(np.arange(G), 3)
        b = rng.normal(0, 1, G)[g]
        y = b + rng.normal(0, 1, len(g))
        base, stack = .5 * b, .9 * b + .1 * rng.normal(0, 1, len(g))
        se = C.delta_r2(y, stack, base, g, G)["se"]
        boots = []
        for _ in range(400):
            ss = rng.integers(0, G, G)
            ro = (ss[:, None] * 3 + np.arange(3)).ravel()
            boots.append(C.delta_r2(y[ro], stack[ro], base[ro], np.repeat(np.arange(G), 3), G)["delta"])
        self.assertAlmostEqual(se / np.std(boots), 1, delta=.15)

    def test_brier_skill_by_hand(self):
        res = C.brier_skill(np.array([1., 0.]), np.array([.8, .2]), np.array([.5, .5]), np.arange(2), 2)
        self.assertAlmostEqual(res["skill"], 1 - .04 / .25)
        self.assertAlmostEqual(res["brier_stack"], .04)
        self.assertAlmostEqual(res["brier_base"], .25)

    def test_calibration_of_a_true_and_an_overconfident_model(self):
        rng = np.random.default_rng(2)
        eta = rng.normal(-.5, 1.2, 40000)
        p = 1 / (1 + np.exp(-eta))
        y = rng.random(40000) < p
        g = np.arange(40000)
        good = C.calibration(y, p, g, 40000)
        self.assertAlmostEqual(good["slope"], 1, delta=.04)
        self.assertAlmostEqual(good["intercept"], 0, delta=.04)
        wild = C.calibration(y, 1 / (1 + np.exp(-2 * eta)), g, 40000)
        self.assertAlmostEqual(wild["slope"], .5, delta=.03)
        shifted = C.calibration(y, 1 / (1 + np.exp(-(eta + .7))), g, 40000)
        self.assertAlmostEqual(shifted["intercept"], -.7, delta=.05)

    def test_net_benefit_by_hand(self):
        y, p = np.array([1., 1., 0., 0.]), np.array([.9, .2, .6, .1])
        self.assertAlmostEqual(C.net_benefit(y, p, .5).mean(), (1 - 1) / 4)
        self.assertAlmostEqual(C.net_benefit(y, p, .1).mean(), (2 - 2 * .1 / .9) / 4)
        rows = C.net_benefit_rows(y, p, np.full(4, .5), np.arange(4), 4)
        self.assertEqual([r["threshold"] for r in rows], list(C.NET_BENEFIT_AT))
        self.assertAlmostEqual(rows[0]["act_on_all"], (2 - 2 * .05 / .95) / 4)

    def test_strata_gains_and_heterogeneity(self):
        rng = np.random.default_rng(3)
        G = 1200
        g = np.arange(G)
        y = rng.random(G) < .4
        a = g < 600
        base = rng.normal(0, 1, G) + .3 * y
        same = base + 1.0 * y + .5 * rng.normal(0, 1, G)
        res = C.strata_gains(y, same, base, g, G, dict(a=a, b=~a))
        s = res["strata"]
        q = (s["a"]["delta"] - s["b"]["delta"]) ** 2 / (s["a"]["se"] ** 2 + s["b"]["se"] ** 2)
        self.assertAlmostEqual(res["heterogeneity"]["q"], q, delta=.15 * max(q, 1))       # disjoint subjects: the two-group formula
        self.assertEqual(res["heterogeneity"]["df"], 1)
        differ = np.where(a, base + 1.5 * y + .5 * rng.normal(0, 1, G), base)
        self.assertLess(C.strata_gains(y, differ, base, g, G, dict(a=a, b=~a))["heterogeneity"]["p"], .001)

    def test_a_small_stratum_gives_counts_only(self):
        rng = np.random.default_rng(4)
        y = rng.random(500) < .5
        p = y + rng.normal(0, 1, 500)
        small = np.zeros(500, bool)
        small[:10] = True
        res = C.strata_gains(y, p, rng.normal(0, 1, 500), np.arange(500), 500, dict(small=small, rest=~small))
        self.assertNotIn("delta", res["strata"]["small"])
        self.assertEqual(res["strata"]["small"]["n"], 10)
        self.assertEqual(res["heterogeneity"]["levels"], 1)
        self.assertIsNone(res["heterogeneity"]["q"])

    def test_previous_age_strata_are_fixed_and_cover_every_row(self):
        a = np.array([0., 12., 12.1, 24., 48., 48.1, np.nan])
        m = C.previous_age_masks(a)
        self.assertEqual(list(m), ["le_12h", "12_24h", "24_48h", "gt_48h", "never"])
        self.assertEqual(np.sum(list(m.values()), axis=0).tolist(), [1] * 7)
        self.assertEqual(m["le_12h"].tolist()[:3], [True, True, False])
        self.assertEqual(m["gt_48h"].tolist()[4:6], [False, True])
        self.assertTrue(m["never"][6])

    def test_score_test_linear_by_hand(self):
        y = np.array([1., 2., 3., 4., 5., 9.])
        x = np.array([2., 1., 0., -1., 0., 4.])
        g = np.array([0, 0, 1, 1, 2, 2])
        z, r, e = C.score_test_linear(y, np.empty((6, 0)), x, g, 3)
        np.testing.assert_allclose(r, x - x.mean(), atol=1e-12)
        np.testing.assert_allclose(e, y - y.mean(), atol=1e-12)
        u = np.bincount(g, r * e, minlength=3)
        self.assertAlmostEqual(z, u.sum() / np.sqrt(3 / 2 * 5 / 5 * np.sum(u * u)))

    def test_score_test_linear_null_size(self):
        rng = np.random.default_rng(5)
        zs = []
        for _ in range(400):
            G = 150
            g = np.repeat(np.arange(G), rng.integers(1, 5, G))
            n = len(g)
            Z = rng.normal(0, 1, (n, 4))
            y = Z[:, 0] + rng.normal(0, 1, G)[g] + rng.normal(0, 1, n)
            x = .5 * Z[:, 0] + rng.normal(0, 1, G)[g] + rng.normal(0, 1, n)
            zs.append(C.score_test_linear(y, Z, x, g, G)[0])
        self.assertAlmostEqual(np.std(zs), 1, delta=.12)
        self.assertAlmostEqual(np.mean(zs), 0, delta=.15)

    def test_the_residual_arm_finds_an_interaction_a_single_score_misses(self):
        """PPG acts only at high base risk: a stack on one score spreads its one weight over all rows, the arm need not."""
        rng = np.random.default_rng(6)
        G = 2500
        g = np.repeat(np.arange(G), 2)
        n = len(g)
        b = rng.normal(0, 1, G)[g] + .3 * rng.normal(0, 1, n)
        x = rng.normal(0, 1, (n, 4))
        eta_true = b + 2.5 * x[:, 0] * (b > .5) - 2.5 * x[:, 0] * (b <= .5) * 0
        y = (rng.random(n) < 1 / (1 + np.exp(-eta_true))).astype(float)
        tr, ev = np.flatnonzero(g < 1800), np.flatnonzero(g >= 1800)
        Z = b[:, None]
        eta = BL.base_clf().fit(Z[tr], y[tr]).decision_function(Z)
        r_tr, r_ev = C.residual_score(x, eta, y, g, tr, ev, False)
        predict = lambda s_tr, s_ev: BL.base_clf().fit(np.column_stack([Z[tr], s_tr]), y[tr]).predict_proba(np.column_stack([Z[ev], s_ev]))[:, 1]
        s_tr, s_ev = BL.ppg_score(BL.boost_clf, x, y.astype(int), g, tr, ev)
        d, G2 = C.dense(g[ev])
        arm, plain = C.delta_auc(y[ev].astype(bool), predict(r_tr, r_ev), predict(np.zeros(len(tr)), np.zeros(len(ev))), d, G2), None
        stack = C.delta_auc(y[ev].astype(bool), predict(s_tr, s_ev), predict(np.zeros(len(tr)), np.zeros(len(ev))), d, G2)
        self.assertGreater(arm["delta"], stack["delta"] + .01)

    def test_the_residual_arm_is_null_on_noise(self):
        rng = np.random.default_rng(7)
        G = 1500
        g = np.repeat(np.arange(G), 2)
        n = len(g)
        b = rng.normal(0, 1, n)
        x = rng.normal(0, 1, (n, 4))
        y = (rng.random(n) < 1 / (1 + np.exp(-b))).astype(float)
        tr, ev = np.flatnonzero(g < 1000), np.flatnonzero(g >= 1000)
        Z = b[:, None]
        eta = BL.base_clf().fit(Z[tr], y[tr]).decision_function(Z)
        r_tr, r_ev = C.residual_score(x, eta, y, g, tr, ev, False)
        base = BL.base_clf().fit(Z[tr], y[tr]).predict_proba(Z[ev])[:, 1]
        arm = BL.base_clf().fit(np.column_stack([Z[tr], r_tr]), y[tr]).predict_proba(np.column_stack([Z[ev], r_ev]))[:, 1]
        d, G2 = C.dense(g[ev])
        self.assertLess(C.delta_auc(y[ev].astype(bool), arm, base, d, G2)["delta"], .01)


class TestLogitScore(unittest.TestCase):
    def test_a_classifier_score_is_on_the_log_odds_scale_and_a_regressor_keeps_its_scale(self):
        rng = np.random.default_rng(0)
        n, g = 600, np.repeat(np.arange(200), 3)
        x = rng.normal(0, 1, (n, 3))
        y = (rng.random(n) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
        tr, ev = np.flatnonzero(g < 150), np.flatnonzero(g >= 150)
        s_tr, s_ev = BL.ppg_score(BL.boost_clf, x, y, g, tr, ev)
        p = BL.boost_clf().fit(x[tr], y[tr]).predict_proba(x[ev])[:, 1]
        np.testing.assert_allclose(s_ev, BL.clip_logit(p))
        self.assertGreater(np.abs(s_tr).max(), 1)        # not a probability
        v_tr, v_ev = BL.ppg_score(BL.boost_reg, x, x[:, 1] * 100, g, tr, ev)
        self.assertGreater(np.abs(v_ev).max(), 10)

    def test_each_fold_is_centred_on_the_log_odds_scale(self):
        """A model that predicts its training prevalence gives a score of exactly the train log-odds in every fold."""
        class Mean:
            def fit(self, X, y):
                self.p = y.mean()
                return self

            def predict_proba(self, X):
                return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])
        rng = np.random.default_rng(1)
        g = np.repeat(np.arange(100), 2)
        y = (rng.random(200) < .3).astype(int)
        tr, ev = np.flatnonzero(g < 80), np.flatnonzero(g >= 80)
        s_tr, s_ev = BL.ppg_score(Mean, np.zeros((200, 1)), y, g, tr, ev)
        np.testing.assert_allclose(s_tr, BL.clip_logit(y[tr].mean()))
        np.testing.assert_allclose(s_ev, BL.clip_logit(y[tr].mean()))

    def test_the_clip(self):
        self.assertAlmostEqual(BL.clip_logit(0.0), np.log(1e-4 / (1 - 1e-4)))
        self.assertAlmostEqual(BL.clip_logit(1.0), -np.log(1e-4 / (1 - 1e-4)))


class TestPlanHelpers(unittest.TestCase):
    def test_strata(self):
        self.assertEqual(C.stratum(999, .4), "sparse")
        self.assertEqual(C.stratum(427, .05), "sparse")          # sample size first
        self.assertEqual(C.stratum(1000, .05), "mid")
        self.assertEqual(C.stratum(2000, .4), "mid")
        self.assertEqual(C.stratum(2001, .14), "rare")
        self.assertEqual(C.stratum(3000, .87), "rare")
        self.assertEqual(C.stratum(3000, .15), "common")
        self.assertEqual(C.stratum(3000, .85), "common")

    def test_r_hat_by_hand(self):
        rows = [("common", .0001, .01), ("common", -.0005, .01), ("sparse", .0002, .02)]
        r = C.r_hat_by_stratum(rows)
        self.assertAlmostEqual(r["common"], 4 * .0001 / (2 * .01 ** 2))      # a negative covariance counts as 0
        self.assertAlmostEqual(r["sparse"], 4 * .0002 / .02 ** 2)

    def test_half_cov_sees_training_not_noise(self):
        """Refits differ by quality: positive covariance. Refits differ only by noise: covariance near 0."""
        rng = np.random.default_rng(10)
        G = 400
        g = np.arange(G)
        y = rng.random(G) < .5
        base = rng.normal(0, 1, G)
        strength = rng.normal(.8, .3, 20)
        stacks = [s * y + rng.normal(0, 1, G) for s in strength]
        c_real = C.half_cov(y, g, G, stacks, [base] * 20, np.random.default_rng(1), n_splits=10)
        same = [.8 * y + rng.normal(0, 1, G) for _ in strength]
        c_noise = C.half_cov(y, g, G, same, [base] * 20, np.random.default_rng(1), n_splits=10)
        self.assertGreater(c_real, 3 * abs(c_noise))
        self.assertGreater(c_real, 0)

    def test_plan_problems(self):
        good = dict(provisional=False, cnn_scores_sha256="c" * 64)
        self.assertEqual(C.plan_problems(good, [], "a" * 64, "a" * 64, "c" * 64), [])
        self.assertEqual(len(C.plan_problems(dict(good, provisional=True, provisional_reasons=["x"]), [], "a" * 64, "a" * 64, "c" * 64)), 1)
        self.assertEqual(len(C.plan_problems(good, ["src/ppglabbench/baselines.py"], "a" * 64, "a" * 64, "c" * 64)), 1)
        self.assertEqual(len(C.plan_problems(dict(cnn_scores_sha256="c" * 64), [], "a" * 64, "a" * 64, "c" * 64)), 1)     # a plan with no flag is not trusted

    def test_the_plan_is_bound_by_its_sha256(self):
        good = dict(provisional=False, cnn_scores_sha256="c" * 64)
        self.assertIn("not set", C.plan_problems(good, [], "a" * 64, None, "c" * 64)[0])
        self.assertIn("differs", C.plan_problems(good, [], "a" * 64, "b" * 64, "c" * 64)[0])

    def test_the_cnn_file_is_bound_too(self):
        good = dict(provisional=False, cnn_scores_sha256="c" * 64)
        self.assertIn("CNN score file", C.plan_problems(good, [], "a" * 64, "a" * 64, "d" * 64)[0])
        self.assertIn("CNN score file", C.plan_problems(good, [], "a" * 64, "a" * 64, None)[0])
        self.assertIn("CNN score file", C.plan_problems(dict(provisional=False, cnn_scores_sha256=None), [], "a" * 64, "a" * 64, None)[0])

    def test_a_failed_newton_step_keeps_the_last_finite_fit(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(200), rng.normal(size=200)])
        y = (rng.random(200) < .3).astype(float)
        solve = np.linalg.solve
        calls = []

        def failing(a, b):
            calls.append(1)
            return solve(a, b) if len(calls) < 2 else np.full(b.shape, np.nan)
        with mock.patch.object(C.np.linalg, "solve", failing):
            b, p, W = C.fit_logistic(y, X)
        self.assertEqual(len(calls), 2)
        self.assertTrue(np.isfinite(b).all() and np.isfinite(p).all() and np.isfinite(W).all())

    def test_provisional_reasons(self):
        self.assertEqual(C.provisional_reasons(list(C.MODELS), 50, "abc1234", "pre"), [])
        self.assertEqual(len(C.provisional_reasons(["beat", "papagei", "cnn"], 50, "abc1234", "pre")), 1)
        self.assertEqual(C.provisional_reasons(["beat", "papagei", "pulseppg"], 50, "abc1234", "pre"), ["model cnn not available"])
        self.assertEqual(len(C.provisional_reasons(list(C.MODELS), 5, "abc1234-dirty", "pre")), 2)
        self.assertEqual(len(C.provisional_reasons(list(C.MODELS), 50, "abc1234", "centered")), 1)

    def test_family_is_ten(self):
        self.assertEqual(len(C.FAMILY_LABS), 10)
        self.assertEqual(len(set(C.FAMILY_LABS)), 10)
        self.assertNotIn("creatinine", C.FAMILY_LABS)
        self.assertEqual(set(C.FAMILY_LABS) & set(C.DESCRIPTIVE_LABS), set())
        self.assertEqual(set(C.FAMILY_LABS) | set(C.DESCRIPTIVE_LABS), set(BL.C["labs"]) | {"egfr", "creatinine"})


def synthetic_cohort(labs=("bun", "potassium", "ast", "lactate"), n_train=110, n_val=60, seed=11, big_ids=True):
    """Train and val only, shaped like load_mimic's output. Subject IDs are large so a leak is easy to find."""
    rng = np.random.default_rng(seed)
    n_subj = n_train + n_val
    nw = rng.integers(1, 5, n_subj)
    sid = np.repeat(np.arange(n_subj), nw)
    ids = np.repeat(rng.permutation(n_subj) + 7_300_000, nw) if big_ids else sid
    n = len(sid)
    d = pd.DataFrame(dict(group=ids, split=np.where(sid < n_train, "train", "val")))
    d["age"], d["female"] = rng.normal(60, 15, n_subj)[sid], (rng.random(n_subj) < .45).astype(float)[sid]
    acuity = ["vasopressor", "emergency_admit", "map_mmhg", "map_lt70", "routine_draw", "hour_sin", "hour_cos"]
    d["vasopressor"] = (rng.random(n) < .1).astype(float)
    d["emergency_admit"] = (rng.random(n) < .8).astype(float)
    d["map_mmhg"] = np.where(rng.random(n) < .4, rng.normal(80, 12, n), np.nan)
    d["map_lt70"] = (d.map_mmhg < 70).astype(float).where(d.map_mmhg.notna())
    d["routine_draw"] = (rng.random(n) < .6).astype(float)
    d["hour_sin"], d["hour_cos"] = rng.normal(0, .7, n), rng.normal(0, .7, n)
    d["hr_mon"] = rng.normal(90, 15, n)
    d["on_heparin"] = (rng.random(n) < .2).astype(float)
    bcols = [f"b{i}" for i in range(6)]
    signal = rng.normal(0, 1, n)
    for i, c in enumerate(bcols):
        d[c] = rng.normal(0, 1, n) + (signal if i == 0 else 0)
    other = np.random.default_rng(seed + 1)     # its own stream, so that the rest of the cohort stays as it was
    for c in BL.RATE:
        d[c] = other.normal(0, 1, n) + .3 * signal
    emb = pd.DataFrame(rng.normal(0, 1, (n, 512)) + signal[:, None] * rng.normal(0, .05, 512), columns=BL.PG)
    d = pd.concat([d, emb], axis=1)
    spec = {}
    for name in labs:
        y = rng.normal(0, 1, n) + signal + .02 * (d.age.to_numpy() - 60)
        d[name], d[f"{name}_abnormal"] = y, (y > 0.3).astype(float)
        d[f"{name}_locf"], d[f"{name}_locf_age_h"] = np.exp(y / 5), rng.exponential(12, n)
        d[f"{name}_hist_mean"], d[f"{name}_hist_n"], d[f"{name}_trend"] = np.exp(y / 6), rng.integers(0, 9, n), rng.normal(0, 1, n)
        spec[name] = dict(value=name, abnormal=f"{name}_abnormal", locf=[f"{name}_locf", f"{name}_locf_age_h"],
                          hist=[f"{name}_hist_mean", f"{name}_hist_n", f"{name}_trend"])
    tiers = dict(acuity=acuity, monitor=acuity + ["hr_mon"], now=acuity + ["hr_mon", "on_heparin"])
    return d, bcols, spec, tiers


class TestFrame(unittest.TestCase):
    def test_eval_counts_do_not_drop_a_lab(self):
        d, bcols, spec, tiers = synthetic_cohort(labs=("ast",))
        d.loc[d.split == "val", "ast_abnormal"] = 0.0      # no abnormal row on eval
        self.assertIsNone(BL.task_frame(d, tiers, "ast", spec["ast"], "clf", "val"))
        frame = C.fixed_frame(d, tiers, "ast", spec["ast"], "val")
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame[3]), (d.split == "val").sum())
        self.assertEqual(BL.MIN_POS_EVAL, C.MIN_SUBJECTS)      # the guards come back
        self.assertIsNone(BL.task_frame(d, tiers, "ast", spec["ast"], "clf", "val"))

    def test_train_guard_stays(self):
        d, bcols, spec, tiers = synthetic_cohort(labs=("ast",))
        d.loc[d.split == "train", "ast_abnormal"] = 0.0
        self.assertIsNone(C.fixed_frame(d, tiers, "ast", spec["ast"], "val"))

    def test_same_bases_as_task_frame(self):
        d, bcols, spec, tiers = synthetic_cohort(labs=("bun",), n_train=400, n_val=200)
        a = BL.task_frame(d, tiers, "bun", spec["bun"], "clf", "val")
        b = C.fixed_frame(d, tiers, "bun", spec["bun"], "val")
        self.assertEqual(sorted(a[5]), sorted(b[5]))
        for k in a[5]:
            np.testing.assert_array_equal(a[5][k], b[5][k])

    def test_fit_models_matches_the_run_lab_stack(self):
        """fit_models against the closure in baselines.run_lab, written out."""
        d, bcols, spec, tiers = synthetic_cohort(labs=("bun",), n_train=200, n_val=100)
        q, y, tr, ev, groups, bases, _, _ = C.fixed_frame(d, tiers, "bun", spec["bun"], "val")
        X = q[bcols].to_numpy(float)
        got = C.fit_models(y, tr, ev, groups, {"now": bases["now"]}, {"beat": X})
        s_tr, s_ev = BL.ppg_score(BL.PPG_MODELS["beat"]["clf"], X, y, groups, tr, ev)
        s = np.zeros(len(q))
        s[tr], s[ev] = s_tr, s_ev
        Z = np.column_stack([bases["now"], s])
        want = BL.base_clf().fit(Z[tr], y[tr]).predict_proba(Z[ev])[:, 1]
        np.testing.assert_allclose(got["now+beat"], want)
        np.testing.assert_allclose(got["score:beat"], s_ev)


class TestPredictions(unittest.TestCase):
    def test_no_subject_ids(self):
        ids = np.array([7300001, 7300001, 7300002, 7300003, 7300003, 7300003])
        arrays = dict(y=np.array([1, 0, 1, 0, 0, 1], np.int8), group=ids, now=np.arange(6) / 7.)
        flat = C.anonymise_arrays(dict(bun=arrays, ast=dict(arrays, group=ids[::-1])), np.random.default_rng(0))
        for v in flat.values():
            self.assertFalse(np.isin(v, ids).any())
        self.assertLess(flat["bun__group"].max(), 3)
        self.assertEqual(flat["bun__group"].dtype, np.int32)
        for k in flat:
            self.assertFalse(any(str(i) in k for i in ids))
        # a subject keeps one code inside a lab, and each row keeps its y and prediction
        codes = pd.Series(flat["bun__now"]).groupby(flat["bun__group"]).size()
        self.assertEqual(sorted(codes.tolist()), [1, 2, 3])
        by_pred = {round(float(p) * 7): int(y) for p, y in zip(flat["bun__now"], flat["bun__y"])}
        self.assertEqual(by_pred, {i: int(arrays["y"][i]) for i in range(6)})


def stand_in_split_file(folder):
    """check_test_allowed hashes data/benchmark/mimic_splits.csv first: a stand-in, so that the guards run with no data/ folder."""
    (Path(folder) / "benchmark").mkdir()
    (Path(folder) / "benchmark" / "mimic_splits.csv").write_text("subject_id,split\n")
    return mock.patch.object(BL, "DATA", Path(folder))


class TestGuards(unittest.TestCase):
    def test_run_refuses_without_open_test_before_reading_data(self):
        with tempfile.TemporaryDirectory() as t, stand_in_split_file(t), \
                mock.patch.object(BL, "load_mimic", side_effect=AssertionError("data was read")):
            with self.assertRaises(SystemExit) as e:
                C.cmd_run(Namespace(open_test=False, jobs=1))
        self.assertIn("--open-test not given", str(e.exception))

    def test_run_refuses_without_a_plan(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(BL, "check_test_allowed"), \
                mock.patch.object(C, "PLAN", Path(t) / "none.json"), mock.patch.object(C, "TEST_OUT", Path(t) / "out"), \
                mock.patch.object(BL, "load_mimic", side_effect=AssertionError("data was read")):
            with self.assertRaises(SystemExit) as e:
                C.cmd_run(Namespace(open_test=True, jobs=1))
        self.assertIn("does not exist", str(e.exception))

    def test_run_refuses_a_second_time(self):
        with tempfile.TemporaryDirectory() as t, stand_in_split_file(t), mock.patch.object(BL, "OUT", Path(t)), \
                mock.patch.object(BL, "load_mimic", side_effect=AssertionError("data was read")):
            (Path(t) / C.TEST_OUT.name).mkdir()
            with self.assertRaises(SystemExit) as e:
                C.cmd_run(Namespace(open_test=True, jobs=1))
        self.assertIn(f"{C.TEST_OUT.name} exists", str(e.exception))

    def test_run_refuses_after_an_exploratory_test_run(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(BL, "check_test_allowed"), mock.patch.object(C, "OUT", Path(t)), \
                mock.patch.object(BL, "load_mimic", side_effect=AssertionError("data was read")):
            (Path(t) / "mimic_test").mkdir()
            with self.assertRaises(SystemExit) as e:
                C.cmd_run(Namespace(open_test=True, jobs=1))
        self.assertIn("has to come first", str(e.exception))

    def bound_run(self, t, bound):
        plan = Path(t) / "plan.json"
        plan.write_text(json.dumps(dict(provisional=False, code_version="abc1234", window="pre")))
        with mock.patch.dict(BL.CFG["benchmark"], plan_sha256=bound), mock.patch.object(BL, "check_test_allowed"), mock.patch.object(C, "PLAN", plan), mock.patch.object(C, "OUT", Path(t)), mock.patch.object(BL, "cnn_sha256", return_value=None), \
                mock.patch.object(C, "TEST_OUT", Path(t) / "out"), mock.patch.object(C, "ATTEMPTS", Path(t) / "attempts"), mock.patch.object(C, "code_changed_since", return_value=[]), \
                mock.patch.object(BL, "load_mimic", side_effect=AssertionError("data was read")):
            with self.assertRaises(SystemExit) as e:
                C.cmd_run(Namespace(open_test=True, jobs=1))
        self.assertFalse((Path(t) / "attempts").exists())
        return str(e.exception)

    def test_run_refuses_a_plan_whose_sha256_is_not_in_the_config(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertIn("plan_sha256 is not set", self.bound_run(t, None))

    def test_run_refuses_a_plan_with_another_sha256(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertIn("differs from benchmark.plan_sha256", self.bound_run(t, "f" * 64))

    def test_config_change_counts_unless_it_is_a_key_written_after_the_plan(self):
        names = mock.Mock(returncode=0, stdout="config.yaml\n")
        old = "benchmark:\n  plan_sha256: null\n  test_opened: false\n  beat_max_iter: 75\n"
        for new, changed in (("benchmark:\n  plan_sha256: abc\n  test_opened: false\n  beat_max_iter: 75\n", []),
                             ("benchmark:\n  plan_sha256: abc\n  test_opened: false\n  beat_max_iter: 50\n", ["config.yaml"])):
            shows = [mock.Mock(stdout=old), mock.Mock(stdout=new)]
            with mock.patch.object(C.subprocess, "run", side_effect=[names, *shows]):
                self.assertEqual(C.code_changed_since("abc1234"), changed)

    def test_plan_refuses_a_missing_cnn_file_before_reading_data(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(BL, "FEAT", Path(t)), mock.patch.object(C, "TEST_OUT", Path(t) / "out"), \
                mock.patch.object(BL.pd, "read_csv", side_effect=AssertionError("data was read")):
            with self.assertRaisesRegex(SystemExit, "cnn_scores.npz does not exist"):
                C.cmd_plan(Namespace(window="pre", B=1, jobs=1))

    def test_only_run_opens_the_cnn_scores_on_test_rows(self):
        """Every other caller of load_mimic leaves open_test false unless its eval split is test."""
        import inspect
        self.assertFalse(inspect.signature(BL.load_mimic).parameters["open_test"].default)
        with tempfile.TemporaryDirectory() as t:
            plan = Path(t) / "plan.json"
            plan.write_text(json.dumps(dict(provisional=True, window="pre", models=["cnn"], cnn_scores_sha256="c" * 64)))
            stop = AssertionError("stop")
            with mock.patch.object(C, "PLAN", plan), mock.patch.object(BL, "cnn_sha256", return_value="c" * 64), mock.patch.object(BL, "load_mimic", side_effect=stop) as load:
                with self.assertRaises(AssertionError):
                    C.cmd_dry_run(Namespace(jobs=1))
                self.assertNotIn("open_test", load.call_args.kwargs)

    def test_dry_run_refuses_another_cnn_file(self):
        with tempfile.TemporaryDirectory() as t:
            plan = Path(t) / "plan.json"
            plan.write_text(json.dumps(dict(provisional=True, window="pre", models=["cnn"], cnn_scores_sha256="c" * 64)))
            with mock.patch.object(C, "PLAN", plan), mock.patch.object(BL, "cnn_sha256", return_value="d" * 64), mock.patch.object(BL, "load_mimic", side_effect=AssertionError("data was read")):
                with self.assertRaisesRegex(SystemExit, "not the one the plan used"):
                    C.cmd_dry_run(Namespace(jobs=1))

    def test_run_refuses_a_provisional_plan_and_writes_no_attempt(self):
        with tempfile.TemporaryDirectory() as t:
            plan = Path(t) / "plan.json"
            plan.write_text(json.dumps(dict(provisional=True, provisional_reasons=["model pulseppg not available"], code_version="abc1234")))
            with mock.patch.object(BL, "check_test_allowed"), mock.patch.object(C, "PLAN", plan), mock.patch.object(C, "TEST_OUT", Path(t) / "out"), \
                    mock.patch.object(C, "ATTEMPTS", Path(t) / "attempts"), mock.patch.object(C, "code_changed_since", return_value=[]), \
                    mock.patch.object(BL, "load_mimic", side_effect=AssertionError("data was read")):
                with self.assertRaises(SystemExit) as e:
                    C.cmd_run(Namespace(open_test=True, jobs=1))
            self.assertIn("provisional", str(e.exception))
            self.assertFalse((Path(t) / "attempts").exists())

    def test_run_does_not_read_the_evaluate_marker(self):
        """`run` opens the split whatever benchmark.test_opened says: the marker is set after it, for everyone else."""
        for value in (False, True):
            with tempfile.TemporaryDirectory() as t:
                plan = Path(t) / "plan.json"
                plan.write_text(json.dumps(dict(provisional=False, code_version="abc1234", window="pre", cnn_scores_sha256="c" * 64)))
                reached = AssertionError("reached the data")
                sha = hashlib.sha256(plan.read_bytes()).hexdigest()
                with mock.patch.dict(BL.CFG["benchmark"], test_opened=value, plan_sha256=sha), mock.patch.object(BL, "check_test_allowed"), mock.patch.object(C, "PLAN", plan), mock.patch.object(BL, "cnn_sha256", return_value="c" * 64), \
                        mock.patch.object(C, "OUT", Path(t)), mock.patch.object(C, "TEST_OUT", Path(t) / "out"), mock.patch.object(C, "ATTEMPTS", Path(t) / "attempts"), \
                        mock.patch.object(C, "code_changed_since", return_value=[]), mock.patch.object(C, "dry_run_problems", return_value=[]), mock.patch.object(BL, "load_mimic", side_effect=reached):
                    with self.assertRaises(AssertionError) as e:
                        C.cmd_run(Namespace(open_test=True, jobs=1))
                self.assertIs(e.exception, reached)

    def test_execute_refuses_val_outside_a_dry_run(self):
        with self.assertRaises(ValueError):
            C.execute(None, None, None, None, None, None, "val", None, 1, dry=False)
        with self.assertRaises(ValueError):
            C.execute(None, None, None, None, None, None, "test", None, 1, dry=True)

    def test_code_change_is_seen_but_tests_and_text_are_not(self):
        fake = mock.Mock(returncode=0, stdout="src/ppglabbench/baselines.py\ntests/test_confirmatory.py\nPROTOCOL.md\n")
        with mock.patch.object(C.subprocess, "run", return_value=fake):
            self.assertEqual(C.code_changed_since("abc1234"), ["src/ppglabbench/baselines.py"])
        self.assertEqual(len(C.code_changed_since("abc1234-dirty")), 1)


def with_cnn(d, labs, strength=1.5, seed=3):
    """The cohort with a synthetic CNN: a logit per lab, out of fold on train by subject, -1 elsewhere, and the two extra heads."""
    rng = np.random.default_rng(seed)
    sid = pd.factorize(d.group)[0]
    out = d.assign(cnn_fold=np.where(d.split == "train", sid % BL.K, -1), cnn_age_pred=d.age + rng.normal(0, 8, len(d)), cnn_hr_pred=d.hr + rng.normal(0, 3, len(d)))
    for lab in labs:
        out[f"cnn_{lab}"] = strength * (d[f"{lab}_abnormal"] - .5) + rng.normal(0, 1, len(d))
    return out.copy()


class TestCnnModel(unittest.TestCase):
    """The CNN as a candidate: nothing to fit, the file's folds, everywhere the other models go."""

    @classmethod
    def setUpClass(cls):
        d, cls.bcols, cls.spec, cls.tiers = synthetic_cohort(labs=("bun", "potassium", "lactate"), n_train=150, n_val=90)
        cls.d = with_cnn(d, ("bun", "potassium", "lactate"))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "dry"
        cls.saved_boot, BL.N_BOOT = BL.N_BOOT, 30
        cls.plan = C.make_plan(cls.d, cls.bcols, cls.spec, cls.tiers, B=2, jobs=1, window="pre", family=("bun", "potassium"), descriptive=("lactate",), named=("lactate",), cnn_sha="c" * 64)
        cls.R = C.execute(cls.d, cls.bcols, cls.spec, cls.tiers, json.loads(json.dumps(C.clean(cls.plan))), "0" * 64, "val", cls.out, 1, dry=True, seed=0)

    @classmethod
    def tearDownClass(cls):
        BL.N_BOOT = cls.saved_boot
        cls.tmp.cleanup()

    def test_it_is_a_candidate_and_a_strong_one_is_picked(self):
        self.assertIn("cnn", self.plan["models"])
        self.assertEqual(self.plan["picks"]["bun"], "cnn")
        self.assertEqual(self.plan["picks"]["lactate"], "cnn")
        self.assertIn("cnn", self.plan["labs"]["bun"]["table"])

    def test_the_named_hypotheses_keep_the_models_with_a_value_fit(self):
        self.assertNotIn("cnn", self.plan["named_labs"]["lactate"]["table"])
        self.assertIn(self.plan["picks_t1"]["lactate"], C.VALUE_MODELS)

    def test_the_plan_binds_the_file(self):
        self.assertEqual(self.plan["cnn_scores_sha256"], "c" * 64)
        self.assertEqual(self.plan["provisional_reasons"], [f"model pulseppg not available", "B = 2 < 50", "the working tree was not clean"][:1] + self.plan["provisional_reasons"][1:])

    def test_it_takes_no_part_in_r_hat_and_gets_the_largest(self):
        self.assertTrue(all(np.isnan(v) for v in self.plan["labs"]["bun"]["cov"].values()))
        plan = dict(picks=dict(a="cnn", b="beat"), strata=dict(a="common", b="common"), r_hat=dict(now=dict(common=.1, rare=.4)), margin=dict(margin=.01))
        self.assertEqual(C.lab_entry(plan, "a")["r"]["now"], .4)
        self.assertEqual(C.lab_entry(plan, "b")["r"]["now"], .1)

    def test_the_largest_upper_bound_of_a_step(self):
        for step in self.R["chain"]:
            self.assertAlmostEqual(step["largest_upper"], max(self.R["labs"][n]["chain"][step["base"]]["upper"] for n in ("bun", "potassium")))

    def test_the_mean_of_a_cnn_pick_takes_the_largest_r_hat(self):
        self.assertEqual(self.plan["picks"]["bun"], "cnn")
        self.assertEqual(self.R["labs"]["bun"]["r_hat"], C.lab_entry(json.loads(json.dumps(C.clean(self.plan))), "bun")["r"])

    def test_every_table_holds_its_rows(self):
        r = self.R["labs"]["bun"]
        self.assertEqual(r["pick"], "cnn")
        self.assertGreater(r["family_a"]["delta"], .02)
        self.assertGreater(r["family_a"]["z1"], 2)
        self.assertEqual(set(r["chain"]), set(C.CHAIN))
        self.assertIn("rate", r["beside_rate"])
        self.assertIn("delta", r["secondary"]["residual_arm"])
        self.assertIn("partial_r2", r["secondary"]["value"])
        self.assertTrue(self.R["descriptive"]["lactate"]["computable"])
        self.assertIn("bun | cnn", (self.out / "report.md").read_text())
        z = np.load(self.out / "predictions.npz", allow_pickle=False)
        self.assertIn("bun__score:cnn", z.files)

    def test_saved_score_equals_the_files_logit(self):
        z = np.load(self.out / "predictions.npz", allow_pickle=False)
        val = self.d[self.d.split == "val"].cnn_bun.to_numpy(float)
        self.assertEqual(sorted(np.round(z["bun__score:cnn"], 3).tolist()), sorted(np.round(val.astype(np.float32), 3).tolist()))

    def test_the_baselines_step_runs_it_for_the_abnormal_value_only(self):
        rows, nulls = BL.run_lab(self.d, self.bcols, self.tiers, "bun", self.spec["bun"], "val", 2, only=("all",))
        r = BL.adjust(pd.DataFrame(rows))
        self.assertEqual(set(r[r.method.astype(str).str.contains("cnn")].task), {"clf"})
        n = pd.DataFrame(nulls, columns=["lab", "task", "ppg", "base", "perm", "delta"])
        self.assertEqual(set(n[n.ppg == "cnn"].task), {"clf"})
        self.assertIn("+cnn", BL.report(r, "mimic", "val", len(self.d), self.d.group.nunique(), n))

    def test_a_papagei_pick_is_reported_beside_the_cnn(self):
        entry = dict(pick="papagei", r=dict(now=0., full=0., monitor=0., acuity=0.), margin=.01)
        r = C.evaluate_lab(self.d, self.bcols, self.tiers, "potassium", self.spec["potassium"], entry, "val")
        self.assertIn("cnn", r["comparators"])
        self.assertIn("beat", r["comparators"])

    def test_one_pick_per_lab_leaves_the_thresholds_as_they_were(self):
        self.assertAlmostEqual(C.threshold(len(C.FAMILY_LABS), C.ALPHA_A), 2.878, places=3)
        self.assertAlmostEqual(C.threshold(len(C.NAMED_LABS), C.ALPHA_NAMED), 2.807, places=3)
        self.assertEqual(len(C.MODELS), 4)


def walk(x, path=()):
    """Every path in a results dictionary, parents first."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield (*path, k)
            yield from walk(v, (*path, k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield (*path, i)
            yield from walk(v, (*path, i))


def with_at(R, path, value=None, drop=False):
    R = copy.deepcopy(R)
    x = R
    for key in path[:-1]:
        x = x[key]
    if drop:
        del x[path[-1]]
    else:
        x[path[-1]] = value
    return R


def none_everywhere(x):
    if isinstance(x, dict):
        return {k: none_everywhere(v) for k, v in x.items()}
    if isinstance(x, list):
        return [none_everywhere(v) for v in x]
    return None if isinstance(x, (int, float)) and not isinstance(x, bool) else x


class TestNothingIsLost(unittest.TestCase):
    """The one opening: a target that cannot be computed, a report that fails, and any secondary that raises must leave the confirmatory result in its final folder."""

    FAMILY, NAMED = ("bun", "potassium", "sodium"), ("lactate",)

    @classmethod
    def setUpClass(cls):
        d, cls.bcols, cls.spec, cls.tiers = synthetic_cohort(labs=("bun", "potassium", "sodium", "lactate"), n_train=120, n_val=70)
        cls.d = with_cnn(d, ("bun", "potassium", "sodium", "lactate"))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.saved_boot, BL.N_BOOT = BL.N_BOOT, 20
        plan = C.make_plan(cls.d, cls.bcols, cls.spec, cls.tiers, B=2, jobs=1, window="pre", family=cls.FAMILY, descriptive=("lactate",), named=cls.NAMED, cnn_sha="c" * 64)
        cls.plan = json.loads(json.dumps(C.clean(plan)))
        cls.R = cls.go(Path(cls.tmp.name) / "whole")

    @classmethod
    def tearDownClass(cls):
        BL.N_BOOT = cls.saved_boot
        cls.tmp.cleanup()

    @classmethod
    def go(cls, out, d=None):
        return C.execute(cls.d if d is None else d, cls.bcols, cls.spec, cls.tiers, cls.plan, "0" * 64, "val", out, 1, dry=True, seed=0)

    def fresh(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        return Path(t.name) / "out"

    def assert_same_confirmatory(self, R):
        for n in self.FAMILY:
            self.assertEqual(R["labs"][n]["family_a"], self.R["labs"][n]["family_a"], n)
            self.assertEqual(R["labs"][n]["levels_cleared"], self.R["labs"][n]["levels_cleared"], n)
        self.assertEqual(R["named"], self.R["named"])
        self.assertEqual(R["sequence"], self.R["sequence"])
        self.assertEqual(R["counts"], self.R["counts"])

    def test_a_clean_run_records_no_failure(self):
        self.assertEqual(self.R["failures"], [])

    def test_one_target_not_computable(self):
        d = self.d.copy()
        d.loc[d.split == "val", "potassium_abnormal"] = 0.0
        plan = dict(self.plan, picks={**self.plan["picks"], "potassium": None})
        out = self.fresh()
        R = C.execute(d, self.bcols, self.spec, self.tiers, plan, "0" * 64, "val", out, 1, dry=True, seed=0)
        self.assertFalse(R["labs"]["potassium"]["computable"])
        self.assertIn("potassium", R["counts"]["not_computable"])
        self.assertTrue(R["labs"]["bun"]["computable"])
        step = next(s for s in R["chain"] if s["base"] == "full")
        self.assertIsNone(step["largest_upper"])
        self.assertEqual(step["labs"], 2)
        self.assertIsNotNone(step["mean"])
        self.assertFalse(any(s["passed"] for s in R["chain"]))
        text = (out / "report.md").read_text()
        self.assertIn("not computable: no pick in the plan", text)
        self.assertNotIn("could not be rendered", text)
        self.assertEqual(json.loads((out / "results.json").read_text())["counts"], R["counts"])

    def test_a_target_with_too_few_subjects_on_the_eval_split(self):
        d = self.d.copy()
        d.loc[d.split == "val", "potassium_abnormal"] = 0.0
        out = self.fresh()
        R = self.go(out, d)
        self.assertIn("fewer than", R["labs"]["potassium"]["reason"])
        self.assertIn("not computable: fewer than", (out / "report.md").read_text())

    def test_no_target_computable(self):
        plan = dict(self.plan, picks={k: None for k in self.plan["picks"]}, picks_t1={k: None for k in self.plan["picks_t1"]})
        out = self.fresh()
        R = C.execute(self.d, self.bcols, self.spec, self.tiers, plan, "0" * 64, "val", out, 1, dry=True, seed=0)
        self.assertEqual(R["headline"], {})
        self.assertEqual(R["counts"]["l1"], 0)
        self.assertNotIn("could not be rendered", (out / "report.md").read_text())

    def rich(self):
        """Results with every optional part present: a PaPaGei lab past L1 with comparators, and a lab that cannot be computed."""
        R = copy.deepcopy(self.R)
        R["labs"]["bun"].update(pick="papagei", levels_cleared=1, comparators=dict(beat=R["labs"]["bun"]["family_a"]))
        R["labs"]["sodium"] = dict(lab="sodium", computable=False, pick=None, reason="no pick in the plan", levels_cleared=0)
        R["labs"]["potassium"]["flag_gain_without_association"] = True
        return R

    def test_every_value_in_the_results_can_be_none_absent_or_a_failure(self):
        R = self.rich()
        for path in walk(R):
            for broken in (with_at(R, path), with_at(R, path, drop=True), with_at(R, path, dict(failed="RuntimeError at x.py:1 in f")), with_at(R, path, "text")):
                text = C.report_md(broken)
                self.assertFalse(re.findall(r".*could not be rendered.*", text), path)

    def test_every_number_none_at_once(self):
        text = C.report_md(none_everywhere(self.rich()))
        self.assertNotIn("could not be rendered", text)
        self.assertIn("Family A", text)

    def test_nothing_at_all(self):
        for R in ({}, dict(labs=None, named=None, descriptive=None, chain=None, headline=None, counts=None)):
            self.assertNotIn("could not be rendered", C.report_md(R))

    def test_a_section_that_still_raises_leaves_the_others(self):
        with mock.patch.object(C, "NET_BENEFIT_AT", None):
            text = C.report_md(self.R)
        self.assertIn("## Net benefit", text)
        self.assertIn("could not be rendered (TypeError", text)
        self.assertIn("## Family A", text)
        self.assertIn("## Labs outside the family", text)

    def test_a_failed_report_keeps_the_results_and_can_be_rendered_again(self):
        out = self.fresh()
        with mock.patch.object(C, "report_md", side_effect=RuntimeError("value 3.2")), contextlib.redirect_stderr(io.StringIO()) as err:
            R = self.go(out)
        self.assert_same_confirmatory(R)
        self.assertTrue((out / "results.json").exists())
        self.assertFalse((out / "report.md").exists())
        self.assertTrue((out / "predictions.npz").exists())
        self.assertTrue((out / "manifest.json").exists())
        note = (out / "report_failed.txt").read_text()
        self.assertIn("confirmatory render", note)
        self.assertNotIn("3.2", note + err.getvalue())         # a message can quote a value, so only its type and place are kept
        with mock.patch.object(C, "DRY_OUT", out), mock.patch.object(BL, "load_mimic", side_effect=AssertionError("data was read")), contextlib.redirect_stdout(io.StringIO()):
            C.cmd_render(Namespace(dry_run=True))
        self.assertFalse((out / "report_failed.txt").exists())
        self.assertEqual((out / "report.md").read_text(), C.report_md(json.loads((out / "results.json").read_text())))

    def test_show_says_where_the_report_went(self):
        out = self.fresh()
        out.mkdir()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            C.show(out)
        self.assertIn("report_failed.txt", printed.getvalue())

    def test_render_needs_something_to_render(self):
        out = self.fresh()
        out.mkdir()
        with mock.patch.object(C, "TEST_OUT", out):
            with self.assertRaisesRegex(SystemExit, "neither"):
                C.cmd_render(Namespace(dry_run=False))

    def test_the_cli_maps_each_step_to_its_own_function(self):
        for argv, fn in ((["render"], "cmd_render"), (["render", "--dry-run"], "cmd_render"), (["run", "--open-test"], "cmd_run"), (["dry-run"], "cmd_dry_run"), (["plan"], "cmd_plan")):
            with mock.patch.object(C, fn) as f, mock.patch.object(C.sys, "argv", ["confirmatory", *argv]):
                # the parser takes its functions when it is built, so main sees the mock
                C.main()
            f.assert_called_once()

    def test_failed_predictions_or_manifest_leave_results_and_report(self):
        for target, name in ((mock.patch.object(C, "anonymise_arrays", side_effect=RuntimeError("x")), "predictions.npz"), (mock.patch.object(C.np, "savez_compressed", side_effect=OSError("x")), "predictions.npz"),
                             (mock.patch.object(C, "write_manifest", side_effect=RuntimeError("x")), "manifest.json")):
            out = self.fresh()
            with target, contextlib.redirect_stderr(io.StringIO()):
                R = self.go(out)
            self.assert_same_confirmatory(R)
            self.assertTrue((out / "results.json").exists(), name)
            self.assertTrue((out / "report.md").exists(), name)
            self.assertFalse((out / name).exists())
            self.assertIn(f"{name} was not written", (out / "steps_failed.txt").read_text())
            self.assertFalse((out.parent / "out.tmp").exists())

    def test_failed_assembly_keeps_the_statistics_and_can_be_assembled_again(self):
        out = self.fresh()
        with mock.patch.object(C, "assemble", side_effect=RuntimeError("x")):
            with self.assertRaisesRegex(SystemExit, "per_lab.json"):
                self.go(out)
        self.assertFalse((out / "results.json").exists())
        self.assertTrue((out / "per_lab.json").exists())
        self.assertTrue((out / "predictions.npz").exists())
        planfile = out.parent / "plan.json"
        planfile.write_text(json.dumps(self.plan))
        with mock.patch.object(C, "DRY_OUT", out), mock.patch.object(C, "PLAN", planfile), mock.patch.object(BL, "load_mimic", side_effect=AssertionError("data was read")), contextlib.redirect_stdout(io.StringIO()):
            C.cmd_render(Namespace(dry_run=True))
        R = json.loads((out / "results.json").read_text())
        self.assertFalse((out / "per_lab.json").exists())
        self.assertEqual(R["counts"], self.R["counts"])
        self.assertEqual(R["labs"]["bun"]["family_a"], self.R["labs"]["bun"]["family_a"])
        self.assertEqual(R["headline"], self.R["headline"])
        self.assertTrue((out / "report.md").exists())

    def inject(self, target, name, side_effect=None):
        """Execute with one function raising, and return the results and the report."""
        out = self.fresh()
        with mock.patch.object(target, name, side_effect=side_effect or RuntimeError("boom 3.2")):
            R = self.go(out)
        text = (out / "report.md").read_text()
        self.assertNotIn("could not be rendered", text, name)
        self.assertNotIn("boom", json.dumps(R) + text)
        self.assertTrue(R["failures"], name)
        self.assertTrue(all(f"RuntimeError" in f for f in R["failures"]), name)
        return R, text

    def test_each_secondary_fails_alone(self):
        for name, row in (("brier_skill", "brier"), ("calibration", "calibration.stack"), ("net_benefit_rows", "net_benefit"), ("residual_score", "residual_arm"), ("rate_rows", "beside_rate"),
                          ("subgroup_masks", "subgroups"), ("previous_age_masks", "by_previous_age")):
            with self.subTest(name):
                R, text = self.inject(C, name)
                self.assert_same_confirmatory(R)
                self.assertTrue(any(row in f for f in R["failures"]), (name, R["failures"]))
                self.assertIn("failed", text)
                for n in self.FAMILY:
                    self.assertEqual(R["labs"][n]["chain"], self.R["labs"][n]["chain"], n)

    def test_the_gain_in_each_stratum_fails_alone(self):
        R, text = self.inject(C, "strata_gains")
        self.assert_same_confirmatory(R)
        self.assertTrue(any("subgroups.sex" in f for f in R["failures"]))
        self.assertTrue(any("by_previous_age.now" in f for f in R["failures"]))

    def test_the_gain_on_the_value_fails_alone_in_a_secondary_and_as_the_named_lab_it_tests(self):
        R, _ = self.inject(C, "delta_r2")
        for n in self.FAMILY:
            self.assertEqual(R["labs"][n]["family_a"], self.R["labs"][n]["family_a"], n)
            self.assertIn("failed", R["labs"][n]["secondary"]["value"])
        self.assertFalse(R["named"]["lactate"]["computable"])
        self.assertIn("lactate (value)", R["counts"]["not_computable"])

    def test_the_bootstrap_check_fails_alone(self):
        R, _ = self.inject(BL, "subject_boot")
        self.assert_same_confirmatory(R)
        self.assertTrue(any("boot_se" in f for f in R["failures"]))

    def test_the_mean_gain_fails_alone(self):
        R, text = self.inject(C, "mean_gain")
        for n in self.FAMILY:
            self.assertEqual(R["labs"][n]["family_a"], self.R["labs"][n]["family_a"], n)
        self.assertEqual(R["sequence"], self.R["sequence"])
        self.assertEqual(R["counts"]["chain_passed"], [])       # with no mean, no step can pass
        self.assertTrue(all(s["upper"] is None and s["labs"] == 0 for s in R["chain"]))
        self.assertIn("The mean gain over `now` failed", text)

    def test_a_lab_without_a_bound_leaves_the_mean_without_one_and_keeps_the_family(self):
        """The per-lab row of a base fails: the step has no training part, so it has no bound and cannot pass, and L1 to L3 are untouched."""
        delta_auc = C.delta_auc
        calls = {"n": 0}

        def second_call_for_each_lab_fails(y, stack, base, g, G, floor=C.GAIN_FLOOR):
            calls["n"] += 1
            if calls["n"] == 2:      # the first is L2 of the first lab, the second is that lab's row over `full`
                raise RuntimeError("boom 3.2")
            return delta_auc(y, stack, base, g, G, floor)
        out = self.fresh()
        with mock.patch.object(C, "delta_auc", second_call_for_each_lab_fails):
            R = self.go(out)
        self.assertIn("failed", R["labs"]["bun"]["chain"]["full"])
        self.assertIsNone(R["chain"][0]["upper"])
        self.assertIsNotNone(R["chain"][0]["mean"])
        self.assertIsNone(R["chain"][0]["largest_upper"])
        self.assertFalse(R["chain"][0]["passed"])
        self.assertEqual(R["labs"]["bun"]["family_a"], self.R["labs"]["bun"]["family_a"])
        text = (out / "report.md").read_text()
        self.assertIn("failed: RuntimeError", text)
        self.assertNotIn("could not be rendered", text)

    def test_a_comparator_fails_alone(self):
        calls = {"n": 0}
        fit = C.fit_models

        def on_the_comparators(y, tr, ev, groups, bases, X, task="clf"):
            if len(bases) == 1 and len(X) > 1:      # only the comparators of a PaPaGei pick fit several models on one base
                raise RuntimeError("boom")
            calls["n"] += 1
            return fit(y, tr, ev, groups, bases, X, task=task)
        plan = dict(self.plan, picks={**self.plan["picks"], "bun": "papagei"})
        out = self.fresh()
        with mock.patch.object(C, "fit_models", on_the_comparators):
            R = C.execute(self.d, self.bcols, self.spec, self.tiers, plan, "0" * 64, "val", out, 1, dry=True, seed=0)
        self.assertIn("failed", R["labs"]["bun"]["comparators"])
        self.assertTrue(R["labs"]["bun"]["computable"])
        self.assertNotIn("could not be rendered", (out / "report.md").read_text())

    def test_a_lab_that_raises_in_its_own_tests_cannot_be_computed_and_the_others_keep_theirs(self):
        score, prevalence = C.score_test, self.R["labs"]["potassium"]["prevalence"]
        self.assertNotEqual(prevalence, self.R["labs"]["bun"]["prevalence"], "the prevalence must tell the labs apart")

        def only_potassium(y, Z, x, g, G):
            if y.mean() == prevalence:
                raise RuntimeError("boom")
            return score(y, Z, x, g, G)
        out = self.fresh()
        with mock.patch.object(C, "score_test", only_potassium):
            R = self.go(out)
        self.assertFalse(R["labs"]["potassium"]["computable"])
        self.assertIn("failed: RuntimeError", R["labs"]["potassium"]["reason"])
        self.assertIn("potassium", R["counts"]["not_computable"])
        self.assertEqual(R["labs"]["bun"]["family_a"], self.R["labs"]["bun"]["family_a"])
        self.assertFalse(any(s["passed"] for s in R["chain"]))
        self.assertIn("not computable: failed: RuntimeError", (out / "report.md").read_text())

    def test_a_sys_exit_in_a_lab_is_caught_too(self):
        r = C.whole_lab(lambda: sys.exit("stop 3.2"), dict(lab="x", pick="cnn"))
        self.assertFalse(r["computable"])
        self.assertIn("SystemExit", r["failed"])
        self.assertNotIn("3.2", r["reason"])

    def test_l1_without_residuals_does_not_stop_the_predictions(self):
        """score_test returns no residuals when it cannot be computed, and that used to reach the predictions."""
        out = self.fresh()
        with mock.patch.object(C, "score_test", return_value=(float("nan"), None, None)):
            R = self.go(out)
        self.assertTrue(R["labs"]["bun"]["computable"])
        self.assertIsNone(R["labs"]["bun"]["family_a"]["z1"])
        self.assertEqual(R["counts"]["l1"], 0)
        z = np.load(out / "predictions.npz")
        self.assertNotIn("bun__l1_r", z.files)
        self.assertIn("bun__now", z.files)
        self.assertFalse((out / "steps_failed.txt").exists())

    def test_a_dead_worker_falls_back_to_one_lab_at_a_time(self):
        with mock.patch.object(C, "parallel_map", side_effect=RuntimeError("worker died")), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(C.protected_map(lambda a: a + 1, [(1,), (2,)], 3), [2, 3])

    def test_clean_takes_anything(self):
        x = C.clean(dict(a=np.arange(2), b=np.float32(1.5), c=float("nan"), d=np.bool_(True), e=Path("x"), f=None))
        self.assertEqual(x, dict(a=[0, 1], b=1.5, c=None, d=True, e="x", f=None))
        json.dumps(x)
        self.assertTrue(np.isnan(C.clean(float("nan"), keep_nan=True)))

    def test_failures_are_listed_by_place(self):
        self.assertEqual(C.list_failures(dict(a=dict(failed="X"), b=[dict(c=dict(failed="Y"))], d=1)), ["a: X", "b.0.c: Y"])


class TestOpeningOnlyAfterADryRun(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.t = Path(t.name)
        self.dry = self.t / "dry"
        self.dry.mkdir()
        p = mock.patch.object(C, "DRY_OUT", self.dry)
        p.start()
        self.addCleanup(p.stop)

    def write(self, **R):
        (self.dry / "results.json").write_text(json.dumps(dict(dict(plan_sha256="a" * 64, failures=[]), **R)))
        (self.dry / "report.md").write_text("report")

    def test_a_finished_dry_run_of_this_plan_passes(self):
        self.write()
        self.assertEqual(C.dry_run_problems("a" * 64), [])

    def test_none(self):
        self.assertIn("run the dry run of this plan first", C.dry_run_problems("a" * 64)[0])

    def test_another_plan(self):
        self.write()
        self.assertIn("another plan", C.dry_run_problems("b" * 64)[0])

    def test_failures(self):
        self.write(failures=["labs.bun.secondary.brier: TypeError"])
        self.assertIn("brier", C.dry_run_problems("a" * 64)[0])

    def test_no_report(self):
        self.write()
        (self.dry / "report.md").unlink()
        self.assertIn("did not finish", C.dry_run_problems("a" * 64)[0])

    def test_a_failed_step(self):
        self.write()
        (self.dry / "steps_failed.txt").write_text("x")
        self.assertIn("did not finish", C.dry_run_problems("a" * 64)[0])

    def harness(self, plan, load):
        planfile = self.t / "plan.json"
        planfile.write_text(json.dumps(plan))
        sha = hashlib.sha256(planfile.read_bytes()).hexdigest()
        (self.dry / "results.json").write_text(json.dumps(dict(plan_sha256=sha, failures=[])))
        (self.dry / "report.md").write_text("report")
        return (mock.patch.dict(BL.CFG["benchmark"], plan_sha256=sha), mock.patch.object(BL, "check_test_allowed"), mock.patch.object(C, "PLAN", planfile), mock.patch.object(C, "OUT", self.t),
                mock.patch.object(C, "TEST_OUT", self.t / "out"), mock.patch.object(C, "ATTEMPTS", self.t / "attempts"), mock.patch.object(C, "code_changed_since", return_value=[]),
                mock.patch.object(BL, "cnn_sha256", return_value="c" * 64), mock.patch.object(BL, "load_mimic", load))

    PLAN_OK = dict(provisional=False, code_version="abc1234", window="pre", cnn_scores_sha256="c" * 64, family=["bun"], descriptive=[], named=[], picks=dict(bun="beat"), picks_t1={})

    def test_run_refuses_without_a_dry_run_before_reading_data(self):
        patches = self.harness(self.PLAN_OK, mock.Mock(side_effect=AssertionError("data was read")))
        (self.dry / "results.json").unlink()
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with self.assertRaisesRegex(SystemExit, "dry run"):
                C.cmd_run(Namespace(open_test=True, jobs=1))
        self.assertFalse((self.t / "attempts").exists())

    def test_what_can_refuse_refuses_before_the_attempt_is_logged(self):
        d = pd.DataFrame(dict(draw_id=["w0"], split=["test"], group=[1]))
        for name, plan, labs, message in (("a model without features", self.PLAN_OK, dict(bun={}), "beat"), ("a lab that is gone", dict(self.PLAN_OK, picks=dict(bun=None)), {}, "bun is not among the labs")):
            with self.subTest(name):
                load = mock.Mock(return_value=(d, ["not_a_column"], labs, {}))
                patches = self.harness(plan, load)
                with contextlib.ExitStack() as stack:
                    for p in patches:
                        stack.enter_context(p)
                    with self.assertRaisesRegex(SystemExit, message):
                        C.cmd_run(Namespace(open_test=True, jobs=1))
                self.assertFalse((self.t / "attempts").exists())
                self.assertNotIn("open_test", load.call_args.kwargs)

    def test_test_rows_of_the_cnn_file_are_checked_before_the_attempt_is_logged_by_ids_and_folds_only(self):
        d = pd.DataFrame(dict(draw_id=["w0", "w1"], split=["train", "test"], group=[1, 2], b0=[0., 0.]))
        load = mock.Mock(return_value=(d, ["b0"], dict(bun={}), {}))
        patches = self.harness(self.PLAN_OK, load)
        f = self.t / "cnn_scores.npz"
        np.savez(f, draw_id=np.array(["w0"]), fold=np.array([0]), score=np.array([[np.inf]]))
        with contextlib.ExitStack() as stack, mock.patch.object(BL, "cnn_file", return_value=f), mock.patch.object(BL, "K", 1):
            for p in patches:
                stack.enter_context(p)
            with self.assertRaisesRegex(SystemExit, "1 of 2 windows are not in the file"):
                C.cmd_run(Namespace(open_test=True, jobs=1))
        self.assertFalse((self.t / "attempts").exists())


class TestEndToEnd(unittest.TestCase):
    """plan -> dry run on a small synthetic cohort, one lab with no abnormal eval row. Slow part of the suite, about a minute."""

    @classmethod
    def setUpClass(cls):
        cls.d, cls.bcols, cls.spec, cls.tiers = synthetic_cohort(n_train=150, n_val=90)
        cls.d.loc[cls.d.split == "val", "ast_abnormal"] = 0.0
        cls.family = ("bun", "potassium", "ast")
        cls.descriptive = ("lactate",)
        cls.named = ("lactate", "potassium")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "dry"
        cls.saved_boot, BL.N_BOOT = BL.N_BOOT, 30
        cls.plan = C.make_plan(cls.d, cls.bcols, cls.spec, cls.tiers, B=3, jobs=1, window="centered", family=cls.family, descriptive=cls.descriptive, named=cls.named)
        cls.R = C.execute(cls.d, cls.bcols, cls.spec, cls.tiers, json.loads(json.dumps(C.clean(cls.plan))), "0" * 64, "val", cls.out, 1, dry=True, seed=0)

    @classmethod
    def tearDownClass(cls):
        BL.N_BOOT = cls.saved_boot
        cls.tmp.cleanup()

    def test_plan(self):
        p = self.plan
        self.assertTrue(p["provisional"])                      # no Pulse-PPG, B = 3, centred window
        self.assertEqual(p["window"], "centered")
        self.assertIn("lactate", p["picks"])
        self.assertNotIn("lactate", p["strata"])
        self.assertEqual(p["picks"]["ast"], None)
        self.assertIn(p["picks"]["bun"], ("beat", "papagei"))
        self.assertEqual(p["alpha_a"] + p["alpha_named"], .025)
        self.assertGreaterEqual(p["margin"]["margin"], C.MIN_GAIN)
        self.assertIn(PRIMARY_KEY, p["r_hat"])
        json.dumps(C.clean(p))

    def test_uncomputable_lab_stays_in_the_family(self):
        R = self.R
        self.assertEqual(R["k"], 3)
        self.assertEqual(R["counts"]["not_computable"], ["ast"])
        self.assertEqual(R["labs"]["ast"]["levels_cleared"], 0)
        self.assertFalse(R["labs"]["ast"]["computable"])
        if R["counts"]["l1"] == 0:
            self.assertEqual(R["sequence"]["final_A"], 3)      # it still holds its share
        self.assertAlmostEqual(R["first_threshold"], C.threshold(3, C.ALPHA_A))
        self.assertAlmostEqual(R["first_threshold_named"], C.threshold(2, C.ALPHA_NAMED))

    def test_outputs_and_no_ids(self):
        for f in ("results.json", "report.md", "predictions.npz", "manifest.json"):
            self.assertTrue((self.out / f).exists(), f)
        text = "".join((self.out / f).read_text() for f in ("results.json", "report.md", "manifest.json"))
        ids = np.unique(self.d.group.to_numpy())
        self.assertFalse(any(str(i) in text for i in ids))
        z = np.load(self.out / "predictions.npz", allow_pickle=False)
        for k in z.files:
            self.assertFalse(np.isin(z[k], ids).any(), k)
        self.assertIn("DRY RUN", (self.out / "report.md").read_text())
        self.assertFalse(Path(str(self.out) + ".tmp").exists())

    def test_saved_predictions_reproduce_l1_and_l2(self):
        """A reader of predictions.npz gets the same statistics without the original table."""
        z = np.load(self.out / "predictions.npz", allow_pickle=False)
        r = self.R["labs"]["bun"]
        pick = r["pick"]
        y, g = z["bun__y"].astype(bool), z["bun__group"]
        d, G = C.dense(g)
        res = C.delta_auc(y, z[f"bun__now+{pick}"].astype(float), z["bun__now"].astype(float), d, G)
        self.assertAlmostEqual(res["delta"], r["family_a"]["delta"], places=4)
        self.assertAlmostEqual(res["se"], r["family_a"]["se"], places=4)
        u = np.bincount(d, z["bun__l1_r"] * z["bun__l1_e"], minlength=G)
        k = r["n_design"]
        z1 = u.sum() / np.sqrt(G / (G - 1) * (len(y) - 1) / (len(y) - k) * np.sum(u * u))
        self.assertAlmostEqual(z1, r["family_a"]["z1"], places=3)

    def test_descriptive_lab_has_no_test(self):
        r = self.R["descriptive"]["lactate"]
        self.assertTrue(r["computable"])
        self.assertEqual(set(r["gains"]), {C.PRIMARY, C.UPPER})
        self.assertFalse(any(k.startswith("z") or "reject" in k for k in r))
        self.assertFalse(any(k.startswith("z") for g in r["gains"].values() for k in g))
        self.assertNotIn("lactate", self.R["labs"])
        self.assertIn("lactate", (self.out / "report.md").read_text())
        z = np.load(self.out / "predictions.npz", allow_pickle=False)
        self.assertIn("lactate__y", z.files)

    def test_rows_beside_the_pick(self):
        for r in (self.R["labs"]["bun"], self.R["labs"]["potassium"], self.R["descriptive"]["lactate"]):
            b = r["beside_rate"]
            self.assertEqual(set(b), {"rate", "pick_over_rate"})
            self.assertIn("z1", b["rate"])
            self.assertNotIn("z1", b["pick_over_rate"])
            self.assertTrue(np.isfinite([b["rate"]["delta"], b["pick_over_rate"]["delta"]]).all())
        md = (self.out / "report.md").read_text()
        self.assertIn("pulse rate and its variability, not tested", md)
        section = md.split("Beside the pick")[1].split("##")[0]
        for lab in ("bun", "potassium", "lactate"):
            self.assertIn(f"| {lab} |", section)
        self.assertNotIn("| ast |", section)

    def test_the_saved_predictions_reproduce_the_rows_beside_the_pick(self):
        z = np.load(self.out / "predictions.npz", allow_pickle=False)
        r = self.R["labs"]["bun"]
        pick = r["pick"]
        y, g = z["bun__y"].astype(bool), z["bun__group"]
        d, G = C.dense(g)
        rate = C.delta_auc(y, z["bun__now+rate"].astype(float), z["bun__now"].astype(float), d, G)
        over = C.delta_auc(y, z[f"bun__now+rate+{pick}"].astype(float), z["bun__now+rate"].astype(float), d, G)
        self.assertAlmostEqual(rate["delta"], r["beside_rate"]["rate"]["delta"], places=4)
        self.assertAlmostEqual(over["delta"], r["beside_rate"]["pick_over_rate"]["delta"], places=4)

    def test_rate_is_no_candidate_and_changes_no_decision(self):
        self.assertNotIn("rate", C.MODELS)
        self.assertNotIn("rate", self.plan["models"])
        self.assertNotIn("rate", set(self.plan["picks"].values()))
        with tempfile.TemporaryDirectory() as tmp:
            R = C.execute(self.d.drop(columns=BL.RATE), self.bcols, self.spec, self.tiers, json.loads(json.dumps(C.clean(self.plan))), "0" * 64, "val", Path(tmp) / "dry", 1, dry=True, seed=0)
        for n in self.family:
            a, b = R["labs"][n], self.R["labs"][n]
            self.assertEqual(a["levels_cleared"], b["levels_cleared"], n)
            if b["computable"]:
                self.assertIsNone(a["beside_rate"])
                for k in ("delta", "z1", "z2", "z3"):
                    self.assertEqual(a["family_a"][k], b["family_a"][k], (n, k))
                self.assertEqual(a["chain"][C.UPPER]["upper"], b["chain"][C.UPPER]["upper"], n)

    def test_decisions_follow_the_statistics(self):
        R = self.R
        for n, r in R["labs"].items():
            if r["computable"]:
                a = r["family_a"]
                self.assertAlmostEqual(a["se_t"], a["se"] * np.sqrt(1 + r["r_hat"][C.PRIMARY]))
                self.assertAlmostEqual(a["z3"], (a["delta"] - .01) / a["se_t"])
                for base, b in r["chain"].items():
                    self.assertAlmostEqual(b["se_t"] ** 2, b["se"] ** 2 * (1 + r["r_hat"].get(base, 0.0)))
                    self.assertAlmostEqual(b["upper"], b["delta"] + 1.959964 * b["se_t"], places=5)
                    self.assertNotIn("z", b)

    def test_the_headline_and_the_chain(self):
        R = self.R
        h = R["headline"][C.PRIMARY]
        self.assertAlmostEqual(h["mean"], np.mean([R["labs"][n]["family_a"]["delta"] for n in ("bun", "potassium")]), places=6)
        self.assertLessEqual(h["lo"], h["hi"])
        self.assertEqual([s["base"] for s in R["chain"]], list(C.CHAIN))
        self.assertFalse(R["chain"][0]["passed"])                      # a lab that cannot be computed fails the first step
        self.assertEqual(R["chain"][0]["labs"], 2)
        self.assertEqual(R["chain"][0]["k"], R["k"])
        self.assertLess(R["chain"][0]["labs"], R["chain"][0]["k"])
        self.assertEqual([s["tested"] for s in R["chain"]], [True, False, False, False])
        for s in R["chain"]:
            h = R["headline"][s["base"]]
            self.assertEqual((s["mean"], s["lo"], s["hi"]), (h["mean"], h["lo"], h["hi"]))
            self.assertAlmostEqual(s["training_sd"], np.mean([np.sqrt(R["labs"][n]["r_hat"].get(s["base"], 0.0)) * R["labs"][n]["chain"][s["base"]]["se"] for n in ("bun", "potassium")]))
            self.assertAlmostEqual(s["upper"], C.mean_upper(s["mean"], s["hi"], s["training_sd"]))
        md = (self.out / "report.md").read_text()
        self.assertIn("**Headline.**", md)
        self.assertIn("Upper-bound chain", md)

    def test_the_named_hypotheses(self):
        R = self.R
        self.assertEqual(set(R["named"]), set(self.named))
        for n, r in R["named"].items():
            self.assertTrue(r["computable"])
            b = r["named"]
            self.assertAlmostEqual(b["z2"], b["delta"] / b["se"])
            self.assertAlmostEqual(b["simultaneous"][1] - b["delta"], R["first_threshold_named"] * b["se"])
        self.assertTrue(set(self.plan["picks_t1"]) == set(self.named))
        z = np.load(self.out / "predictions.npz", allow_pickle=False)
        r = R["named"]["lactate"]
        y, g = z["t1_lactate__y"].astype(float), z["t1_lactate__group"]
        d, G = C.dense(g)
        res = C.delta_r2(y, z[f"t1_lactate__now+{r['pick']}"].astype(float), z["t1_lactate__now"].astype(float), d, G)
        self.assertAlmostEqual(res["delta"], r["named"]["delta"], places=3)

    def test_the_margin_comes_from_bedside_groups(self):
        m = self.plan["margin"]
        self.assertTrue(set(m["bedside_gains"]["bun"]) <= set(C.MARGIN_GROUPS))
        self.assertEqual(self.R["margin"], m["margin"])
        self.assertEqual(set(m["joint_gains"]), set(self.family))
        self.assertAlmostEqual(m["joint_mean"], np.mean([max(v, 0) for v in m["joint_gains"].values() if v is not None]))

    def test_the_secondary_rows(self):
        s = self.R["labs"]["bun"]["secondary"]
        self.assertEqual(set(s["by_previous_age"]), {C.PRIMARY, "locf"})
        self.assertEqual(list(s["by_previous_age"]["locf"]["strata"]), ["le_12h", "12_24h", "24_48h", "gt_48h", "never"])
        self.assertEqual(sum(x["n"] for x in s["by_previous_age"]["locf"]["strata"].values()), self.R["labs"]["bun"]["n"])
        self.assertEqual([x["threshold"] for x in s["net_benefit"]], list(C.NET_BENEFIT_AT))
        self.assertEqual(set(s["subgroups"]), {"sex", "age_band", "vasopressor"})     # the cohort has no AF or era column
        self.assertIn("slope", s["calibration"]["stack"])
        self.assertIn("skill", s["brier"])
        self.assertIn("partial_r2", s["value"])
        self.assertIn("delta", s["residual_arm"])
        self.assertIn("by_previous_age", self.R["descriptive"]["lactate"])
        for r in (s, self.R["labs"]["bun"]):
            self.assertFalse(any(k in ("z", "reject") for k in r))
        md = (self.out / "report.md").read_text()
        for title in ("Gain by age of the previous value", "Net benefit", "Subgroups", "Residual arm"):
            self.assertIn(title, md)


PRIMARY_KEY = C.PRIMARY

if __name__ == "__main__":
    unittest.main(verbosity=1)
