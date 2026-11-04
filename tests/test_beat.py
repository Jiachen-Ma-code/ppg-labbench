"""The diastolic point of ppglabbench.features.beat on made-up beats. No data/ folder is needed.

Usage:  python -m unittest tests.test_beat
"""
import unittest
from unittest import mock

import numpy as np

from ppglabbench.features import beat as B
from ppglabbench.pipeline.s4_score_sqi import EP

FS = B.FS


def wave(t, peak=.26, width=.10, second=0.0, at=.55, width2=.09):
    return np.exp(-((t - peak) / width) ** 2) + second * np.exp(-((t - at) / width2) ** 2)


def scaled(y):
    """A beat as beat_table holds it: NB points in [0, 1], its derivative and its systolic peak."""
    y = (y - y.min()) / np.ptp(y)
    return y, np.gradient(y), int(np.argmax(y[:70]))


class TestFirstLocalMax(unittest.TestCase):
    def test_a_plain_peak(self):
        self.assertEqual(B.first_local_max(np.array([3., 1, 2, 5, 4, 6, 1])), 3)

    def test_steps_upwards_are_not_a_maximum(self):
        self.assertIsNone(B.first_local_max(np.array([1., 1, 2, 2, 3, 3, 4, 4])))

    def test_the_top_of_a_flat_stretch(self):
        self.assertEqual(B.first_local_max(np.array([1., 2, 2, 3, 3, 3, 2])), 3)

    def test_nothing_to_find(self):
        for s in (np.array([]), np.array([1.]), np.array([1., 2.]), np.array([3., 2., 1.]), np.array([1., 2., 3.]), np.array([2., 2., 2.])):
            self.assertIsNone(B.first_local_max(s))


class TestDiastolicPoint(unittest.TestCase):
    @staticmethod
    def knots(*points):
        """A beat of straight stretches between (index, height) points."""
        x, y = zip(*points)
        return scaled(np.interp(np.arange(B.NB), x, y))

    def test_a_clear_notch(self):
        bt, g1, i_sys = self.knots((0, 0), (26, 1), (50, .30), (62, .50), (99, 0))
        i_dia, notch = B.diastolic_point(bt, g1, i_sys)
        self.assertEqual(notch, 1.0)
        self.assertLessEqual(abs(i_dia - 62), 1)
        self.assertAlmostEqual(bt[i_dia], .50, delta=.02)

    def test_a_rise_under_one_percent_is_no_notch(self):
        bt, g1, i_sys = self.knots((0, 0), (26, 1), (45, .40), (48, .405), (99, 0))
        i_dia, notch = B.diastolic_point(bt, g1, i_sys)
        self.assertEqual(notch, 0.0)
        self.assertTrue(44 <= i_dia <= 49, i_dia)

    def test_a_rise_of_one_percent_is_a_notch(self):
        bt, g1, i_sys = self.knots((0, 0), (26, 1), (45, .40), (48, .42), (99, 0))
        self.assertEqual(B.diastolic_point(bt, g1, i_sys)[1], 1.0)

    def test_a_ripple_does_not_hide_a_real_notch_after_it(self):
        bt, g1, i_sys = self.knots((0, 0), (26, 1), (40, .50), (42, .505), (44, .48), (55, .30), (65, .50), (99, 0))
        i_dia, notch = B.diastolic_point(bt, g1, i_sys)
        self.assertEqual(notch, 1.0)
        self.assertLessEqual(abs(i_dia - 65), 1)

    def test_no_notch_gives_the_first_maximum_of_the_derivative_not_a_late_point(self):
        bt, g1, i_sys = self.knots((0, 0), (26, 1), (40, .55), (55, .45), (99, 0))
        self.assertTrue((g1[i_sys + 2:B.SEARCH_END] < 0).all())       # a shoulder and no notch
        i_dia, notch = B.diastolic_point(bt, g1, i_sys)
        self.assertEqual(notch, 0.0)
        self.assertLessEqual(abs(i_dia - 40), 1)
        self.assertAlmostEqual(bt[i_dia], .55, delta=.02)

    def test_a_plain_decay_has_no_diastolic_point(self):
        t = np.linspace(0, 1, B.NB)
        bt, g1, i_sys = scaled(np.where(t < .25, t / .25, np.exp(-(t - .25) / .2)))
        self.assertEqual(B.diastolic_point(bt, g1, i_sys), (None, 0.0))

    def test_the_search_stops_at_its_bound(self):
        bt, g1, i_sys = self.knots((0, 0), (26, 1), (70, .10), (85, .30), (92, .20), (99, 0))    # a rise that tops out after the bound
        self.assertGreater(85, B.SEARCH_END)
        i_dia, notch = B.diastolic_point(bt, g1, i_sys)
        self.assertEqual(notch, 0.0)
        self.assertTrue(i_dia is None or i_dia < B.SEARCH_END)

    def test_a_peak_late_in_the_beat_does_not_break_it(self):
        for peak in (60, 68, 69):
            bt, g1, i_sys = self.knots((0, 0), (peak, 1), (99, 0))
            self.assertEqual(i_sys, peak)
            i_dia, notch = B.diastolic_point(bt, g1, i_sys)
            self.assertEqual(notch, 0.0)
            self.assertTrue(i_dia is None or i_dia < B.SEARCH_END)


def record(ibi, second, seconds=120, **kw):
    n = int(round(ibi * FS))
    one = wave(np.arange(n) / n, second=second, **kw)
    x = np.tile(one, seconds * FS // n + 1)[:seconds * FS // EP * EP]
    return x.astype(np.float32), np.ones(len(x) // EP, bool)


class TestBeatTable(unittest.TestCase):
    def test_columns_are_those_named(self):
        bt = B.beat_table(*record(.8, .45))
        self.assertEqual(set(B.BEAT_FEATS) - set(bt.columns), set())
        self.assertEqual(B.BEAT_FEATS, ["ibi", "t_sys", "t_sys_frac", "dT", "dT_frac", "ri", "notch", "ba", "width50", "slope"])

    def test_a_visible_notch_through_the_whole_path(self):
        bt = B.beat_table(*record(.8, .45))
        self.assertEqual(bt.notch.median(), 1.0)
        self.assertTrue(.15 < bt.dT_frac.median() < .5)
        self.assertTrue(.2 < bt.ri.median() < 1)
        np.testing.assert_allclose(bt.dT, bt.dT_frac * bt.ibi)

    def test_a_stiff_beat_is_not_placed_late(self):
        bt = B.beat_table(*record(.8, .30, at=.50, width2=.16))
        self.assertEqual(bt.notch.median(), 0.0)
        self.assertLess(bt.dT_frac.median() + bt.t_sys_frac.median(), B.SEARCH_END / B.NB)

    def test_a_fast_stiff_beat_too(self):
        bt = B.beat_table(*record(.42, .30, at=.50, width2=.16))
        self.assertEqual(bt.notch.median(), 0.0)
        self.assertLess(bt.dT_frac.median() + bt.t_sys_frac.median(), B.SEARCH_END / B.NB)

    def test_a_beat_without_a_diastolic_point_keeps_its_other_features(self):
        x, good = record(.8, .45)
        with_point = B.beat_table(x, good)
        with mock.patch.object(B, "diastolic_point", return_value=(None, 0.0)):
            without = B.beat_table(x, good)
        self.assertEqual(len(without), len(with_point))
        self.assertTrue(without[["dT", "dT_frac", "ri"]].isna().all().all())
        for c in ("ibi", "t_sys", "t_sys_frac", "ba", "width50", "slope"):
            np.testing.assert_array_equal(without[c], with_point[c])


if __name__ == "__main__":
    unittest.main()
