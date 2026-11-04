"""History in the `now` base: earlier admissions from made-up tables, and the time and count of a lab's earlier draws. No restricted data is needed.

Usage:  python -m unittest tests.test_history
"""
import unittest

import numpy as np
import pandas as pd

from ppglabbench import baselines as BL
from ppglabbench.features import covariates as COV


def day(n):
    return pd.Timestamp(2000, 1, 1) + pd.Timedelta(days=n)


def windows(*rows):
    return pd.DataFrame(rows, columns=["subject_id", "hadm_id", "charttime"])


class TestAdmissionHistory(unittest.TestCase):
    ADM = pd.DataFrame([(1, 100, day(10)), (1, 101, day(20)), (1, 102, day(40)), (1, 103, day(60)), (2, 200, day(5))], columns=["subject_id", "hadm_id", "dischtime"])
    DX = pd.DataFrame([(100, "5859"), (101, "42731"), (102, "5715"), (102, "5849"), (103, "4280"), (200, "25000")], columns=["hadm_id", "icd9_code"])

    def history(self, *rows):
        return COV.admission_history(windows(*rows), self.ADM, self.DX)

    def test_counts_and_days_by_hand(self):
        f = self.history((1, 102, day(30)))
        self.assertAlmostEqual(f.prior_admissions[0], np.log1p(2))
        self.assertAlmostEqual(f.days_since_discharge[0], np.log1p(10))

    def test_only_admissions_that_ended_before_the_draw(self):
        f = self.history((1, 102, day(30)))
        self.assertEqual((f.prior_ckd[0], f.prior_af[0]), (1.0, 1.0))
        self.assertEqual(f.prior_chf[0], 0.0)      # admission 103 ends after the draw

    def test_no_code_of_the_current_stay(self):
        f = self.history((1, 102, day(30)))
        self.assertEqual((f.prior_liver[0], f.prior_aki[0]), (0.0, 0.0))

    def test_a_draw_after_its_own_discharge_does_not_count_its_own_stay(self):
        f = self.history((1, 102, day(41)))
        self.assertAlmostEqual(f.prior_admissions[0], np.log1p(2))
        self.assertAlmostEqual(f.days_since_discharge[0], np.log1p(21))
        self.assertEqual((f.prior_liver[0], f.prior_aki[0]), (0.0, 0.0))

    def test_no_earlier_admission(self):
        f = self.history((1, 100, day(5)))
        self.assertEqual(f.prior_admissions[0], 0.0)
        self.assertTrue(np.isnan(f.days_since_discharge[0]))
        self.assertEqual(f[list(COV.PRIOR_DX)].iloc[0].sum(), 0.0)

    def test_a_window_without_an_admission_id_sees_every_earlier_stay(self):
        f = self.history((1, np.nan, day(30)))
        self.assertAlmostEqual(f.prior_admissions[0], np.log1p(2))

    def test_subjects_do_not_mix_and_rows_stay_in_order(self):
        f = self.history((2, 200, day(1)), (1, 102, day(30)), (2, np.nan, day(9)), (3, np.nan, day(9)))
        self.assertEqual(list(f.prior_admissions.round(6)), list(np.round(np.log1p([0, 2, 1, 0]), 6)))
        self.assertEqual(list(f.prior_dm), [0., 0., 1., 0.])
        self.assertEqual(list(f.prior_af), [0., 1., 0., 0.])
        self.assertAlmostEqual(f.days_since_discharge[2], np.log1p(4))
        self.assertTrue(np.isnan(f.days_since_discharge[3]))

    def test_a_discharge_at_the_draw_is_not_earlier(self):
        f = self.history((1, np.nan, day(10)))
        self.assertEqual(f.prior_admissions[0], 0.0)

    def test_a_draw_right_after_a_discharge_stays_finite(self):
        f = self.history((1, np.nan, day(10) + pd.Timedelta(seconds=1)))
        self.assertTrue(np.isfinite(f.days_since_discharge[0]))

    def test_the_code_groups(self):
        codes = dict(prior_ckd=["5859", "5851"], prior_esrd=["5856", "40391"], prior_aki=["5849"], prior_af=["42731"], prior_htn=["4019", "40390"], prior_dm=["25000"], prior_chf=["4280"],
                     prior_liver=["5712", "570", "5728", "5733"])
        self.assertEqual(set(codes), set(COV.PRIOR_DX))
        for col, values in codes.items():
            for c in values:
                dx = pd.DataFrame([(100, c)], columns=["hadm_id", "icd9_code"])
                f = COV.admission_history(windows((1, 102, day(30))), self.ADM, dx)
                self.assertEqual(f[col][0], 1.0, (col, c))
        dx = pd.DataFrame([(100, c) for c in ("4273", "5742", "nan", "V1582")], columns=["hadm_id", "icd9_code"])
        f = COV.admission_history(windows((1, 102, day(30))), self.ADM, dx)
        self.assertEqual(f[list(COV.PRIOR_DX)].iloc[0].sum(), 0.0)

    def test_nothing_looks_like_ethnicity_or_a_label(self):
        self.assertFalse([c for c in COV.PRIOR_DX if c.startswith("eth_") or c == "prior_dialysis"])


def cohort(n=600, seed=0):
    """Train and val rows with one lab, some windows having no earlier draw."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(dict(group=np.arange(n), split=np.where(np.arange(n) % 3 == 0, "val", "train"), age=rng.normal(60, 15, n), female=(rng.random(n) < .5).astype(float),
                          vasopressor=(rng.random(n) < .2).astype(float), map_mmhg=rng.normal(80, 10, n), routine_draw=(rng.random(n) < .5).astype(float), hr_mon=rng.normal(85, 10, n),
                          on_sedation=(rng.random(n) < .3).astype(float)))
    d["k"], d["k_abnormal"] = rng.normal(4, 1, n), (rng.random(n) < .4).astype(float)
    none = rng.random(n) < .2
    for c, values in (("k_locf", rng.normal(4, 1, n)), ("k_locf_age_h", rng.exponential(20, n)), ("k_hist_mean", rng.normal(4, 1, n)), ("k_hist_n", rng.integers(1, 9, n).astype(float)), ("k_trend", rng.normal(0, 1, n))):
        d[c] = np.where(none, np.nan, values)
    d["k_bg_locf"], d["k_bg_locf_age_h"] = rng.normal(4, 1, n), rng.exponential(20, n)
    tiers = dict(acuity=["vasopressor", "map_mmhg", "routine_draw"], monitor=["vasopressor", "map_mmhg", "routine_draw", "hr_mon"], now=["vasopressor", "map_mmhg", "routine_draw", "hr_mon", "on_sedation"])
    return d, tiers, dict(value="k", abnormal="k_abnormal", locf=["k_locf", "k_locf_age_h"], hist=["k_hist_mean", "k_hist_n", "k_trend"]), none


class TestNowBase(unittest.TestCase):
    def setUp(self):
        self.d, self.tiers, self.spec, self.none = cohort()

    def bases(self, d=None, spec=None, task="clf"):
        return BL.task_frame(self.d if d is None else d, self.tiers, "k", spec or self.spec, task, "val")[5]

    def test_now_gains_the_time_and_the_count_as_its_last_columns(self):
        now = self.bases()["now"]
        without = self.bases(spec=dict(self.spec, locf=None))["now"]
        self.assertEqual(now.shape[1], without.shape[1] + 2)
        np.testing.assert_array_equal(now[:, :-2], without)
        np.testing.assert_allclose(now[:, -2], np.log1p(self.d.k_locf_age_h))
        np.testing.assert_allclose(now[:, -1], np.log1p(self.d.k_hist_n.fillna(0)))

    def test_no_earlier_draw_leaves_the_time_missing_and_the_count_zero(self):
        now = self.bases()["now"]
        self.assertTrue(np.isnan(now[self.none, -2]).all())
        self.assertTrue((now[self.none, -1] == 0).all())
        self.assertTrue(np.isfinite(now[~self.none, -2]).all())

    def test_the_imputer_adds_the_missing_indicator(self):
        now = self.bases()["now"]
        pipe = BL.base_clf()
        pipe.fit(now, self.d.k_abnormal)
        imp = pipe.named_steps["simpleimputer"]
        self.assertIn(now.shape[1] - 2, imp.indicator_.features_)
        self.assertNotIn(now.shape[1] - 1, imp.indicator_.features_)

    def test_no_value_of_the_lab_enters_now(self):
        changed = self.d.copy()
        for c in ("k_locf", "k_hist_mean", "k_trend", "k_bg_locf", "k_bg_locf_age_h"):
            changed[c] = changed[c] * 3 + 1
        a, b = self.bases(), self.bases(changed)
        np.testing.assert_array_equal(a["now"], b["now"])
        self.assertFalse(np.array_equal(a["full"], b["full"], equal_nan=True))

    def test_full_inherits_now_and_repeats_nothing(self):
        b = self.bases()
        now, full = b["now"], b["full"]
        np.testing.assert_array_equal(full[:, :now.shape[1]], now)
        self.assertEqual(full.shape[1], now.shape[1] + 5)       # value, mean, trend, blood-gas value and its age
        rest = full[:, now.shape[1]:]
        np.testing.assert_allclose(rest[:, 0], self.d.k_locf)
        np.testing.assert_allclose(rest[:, 1], self.d.k_hist_mean)
        np.testing.assert_allclose(rest[:, 2], self.d.k_trend)

    def test_the_other_rungs_do_not_change(self):
        b = self.bases()
        self.assertEqual(b["locf"].shape[1], 4)
        self.assertEqual(b["hist"].shape[1], 7)
        np.testing.assert_allclose(b["hist"][:, 5], np.log1p(self.d.k_hist_n))

    def test_a_skewed_lab_keeps_its_log_scale_in_full_only(self):
        d = self.d.assign(glucose=self.d.k + 10, glucose_abnormal=self.d.k_abnormal, glucose_locf=self.d.k_locf + 10, glucose_locf_age_h=self.d.k_locf_age_h, glucose_hist_mean=self.d.k_hist_mean + 10,
                          glucose_hist_n=self.d.k_hist_n, glucose_trend=self.d.k_trend)
        spec = dict(value="glucose", abnormal="glucose_abnormal", locf=["glucose_locf", "glucose_locf_age_h"], hist=["glucose_hist_mean", "glucose_hist_n", "glucose_trend"])
        b = BL.task_frame(d, self.tiers, "glucose", spec, "clf", "val")[5]
        np.testing.assert_allclose(b["full"][:, b["now"].shape[1]], np.log(d.glucose_locf))


if __name__ == "__main__":
    unittest.main()
