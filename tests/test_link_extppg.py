"""Tests of ppglabbench.link_extppg on generated tables. No restricted data is read.

Usage:  python -m unittest tests.test_link_extppg
"""
import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ppglabbench import link_extppg as L
from ppglabbench import labels as LB

TRAIN, TEST = 101, 104


def hashed(split):
    """A subject outside the split file that the split rule puts in this split."""
    return next(s for s in range(100000, 110000) if LB.split_of(s, set()) == split)


class LinkCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.out = self.tmp / "staging"
        self.other_train, self.other_test = hashed("train"), hashed("test")
        pd.DataFrame(dict(subject_id=[TRAIN, TEST], split=["train", "test"], dev_set=False)).to_csv(self.tmp / "splits.csv", index=False)
        self.subjects = [TRAIN, TRAIN, TEST, self.other_train, self.other_test, TRAIN]
        n = len(self.subjects)
        self.meta = self.tmp / "metadata.csv"
        pd.DataFrame(dict(
            signal_file_name=[f"rec{i}_0_0" for i in range(n)], record_id=[f"rec{i}" for i in range(n)], event_id=0, segment_id=0, subject_id=self.subjects,
            start_segment=["2000-01-01 12:00:00", "2000-01-01 18:00:00", "2000-01-01 12:00:00", "2000-01-01 12:00:00", "2000-01-01 12:00:00", "2000-01-01 12:30:00"],
            vector_10s_pleth_sqi=["[1, 1, 1]", "[1, 0, 1]", "[1, 1, 1]", "[1, 1, 1]", "[1, 1, 1]", "[-3, 1, 1]"],
            strat_fold=[0, 1, 2, 3, 4, 5], ignored_column="x")).to_csv(self.meta, index=False)
        # lab measurement 5 min after the first segment's centre, and one 50 min after the same centre; test subject's has a marked value
        self.labs = dict(sodium=pd.DataFrame(dict(subject_id=[TRAIN, TRAIN, TEST, self.other_train, self.other_test],
                                                  charttime=pd.to_datetime(["2000-01-01 12:05:15", "2000-01-01 12:50:15", "2000-01-01 12:05:15", "2000-01-01 12:05:15", "2000-01-01 12:05:15"]),
                                                  valuenum=[140., 130., 999., 120., 888.])))
        for p in (mock.patch.object(L, "SPLITS", self.tmp / "splits.csv"), mock.patch.object(L, "S1", self.tmp / "no_catalog"),
                  mock.patch.object(L, "load_labs", side_effect=lambda subjects: {k: v[v.subject_id.isin(subjects)] for k, v in self.labs.items()})):
            p.start()
            self.addCleanup(p.stop)

    def run_main(self, *args):
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            L.main(["--out", str(self.out), "--metadata", str(self.meta), *args])
        return text.getvalue()

    def table(self):
        with gzip.open(self.out / L.FOLDER / "ext_ppg_lab_links.csv.gz", "rt") as f:
            return pd.read_csv(f)

    def counts(self):
        return json.loads((self.out / L.FOLDER / "counts.json").read_text())


class TestSegments(LinkCase):
    def test_test_subjects_of_the_file_and_of_the_rule_are_gone(self):
        d, left_out, read = L.read_segments(self.meta)
        self.assertEqual(read, 6)
        self.assertEqual(left_out, 2)
        self.assertEqual(set(d.subject_id), {TRAIN, self.other_train})

    def test_all_good_means_three_ones(self):
        d, _, _ = L.read_segments(self.meta)
        by = d.set_index("signal_file_name").pleth_sqi_all_good
        self.assertTrue(by["rec0_0_0"])
        self.assertFalse(by["rec1_0_0"])
        self.assertFalse(by["rec5_0_0"])

    def test_the_centre_is_15_s_after_the_start(self):
        d, _, _ = L.read_segments(self.meta)
        self.assertEqual((d.centre - d.start_segment).unique()[0], pd.Timedelta(seconds=15))


class TestLink(LinkCase):
    def test_nearest_within_the_window_and_signed_offset(self):
        t = self.run_main() and self.table()
        first = t[t.signal_file_name == "rec0_0_0"].iloc[0]
        self.assertEqual(first.value, 140)
        self.assertAlmostEqual(first.offset_min, 5.0, places=3)
        self.assertEqual(t[t.signal_file_name == "rec5_0_0"].iloc[0].value, 130)     # 12:30 is nearer the 12:50 value than the 12:05 one

    def test_a_segment_with_nothing_within_60_min_has_no_row(self):
        self.run_main()
        self.assertNotIn("rec1_0_0", set(self.table().signal_file_name))

    def test_the_columns_and_the_fold_are_kept(self):
        self.run_main()
        t = self.table()
        self.assertEqual(list(t.columns), L.OUTPUT_COLUMNS)
        self.assertEqual(t[t.signal_file_name == "rec5_0_0"].strat_fold.iloc[0], 5)

    def test_no_row_of_a_test_subject_and_no_value_of_one(self):
        self.run_main()
        t = self.table()
        self.assertFalse(set(t.subject_id) & {TEST, self.other_test})
        self.assertFalse(set(t.value) & {999, 888})
        self.assertFalse(self.counts()["test_subjects_included"])

    def test_counts_at_each_window_and_quality_rule(self):
        self.run_main()
        c = self.counts()["windows"]
        self.assertEqual(c["within_60_min_any_sqi"]["link_rows"], 3)    # rec0, rec3, rec5
        self.assertEqual(c["within_60_min_all_good_pleth_sqi"]["link_rows"], 2)
        self.assertEqual(c["within_10_min_any_sqi"]["link_rows"], 2)    # rec0, rec3; rec5 is 20 min away
        self.assertEqual(c["within_10_min_all_good_pleth_sqi"]["segments"], 2)
        self.assertEqual(c["within_60_min_any_sqi"]["subjects"], 2)
        self.assertEqual(c["within_60_min_any_sqi"]["subjects_in_this_benchmark"], 1)
        self.assertEqual(c["within_60_min_any_sqi"]["per_lab"]["sodium"]["measurements"], 3)

    def test_nothing_printed_holds_an_id_or_a_value(self):
        text = self.run_main()
        for secret in (str(TRAIN), "rec0", "140", "2000-01"):
            self.assertNotIn(secret, text.replace("in this benchmark", ""))


class TestIncludeTest(LinkCase):
    def test_refused_while_the_split_is_closed(self):
        with mock.patch.dict(L.C, test_opened=False), self.assertRaisesRegex(SystemExit, "benchmark.test_opened"):
            self.run_main("--include-test")
        self.assertFalse(self.out.exists())

    def test_allowed_once_opened(self):
        with mock.patch.dict(L.C, test_opened=True):
            self.run_main("--include-test")
        self.assertIn(TEST, set(self.table().subject_id))
        self.assertTrue(self.counts()["test_subjects_included"])

    def test_counts_only_writes_no_table(self):
        self.run_main("--counts-only")
        self.assertFalse((self.out / L.FOLDER / "ext_ppg_lab_links.csv.gz").exists())
        self.assertTrue((self.out / L.FOLDER / "counts.json").is_file())


class TestClock(LinkCase):
    def test_share_of_segments_inside_a_record(self):
        folder = self.tmp / "s1"
        folder.mkdir()
        pd.DataFrame(dict(subject_id=[TRAIN], start=["2000-01-01 11:00:00"], end=["2000-01-01 13:00:00"], error=[np.nan])).to_csv(folder / "waveform_catalog.csv", index=False)
        d = pd.DataFrame(dict(subject_id=[TRAIN, TRAIN], start_segment=pd.to_datetime(["2000-01-01 12:00:00", "2000-01-01 14:00:00"])))
        with mock.patch.object(L, "S1", folder):
            self.assertEqual(L.clock_check(d), 0.5)


if __name__ == "__main__":
    unittest.main()
