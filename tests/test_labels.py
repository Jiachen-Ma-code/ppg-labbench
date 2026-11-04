"""ppglabbench.labels on made-up strings and arrays. No data/ folder is needed.

Usage:  python -m unittest tests.test_labels
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ppglabbench import labels


class TestCensored(unittest.TestCase):
    @staticmethod
    def limit(text):
        return pd.Series([text]).str.extract(labels.CENSORED).bfill(axis=1).iloc[0, 0]

    def test_forms_of_a_limit(self):
        for text, want in ((">150", "150"), ("<0.5", "0.5"), (">=3.2", "3.2"), ("<= 2", "2"), ("GREATER THAN 10", "10"), ("LESS THAN 0.1", "0.1"),
                           (">GREATER THAN 10", "10"), ("> GREATER THAN 10", "10"), ("<LESS THAN 2", "2"), ("150 IS HIGHEST MEASURED PTT", "150")):
            with self.subTest(text):
                self.assertEqual(self.limit(text), want)

    def test_other_text_is_not_a_limit(self):
        for text in ("12", "12.3", "HEMOLYZED", "10-20", "NOT DONE", "nan", "GREATER THAN ULN"):
            with self.subTest(text):
                self.assertTrue(pd.isna(self.limit(text)))

    def test_two_groups_as_the_extraction_expects(self):
        self.assertEqual(pd.Series(["1"]).str.extract(labels.CENSORED).shape[1], 2)


class TestPreDrawWindow(unittest.TestCase):
    def window(self, good):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(labels, "S4", Path(t)):
            np.savez(Path(t) / "epoch_good.npz", w=np.asarray(good, bool))
            start, score = labels.pre_draw_window(["w"])
        return start[0], score[0]

    def test_every_epoch_good_is_a_tie_that_goes_nearest_the_draw(self):
        self.assertEqual(self.window(np.ones(120, bool)), (-300, 1.0))

    def test_only_epochs_before_the_draw_count(self):
        good = np.zeros(120, bool)
        good[60:] = True
        self.assertEqual(self.window(good), (-300, 0.0))

    def test_the_best_window(self):
        good = np.zeros(120, bool)
        good[5:35] = True
        start, score = self.window(good)
        self.assertEqual(start, -600 + 50)
        self.assertAlmostEqual(score, 1.0)


class TestSplitFile(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "splits.csv"

    def test_written_when_absent(self):
        labels.keep_split_file(self.path, b"a\n1\n")
        self.assertEqual(self.path.read_bytes(), b"a\n1\n")

    def test_the_same_bytes_change_nothing(self):
        self.path.write_bytes(b"a\n1\n")
        labels.keep_split_file(self.path, b"a\n1\n")
        self.assertEqual(self.path.read_bytes(), b"a\n1\n")

    def test_other_bytes_stop_the_run_and_the_file_stays(self):
        self.path.write_bytes(b"a\n1\n")
        with self.assertRaisesRegex(SystemExit, "not replaced"):
            labels.keep_split_file(self.path, b"a\n2\n")
        self.assertEqual(self.path.read_bytes(), b"a\n1\n")


if __name__ == "__main__":
    unittest.main()
