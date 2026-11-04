"""Input lengths of ppglabbench.features.pulseppg on generated signals. Needs torch, so it is skipped outside the torch environment.

Usage:  .venv-torch/bin/python -m unittest tests.test_pulseppg
"""
import unittest

import numpy as np

try:
    from ppglabbench.features import pulseppg as P
except ImportError:
    P = None


@unittest.skipIf(P is None, "needs torch")
class TestInput(unittest.TestCase):
    @staticmethod
    def signal(seconds):
        return np.sin(2 * np.pi * 1.2 * np.arange(int(seconds * P.FS)) / P.FS)

    def test_every_full_length_input_has_the_same_length_and_residue(self):
        for seconds in (240, 241, 300):
            s = P.prepare(self.signal(seconds))
            self.assertEqual(len(s) / P.FS_OUT, 240)
            self.assertEqual(len(s) % 64, 32)

    def test_a_shorter_run_gives_fewer_seconds_as_the_output_records_them(self):
        s = P.prepare(self.signal(200))
        self.assertLess(len(s) / P.FS_OUT, 240)
        self.assertGreater(len(s), 0)

    def test_the_longest_finite_run_is_used(self):
        x = self.signal(300)
        x[100 * P.FS] = np.nan
        self.assertAlmostEqual(len(P.prepare(x)) / P.FS_OUT, 199.99, delta=.05)

    def test_too_short_or_flat_is_none(self):
        self.assertIsNone(P.prepare(self.signal(29)))
        self.assertIsNone(P.prepare(np.ones(60 * P.FS)))
        self.assertIsNone(P.prepare(np.full(60 * P.FS, np.nan)))


if __name__ == "__main__":
    unittest.main()
