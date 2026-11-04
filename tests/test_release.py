"""Tests of ppglabbench.release on a generated data folder. No restricted data is read.

Usage:  python -m unittest tests.test_release
"""
import contextlib
import hashlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import wfdb

from ppglabbench import baselines as BL
from ppglabbench import labels as LB
from ppglabbench import release as R

SUBJECTS = {101: "train", 102: "train", 103: "val", 104: "test"}
PLANTED_VALUE = "987.654321"     # an outcome value that must never reach a summary


def columns_of(file):
    d = R.dictionary()
    return list(dict.fromkeys(d.column[d.file == file]))


def stored_window(fmt, seed, first_missing=False):
    """A window as S3 stores it: digital values of one source encoding read as physical values, then cast to float16, with a gap."""
    rng = np.random.default_rng(seed)
    x = ((rng.integers(-127, 128, 150000) + 128) / 255 if fmt == "80" else rng.integers(0, 4096, 150000) / 1023).astype(np.float32)
    x[1000:1400] = np.nan
    if first_missing:
        x[:10] = np.nan
    return x.astype(np.float16)


def make_data(root):
    """A data folder with every file the package reads: two windows per subject, one of which has no usable pre-draw window."""
    rows = []
    for s, split in SUBJECTS.items():
        for k, score in enumerate((0.9, 0.5)):
            rows.append(dict(draw_id=f"{s}_20000101{k:02d}00", subject_id=s, split=split, pre_score=score, charttime=f"2000-01-01 {k:02d}:00:00"))
    base = pd.DataFrame(rows)
    table = pd.DataFrame({c: 1.0 for c in columns_of(R.LABELS) + sorted(R.DROP[R.LABELS])}, index=base.index)
    for c in base:
        table[c] = base[c]
    table["file"] = table.draw_id + ".npz"
    table["records"] = "rec"
    for flag in ("record_overlap", "exclude_dialysis", "female"):
        table[flag] = False
    table["bun"] = PLANTED_VALUE
    table["hadm_id"] = 7.0
    (root / "benchmark" / "features" / "mimic_cov").mkdir(parents=True)
    table.to_csv(root / R.LABELS, index=False)
    splits = pd.DataFrame(dict(subject_id=list(SUBJECTS), split=list(SUBJECTS.values()), dev_set=False))
    splits.to_csv(root / R.SPLITS, index=False)
    ids = list(table.draw_id)
    frame = lambda cols: pd.DataFrame({c: 1.0 for c in cols}, index=range(len(ids))).assign(draw_id=ids)
    frame([c for c in columns_of(R.COVARIATES) if c != "draw_id"]).to_csv(root / R.COVARIATES, index=False)
    pre = [i for i, s in zip(ids, table.pre_score) if s >= 0.8]
    for w, keep in (("pre", pre), ("centered", ids)):
        for sub, name, extra in (("mimic_abp", "abp_map.csv", ()), ("mimic_num", "numerics.csv", ("pulse_mon",))):
            f = root / "benchmark" / "features" / w / sub / name
            f.parent.mkdir(parents=True)
            cols = columns_of(f"benchmark/features/{w}/{sub}/{name}") + list(extra)
            t = pd.DataFrame({c: 1.0 for c in cols}, index=range(len(keep))).assign(draw_id=keep, subject_id=1, error=np.nan)
            t.to_csv(f, index=False)
        folder = root / "benchmark" / "features" / w / "mimic"
        folder.mkdir(parents=True)
        pd.DataFrame(dict(draw_id=keep, n_beats=30.0)).to_csv(folder / "beat_features.csv", index=False)
        np.savez_compressed(folder / "papagei.npz", ids=np.array(keep, dtype=object), emb=np.ones((len(keep), 512)))
    windows = root / R.WINDOW_DIR
    windows.mkdir(parents=True)
    for n, i in enumerate(ids):
        np.savez_compressed(windows / f"{i}.npz", pleth=stored_window("80" if n % 2 else "16", n, first_missing=n == 0), fs=125, t0_s=-600)
    pd.DataFrame(dict(draw_id=ids, subject_id=1, file=[f"{i}.npz" for i in ids], coverage=1.0, n_segments=1, segments="s", error=np.nan)).to_csv(root / R.INDEX, index=False)
    (root / "s4_sqi").mkdir()
    np.savez_compressed(root / R.EPOCHS, **{i: np.ones(120, bool) for i in ids}, **{"999_200001010000": np.ones(120, bool)})
    return ids, hashlib.sha256((root / R.SPLITS).read_bytes()).hexdigest()


class ReleaseCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.data = self.tmp / "data"
        self.data.mkdir()
        self.ids, sha = make_data(self.data)
        self.out = self.data / "release_staging"
        for p in (mock.patch.dict(R.C, splits_sha256=sha), mock.patch.object(R, "DATA", self.data), mock.patch.object(R, "ROOT", self.tmp)):
            p.start()
            self.addCleanup(p.stop)

    def run_main(self, *args):
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            R.main(["--out", str(self.out), *args])
        return text.getvalue()


class TestDryRun(ReleaseCase):
    def test_writes_nothing(self):
        before = sorted(p for p in self.tmp.rglob("*"))
        self.run_main("--dry-run")
        self.assertEqual(sorted(p for p in self.tmp.rglob("*")), before)
        self.assertFalse(self.out.exists())

    def test_lists_files_and_sizes(self):
        text = self.run_main("--dry-run")
        for f in (R.LABELS, R.SPLITS, R.COHORT, R.COVARIATES, R.INDEX, R.EPOCHS, R.DRAWS, "README.md", "RECORDS", "SHA256SUMS", "data_dictionary.csv", "NOTICE.txt"):
            self.assertIn(f, text)
        self.assertIn("<draw_id>.hea,.dat", text)
        self.assertIn(f"in {2 * len(self.ids)} files", text)

    def test_the_sizes_are_those_of_the_build(self):
        items, window_items, _ = R.plan_package()
        target = R.build(items, window_items, self.out)
        self.assertEqual(sum(i.size for i in items) + sum(w.size for w in window_items) + R.sums_size([i.path for i in items] + [p for w in window_items for p in w.paths]),
                         sum(f.stat().st_size for f in target.rglob("*") if f.is_file()))

    def test_no_id_and_no_outcome_is_printed(self):
        for text in (self.run_main("--dry-run"), self.run_main()):
            for i in self.ids:
                self.assertNotIn(i, text)
            self.assertNotIn(PLANTED_VALUE, text)
            for word in ("abnormal", "prevalence", "AUROC"):
                self.assertNotIn(word, text)
            self.assertIn("Structure only", text)


class TestPackage(ReleaseCase):
    def setUp(self):
        super().setUp()
        self.run_main()
        self.package = self.out / "package"

    def files(self):
        return sorted(str(f.relative_to(self.package)) for f in self.package.rglob("*") if f.is_file())

    def test_checksums_cover_every_file_and_hold(self):
        listed = {}
        for line in (self.package / "SHA256SUMS").read_text().splitlines():
            sha, path = line.split("  ", 1)
            listed[path] = sha
        self.assertEqual(sorted(listed), sorted(f for f in self.files() if f != "SHA256SUMS"))
        for path, sha in listed.items():
            self.assertEqual(hashlib.sha256((self.package / path).read_bytes()).hexdigest(), sha, path)

    def test_records_lists_every_window_without_an_extension(self):
        lines = (self.package / "RECORDS").read_text().split()
        self.assertEqual(sorted(lines), sorted(f"{R.WINDOW_DIR}/p{i[:3].zfill(6)[:2]}/p{i[:3].zfill(6)}/{i}" for i in self.ids))
        for line in lines:
            self.assertTrue((self.package / f"{line}.hea").is_file())
            self.assertTrue((self.package / f"{line}.dat").is_file())

    def test_no_window_is_left_as_a_numpy_file(self):
        self.assertFalse([f for f in self.files() if f.startswith(R.WINDOW_DIR) and not f.endswith((".hea", ".dat"))])

    def test_the_readme_is_complete_and_holds_no_id_and_no_outcome(self):
        text = (self.package / "README.md").read_text()
        self.assertTrue(text.startswith(f"# {R.TITLE}\n"))
        self.assertNotIn("TBD", text)
        self.assertNotIn(PLANTED_VALUE, text)
        for i in self.ids:
            self.assertNotIn(i, text)
        self.assertIn(f"{len(self.ids)} windows from {len(SUBJECTS)} adult subjects", text)
        self.assertIn("91.4", text)
        self.assertIn("_age_h", text)

    def test_the_manifest_holds_structure_only(self):
        info = json.loads((self.package / "release_manifest.json").read_text())
        self.assertEqual(sorted(info), sorted(["title", "version", "splits_sha256", "windows", "subjects", "windows_in_cohort_pre", "windows_in_cohort_centered", "windows_per_split",
                                               "subjects_per_split", "windows_included", "window_encodings", "left_out_incomplete", "not_included"]))

    def test_the_title_meets_the_rule(self):
        self.assertRegex(R.TITLE, r"^[A-Za-z0-9 _-]{1,200}$")
        self.assertIn("Ext", R.TITLE)

    def test_no_two_small_files_are_identical(self):
        """PhysioNet's validator warns about byte-identical files, and the completeness markers would be."""
        sums = [line.split("  ")[0] for line in (self.package / "SHA256SUMS").read_text().splitlines()]
        self.assertEqual(len(sums), len(set(sums)))

    def test_file_names_have_no_space_or_special_character(self):
        for f in self.files():
            self.assertRegex(f, r"^[A-Za-z0-9_./-]+$")

    def test_the_split_file_is_byte_identical(self):
        self.assertEqual(hashlib.sha256((self.package / R.SPLITS).read_bytes()).hexdigest(), R.C["splits_sha256"])
        self.assertEqual(json.loads((self.package / "release_manifest.json").read_text())["splits_sha256"], R.C["splits_sha256"])

    def test_no_embedding_no_vitaldb_no_results(self):
        for f in self.files():
            for word in R.FORBIDDEN_IN_PATH:
                self.assertNotIn(word, f.lower())

    def test_the_dictionary_covers_every_column_of_every_table(self):
        d = pd.read_csv(self.package / "data_dictionary.csv", keep_default_na=False)
        for f in self.files():
            if f.endswith(".csv") and f != "data_dictionary.csv":
                self.assertEqual(set(pd.read_csv(self.package / f, nrows=1).columns), set(d.column[d.file == f]), f)
        self.assertTrue((d.definition.str.len() > 0).all())
        self.assertEqual(d.columns.tolist(), ["file", "column", "unit", "source", "itemid", "definition"])

    def test_itemids_come_from_the_config(self):
        d = pd.read_csv(self.package / "data_dictionary.csv", keep_default_na=False).set_index(["file", "column"])
        for lab, (item, _) in R.C["labs"].items():
            self.assertEqual(d.loc[(R.LABELS, lab), "itemid"], str(item))
            self.assertEqual(d.loc[(R.LABELS, f"{lab}_locf"), "itemid"], str(item))
        for lab, item in R.C["blood_gas"].items():
            self.assertEqual(d.loc[(R.LABELS, f"{lab}_bg_locf"), "itemid"], str(item))

    def test_columns_left_out(self):
        header = pd.read_csv(self.package / R.LABELS, nrows=1).columns
        for c in R.DROP[R.LABELS]:
            self.assertNotIn(c, header)
        num = pd.read_csv(self.package / "benchmark/features/pre/mimic_num/numerics.csv", nrows=1).columns
        self.assertNotIn("pulse_mon", num)
        self.assertNotIn("error", num)

    def test_values_pass_through_unchanged(self):
        keep = pd.read_csv(self.package / R.LABELS, dtype=str)
        source = pd.read_csv(self.data / R.LABELS, dtype=str)
        pd.testing.assert_frame_equal(keep, source[keep.columns])

    def test_epochs_hold_only_released_windows(self):
        with np.load(self.package / R.EPOCHS) as z:
            self.assertEqual(sorted(z.files), sorted(self.ids))

    def test_cohort_flags(self):
        c = pd.read_csv(self.package / R.COHORT).set_index("draw_id")
        self.assertEqual(int(c.in_cohort_pre.sum()), 4)     # pre_score 0.9 on one window per subject
        self.assertEqual(int(c.in_cohort_centered.sum()), 8)

    def test_an_existing_package_needs_force(self):
        with self.assertRaisesRegex(SystemExit, "--force"):
            self.run_main()
        self.run_main("--force")

    def test_the_package_reads_as_a_data_folder(self):
        """Linked as data/ with the regenerated features beside it, the benchmark's own loader takes it."""
        for w, keep in (("pre", pd.read_csv(self.package / R.LABELS).query("pre_score >= 0.8").draw_id.tolist()),):
            folder = self.package / "benchmark" / "features" / w / "mimic"
            folder.mkdir(parents=True)
            pd.DataFrame(dict(draw_id=keep, n_beats=30.0, feat=1.0)).to_csv(folder / "beat_features.csv", index=False)
            (folder / "manifest.json").write_text(json.dumps(dict(params=dict(search_end=BL.SEARCH_END, min_notch=BL.MIN_NOTCH))))
            np.savez_compressed(folder / "papagei.npz", ids=np.array(keep, dtype=object), emb=np.ones((len(keep), 512)))
        with mock.patch.object(BL, "DATA", self.package), mock.patch.object(BL, "FEAT", self.package / "benchmark" / "features"), \
                mock.patch.object(LB, "DATA", self.package), mock.patch.object(LB, "OUT", self.package / "benchmark"):
            d, bcols, labs, tiers = BL.load_mimic("pre")
        self.assertEqual(len(d), 4)
        self.assertIn("feat", bcols)
        self.assertEqual(set(R.C["labs"]) | {"egfr", "creatinine"}, set(labs))
        for cols in tiers.values():
            self.assertFalse(set(cols) - set(d.columns))


class TestWindows(ReleaseCase):
    def stored(self, draw_id):
        with np.load(self.data / R.WINDOW_DIR / f"{draw_id}.npz") as z:
            return z["pleth"], int(z["fs"]), int(z["t0_s"])

    def record(self, draw_id):
        return next(self.out / "package" / r for r in (self.out / "package" / "RECORDS").read_text().split() if r.endswith(draw_id))

    def test_every_digital_value_of_both_source_encodings_gives_back_its_float16(self):
        for fmt, gain, baseline, top in R.ENCODINGS:
            digital = np.arange(-127 if fmt == "80" else 0, top + 1)
            pleth = ((digital - baseline) / gain).astype(np.float32).astype(np.float16)
            found = R.encode(pleth)
            self.assertEqual(found[:3], (fmt, gain, baseline))
            self.assertEqual(R.decode(found[3], *found[:3]).tobytes(), pleth.tobytes())
            self.assertLessEqual(found[3].max(), top)

    def test_digital_values_are_the_source_s_below_2_nu(self):
        digital = np.arange(0, 2046)
        self.assertTrue((R.encode((digital / 1023).astype(np.float32).astype(np.float16))[3] == digital).all())
        digital = np.arange(-127, 128)
        self.assertTrue((R.encode(((digital + 128) / 255).astype(np.float32).astype(np.float16))[3] == digital).all())

    def test_a_record_is_byte_identical_to_what_wfdb_writes(self):
        for fmt, gain, baseline, _ in R.ENCODINGS:
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as d:
                pleth = stored_window(fmt, 3, first_missing=True)
                found = R.encode(pleth)
                self.assertEqual(found[:3], (fmt, gain, baseline))
                wfdb.wrsamp("rec", fs=125, units=["NU"], sig_name=["PLETH"], d_signal=found[3].reshape(-1, 1), fmt=[fmt], adc_gain=[gain], baseline=[baseline], comments=["t0_s -600"], write_dir=d)
                _, hea, dat = R.render("rec", pleth, 125, -600)
                self.assertEqual(hea, (Path(d) / "rec.hea").read_bytes())
                self.assertEqual(dat, (Path(d) / "rec.dat").read_bytes())

    def test_the_records_give_back_the_stored_arrays(self):
        self.run_main()
        for i in self.ids:
            pleth, fs, t0_s = self.stored(i)
            back = wfdb.rdrecord(str(self.record(i)))
            self.assertEqual(back.p_signal[:, 0].astype(np.float16).tobytes(), pleth.tobytes(), "bit for bit, NaN included")
            self.assertEqual(back.fs, fs)
            self.assertEqual(back.sig_name, ["PLETH"])
            self.assertEqual(back.units, ["NU"])
            self.assertIn((back.fmt[0], back.adc_gain[0], back.baseline[0]), [e[:3] for e in R.ENCODINGS])
            self.assertEqual(R.read_record(self.record(i))[2], t0_s)

    def test_digital_values_read_back_as_written(self):
        self.run_main()
        for i in self.ids:
            pleth, _, _ = self.stored(i)
            found = R.encode(pleth)
            back = wfdb.rdrecord(str(self.record(i)), physical=False)
            self.assertTrue((back.d_signal[:, 0] == found[3]).all())

    def test_both_encodings_are_counted(self):
        self.run_main()
        info = json.loads((self.out / "package" / "release_manifest.json").read_text())
        self.assertEqual(sum(info["window_encodings"].values()), len(self.ids))
        self.assertEqual(len(info["window_encodings"]), 2)

    def test_restore_writes_what_the_feature_code_reads(self):
        self.run_main()
        self.assertIn(f"{len(self.ids)} NumPy files", self.run_restore())
        draws = pd.read_csv(self.out / "package" / R.DRAWS)
        for r in draws.itertuples():
            with np.load(self.out / "package" / R.WINDOW_DIR / r.file) as restored, np.load(self.data / R.WINDOW_DIR / r.file) as original:
                self.assertEqual(sorted(restored.files), sorted(original.files))
                for key in original.files:
                    self.assertEqual(restored[key].dtype, original[key].dtype, key)
                    self.assertEqual(restored[key].tobytes(), original[key].tobytes(), key)

    def run_restore(self):
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            R.main(["--restore-windows", str(self.out / "package")])
        return text.getvalue()

    def test_a_window_of_two_gains_stops_the_build_and_names_none(self):
        pleth = stored_window("80", 1)
        pleth[75000:] = stored_window("16", 2)[75000:]
        np.savez_compressed(self.data / R.WINDOW_DIR / f"{self.ids[0]}.npz", pleth=pleth, fs=125, t0_s=-600)
        with self.assertRaisesRegex(SystemExit, "1 windows have no exact WFDB encoding") as e:
            self.run_main("--dry-run")
        self.assertNotIn(self.ids[0], str(e.exception))
        self.assertFalse((self.out / "package").exists())

    def test_a_value_on_neither_grid_stops_the_build(self):
        pleth = stored_window("80", 1)
        pleth[5] = 0.123
        self.assertIsNone(R.encode(pleth))

    def test_zero_is_not_taken_for_the_invalid_value_of_format_80(self):
        pleth = (np.array([0.0, 85.0, 170.0, 255.0]) / 255).astype(np.float16)   # on both grids, but digital -128 would read back as missing in format 80
        found = R.encode(pleth)
        self.assertEqual(found[0], "16")
        self.assertEqual(R.decode(found[3], *found[:3]).tobytes(), pleth.tobytes())

    def test_a_saturated_value_stays_in_the_adc_range(self):
        pleth = (np.array([0, 2000, 4095]) / 1023).astype(np.float32).astype(np.float16)
        self.assertEqual(R.encode(pleth)[3].max(), 4095)

    def test_the_build_stops_when_a_record_does_not_read_back(self):
        items, window_items, _ = R.plan_package()
        with mock.patch.object(R, "read_record", return_value=(np.zeros(150000, np.float16), 125, -600)):
            with self.assertRaisesRegex(SystemExit, "did not read back"):
                R.build(items, window_items, self.out)
        self.assertFalse((self.out / "package").exists())
        self.assertFalse((self.out / "package.tmp").exists())


class TestStatistics(ReleaseCase):
    def test_a_test_row_is_never_parsed(self):
        t = pd.read_csv(self.data / R.LABELS, dtype=str)
        t.loc[t.split == "test", ["bun", "egfr", "age", "sodium", "female"]] = "SEALED"
        t.to_csv(self.data / R.LABELS, index=False)
        stats = R.non_test_statistics(self.data / R.LABELS)     # float("SEALED") would raise
        self.assertEqual(stats["windows"], int((t.split != "test").sum()))
        self.assertAlmostEqual(next(r for r in stats["rows"] if r["column"] == "bun")["median"], float(PLANTED_VALUE))

    def test_it_counts_train_and_val_only(self):
        self.assertEqual(R.non_test_statistics(self.data / R.LABELS)["windows"], 6)

    def test_the_table_is_markdown_with_no_id(self):
        text = R.statistics_table(R.non_test_statistics(self.data / R.LABELS))
        self.assertIn("| bun | mg/dL |", text)
        for i in self.ids:
            self.assertNotIn(i, text)

    def test_the_build_summary_holds_no_statistic(self):
        items, window_items, info = R.plan_package()
        self.assertNotIn("median", R.summary(items, window_items, info, True))


class TestRefusals(ReleaseCase):
    def test_a_changed_split_file_stops_it(self):
        with (self.data / R.SPLITS).open("a") as f:
            f.write("105,train,False\n")
        with self.assertRaisesRegex(SystemExit, "splits_sha256"):
            self.run_main("--dry-run")

    def test_a_column_without_a_dictionary_row_stops_it(self):
        cov = pd.read_csv(self.data / R.COVARIATES).assign(new_column=1.0)
        cov.to_csv(self.data / R.COVARIATES, index=False)
        with self.assertRaisesRegex(SystemExit, "new_column"):
            self.run_main("--dry-run")

    def test_an_incomplete_table_stops_the_build_and_warns_in_a_dry_run(self):
        f = self.data / "benchmark/features/centered/mimic_num/numerics.csv"
        pd.read_csv(f).iloc[:-1].to_csv(f, index=False)
        self.assertIn("WARNING, incomplete tables", self.run_main("--dry-run"))
        with self.assertRaisesRegex(SystemExit, "incomplete"):
            self.run_main()
        self.assertFalse((self.out / "package").exists())
        self.run_main("--allow-incomplete")
        self.assertFalse((self.out / "package/benchmark/features/centered/mimic_num").exists())
        self.assertTrue((self.out / "package/benchmark/features/pre/mimic_num/numerics.csv").is_file())

    def test_a_failed_row_counts_as_incomplete(self):
        f = self.data / "benchmark/features/pre/mimic_abp/abp_map.csv"
        pd.read_csv(f).assign(error=lambda t: ["timeout"] + [np.nan] * (len(t) - 1)).to_csv(f, index=False)
        self.assertIn("1 of 4 windows", self.run_main("--dry-run"))

    def test_a_missing_window_file_stops_it(self):
        (self.data / R.WINDOW_DIR / f"{self.ids[0]}.npz").unlink()
        with self.assertRaisesRegex(SystemExit, "1 window files are missing") as e:
            self.run_main("--dry-run")
        self.assertNotIn(self.ids[0], str(e.exception))

    def test_tables_only(self):
        text = self.run_main("--no-windows")
        self.assertNotIn("<draw_id>.hea", text)
        self.assertFalse((self.out / "package" / R.WINDOW_DIR).exists())
        self.assertFalse((self.out / "package" / "RECORDS").exists())

    def test_out_must_stay_where_git_cannot_reach(self):
        for bad in (self.tmp, self.tmp / "src", self.data):
            with self.assertRaisesRegex(SystemExit, "credentialed"):
                R.check_out(bad)
        self.assertEqual(R.check_out(self.data / "release_staging"), (self.data / "release_staging").resolve())
        with tempfile.TemporaryDirectory() as other:
            self.assertEqual(R.check_out(other), Path(other).resolve())


if __name__ == "__main__":
    unittest.main()
