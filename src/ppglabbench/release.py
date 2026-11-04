"""Build the PhysioNet package of the MIMIC part of the benchmark in a staging folder.

The package has the layout of data/, with the windows as WFDB records. Its tables can be read as they are; --restore-windows writes the NumPy files
the feature code reads, so that a checkout works once the package is linked as data/:
  benchmark/mimic_labels.csv                        labs with offsets, previous values, history, flags, one row per window
  benchmark/mimic_splits.csv                        subject, split, development set; byte-identical to the file `splits_sha256` names
  benchmark/cohort.csv                              whether each window is in the benchmark cohort, per PPG window
  benchmark/features/mimic_cov/covariates.csv       treatments and context from the clinical tables
  benchmark/features/<window>/mimic_abp/abp_map.csv MAP from the arterial line (window: pre, centered)
  benchmark/features/<window>/mimic_num/numerics.csv monitor values, without the PLETH-derived pulse
  s3_windows/windows/pXX/pXXXXXX/<draw_id>.hea,.dat 20 min of PLETH around the draw, one WFDB record per window
  s3_windows/s3_index.csv                           source waveform segments of each window
  s4_sqi/epoch_good.npz                             per-window boolean SQI, one value per 10 s epoch
  s5_groupB/groupB_draws.csv                        the window table: subject, admission, shifted time, file, signal quality
  README.md, data_dictionary.csv, NOTICE.txt, RECORDS, SHA256SUMS, release_manifest.json

Each record keeps the encoding of its source: format 80, gain 255, baseline -128, or format 16, gain 1023, baseline 0. Its physical values cast to float16 are,
bit for bit, the stored array that the benchmark's features were computed from; the build checks this for every window and stops otherwise. float16 merges neighbouring
format 16 values at 2 NU and above, so there a digital value can differ from the source's by one count.

Left out: every embedding. PaPaGei's weights have no clear licence for redistributing outputs, so its embeddings are regenerated with
features/papagei.py. Pulse-PPG's weights are MIT, but its embeddings add nothing the code cannot regenerate from the windows. Beat features
are regenerated the same way. Cohort membership does depend on PaPaGei preprocessing, so it ships as a flag. VitalDB is code only.
Results of any evaluation are not part of the package.

Nothing printed or written as a summary holds an outcome or an ID: counts per split are structure only, and windows are listed as one line.
--statistics is the one exception to "no outcome": it gives summary statistics of the training and validation rows for the project page, and never parses a test row.

Usage:
    python -m ppglabbench.release --out data/release_staging --dry-run      # file list and sizes, writes nothing
    python -m ppglabbench.release --out data/release_staging [--force] [--no-windows] [--allow-incomplete] [--jobs N]
    python -m ppglabbench.release --restore-windows data                    # NumPy files beside the records, for the feature code
    python -m ppglabbench.release --statistics                              # training and validation rows only
Output: <out>/package/  (credentialed data: never committed, shared only through PhysioNet)
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ppglabbench.common import CFG, DATA, ROOT
from ppglabbench.features.covariates import AT_DRAW, ITEMS, PRESSORS, PRIOR_24H, PRIOR_DX
from ppglabbench.features.numerics import LAST, WINDOW
from ppglabbench.labels import C
from ppglabbench.pipeline.s6_build_dataset import DIALYSIS_PROC, DX

TITLE, VERSION = "MIMIC-III-Ext-LabBench photoplethysmography windows and blood laboratory values", "1.0.0"
PACKAGE = "package"
LABELS, SPLITS, COHORT = "benchmark/mimic_labels.csv", "benchmark/mimic_splits.csv", "benchmark/cohort.csv"
COVARIATES = "benchmark/features/mimic_cov/covariates.csv"
INDEX, WINDOW_DIR, EPOCHS, DRAWS = "s3_windows/s3_index.csv", "s3_windows/windows", "s4_sqi/epoch_good.npz", "s5_groupB/groupB_draws.csv"
WINDOWS = ("pre", "centered")
DROP = {LABELS: {"x_row", "s3_selected", "ckd_egfr", "hb_gdl", "bun_mgdl"},   # no reader: a row of a matrix not shipped, always true, a duplicate, ± 12 h labs
        INDEX: {"error"}, "abp": {"error"}, "num": {"error", "pulse_mon"}}    # the pulse comes from PLETH, and the covariates are never PPG-derived
DRAW_COLUMNS = ["draw_id", "subject_id", "hadm_id", "charttime", "file", "records", "best_score", "best_start_s", "best_hr", "best_tcorr", "best_skew",
                "best_ibi_cv", "valid_epoch_frac", "good_epoch_frac"]
# How the source records store PLETH: WFDB format, gain, baseline, highest digital value
ENCODINGS = (("80", 255.0, -128, 127), ("16", 1023.0, 0, 4095))
MISSING = {"80": -128, "16": -32768}    # the WFDB invalid sample of each format
ADC_RES = {"80": 8, "16": 16}           # the value wfdb writes for each format
FORBIDDEN_IN_PATH = ("papagei", "pulseppg", "embedding", "vitaldb", "results")
NOT_INCLUDED = {"PaPaGei and Pulse-PPG embeddings, or any other model output": "regenerate with ppglabbench.features.papagei and ppglabbench.features.pulseppg",
                "Beat features": "regenerate with ppglabbench.features.beat",
                "VitalDB": "code only, ppglabbench.sources.vitaldb",
                "Results of any evaluation": "not part of this resource"}
LAB_UNITS = dict(bun="mg/dL", sodium="mmol/L", potassium="mmol/L", bicarbonate="mmol/L", calcium="mg/dL", glucose="mg/dL", haemoglobin="g/dL",
                 platelets="10^3/uL", wbc="10^3/uL", inr="ratio", ptt="s", lactate="mmol/L", albumin="g/dL", ast="U/L", alt="U/L")
NOTICE = f"""{TITLE}, version {VERSION}.

The PPG windows under {WINDOW_DIR}/, the signal quality in {EPOCHS} and the columns of abp_map.csv and numerics.csv are derived from the MIMIC-III
Waveform Database Matched Subset v1.0 (Moody et al., PhysioNet, https://doi.org/10.13026/c2294b), made available under the Open Data Commons
Open Database License v1.0 (https://opendatacommons.org/licenses/odbl/1-0/).

Everything else, and the pairing of a window with a draw time, is derived from the MIMIC-III Clinical Database v1.4 (Johnson et al., PhysioNet,
https://doi.org/10.13026/C2XW26), under the PhysioNet Credentialed Health Data License 1.5.0 and its Data Use Agreement.

Column definitions are in data_dictionary.csv. Dates are the shifted dates of MIMIC-III.
"""


@dataclass
class Item:
    """One file of the package: its bytes, or the file they are copied from."""
    path: str
    data: Optional[bytes] = None
    source: Optional[Path] = None

    @property
    def size(self):
        return len(self.data) if self.source is None else self.source.stat().st_size


@dataclass
class Window:
    """One WFDB record of the package: its path without an extension, and the stored array it is written from. The scan fills in the rest."""
    record: str
    source: Path
    encoding: str = ""
    hea_size: int = 0
    dat_size: int = 0

    @property
    def size(self):
        return self.hea_size + self.dat_size

    @property
    def paths(self):
        return [f"{self.record}.hea", f"{self.record}.dat"]


# ------------------------------------------------------------------------------------------------- the dictionary

def dictionary():
    """One row per column of every table in the package, built from the same config and constants the pipeline reads."""
    rows = []
    def add(file, column, unit, source, itemid, definition):
        rows.append(dict(file=file, column=column, unit=unit, source=source, itemid=itemid, definition=definition))
    def ids(values):
        return " ".join(str(v) for v in values)
    FILE_NOTE = f"Name of the window's NumPy file in {WINDOW_DIR}/ once written by ppglabbench.release --restore-windows, which the feature code reads. The WFDB record has the same name without the extension."
    half, tol = C["window_half_min"], C["lab_window_min"]
    min_score, ep, win = CFG["s5_select"]["min_window_score"], CFG["s4_sqi"]["epoch_s"], CFG["s4_sqi"]["window_min"]
    lo, hi = CFG["creatinine"]["valid_range_mgdl"]
    s6 = CFG["s6_dataset"]
    WF, LE = "MIMIC-III Waveform Database Matched Subset v1.0", "LABEVENTS"

    key = [("draw_id", "", LE, CFG["creatinine"]["itemid"], "Window key: <subject_id>_<YYYYmmddHHMM> of the creatinine draw, with the shifted date of MIMIC-III."),
           ("subject_id", "", "PATIENTS", "", "MIMIC-III SUBJECT_ID.")]
    window = key + [
        ("hadm_id", "", LE, "", "HADM_ID of the draw. Empty when LABEVENTS ties the draw to no admission."),
        ("charttime", "date-shifted datetime", LE, CFG["creatinine"]["itemid"], "CHARTTIME of the creatinine draw the window is anchored on. Waveform and clinical clocks agree to about 10 min."),
        ("file", "", WF, "", FILE_NOTE),
        ("records", "", WF, "", "Waveform records with PLETH overlapping the draw, separated by |."),
        ("best_score", "fraction", f"{WF}: PLETH", "", f"Centred window: fraction of good {ep} s epochs in the best contiguous {win} min of the 20 min window; ties go to the window nearest the draw. A window is released if this is at least {min_score}."),
        ("best_start_s", "s", f"{WF}: PLETH", "", "Centred window: its start relative to the draw."),
        ("best_hr", "beats/min", f"{WF}: PLETH", "", "Centred window: median over epochs of the heart rate from PPG peaks."),
        ("best_tcorr", "correlation", f"{WF}: PLETH", "", "Centred window: median over epochs of the mean beat-to-template correlation."),
        ("best_skew", "", f"{WF}: PLETH", "", "Centred window: median over epochs of the skewness of the filtered signal. Reported, not thresholded."),
        ("best_ibi_cv", "ratio", f"{WF}: PLETH", "", "Centred window: median over epochs of the inter-beat interval coefficient of variation. Reported, not thresholded."),
        ("valid_epoch_frac", "fraction", f"{WF}: PLETH", "", "Fraction of the window's epochs with no missing samples."),
        ("good_epoch_frac", "fraction", f"{WF}: PLETH", "", "Fraction of the window's epochs passing the SQI rule.")]
    for c in window:
        add(DRAWS, *c)
        add(LABELS, *c)

    add(LABELS, "creat_mgdl", "mg/dL", LE, CFG["creatinine"]["itemid"], f"Creatinine at the draw, the anchor lab (offset 0). Values outside {lo}-{hi} mg/dL are not used.")
    add(LABELS, "age", "years", "PATIENTS, LABEVENTS", "", "Age at the draw from the shifted DOB. Ages of 90 or more, which MIMIC-III hides, are set to 91.4. Adults only.")
    add(LABELS, "female", "True/False", "PATIENTS", "", "GENDER is F.")
    add(LABELS, "egfr", "mL/min/1.73 m2", "derived", "", "CKD-EPI 2021 creatinine equation (race-free), from creat_mgdl, age and sex.")
    add(LABELS, "egfr_lt60", "1/0", "derived", "", "egfr below 60. The abnormal task for eGFR; analyses leave out rows with exclude_dialysis.")
    add(LABELS, "aki_kdigo", "True/False", LE, CFG["creatinine"]["itemid"], "KDIGO creatinine criterion at the draw: rise of at least 0.3 mg/dL over the lowest value in the prior 48 h, or at least 1.5 times the lowest value in the prior 7 days.")
    add(LABELS, "emergency_admit", "True/False", "ADMISSIONS", "", "ADMISSION_TYPE is EMERGENCY or URGENT.")
    names = dict(dx_ckd="chronic kidney disease", dx_esrd="end-stage renal disease, including CKD stage V and hypertensive stage V or ESRD", dx_aki="acute kidney injury",
                 dx_af="atrial fibrillation", dx_htn="hypertension", dx_dm="diabetes", dx_chf="heart failure", dx_dialysis_status="dialysis status")
    for col, prefixes in DX.items():
        add(LABELS, col, "True/False", "DIAGNOSES_ICD", f"ICD-9 prefixes {ids(prefixes)}", f"The admission has a code for {names[col]}. Coded at discharge, so context and not prediction-time information.")
    add(LABELS, "proc_dialysis", "True/False", "PROCEDURES_ICD", f"ICD-9 {ids(sorted(DIALYSIS_PROC))}", "A dialysis procedure in the admission. These codes have no time.")
    add(LABELS, "prior_dialysis", "True/False", "DIAGNOSES_ICD, ADMISSIONS", "", "The draw is later than the discharge of the subject's first admission coded ESRD or dialysis status.")
    add(LABELS, "exclude_dialysis", "True/False", "derived", "", "dx_esrd, dx_dialysis_status, proc_dialysis or prior_dialysis. Creatinine then reflects the dialysis schedule. The row is kept and flagged.")
    add(LABELS, "vasopressor", "True/False", "INPUTEVENTS_MV, INPUTEVENTS_CV", f"MV {ids(s6['vaso_mv'])}; CV {ids(s6['vaso_cv'])}",
        f"Vasopressor or inotrope at the draw. MetaVision: running, not cancelled or rewritten. CareVue: a positive rate or amount in the preceding {s6['vaso_cv_window_h']} h. "
        "Admissions without input events count as not on vasopressors.")
    def history(prefix, name, unit, item):
        add(LABELS, f"{prefix}_locf", unit, LE, item, f"Previous {name} of the subject, strictly before both the target measurement and the window start ({half} min before the draw).")
        add(LABELS, f"{prefix}_locf_age_h", "h", LE, item, f"Hours from the previous {name} to the earlier of the target measurement and the window start.")
        add(LABELS, f"{prefix}_hist_mean", unit, LE, item, f"Mean of all {name} values of the subject up to and including the previous value.")
        add(LABELS, f"{prefix}_hist_n", "count", LE, item, f"Number of {name} values of the subject up to and including the previous value.")
        add(LABELS, f"{prefix}_trend", f"{unit} per h", LE, item, f"Previous {name} minus the one before it, divided by the hours between them (at least 1).")
    for lab, (item, rule) in C["labs"].items():
        unit = LAB_UNITS[lab]
        add(LABELS, lab, unit, LE, item, f"Measurement nearest to the draw within ± {tol} min, from VALUENUM; censored results such as >150 count as their limit.")
        add(LABELS, f"{lab}_offset_min", "min", LE, item, f"Time of that measurement minus the draw; positive is after the draw. Use it to apply a stricter window.")
        add(LABELS, f"{lab}_abnormal", "1/0", "derived", "", "Below the sex-specific limit, 12 g/dL (female) or 13 g/dL (male)." if rule == "anaemia" else f"Value {rule} {unit}. Empty when the value is missing.")
        history(lab, lab, unit, item)
    history("creat", "creatinine", "mg/dL", CFG["creatinine"]["itemid"])
    for lab, item in C["blood_gas"].items():
        add(LABELS, f"{lab}_bg_locf", LAB_UNITS[lab], LE, item, f"Previous blood-gas analyser value of {lab}, before the same reference time as {lab}_locf. Kept apart because the analysers differ in bias.")
        add(LABELS, f"{lab}_bg_locf_age_h", "h", LE, item, f"Hours from that value to the reference time.")
    add(LABELS, "pre_start_s", "s", f"{WF}: PLETH", "", f"Primary window: start, relative to the draw, of the best contiguous {win} min wholly before the draw; ties go to the window nearest the draw, up to floating-point rounding.")
    add(LABELS, "pre_score", "fraction", f"{WF}: PLETH", "", f"Primary window: its fraction of good {ep} s epochs. It is usable at {min_score} or more.")
    add(LABELS, "record_overlap", "True/False", WF, "", "A PLETH record of this window overlaps another PLETH record of the subject, and the two may not be the same patient. Left out of the cohort.")
    add(LABELS, "split", "train/val/test", "derived", "", "The subject's split, from mimic_splits.csv.")

    add(SPLITS, "subject_id", "", "PATIENTS", "", "MIMIC-III SUBJECT_ID.")
    add(SPLITS, "split", "train/val/test", "derived", "", f"Salted SHA-256 hash of subject_id, {C['split']}, salt {C['split_salt']}, except that development subjects the hash puts in test move to train.")
    add(SPLITS, "dev_set", "True/False", "derived", "", "The subject was used to develop the methods. Never in test.")
    add(COHORT, "draw_id", "", LE, "", "Window key.")
    add(COHORT, "in_cohort_pre", "True/False", "derived", "", f"In the benchmark cohort for the primary window: pre_score at least {min_score}, not record_overlap, at least 20 kept beats, and at least one gap-free 10 s segment.")
    add(COHORT, "in_cohort_centered", "True/False", "derived", "", "The same for the centred window, without the pre_score rule.")

    add(COVARIATES, "draw_id", "", LE, "", "Window key.")
    both = lambda g: f"MV {ids(ITEMS['metavision'][g])}; CV {ids(ITEMS['carevue'][g])}"
    for g in AT_DRAW:
        add(COVARIATES, f"on_{g}", "1/0", "INPUTEVENTS_MV, INPUTEVENTS_CV", both(g), f"{g.replace('_', ' ')}: running at the draw, or stopped or charted in the preceding hour. MetaVision: not cancelled or rewritten; CareVue: a positive rate or amount. Never a treatment that starts after the draw.")
    for g in PRIOR_24H:
        add(COVARIATES, f"{g}_24h", "1/0", "INPUTEVENTS_MV, INPUTEVENTS_CV", both(g), f"{g.replace('_', ' ')}: running at the draw, or stopped or charted in the preceding 24 h, with the same rules.")
    pressors = lambda keep: "; ".join(f"{k} MV {ids(m)} CV {ids(c)}" for k, (m, c) in PRESSORS.items() if keep(k))
    add(COVARIATES, "n_pressors", "count", "INPUTEVENTS_MV, INPUTEVENTS_CV", pressors(lambda k: True), "Number of vasoactive agents running at the draw or in the preceding hour.")
    add(COVARIATES, "epi_on", "1/0", "INPUTEVENTS_MV, INPUTEVENTS_CV", pressors(lambda k: k == "epinephrine"), "Epinephrine running at the draw or in the preceding hour.")
    add(COVARIATES, "hosp_h", "log(1 + h)", "ADMISSIONS", "", "Log of 1 + hours since ADMITTIME, at least 0.")
    add(COVARIATES, "elective", "1/0", "ADMISSIONS", "", "ADMISSION_TYPE is ELECTIVE.")
    add(COVARIATES, "transfer", "1/0", "ADMISSIONS", "", "ADMISSION_LOCATION starts with TRANSFER.")
    for g in ("black", "hispanic", "asian", "other", "unknown"):
        add(COVARIATES, f"eth_{g}", "1/0", "ADMISSIONS", "", f"ETHNICITY in the group {g}. All groups 0 means white. A proxy for skin tone at best.")
    add(COVARIATES, "era_mv", "1/0", "INPUTEVENTS_MV", "", "The admission has MetaVision input events.")
    earlier = "admissions of the subject that ended before the draw, not counting the window's own"
    add(COVARIATES, "prior_admissions", "log(1 + count)", "ADMISSIONS", "", f"Log of 1 + the number of {earlier}.")
    add(COVARIATES, "days_since_discharge", "log(1 + days)", "ADMISSIONS", "", f"Log of 1 + days from the last discharge among the {earlier}. Empty without one.")
    for col, prefixes in PRIOR_DX.items():
        add(COVARIATES, col, "1/0", "DIAGNOSES_ICD, ADMISSIONS", f"ICD-9 prefixes {ids(prefixes)}", f"A code of this group in any of the {earlier}. Codes of the current stay are not read.")

    for w in WINDOWS:
        name = "the primary window" if w == "pre" else "the centred window"
        f = f"benchmark/features/{w}/mimic_abp/abp_map.csv"
        for c in key:
            add(f, *c)
        add(f, "abp_channel", "", f"{WF}: ABP or ART", "", "Name of the arterial channel used. Empty without an arterial waveform.")
        add(f, "abp_epochs_valid", "count", f"{WF}: ABP or ART", "", f"Valid {ep} s epochs: at least 90% of samples finite, mean within 20-200 mmHg, pulse pressure (p95 - p5) 10-150 mmHg.")
        add(f, "map_mmhg", "mmHg", f"{WF}: ABP or ART", "", f"Median epoch mean over the valid epochs of {name}. Empty with fewer than 6 valid epochs or no arterial waveform, which does not strictly mean no arterial line: the numerics may hold an arterial mean.")
        f = f"benchmark/features/{w}/mimic_num/numerics.csv"
        for c in key:
            add(f, *c)
        add(f, "has_numerics", "True/False", f"{WF}: numerics records", "", "A numerics record covers the window.")
        units = dict(HR="beats/min", RESP="breaths/min", SPO2="%", IRREGULARHRPERCENT="%", PVCRATEPERMINUTE="per min")
        for channel, col in WINDOW.items():
            if channel != "PULSE":
                add(f, col, units[channel], f"{WF}: numerics {channel}", "", f"Median monitor value over {name}." + (" The benchmark counts exactly 0 as missing." if channel == "RESP" else ""))
        for channel, col in LAST.items():
            add(f, col, "mmHg", f"{WF}: numerics {channel}", "", f"Last reading in the hour before the end of {name}.")

    add(INDEX, "draw_id", "", LE, "", "Window key.")
    add(INDEX, "subject_id", "", "PATIENTS", "", "MIMIC-III SUBJECT_ID.")
    add(INDEX, "file", "", WF, "", FILE_NOTE)
    add(INDEX, "coverage", "fraction", f"{WF}: PLETH", "", "Fraction of the window's samples with PLETH.")
    add(INDEX, "n_segments", "count", WF, "", "Number of source waveform segments read.")
    add(INDEX, "segments", "", WF, "", "Names of those segments, separated by |.")

    source = f"{WF}: PLETH"
    half_samples = CFG["s3_fetch"]["half_window_min"] * 60 * CFG["fs"]
    record = f"{WINDOW_DIR}/pXX/pXXXXXX/<draw_id>"
    add(f"{record}.dat", "PLETH", "NU", source, "", f"{2 * half_samples} digital samples at {CFG['fs']} Hz, in the units of the source records (normalised, sensor site not recorded). Physical value = (digital - baseline) / gain. Missing samples hold the WFDB invalid value and read as NaN. Sample {half_samples} is the draw time.")
    add(f"{record}.hea", "format, gain, baseline", "", source, "", "Format 80, gain 255, baseline -128, or format 16, gain 1023, baseline 0, as the source record stores PLETH. Physical values cast to float16 are the arrays the benchmark's features were computed from. float16 merges neighbouring format 16 values at 2 NU and above, so there a digital value can differ from the source's by one count.")
    add(f"{record}.hea", "t0_s", "s", source, "", "Header comment: time of the first sample relative to the draw.")
    add(EPOCHS, "<draw_id>", "True/False", source, "", f"Boolean array, one value per {ep} s epoch of the window: no missing samples, flat fraction at most {CFG['s4_sqi']['max_flat_frac']}, rate within {CFG['s4_sqi']['hr_range_bpm']} beats/min, beat-to-template correlation at least {CFG['s4_sqi']['min_template_corr']}.")
    return pd.DataFrame(rows, columns=["file", "column", "unit", "source", "itemid", "definition"]).drop_duplicates(["file", "column"])


def check_dictionary(entries, tables):
    """Every column of every table has a row, and every row a column: a table that changed stops the build."""
    problems = []
    for f, header in tables.items():
        have, documented = set(header), set(entries.column[entries.file == f])
        if have - documented:
            problems.append(f"{f}: no dictionary row for {sorted(have - documented)}")
        if documented - have:
            problems.append(f"{f}: dictionary rows for absent columns {sorted(documented - have)}")
    if problems:
        sys.exit("the data dictionary does not match the tables:\n  " + "\n  ".join(problems))


# ------------------------------------------------------------------------------------------------- the plan

def select_csv(path, drop=(), keep=None, rows=None, key="draw_id"):
    """Bytes of a CSV without some columns or rows, and its header. Values pass through as text, so no number is reformatted."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = [c for c in (keep or header) if c not in drop]
        missing = [c for c in names if c not in header]
        if missing:
            sys.exit(f"{path}: missing columns {missing}")
        at = [header.index(c) for c in names]
        k = header.index(key) if rows is not None else None
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(names)
        for r in reader:
            if rows is None or r[k] in rows:
                writer.writerow([r[i] for i in at])
    return out.getvalue().encode(), names


def cohort_table(data, labels):
    """The flags baselines.load_mimic's filter gives: the embedding enters only through whether a window could be embedded."""
    t = labels[["draw_id"]].copy()
    for w in WINDOWS:
        folder = data / "benchmark" / "features" / w / "mimic"
        beat = pd.read_csv(folder / "beat_features.csv", usecols=["draw_id", "n_beats"])
        with np.load(folder / "papagei.npz", allow_pickle=True) as z:
            embedded = set(z["ids"][np.isfinite(z["emb"][:, 0])].astype(str))
        ok = labels.draw_id.isin(set(beat.draw_id[beat.n_beats.notna()]) & embedded) & ~labels.record_overlap.astype(bool)
        if w == "pre":
            ok &= labels.pre_score >= CFG["s5_select"]["min_window_score"]
        t[f"in_cohort_{w}"] = ok
    return t


def epoch_bytes(path, ids):
    buffer = io.BytesIO()
    with np.load(path) as z:
        np.savez_compressed(buffer, **{i: z[i] for i in sorted(ids)})
    return buffer.getvalue()


def record_path(subject_id, draw_id):
    """The folder scheme of the source subset: pXX/pXXXXXX/."""
    s = f"p{int(subject_id):06d}"
    return f"{WINDOW_DIR}/{s[:3]}/{s}/{draw_id}"


def decode(digital, fmt, gain, baseline):
    """The stored float16 array from digital values."""
    x = (digital.astype(np.float64) - baseline) / gain
    x[digital == MISSING[fmt]] = np.nan
    return x.astype(np.float16)


def encode(pleth):
    """(format, gain, baseline, digital values) that give back this float16 array bit for bit, or None if neither source encoding does."""
    present = np.isfinite(pleth)
    x = pleth[present].astype(np.float64)
    for fmt, gain, baseline, top in ENCODINGS:
        if not x.size:
            break
        digital = np.full(pleth.size, MISSING[fmt], dtype=np.int64)
        digital[present] = np.minimum(np.rint(x * gain) + baseline, top).astype(np.int64)
        if digital[present].min() > MISSING[fmt] and decode(digital, fmt, gain, baseline).tobytes() == pleth.tobytes():
            return fmt, gain, baseline, digital
    return None


def render(name, pleth, fs, t0_s):
    """(encoding, .hea bytes, .dat bytes) of a window, byte-identical to what wfdb.wrsamp writes, or None if it has no exact encoding."""
    found = encode(pleth)
    if found is None:
        return None
    fmt, gain, baseline, digital = found
    checksum = int(digital.sum()) % 65536
    header = (f"{name} 1 {fs} {digital.size}\n{name}.dat {fmt} {gain}({baseline})/NU {ADC_RES[fmt]} 0 {digital[0]} {checksum} 0 PLETH\n# t0_s {t0_s}\n")
    dat = (digital + 128).astype(np.uint8).tobytes() if fmt == "80" else digital.astype("<i2").tobytes()
    return f"format {fmt}, gain {gain:g}, baseline {baseline}", header.encode(), dat


def load_window(path):
    with np.load(path) as z:
        return z["pleth"], int(z["fs"]), int(z["t0_s"])


def read_record(record):
    """A WFDB window read with wfdb: the stored float16 array, the sampling rate and t0_s."""
    import wfdb
    r = wfdb.rdrecord(str(record))
    return r.p_signal[:, 0].astype(np.float16), int(r.fs), next(int(c.split()[1]) for c in r.comments if c.startswith("t0_s"))


def scan_one(args):
    source, name = args
    pleth, fs, t0_s = load_window(source)
    done = render(name, pleth, fs, t0_s)
    return None if done is None or fs != CFG["fs"] else (done[0], len(done[1]), len(done[2]))


def write_one(args):
    """Writes one record, reads it back with wfdb and compares with the stored array, and returns its checksum lines. None if it differs."""
    source, record, root = args
    target = Path(root) / record
    pleth, fs, t0_s = load_window(source)
    done = render(target.name, pleth, fs, t0_s)
    if done is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    hea, dat = target.with_name(target.name + ".hea"), target.with_name(target.name + ".dat")
    hea.write_bytes(done[1])
    dat.write_bytes(done[2])
    back, back_fs, back_t0 = read_record(target)
    if back.tobytes() != pleth.tobytes() or back_fs != fs or back_t0 != t0_s:
        return None
    return [f"{hashlib.sha256(b).hexdigest()}  {record}{ext}\n" for b, ext in ((done[1], ".hea"), (done[2], ".dat"))]


def run(function, jobs, args):
    """Map over the windows, in this process or at most `jobs` others."""
    if jobs <= 1:
        yield from map(function, args)
        return
    with ProcessPoolExecutor(jobs) as pool:
        yield from pool.map(function, args, chunksize=64)


def scan_windows(window_items, jobs=1):
    """The encoding and sizes of every window, so that sizes are exact before anything is written. Stops on a window with no exact encoding, naming none."""
    failed = 0
    for w, found in zip(window_items, run(scan_one, jobs, [(w.source, Path(w.record).name) for w in window_items])):
        if found is None:
            failed += 1
        else:
            w.encoding, w.hea_size, w.dat_size = found
    if failed:
        sys.exit(f"{failed} windows have no exact WFDB encoding in the source's gain and baseline, or a different sampling rate: they cannot be written losslessly")


def restore_windows(package, jobs=1):
    """Writes the NumPy files the feature code reads beside the records, from the records. Returns how many."""
    package = Path(package)
    records = (package / "RECORDS").read_text().split()
    (package / WINDOW_DIR).mkdir(parents=True, exist_ok=True)
    for _ in run(restore_one, jobs, [(package, r) for r in records]):
        pass
    return len(records)


def restore_one(args):
    package, record = args
    pleth, fs, t0_s = read_record(package / record)
    np.savez_compressed(package / WINDOW_DIR / f"{Path(record).name}.npz", pleth=pleth, fs=fs, t0_s=t0_s)


def non_test_statistics(path):
    """Coverage and quartiles per lab, age and sex, from training and validation rows. A test row is skipped before any of its values is parsed."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        wanted = ["creat_mgdl", "egfr", *C["labs"], "age"]
        at, split, female = {c: header.index(c) for c in wanted}, header.index("split"), header.index("female")
        values, n, women = {c: [] for c in wanted}, 0, 0
        for row in reader:
            if row[split] == "test":
                continue
            n += 1
            women += row[female] == "True"
            for c, i in at.items():
                if row[i] != "":
                    values[c].append(float(row[i]))
    rows = []
    for c in wanted:
        q1, median, q3 = np.quantile(values[c], [.25, .5, .75]) if values[c] else (np.nan,) * 3
        rows.append(dict(column=c, unit=LAB_UNITS.get(c, dict(creat_mgdl="mg/dL", egfr="mL/min/1.73 m2", age="years").get(c, "")), windows=len(values[c]), coverage=len(values[c]) / n if n else np.nan, median=median, q1=q1, q3=q3))
    return dict(windows=n, female=women / n if n else np.nan, rows=rows)


def statistics_table(stats):
    """The Markdown table of the project page."""
    L = [f"Training and validation windows: {stats['windows']}. Female: {100 * stats['female']:.1f}% of windows.", "", "| Column | Unit | Windows with a value | Coverage | Median | Quartiles |", "|---|---|---|---|---|---|"]
    L += [f"| {r['column']} | {r['unit']} | {r['windows']} | {100 * r['coverage']:.0f}% | {r['median']:.4g} | {r['q1']:.4g}-{r['q3']:.4g} |" for r in stats["rows"]]
    return "\n".join(L)


def readme(info):
    """The README of the package: what it is, what is where, and how to read it."""
    def counts(table):
        return ", ".join(f"{k} {v}" for k, v in sorted(table.items(), key=lambda kv: ("train", "val", "test").index(kv[0])))
    return f"""# {TITLE}

Version {VERSION}. A dataset derived from the MIMIC-III Clinical Database v1.4 and the MIMIC-III Waveform Database Matched Subset v1.0. It pairs 20 minute photoplethysmography (PLETH) windows at 125 Hz with blood laboratory values, their time offsets, previous values, covariates and fixed subject splits, for benchmarking how much the pulse waveform adds to what is already known at the bedside.

It holds {info['windows']} windows from {info['subjects']} adult subjects. Windows per split: {counts(info['windows_per_split'])}. Subjects per split: {counts(info['subjects_per_split'])}. Dates are the shifted dates of MIMIC-III. It holds no model embeddings and reports no results.

## Files

| Path | Content |
|---|---|
| `s3_windows/windows/pXX/pXXXXXX/<draw_id>.hea`, `.dat` | One WFDB record per window. Sample 75000 is the draw time |
| `RECORDS` | The path of every WFDB record |
| `s3_windows/s3_index.csv` | Per window: coverage and source waveform segments |
| `s4_sqi/epoch_good.npz` | Per window: one boolean per 10 s epoch |
| `s5_groupB/groupB_draws.csv` | The window table: subject, admission, shifted draw time, signal quality |
| `benchmark/mimic_labels.csv` | One row per window: laboratory values, offsets, previous values, history, flags, analysis windows, split |
| `benchmark/mimic_splits.csv` | Subject, split, development set |
| `benchmark/cohort.csv` | Whether each window is in the benchmark cohort |
| `benchmark/features/mimic_cov/covariates.csv` | Treatments and context |
| `benchmark/features/{{pre,centered}}/mimic_abp/abp_map.csv` | Mean arterial pressure over each analysis window |
| `benchmark/features/{{pre,centered}}/mimic_num/numerics.csv` | Monitor values over each analysis window |
| `benchmark/features/{{pre,centered}}/*/manifest.json` | A completeness marker the code reads |
| `data_dictionary.csv` | Every column: file, name, unit, source table, ITEMID, definition |
| `release_manifest.json` | Version and structural counts |
| `NOTICE.txt` | Sources and their licences |
| `SHA256SUMS` | The checksums of the files above |

## Reading a window

```python
import numpy as np, wfdb
r = wfdb.rdrecord("s3_windows/windows/pXX/pXXXXXX/<draw_id>")
pleth = r.p_signal[:, 0].astype(np.float16)     # the array the benchmark's features were computed from; NaN where absent
```

Each record keeps the encoding of its source: format 80, gain 255, baseline -128, or format 16, gain 1023, baseline 0. Uncast, the physical values differ from these arrays by floating-point rounding. The cast merges neighbouring format 16 values at 2 NU and above, so there a digital value can differ from the source record's by one count.

## Reading the tables

Column definitions are in `data_dictionary.csv`. In `age`, ages of 90 or more are one value, 91.4. Columns ending in `_age_h` hold the hours since a previous measurement, not a patient's age.

## Using the benchmark code

The feature code reads NumPy files. With the package linked as the `data/` folder of a checkout, `python -m ppglabbench.release --restore-windows data` writes them beside the records, from the records.

Sources and licences are in `NOTICE.txt`.
"""


def plan_package(with_windows=True, jobs=1):
    """Everything the package holds, read from data/, with nothing written. Returns the items and what the summary reports."""
    data = DATA
    labels_csv, splits_csv = data / LABELS, data / SPLITS
    for f in (labels_csv, splits_csv):
        if not f.is_file():
            sys.exit(f"{f} is missing: build the benchmark first (make labels, see README.md)")
    sha = hashlib.sha256(splits_csv.read_bytes()).hexdigest()
    if sha != C["splits_sha256"]:
        sys.exit(f"{SPLITS} differs from splits_sha256 in config.yaml: this is not the benchmark's split file")
    labels = pd.read_csv(labels_csv, usecols=["draw_id", "file", "subject_id", "split", "pre_score", "record_overlap"], dtype=dict(draw_id=str, file=str))
    if not (labels.file == labels.draw_id + ".npz").all():
        sys.exit("a window file is not named after its draw_id")
    splits = pd.read_csv(splits_csv)
    if not labels.draw_id.is_unique or set(splits.subject_id) != set(labels.subject_id) or not labels.subject_id.map(splits.set_index("subject_id").split).eq(labels.split).all():
        sys.exit("the labels and the split file disagree on subjects or splits")
    all_ids = set(labels.draw_id)
    pre_ids = set(labels.draw_id[labels.pre_score >= CFG["s5_select"]["min_window_score"]])

    items, tables, warnings = [Item(SPLITS, source=splits_csv)], {SPLITS: list(splits.columns)}, []
    def table(path, data_bytes, header):
        items.append(Item(path, data=data_bytes))
        tables[path] = header
    table(LABELS, *select_csv(labels_csv, drop=DROP[LABELS]))
    cohort = cohort_table(data, labels)
    table(COHORT, cohort.to_csv(index=False, lineterminator="\n").encode(), list(cohort.columns))
    table(COVARIATES, *select_csv(data / COVARIATES, rows=all_ids))
    for w, expected in (("pre", pre_ids), ("centered", all_ids)):
        for sub, name, tag, stage in (("mimic_abp", "abp_map.csv", "abp", "abp_map"), ("mimic_num", "numerics.csv", "num", "numerics")):
            path = f"benchmark/features/{w}/{sub}/{name}"
            source = data / path
            done = pd.read_csv(source, usecols=["draw_id", "error"]) if source.is_file() else pd.DataFrame(dict(draw_id=[], error=[]))
            ok = set(done.draw_id[done.error.isna()])
            if not expected <= ok:
                warnings.append(f"{path}: {len(expected - ok)} of {len(expected)} windows missing or failed; left out of the package")
                continue
            table(path, *select_csv(source, drop=DROP[tag], rows=expected))
            # baselines.load_mimic reads `complete`; the counts of the project's own manifest hold no reader's need
            items.append(Item(f"benchmark/features/{w}/{sub}/manifest.json", data=json.dumps(dict(stage=stage, window=w, counts=dict(complete=True))).encode()))
    table(DRAWS, *select_csv(labels_csv, keep=DRAW_COLUMNS))
    table(INDEX, *select_csv(data / INDEX, drop=DROP[INDEX], rows=all_ids))
    items.append(Item(EPOCHS, data=epoch_bytes(data / EPOCHS, all_ids)))
    entries = dictionary()
    check_dictionary(entries, tables)
    items += [Item("data_dictionary.csv", data=entries.to_csv(index=False, lineterminator="\n").encode()), Item("NOTICE.txt", data=NOTICE.encode())]

    window_items = []
    if with_windows:
        missing = sum(not (data / WINDOW_DIR / f).is_file() for f in labels.file)
        if missing:
            sys.exit(f"{missing} window files are missing under {WINDOW_DIR}/")
        window_items = [Window(record_path(r.subject_id, r.draw_id), data / WINDOW_DIR / r.file) for r in labels.sort_values("draw_id").itertuples()]
        scan_windows(window_items, jobs)
        items.append(Item("RECORDS", data="".join(f"{w.record}\n" for w in window_items).encode()))
    bad = [i.path for i in items if any(word in i.path.lower() for word in FORBIDDEN_IN_PATH)]
    if bad:
        sys.exit(f"these must not be in the package: {bad}")
    structure = dict(windows_per_split=labels.split.value_counts().to_dict(), subjects_per_split=splits.split.value_counts().to_dict())
    info = dict(title=TITLE, version=VERSION, splits_sha256=sha, windows=len(labels), subjects=int(splits.subject_id.nunique()),
                windows_in_cohort_pre=int(cohort.in_cohort_pre.sum()), windows_in_cohort_centered=int(cohort.in_cohort_centered.sum()), **structure,
                windows_included=bool(with_windows), window_encodings=pd.Series([w.encoding for w in window_items], dtype=str).value_counts().to_dict(),
                left_out_incomplete=warnings, not_included=NOT_INCLUDED)
    items += [Item("release_manifest.json", data=json.dumps(info, indent=1).encode()), Item("README.md", data=readme(info).encode())]
    return items, window_items, info


def sums_size(paths):
    """Bytes of SHA256SUMS: a 64-character hash, two spaces, the path and a newline per file."""
    return sum(64 + 2 + len(p) + 1 for p in paths)


def human(n):
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"


def summary(items, window_items, info, dry):
    """File list and sizes. Windows are one line: their names hold subject IDs. Counts per split are structure, and no outcome appears."""
    paths = [i.path for i in items] + [p for w in window_items for p in w.paths]
    total = sum(i.size for i in items) + sum(i.size for i in window_items)
    sums = sums_size(paths)
    width = max(len(i.path) for i in items) + 16
    head = "Dry run, nothing written." if dry else "Package written."
    L = [f"{TITLE} {VERSION}. {head}", "", f"{'file':<{width}}{'bytes':>15}  size"]
    L += [f"{i.path:<{width}}{i.size:>15}  {human(i.size)}" for i in sorted(items, key=lambda x: x.path)]
    L.append(f"{'SHA256SUMS':<{width}}{sums:>15}  {human(sums)}")
    if window_items:
        size = sum(i.size for i in window_items)
        L.append(f"{WINDOW_DIR + '/pXX/pXXXXXX/<draw_id>.hea,.dat':<{width}}{size:>15}  {human(size)} in {2 * len(window_items)} files")
    L += [f"{'total':<{width}}{total + sums:>15}  {human(total + sums)} in {len(paths) + 1} files", "",
          f"Windows {info['windows']}, subjects {info['subjects']}. Structure only: windows per split {info['windows_per_split']}, subjects per split {info['subjects_per_split']}.",
          f"In the cohort: {info['windows_in_cohort_pre']} windows for the primary window, {info['windows_in_cohort_centered']} for the centred window.",
          f"Windows by encoding: {info['window_encodings']}.",
          f"Split file SHA-256 {info['splits_sha256']}, the value in config.yaml.", "", "Not included:"]
    L += [f"  {k}: {v}" for k, v in info["not_included"].items()]
    if info["left_out_incomplete"]:
        L += ["", "WARNING, incomplete tables:"] + [f"  {w}" for w in info["left_out_incomplete"]]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------- the build

def check_out(out):
    """Credentialed data stays under data/ or outside the repository, where git cannot pick it up."""
    out, root, data = Path(out).resolve(), ROOT.resolve(), DATA.resolve()
    if out == data or out == root or (out.is_relative_to(root) and not out.is_relative_to(data)):
        sys.exit(f"--out {out}: this is credentialed data, so use a folder inside data/, or one outside the repository")
    return out


def build(items, window_items, out, force=False, jobs=1):
    """Written to a temporary folder and moved into place, so a package on disk is always a whole one."""
    target, tmp = out / PACKAGE, out / f"{PACKAGE}.tmp"
    if target.exists() and not force:
        sys.exit(f"{target} exists: pass --force to replace it")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    lines = []
    for i in items:
        f = tmp / i.path
        f.parent.mkdir(parents=True, exist_ok=True)
        data = i.data if i.source is None else i.source.read_bytes()
        f.write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {i.path}\n")
    failed = 0
    for n, done in enumerate(run(write_one, jobs, [(w.source, w.record, str(tmp)) for w in window_items])):
        if done is None:
            failed += 1
        else:
            lines += done
        if (n + 1) % 5000 == 0:
            print(f"  {n + 1} windows", flush=True)
    if failed:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(f"{failed} windows did not read back exactly with wfdb: nothing was written")
    (tmp / "SHA256SUMS").write_text("".join(sorted(lines, key=lambda x: x[66:])))
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    return target


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="staging folder; the package goes to <out>/package")
    ap.add_argument("--dry-run", action="store_true", help="print the file list and sizes, and write nothing")
    ap.add_argument("--force", action="store_true", help="replace an existing package")
    ap.add_argument("--no-windows", action="store_true", help="tables only; the windows regenerate from the open waveforms with the pipeline's S3")
    ap.add_argument("--allow-incomplete", action="store_true", help="build even if a feature table is incomplete, leaving it out")
    ap.add_argument("--jobs", type=int, default=1, help="processes for the windows (default 1)")
    ap.add_argument("--restore-windows", metavar="PACKAGE", help="write the NumPy files the feature code reads, from the records of this package")
    ap.add_argument("--statistics", action="store_true", help="print summary statistics of the training and validation rows; no test row is parsed")
    args = ap.parse_args(argv)
    if args.restore_windows:
        print(f"{restore_windows(args.restore_windows, args.jobs)} NumPy files written beside the records.")
        return
    if args.statistics:
        print(statistics_table(non_test_statistics(DATA / LABELS)))
        return
    if not args.out:
        ap.error("--out is required to build a package")
    out = check_out(args.out)
    items, window_items, info = plan_package(with_windows=not args.no_windows, jobs=args.jobs)
    if info["left_out_incomplete"] and not (args.dry_run or args.allow_incomplete):
        sys.exit("feature tables are incomplete, so the package would lack them (--allow-incomplete leaves them out):\n  " + "\n  ".join(info["left_out_incomplete"]))
    if not args.dry_run:
        print(f"Written to {build(items, window_items, out, args.force, args.jobs)}")
    print(summary(items, window_items, info, args.dry_run))


if __name__ == "__main__":
    main()
