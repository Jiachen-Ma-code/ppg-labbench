"""Optional link table: the blood labs around the 30 s PPG segments of MIMIC-III-Ext-PPG, by subject and time.

For each segment and lab, the measurement nearest to the segment centre within ± lab_window_min (config, 60 min), the rule of
ppglabbench.labels, with the same labs and the same handling of censored results; creatinine is added from the same tables. Both datasets use
the shifted dates of MIMIC-III, and waveform and clinical clocks agree to about 10 min. `pleth_sqi_all_good` is true where all three 10 s PLETH
quality codes of the segment are 1, the dataset's good-quality code. Rows are segment x lab, so a stricter window or quality rule is a filter.

Test subjects: this table holds outcomes, so subjects of this benchmark's test split are left out until that split is opened: those of the
split file, and any other subject the split rule would put in test. `--include-test` keeps them, and refuses until `benchmark.test_opened` is
true in config.yaml. The dataset's own `strat_fold` is kept as a column.

The input needs PhysioNet credentialed access to MIMIC-III-Ext-PPG. The output is credentialed data: it stays under data/ or goes through
PhysioNet. It is not part of the release package.

Usage:
    python -m ppglabbench.link_extppg --out data/release_staging [--metadata <metadata.csv>] [--counts-only] [--include-test]
Output: <out>/ext_ppg_link/{ext_ppg_lab_links.csv.gz, counts.json, data_dictionary.csv}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ppglabbench.common import CFG, DATA, S1, extract_rows, git_version, load_creatinine
from ppglabbench.labels import C, CENSORED, LAB_CACHE, split_of
from ppglabbench.release import LAB_UNITS, check_out

METADATA = DATA / "external" / "mimic_iii_ext_ppg" / "metadata.csv"
SPLITS = DATA / "benchmark" / "mimic_splits.csv"
FOLDER = "ext_ppg_link"
SEGMENT_S = 30
WINDOWS_MIN = (10, 60)
GOOD = "[1,1,1]"
COLUMNS = ["signal_file_name", "record_id", "event_id", "segment_id", "subject_id", "start_segment", "vector_10s_pleth_sqi", "strat_fold"]
OUTPUT_COLUMNS = ["signal_file_name", "record_id", "event_id", "segment_id", "subject_id", "strat_fold", "pleth_sqi_all_good", "lab", "value", "offset_min"]
UNITS = dict(LAB_UNITS, creatinine="mg/dL")


def check_include_test(include):
    if include and C.get("test_opened") is not True:
        sys.exit("--include-test refused: benchmark.test_opened is not true in config.yaml, and this table holds lab values of test subjects")


def test_subjects(subjects):
    """Those of `subjects` in the test split: from the split file for its subjects, by the split rule for any other."""
    sp = pd.read_csv(SPLITS)
    known = dict(zip(sp.subject_id, sp.split))
    dev = set(sp.subject_id[sp.dev_set])
    return {s for s in subjects if known.get(s, split_of(s, dev)) == "test"}


def read_segments(path, include_test=False, limit=None, chunksize=500_000):
    """The columns needed, in chunks, with test subjects gone chunk by chunk. Returns the segments and how many subjects were left out."""
    keep, verdict = [], {}
    n = 0
    for ch in pd.read_csv(path, usecols=COLUMNS, chunksize=chunksize, dtype=dict(record_id=str, signal_file_name=str)):
        n += len(ch)
        new = [int(s) for s in ch.subject_id.unique() if s not in verdict]
        blocked = test_subjects(new) if not include_test else set()
        verdict.update({s: s in blocked for s in new})
        ch = ch[~ch.subject_id.map(verdict).astype(bool)]
        ch = ch.assign(start_segment=pd.to_datetime(ch.start_segment, format="%Y-%m-%d %H:%M:%S"),
                       pleth_sqi_all_good=ch.vector_10s_pleth_sqi.str.replace(" ", "", regex=False) == GOOD).drop(columns="vector_10s_pleth_sqi")
        keep.append(ch)
        if limit and n >= limit:
            break
    d = pd.concat(keep, ignore_index=True)
    d["centre"] = d.start_segment + pd.Timedelta(seconds=SEGMENT_S / 2)
    return d, sum(verdict.values()), n


def load_labs(subjects):
    """Lab name -> (subject_id, charttime, valuenum), the tables and censoring rule of ppglabbench.labels, on these subjects only."""
    items = {name: v[0] for name, v in C["labs"].items()}
    labs = extract_rows("LABEVENTS", LAB_CACHE,
                        lambda c: c.itemid.isin(list(items.values()) + list(C["blood_gas"].values())) &
                        ((c.valuenum > 0) | c.value.astype(str).str.contains(CENSORED, regex=True)),
                        {"subject_id", "itemid", "charttime", "value", "valuenum"}, dates=("charttime",))
    labs = labs[labs.subject_id.isin(subjects)]
    limit = labs.value.astype(str).str.extract(CENSORED).bfill(axis=1).iloc[:, 0].astype(float)
    labs = labs.assign(valuenum=labs.valuenum.where(labs.valuenum > 0, limit))
    labs = labs[labs.valuenum > 0]
    out = {name: labs[labs.itemid == item][["subject_id", "charttime", "valuenum"]] for name, item in items.items()}
    cr = load_creatinine()
    out["creatinine"] = cr[cr.subject_id.isin(subjects)][["subject_id", "charttime", "creat_mgdl"]].rename(columns={"creat_mgdl": "valuenum"})
    return out


def link_lab(segments, lab, tol_min):
    """For each segment with such a measurement within the tolerance: its row, the nearest value, the offset in minutes (positive: lab after the segment)."""
    left = segments[["subject_id", "centre"]].assign(row=np.arange(len(segments))).sort_values("centre")
    right = lab.sort_values("charttime")
    m = pd.merge_asof(left, right, left_on="centre", right_on="charttime", by="subject_id", direction="nearest", tolerance=pd.Timedelta(minutes=tol_min))
    m = m[m.charttime.notna()]
    return pd.DataFrame(dict(row=m.row.to_numpy(np.int32), value=m.valuenum.to_numpy(np.float32),
                             offset_min=((m.charttime - m.centre).dt.total_seconds() / 60).to_numpy(np.float32), lab_time=m.charttime.to_numpy()))


def clock_check(segments):
    """Share of segments that start inside a waveform record of the same subject in the S1 catalog: both must be on the same shifted clock."""
    catalog = S1 / "waveform_catalog.csv"
    if not catalog.exists():
        return None
    c = pd.read_csv(catalog, usecols=["subject_id", "start", "end", "error"], parse_dates=["start", "end"])
    c = c[c.error.isna() & c.subject_id.isin(segments.subject_id)].sort_values("start")
    m = pd.merge_asof(segments[["subject_id", "start_segment"]].sort_values("start_segment"), c[["subject_id", "start", "end"]],
                      left_on="start_segment", right_on="start", by="subject_id", direction="backward")
    return float((m.start_segment <= m.end).mean())


def counts_of(segments, links, in_benchmark):
    """Structure of the table at each window and quality rule. Counts only: no value, ID or date."""
    good = segments.pleth_sqi_all_good.to_numpy()
    subject = segments.subject_id.to_numpy()
    out = {}
    for tol in WINDOWS_MIN:
        for rule, mask in (("any_sqi", np.ones(len(segments), bool)), ("all_good_pleth_sqi", good)):
            rows, per_lab, seg = 0, {}, []
            for name, t in links.items():
                x = t[(t.offset_min.abs() <= tol) & mask[t.row.to_numpy()]]
                rows += len(x)
                seg.append(x.row.to_numpy())
                per_lab[name] = dict(links=len(x), measurements=int(pd.DataFrame(dict(s=subject[x.row.to_numpy()], t=x.lab_time.to_numpy())).drop_duplicates().shape[0]))
            seg = np.unique(np.concatenate(seg)) if seg else np.empty(0, int)
            subjects = np.unique(subject[seg])
            out[f"within_{tol}_min_{rule}"] = dict(link_rows=int(rows), segments=int(len(seg)), subjects=int(len(subjects)),
                                                  subjects_in_this_benchmark=int(np.isin(subjects, list(in_benchmark)).sum()), per_lab=per_lab)
    return out


def write_links(segments, links, path):
    """Segment x lab rows, one lab at a time, gzip level 1: the table is large and the column values repeat."""
    import gzip
    base = segments[["signal_file_name", "record_id", "event_id", "segment_id", "subject_id", "strat_fold", "pleth_sqi_all_good"]]
    with gzip.open(path, "wt", compresslevel=1, newline="") as f:
        f.write(",".join(OUTPUT_COLUMNS) + "\n")
        for name, t in links.items():
            x = base.iloc[t.row.to_numpy()].assign(lab=name, value=t.value.to_numpy(), offset_min=t.offset_min.round(2).to_numpy())
            x[OUTPUT_COLUMNS].to_csv(f, index=False, header=False, float_format="%.6g", lineterminator="\n")


def dictionary():
    return pd.DataFrame([
        ("signal_file_name", "", "MIMIC-III-Ext-PPG metadata.csv", "Segment name: record_id, event_id and segment_id."),
        ("record_id", "", "MIMIC-III-Ext-PPG metadata.csv", "Source waveform record."),
        ("event_id", "", "MIMIC-III-Ext-PPG metadata.csv", "Index of the rhythm chart event within the record."),
        ("segment_id", "", "MIMIC-III-Ext-PPG metadata.csv", "Index of the 30 s segment, counted backwards from the event time."),
        ("subject_id", "", "MIMIC-III-Ext-PPG metadata.csv", "MIMIC-III SUBJECT_ID."),
        ("strat_fold", "0-9", "MIMIC-III-Ext-PPG metadata.csv", "The dataset's own stratified fold, by patient."),
        ("pleth_sqi_all_good", "True/False", "MIMIC-III-Ext-PPG metadata.csv", "All three codes of vector_10s_pleth_sqi are 1."),
        ("lab", "", "LABEVENTS", "Lab name: " + ", ".join(f"{k} (ITEMID {v[0]})" for k, v in C["labs"].items()) + f", creatinine (ITEMID {CFG['creatinine']['itemid']})."),
        ("value", "; ".join(f"{k} {v}" for k, v in UNITS.items()), "LABEVENTS", f"Measurement nearest to the segment centre within ± {C['lab_window_min']} min. Censored results count as their limit."),
        ("offset_min", "min", "LABEVENTS, MIMIC-III-Ext-PPG metadata.csv", "Time of the measurement minus the segment centre; positive is after the segment."),
    ], columns=["column", "unit", "source", "definition"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="staging folder; the table goes to <out>/ext_ppg_link")
    ap.add_argument("--metadata", default=METADATA, type=Path, help="metadata.csv of MIMIC-III-Ext-PPG")
    ap.add_argument("--counts-only", action="store_true", help="write the counts and no table")
    ap.add_argument("--include-test", action="store_true", help="keep test subjects: refused until benchmark.test_opened is true")
    ap.add_argument("--limit-rows", type=int, help="read about this many metadata rows, for a test")
    args = ap.parse_args(argv)
    check_include_test(args.include_test)
    out = check_out(args.out) / FOLDER
    if not Path(args.metadata).is_file():
        sys.exit(f"{args.metadata} is missing: it needs PhysioNet credentialed access to MIMIC-III-Ext-PPG")
    segments, excluded, read = read_segments(args.metadata, args.include_test, args.limit_rows)
    print(f"read {read} segment rows; kept {len(segments)}; {excluded} test subjects left out", flush=True)
    labs = load_labs(set(segments.subject_id.unique()))
    links = {}
    for name, lab in labs.items():
        links[name] = link_lab(segments, lab, C["lab_window_min"])
    in_benchmark = set(pd.read_csv(SPLITS, usecols=["subject_id"]).subject_id)
    counts = dict(code_version=git_version(), test_subjects_included=bool(args.include_test), test_subjects_left_out=excluded, complete_input=args.limit_rows is None,
                  segment_rows_read=read, segments_kept=len(segments), segments_all_good_pleth_sqi=int(segments.pleth_sqi_all_good.sum()), subjects_kept=int(segments.subject_id.nunique()),
                  segments_inside_a_waveform_record=clock_check(segments), lab_window_min=C["lab_window_min"], windows=counts_of(segments, links, in_benchmark))
    out.mkdir(parents=True, exist_ok=True)
    (out / "counts.json").write_text(json.dumps(counts, indent=1))
    dictionary().to_csv(out / "data_dictionary.csv", index=False)
    if not args.counts_only:
        write_links(segments, links, out / "ext_ppg_lab_links.csv.gz")
    print(json.dumps({k: v for k, v in counts.items() if k != "windows"}, indent=1))
    for key, c in counts["windows"].items():
        print(f"{key}: {c['link_rows']} link rows, {c['segments']} segments, {c['subjects']} subjects, {c['subjects_in_this_benchmark']} of them in this benchmark")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
