"""S1: catalogue every mimic3wdb-matched record — when it was recorded and whether it has PLETH.

Reads only two small headers per record (master + _layout), no signal data.
Resumable: records already in s1_catalog/waveform_catalog.csv are skipped.

Usage:
    python -m ppglabbench.pipeline.s1_scan_waveforms --limit 300    # test batch
    python -m ppglabbench.pipeline.s1_scan_waveforms                # all ~22k records (~40 min)
Output:
    data/s1_catalog/waveform_catalog.csv   one row per record
    data/s1_catalog/pleth_subjects.txt     subject_ids with >=1 PLETH record
"""
import argparse
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import pandas as pd
import wfdb

from ppglabbench.common import CFG, DB, S1, subject_dir, write_manifest

RECORDS = S1 / "RECORDS-waveforms"
CATALOG = S1 / "waveform_catalog.csv"


def scan_record(path):
    """path like 'p00/p00NNNN/p00NNNN-YYYY-MM-DD-hh-mm' -> one catalogue row."""
    sid = int(path.split("/")[1][1:])
    rec = path.split("/")[-1]
    pn_dir = subject_dir(sid)
    row = dict(subject_id=sid, record=rec)
    try:
        h = wfdb.rdheader(rec, pn_dir=pn_dir)
        layout = next(s for s in h.seg_name if s.endswith("_layout"))
        sigs = wfdb.rdheader(layout, pn_dir=pn_dir).sig_name
        row.update(start=h.base_datetime,
                   end=h.base_datetime + timedelta(seconds=h.sig_len / h.fs),
                   hours=round(h.sig_len / h.fs / 3600, 2),
                   has_pleth="PLETH" in sigs, signals="|".join(sigs), error="")
    except Exception as e:
        row.update(error=str(e)[:200])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="scan at most this many new records")
    args = ap.parse_args()
    S1.mkdir(parents=True, exist_ok=True)
    if not RECORDS.exists():
        urllib.request.urlretrieve(f"https://physionet.org/files/{DB}/RECORDS-waveforms", RECORDS)

    records = RECORDS.read_text().split()
    seen = set(pd.read_csv(CATALOG).record) if CATALOG.exists() else set()
    todo = [r for r in records if r.split("/")[-1] not in seen][: args.limit]
    print(f"{len(records)} records total, {len(seen)} already scanned, scanning {len(todo)}")

    t0, buf = time.time(), []
    with ThreadPoolExecutor(CFG["s1_scan"]["workers"]) as ex:
        futs = [ex.submit(scan_record, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            buf.append(fut.result())
            if len(buf) >= 200 or i == len(todo):  # flush often so an interruption loses little
                pd.DataFrame(buf).to_csv(CATALOG, mode="a", header=not CATALOG.exists(), index=False)
                buf = []
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s")

    cat = pd.read_csv(CATALOG)
    ok = cat[cat.error.isna()]
    pleth = ok[ok.has_pleth == True]
    (S1 / "pleth_subjects.txt").write_text("\n".join(map(str, sorted(pleth.subject_id.unique()))) + "\n")
    write_manifest(S1, "s1_scan", CFG["s1_scan"], dict(
        records_total=len(records), records_scanned=len(cat), scan_complete=len(cat) >= len(records),
        records_error=int(cat.error.notna().sum()), records_with_pleth=len(pleth),
        subjects_scanned=int(cat.subject_id.nunique()), subjects_with_pleth=int(pleth.subject_id.nunique()),
        pleth_hours=round(float(pleth.hours.sum()), 1)))


if __name__ == "__main__":
    main()
