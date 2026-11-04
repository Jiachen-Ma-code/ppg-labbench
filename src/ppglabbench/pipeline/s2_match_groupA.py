"""S2: match creatinine draws to PLETH recordings -> group A.

A draw joins group A if a PLETH-containing record (S1) overlaps [t - h, t + h].
Group A subjects = subjects with >=1 such draw.

Usage:
    python -m ppglabbench.pipeline.s2_match_groupA
Output:
    data/s2_groupA/groupA_draws.csv     one row per draw: draw_id, subject_id, hadm_id, charttime, creat_mgdl, records
    data/s2_groupA/groupA_subjects.txt
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from ppglabbench.common import CFG, S1, S2, load_creatinine, write_manifest


def select_evenly(draws, k):
    """Boolean mask: per subject, at most k draws evenly spaced over the time-sorted draws."""
    keep = pd.Series(False, index=draws.index)
    for _, g in draws.sort_values("charttime").groupby("subject_id"):
        pick = np.unique(np.linspace(0, len(g) - 1, min(k, len(g))).round().astype(int))
        keep[g.index[pick]] = True
    return keep


def main():
    S2.mkdir(parents=True, exist_ok=True)
    s1 = pd.read_json(S1 / "manifest.json", typ="series")
    if not s1.counts["scan_complete"]:
        print(f"WARNING: S1 scanned only {s1.counts['records_scanned']}/{s1.counts['records_total']} records; group A is partial")

    cat = pd.read_csv(S1 / "waveform_catalog.csv", parse_dates=["start", "end"])
    pleth = cat[cat.error.isna() & (cat.has_pleth == True)]

    cr = load_creatinine()
    n_raw = len(cr)
    cr = cr.drop_duplicates(["subject_id", "charttime"])  # same draw recorded twice

    h = timedelta(minutes=CFG["s2_match"]["search_half_window_min"])
    m = cr.merge(pleth[["subject_id", "record", "start", "end"]], on="subject_id")
    m = m[(m.charttime + h > m.start) & (m.charttime - h < m.end)]
    draws = (m.groupby(["subject_id", "charttime"], as_index=False)
              .agg(hadm_id=("hadm_id", "first"), creat_mgdl=("creat_mgdl", "first"),
                   records=("record", lambda s: "|".join(sorted(set(s))))))
    draws.insert(0, "draw_id", draws.subject_id.astype(str) + "_" + draws.charttime.dt.strftime("%Y%m%d%H%M"))
    draws["s3_selected"] = select_evenly(draws, CFG["s2_match"]["max_draws_per_subject"])
    draws.to_csv(S2 / "groupA_draws.csv", index=False)
    (S2 / "groupA_subjects.txt").write_text("\n".join(map(str, sorted(draws.subject_id.unique()))) + "\n")

    write_manifest(S2, "s2_match", CFG["s2_match"], dict(
        s1_scan_complete=bool(s1.counts["scan_complete"]),
        creat_rows=n_raw, creat_draws_dedup=len(cr), creat_subjects=int(cr.subject_id.nunique()),
        pleth_subjects=int(pleth.subject_id.nunique()),
        groupA_subjects=int(draws.subject_id.nunique()), groupA_draws=len(draws),
        s3_selected_draws=int(draws.s3_selected.sum()),
        groupA_creat_median=float(draws.creat_mgdl.median()),
        groupA_draws_creat_ge_2=int((draws.creat_mgdl >= 2).sum())))


if __name__ == "__main__":
    main()
