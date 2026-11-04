"""S5: keep draws whose best 5 min window passes the SQI threshold -> group B, and write the flow report.

Usage:
    python -m ppglabbench.pipeline.s5_select_groupB                          # threshold from config.yaml
    python -m ppglabbench.pipeline.s5_select_groupB --min-score 0.6 --tag s06  # sensitivity analysis
Output (in data/s5_groupB[_<tag>]/):
    groupB_draws.csv     one row per draw: clinical info + SQI summary; row i <-> X[i]
    groupB_X.npy         float16 [n_draws, window_min*60*fs], best window per draw (memory-mappable)
    flow.md / flow.json  counts at every stage (CONSORT-style), from the S1-S4 manifests
"""
import argparse
import json

import numpy as np
import pandas as pd

from ppglabbench.common import CFG, S1, S2, S3, S4, S5, write_manifest

FS = CFG["fs"]


def flow_report(out_dir, b_counts):
    m = {s: json.loads((d / "manifest.json").read_text()) for s, d in
         [("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4)]}
    c = {k: v["counts"] for k, v in m.items()}
    lines = [
        "| Stage | Step | Subjects | Draws / records |",
        "|---|---|---|---|",
        f"| S1 | Waveform records scanned | {c['S1']['subjects_scanned']} | {c['S1']['records_scanned']} records |",
        f"| S1 | Records with PLETH | {c['S1']['subjects_with_pleth']} | {c['S1']['records_with_pleth']} records |",
        f"| S2 | Creatinine draws (MIMIC-III, deduplicated) | {c['S2']['creat_subjects']} | {c['S2']['creat_draws_dedup']} |",
        f"| S2 | **Group A**: draw within ±{m['S2']['params']['search_half_window_min']} min of a PLETH record "
        f"| {c['S2']['groupA_subjects']} | {c['S2']['groupA_draws']} |",
        f"| S2 | Selected for S3 (≤ {m['S2']['params']['max_draws_per_subject']} per subject, evenly spaced) "
        f"| {c['S2']['groupA_subjects']} | {c['S2']['s3_selected_draws']} |",
        f"| S3 | PLETH actually present near draw | {c['S3']['subjects_with_pleth']} | {c['S3']['draws_with_pleth']} |",
        f"| S4 | Scored | {c['S4']['subjects_scored']} | {c['S4']['draws_scored']} |",
        f"| S5 | **Group B**: best {CFG['s4_sqi']['window_min']} min window score ≥ {b_counts['min_score']} "
        f"| {b_counts['groupB_subjects']} | {b_counts['groupB_draws']} |",
    ]
    partial = [s for s, k in [("S1", "scan_complete"), ("S2", "s1_scan_complete"), ("S3", "fetch_complete")]
               if not c[s].get(k, True)]
    if partial:
        lines.append(f"\n⚠ Partial run: {', '.join(partial)} incomplete.")
    versions = {s: v["code_version"] for s, v in m.items()}
    lines.append(f"\nCode versions: {versions}")
    (out_dir / "flow.md").write_text("\n".join(lines) + "\n")
    (out_dir / "flow.json").write_text(json.dumps(dict(stages=c, groupB=b_counts, versions=versions), indent=2))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=CFG["s5_select"]["min_window_score"])
    ap.add_argument("--tag", default="", help="suffix for a sensitivity-analysis output folder")
    args = ap.parse_args()
    out = S5.with_name(S5.name + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    sq = pd.read_csv(S4 / "sqi_scores.csv")
    a = pd.read_csv(S2 / "groupA_draws.csv")
    b = sq[sq.best_score >= args.min_score].merge(a, on="draw_id").reset_index(drop=True)
    b.to_csv(out / "groupB_draws.csv", index=False)

    w = CFG["s4_sqi"]["window_min"] * 60 * FS
    X = np.lib.format.open_memmap(out / "groupB_X.npy", mode="w+", dtype=np.float16, shape=(len(b), w))
    for i, r in enumerate(b.itertuples()):
        z = np.load(S3 / "windows" / r.file)
        s = int(round((r.best_start_s - float(z["t0_s"])) * FS))
        X[i] = z["pleth"][s:s + w]
    X.flush()

    counts = dict(min_score=args.min_score, groupB_draws=len(b), groupB_subjects=int(b.subject_id.nunique()),
                  groupB_creat_median=float(b.creat_mgdl.median()) if len(b) else None,
                  groupB_draws_creat_ge_2=int((b.creat_mgdl >= 2).sum()),
                  X_shape=list(X.shape), X_nan_frac=float(np.isnan(X).mean()) if len(b) else None)
    write_manifest(out, "s5_select", dict(min_window_score=args.min_score, tag=args.tag), counts)
    flow_report(out, counts)


if __name__ == "__main__":
    main()
