# ppglabbench.pipeline — MIMIC-III cohort pipeline (S1–S6)

Builds the MIMIC part of PPG-LabBench: PPG windows anchored on blood draws, with signal quality,
labels and covariates. Each stage is one script, one output folder and one `manifest.json`
(parameters, code version, counts). All parameters are in [config.yaml](../../../config.yaml).
The windows are anchored on creatinine draws; most other routine labs are drawn at the same time, and
`ppglabbench.labels` attaches them.

## Stages

| Stage | Script | What | Output (`data/…`) |
|---|---|---|---|
| S1 | `s1_scan_waveforms.py` | scan the headers of all 22,317 waveform records: time spans, whether PLETH is present; no signal download | `s1_catalog/waveform_catalog.csv`, `pleth_subjects.txt` |
| S2 | `s2_match_groupA.py` | creatinine draws ∩ PLETH record spans (± `search_half_window_min`) → **group A**; ≤ 10 draws per subject passed on | `s2_groupA/groupA_draws.csv`, `groupA_subjects.txt` |
| S3 | `s3_fetch_windows.py` | download PLETH for ± `half_window_min` around each draw, aligned to the draw, NaN where absent | `s3_windows/windows/<draw_id>.npz`, `s3_index.csv` |
| S4 | `s4_score_sqi.py` | SQI per 10 s epoch (flat/saturated, plausible HR, beat-template correlation); best 5 min window per draw | `s4_sqi/sqi_scores.csv`, `epoch_good.npz`, `review_sample.csv/png` |
| S5 | `s5_select_groupB.py` | window score ≥ `min_window_score` → **group B**; model input matrix and CONSORT-style flow table | `s5_groupB/groupB_draws.csv`, `groupB_X.npy`, `flow.md` |
| S6 | `s6_build_dataset.py` | eGFR (CKD-EPI 2021), KDIGO AKI, age, sex, Hb, BUN, vasopressors, ICD groups, dialysis flag | `s6_dataset/dataset.csv` |

S1 and S3 resume after interruption; S2, S4 and S5 recompute from scratch (local, fast).

## Run (from the repository root)

```bash
python -m ppglabbench.pipeline.s1_scan_waveforms            # ~40 min, network
python -m ppglabbench.pipeline.s2_match_groupA              # first run extracts creatinine (minutes)
python -m ppglabbench.pipeline.s3_fetch_windows             # slowest: ~15 h for 25.9k draws (PhysioNet ≈ 0.5 MB/s)
python -m ppglabbench.pipeline.s4_score_sqi --plot 100      # also exports 100 windows for manual review
python -m ppglabbench.pipeline.s5_select_groupB
python -m ppglabbench.pipeline.s5_select_groupB --min-score 0.6 --tag s06   # sensitivity analysis
python -m ppglabbench.pipeline.s6_build_dataset            # needs the clinical tables below
```

Clinical tables for S2 and S6 (into `data/clinical/`; type your own PhysioNet password):

```bash
for t in LABEVENTS PATIENTS ADMISSIONS DIAGNOSES_ICD PROCEDURES_ICD INPUTEVENTS_MV INPUTEVENTS_CV; do
  wget -N -c --user <physionet user> --ask-password -P data/clinical https://physionet.org/files/mimiciii/1.4/$t.csv.gz
done
```

Small test runs: S1 `--limit 300`, S3 `--limit-subjects 5`, S4 `--limit 50`.

## Methodological notes

- **Traceability**: every manifest records parameters and the git version; `-dirty` means uncommitted code —
  commit before producing reportable results. "Partial run" in `flow.md` means a stage was incomplete.
- **SQI does not filter on rhythm regularity**: the IBI coefficient of variation is recorded (`best_ibi_cv`), not
  thresholded, so patients in atrial fibrillation are not systematically removed (selection bias).
- **Manual SQI check**: `review_sample.csv` is a random sample to label by hand; report agreement with the automatic SQI.
- **Threshold sensitivity**: S5 `--min-score` / `--tag` produce alternative group B sets.
- **Repeated measures**: several draws per subject → all splits and CIs are by subject.
- **Dialysis**: admissions with ESRD, dialysis status or dialysis procedures, and later windows after an admission with ESRD or
  dialysis status, are flagged `exclude_dialysis`.

## Data

- Clinical: MIMIC-III v1.4 (credentialed), `data/clinical/LABEVENTS.csv.gz`; creatinine ITEMID 50912.
- Waveforms: MIMIC-III Waveform Database Matched Subset v1.0 (open), read per segment online.
- Linked by `subject_id` (not compatible with MIMIC-IV IDs).
- Download volume: `.dat` files interleave ~5 channels, so PLETH alone still costs all channels:
  ± 10 min ≈ 1.5 MB per draw downloaded; ≈ 0.15 MB stored (PLETH only, float16).
