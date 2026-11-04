# PPG-LabBench: specification

PPG-LabBench is a benchmark of photoplethysmography (PPG) windows paired with blood laboratory values. It asks how much the pulse
waveform adds to what is already known at the bedside. It fixes the windows, labels, splits, baselines and metrics, and asks every result
to report the controls in §5. The confirmatory analysis of the test split is in [PROTOCOL.md](PROTOCOL.md).

## 1. Sources, windows and quality

| | MIMIC-III | VitalDB |
|---|---|---|
| Setting | intensive care; PLETH, sensor site not recorded | operating room; PLETH, under general anaesthesia |
| Unit | window: 20 min of PLETH centred on a creatinine draw, at most 10 draws per subject, evenly spaced in time | case: pre-incision PLETH, up to 30 min |
| Sampling | 125 Hz, native | 500 Hz decimated to 125 Hz with an FIR filter |
| Labs | timestamped LABEVENTS aligned to the window (§2) | latest preoperative value, not time-aligned |
| Release | will be a credentialed PhysioNet package and code ([DATA.md](DATA.md)) | code only; the data regenerate from the VitalDB API |

**Quality.** Each 10 s epoch is good if it has no missing samples, at most 10% of its samples are in flat runs, its rate is 30-200 beats/min,
and its beats correlate with the epoch's average beat at 0.86 or more. Skewness and inter-beat variability are recorded and not thresholded, so
arrhythmic patients are not removed by rule. The score of a 5 min window is its fraction of good epochs. A MIMIC window is in the benchmark if
its best 5 min scores at least 0.8.

**PPG windows.** The primary window `pre` is the best 5 min wholly before the draw, where one scores at least 0.8; it exists for 93% of MIMIC windows. Among equal
scores it is the one nearest the draw, up to floating-point rounding, which affects about 0.3% of windows.
The sensitivity window `centered` is the best 5 min anywhere in the 20 min. PPG, MAP and monitor values all come from the same window. With `pre`, a lab
charted before the PPG window ended is left out for that window, and the filter for beat features stops at the window's end. VitalDB uses its best 5 min.

**Amplitude.** PLETH is auto-scaled, so pulse amplitude is not used. Claims cover pulse shape and timing.

**Cohort.** Adults only: age in completed years, 18 or more at the draw; MIMIC ages of 90 or more are set to 91.4. A window also needs at least 20 kept beats and one
gap-free 10 s segment. Left out: MIMIC windows whose PLETH record overlaps another PLETH record of the subject (`record_overlap`, 470 windows, from each window's records and
the catalog of waveform records; the two streams disagree, so one may be another patient's), and VitalDB ASA 6 cases, cases not under general anaesthesia and cases with under 50% good epochs. Other conditions
are flagged, not removed.

## 2. Labels

MIMIC: the measurement nearest to the draw within **± 60 min**. `<lab>_offset_min` is stored, so a stricter window is a filter. Censored
results such as ">150" count as their limit. Coverage below is on non-test windows.

| Lab | ITEMID | Abnormal (binary task) | Coverage | Tested in PROTOCOL.md |
|---|---|---|---|---|
| eGFR (CKD-EPI 2021, race-free), from creatinine | 50912 | eGFR < 60 | anchor | yes |
| creatinine | 50912 | > 1.5 mg/dL | anchor | |
| BUN | 51006 | > 20 mg/dL | 99% | yes |
| sodium | 50983 | < 135 mmol/L | 99% | yes |
| potassium | 50971 | > 5.0 mmol/L | 99% | yes |
| bicarbonate | 50882 | < 22 mmol/L | 99% | yes |
| calcium (total, MIMIC only) | 50893 | < 8.4 mg/dL | 92% | yes |
| glucose | 50931 | ≥ 180 mg/dL | 98% | yes |
| haemoglobin | 51222 | < 12 g/dL (female), < 13 g/dL (male) | 81% | yes, and its value |
| platelets | 51265 | < 150 ×10³/µL | 81% | yes |
| WBC | 51301 | > 12 ×10³/µL | 81% | yes |
| INR | 51237 | > 1.5 | 49% | |
| PTT | 51275 | > 40 s | 49% | |
| AST | 50878 | > 40 U/L | 22% | |
| ALT | 50861 | > 40 U/L | 22% | |
| lactate | 50813 | > 2.0 mmol/L | 15% | its value |
| albumin | 50862 | < 3.5 g/dL | 14% | |

eGFR is rounded to one decimal before the rule. eGFR and creatinine leave out windows flagged `exclude_dialysis`. VitalDB has BUN, sodium, potassium, bicarbonate, glucose, haemoglobin, platelets, PTT, albumin, AST,
ALT and eGFR, plus prothrombin activity below 80%; its eGFR leaves out eGFR < 15 and renal-failure diagnoses (`egfr_incl_failure` keeps them).

## 3. Covariates

For baselines; never derived from the PPG under test.

- **Age and sex** (PATIENTS). ICD-9 groups of the current admission (CKD, ESRD, AKI, AF, hypertension, diabetes, heart failure) are coded at discharge, so they are context,
  not prediction-time information, and they enter no base.
- **Dialysis flag.** ESRD, including CKD stage V and hypertensive stage V or ESRD, dialysis status (V451, V56) or a dialysis procedure (39.95, 54.98) in the stay; ESRD or dialysis
  status in an earlier stay carries forward, a procedure alone does not.
- **Previous value (LOCF)** of each lab: strictly before both the target measurement and the window start, the earlier of which is the reference time, also for a lab left out of the
  window under §1. Its age in hours runs from the reference time, not the draw. **History:** mean and count of all earlier values and the recent trend, from the same reference time.
  Creatinine's previous value and history use VALUENUM within 0.1-20 mg/dL, without censored results.
- **Acuity.** Vasopressors running at the draw according to later documentation (MetaVision: not cancelled or rewritten; CareVue: a positive rate or amount in the preceding hour), emergency or
  urgent admission (a null admission ID counts as neither), MAP and MAP < 70 from the arterial waveform over the PPG window (missing without a usable waveform), and the draw hour, as a fraction
  of the hour (routine 00-05 h flag, sine and cosine).
- **Monitor,** 8 columns from the open numerics records: HR (ECG), RESP, SpO2, irregular-HR percent, PVC rate and systolic, diastolic and mean NBP. RESP of exactly 0 counts as missing. SpO2 comes from
  the same oximeter as PLETH, and is kept because a clinician sees it. The pulse rate derived from PLETH never enters a base.
- **Now.** Treatments running at the draw according to later documentation, since up to about half of at-draw entries were stored after the PPG ended, and in the prior 24 h; pressor count, epinephrine,
  hospital hours, era, admission route and ethnicity group.
- **History in `now`,** all known at the draw. From admissions that ended before the draw, never the current one: their number, days since the last discharge, and flags for CKD, ESRD, AKI, AF, hypertension,
  diabetes, heart failure and liver disease (ICD-9 570-573) from DIAGNOSES_ICD. Per lab: the time since its last draw and the count of its earlier draws, without their values. With no earlier draw the time is
  missing and the count 0.
- **Blood-gas previous value** of potassium, glucose, bicarbonate (total CO2), haemoglobin and sodium, kept apart from the central-lab value. Fingersticks are in CHARTEVENTS and are
  not used, so the glucose LOCF understates what a clinician sees.

In VitalDB, `acuity` is ASA class and emergency operation, and `now` adds department, operation type, approach, BMI, hypertension and diabetes.

## 4. Splits

- **By subject:** train 70%, validation 15%, test 15%, by the SHA-256 hash of the text `salt:id` (`split_salt` in `config.yaml`), its first 12 hexadecimal digits read as a fraction with cut-offs 0.70 and 0.85,
  so a subject's split does not depend on other subjects.
  The exception: the 77 development subjects the hash puts in test move to train. Exact reproduction will therefore need the released split file, whose SHA-256 is
  `splits_sha256` in `config.yaml`. A results table counts as a benchmark result only if it uses these splits.
- **Development subjects** (392, `dev_set` in the split file) are never in test. They are 8% of validation subjects, hence the `not_dev` sensitivity.
- **MIMIC test.** Evaluated once. No benchmark model is fitted on test subjects. Label prevalence pooled over all splits and split counts were computed while the cohort was built,
  so test base rates are not unknown. PaPaGei was pretrained without labels on these cohorts (§8).
- **VitalDB.** The same proportions by `subjectid`, but all its cases were used in cross-validation while the methods were developed, so its validation and test splits are not independent of that
  work. No VitalDB result is confirmatory.
- **Guards.** `ppglabbench.evaluate --eval test` refuses until `benchmark.test_opened` is true in `config.yaml`. `ppglabbench.baselines --eval test` refuses without `--open-test`, a clean tree and the
  split file's SHA-256, and only once.

## 5. Tasks, metrics and required reporting

| Task | Metric | Notes |
|---|---|---|
| T1 regression per lab | MAE, R² | log scale for skewed labs (BUN, glucose, platelets, WBC, INR, PTT, lactate, AST, ALT, creatinine) |
| T2 abnormal-value classification | AUROC, AUPRC | subject-bootstrap 95% interval |
| T3 increment | ΔAUROC, ΔR² over each base of §6 | the PPG score enters as a 5-fold out-of-fold score on train (GroupKFold by subject) and a refit on the evaluation split, on the log-odds scale in T2, with each fold centred on the train mean on that scale; paired subject bootstrap; for AUROC the base counts as at least 0.5 |

Every result reports:

1. **The null control:** the same pipeline over at least 100 subject-block permutations of the PPG, as a distribution. It shows what the pipeline makes from noise; it does not test
   independence of PPG and the lab given the base.
2. **The baselines** age + sex, acuity and LOCF next to every PPG result.
3. **Selection:** model selection on train or validation, never test; test evaluated once per method.
4. **A positive control:** a synthetic signal of known strength through the same pipeline, because weak signal is only partly recovered, so "Δ ≈ 0" means this recipe extracted nothing, not that PPG
   holds nothing. A model that would need retraining for that, such as the CNN of §6, reports a known target instead: the R² of its age and heart-rate heads.
5. **Multiplicity:** unadjusted intervals are exploratory. The confirmatory family and its correction are in [PROTOCOL.md](PROTOCOL.md). Training variance belongs in magnitude statements about the recipe,
   not in the test of no gain for frozen models.
6. **The carry-forward ladder,** for any claim of beating carry-forward: the previous value copied, a model fitted on lab history without PPG, and that model with PPG. Beating the copied value alone says
   nothing about PPG.
7. **Within and between patients.** The gain in a pooled AUROC is mostly between patients, so it does not show that a score follows a patient's changes. `ppglabbench.tracking` reports this as a secondary,
   unadjusted, descriptive result. With the positive controls and the specification list, it is a key secondary of [PROTOCOL.md](PROTOCOL.md).

## 6. Bases and models

| Base | Holds |
|---|---|
| `age_sex` | age and sex |
| `acuity` | + vasopressor, emergency or urgent admission, MAP and MAP < 70, draw hour |
| `monitor` | `acuity` + the 8 monitor values (MIMIC) |
| `now` | `monitor` + treatments, context and the history of §3, with age and MAP entering flexibly: the primary base, because a base linear in age lets a score with no independent information show a gain |
| `locf` | age, sex, the previous value of the lab and its age (MIMIC) |
| `hist` | `locf` + history (MIMIC) |
| `full` | `now` + previous value, history and the blood-gas previous value: the covariates listed (MIMIC) |

Also reported: `mean_train` and `locf_raw` (the previous value copied, T1 only). `now_gb` is the sensitivity with the same model family on both sides: gradient boosting on `now`, then on `now` + beat features.

| Model | Input and probe |
|---|---|
| `beat` | beat-morphology features, gradient boosting without early stopping, one round count for all labs chosen on train by `ppglabbench.tune_beat`. The systolic peak is searched in the first 70 of a beat's 100 points, and a beat with the peak below index 3 is dropped. The diastolic point is searched before index 80: the top of the rise after a notch, if that rise is at least 1% of the beat's amplitude, or else the first local maximum of the first derivative after the peak; a beat with neither has no systolic-diastolic time or reflection index |
| `rate` | HR, SDNN, RMSSD and inter-beat interval from the PPG, gradient boosting: pulse rate and its variability, without morphology |
| `papagei` | PaPaGei-S 512-d embedding, then imputer, scaler, PCA to 64, logistic or ridge |
| `pulseppg` | Pulse-PPG 512-d embedding, the same probe. It uses windows with 240 s of continuous signal, because the embedding of a shorter input depends on its length; the others count as missing and go to the probe's imputer. It was trained on field data, not on these cohorts, and is a wrist model at 50 Hz, so a poor result may reflect the domain gap |
| `cnn` | a small end-to-end CNN trained on the train split, MIMIC `pre` window only; its logit is the score, with no probe, and only for T2 |

All PPG models read the same 5 min window. To run your own embeddings through all of this, use `ppglabbench.evaluate` (see [README.md](README.md)).

**The CNN** (`ppglabbench.features.cnn`) is a 1D ResNet of about 207,000 parameters: a stem, then four stages of two residual blocks, 16 to 64 channels, global average pooling, and one logit per abnormal-value target plus heads for age and heart rate. The input is the `pre` window at 50 Hz, z-scored per window,
with gaps set to 0 and values clipped to ± 6 SD: random 30 s crops in training, and the mean logit over 10 evenly spaced crops when scoring. The loss is masked binary cross-entropy plus 0.2 × squared error of standardised age and heart rate, over 40 epochs of AdamW. The stopping rule uses train only: the checkpoint
kept has the highest mean AUROC, averaged over 5 epochs, on 10% of the fit's training subjects, held out of its gradient steps. Five fits by subject give out-of-fold scores on train and a final fit on all of train scores every other row; a window with no usable signal takes the train median. Its folds are one assignment
on the cohort's train rows, not per lab as for the other models, and every train score still comes from a fit that never saw the subject. Its weights will not be distributed: `cnn_scores.npz`, identified by the SHA-256 in its manifest, fixes the results. VitalDB and the centred window run without it.

## 7. Release

The MIMIC part will be released on PhysioNet under credentialed access as MIMIC-III-Ext-LabBench. Windows below the quality rule will not be released. An optional table will link the labs to
the segments of MIMIC-III-Ext-PPG and keep that dataset's `strat_fold`. See [DATA.md](DATA.md).

The ± 60 min window is the primary one. ± 30 min is a sensitivity subset for lactate only, because every other lab shares the creatinine timestamp in 97.5-100% of windows.

## 8. Known limitations

- **Case spectrum.** The per-subject draw cap removes sicker draws more often, and the quality rule passes healthier and non-AF windows more often, so the case spectrum is narrower than the source population.
- **Sparse labs** are measured in sicker patients: windows with a lactate are far more often on vasopressors. Lactate targets are mostly charted after the PPG window (median +14 min on non-test windows).
- **Dialysis codes have no time,** so windows before dialysis started in that stay are excluded from eGFR too.
- **Input events.** Windows of admissions without input events count as not on vasopressors. Treatments at the draw rest on later documentation: up to about half of at-draw entries were stored after the PPG ended.
- **Missing MAP** does not strictly mean no arterial line, since the numerics may hold an arterial mean. MAP comes from the waveform alone.
- **Between patients.** Pooled gains are mostly between patients. Tracking within a patient is a secondary, unadjusted, descriptive result.
- **Development subjects** are the lowest subject IDs, so they are mostly CareVue. Test holds none of them and has a smaller CareVue share than train and validation.
- **VitalDB has no dialysis flag,** so the eGFR exclusions of the two cohorts are not defined identically. Most of its PPG is under anaesthesia, though some cases start recording before anaesthesia.
- **PaPaGei is partly in-domain** on both cohorts: it was pretrained on them, with no published list of training subjects to exclude. Its results may be optimistic relative to unseen patients, which
  the beat model does not share.
- **The CNN covers one window of one source.** VitalDB and the centred window run without it, so a CNN result has no check there. A re-run gives different scores, since training on MPS is not deterministic.
- **Labels are not diseases.** eGFR < 60 is a snapshot, not CKD. Abnormal PTT mostly marks heparin, and INR reflects anticoagulants and liver disease, so neither is a coagulation task without that
  context; warfarin is not among the covariates. Total calcium tracks albumin.
- **Labels are one-sided:** rule-negative windows include the opposite abnormality, such as sodium above 145.
- **Equity.** The quality rule passes about 88% of windows from White patients and about 84% from Black patients, unadjusted. ETHNICITY is a proxy for skin tone at best.
- **Clocks.** The clinical and waveform clocks agree only to about 10 minutes.
