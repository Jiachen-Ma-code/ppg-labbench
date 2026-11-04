# PPG-LabBench: confirmatory protocol

This is the analysis of the MIMIC test split, which will be evaluated once by `ppglabbench.confirmatory`. The analysis was developed on the training and validation splits and will be
frozen in a commit before the test split is evaluated. The status of that commit is in [RESULTS.md](RESULTS.md). The benchmark itself is in [SPEC.md](SPEC.md).

## Question

Could PPG replace or trigger a draw between samples? The waveform runs all the time, and a lab value is known only as of its last draw. To replace or trigger a draw, a score must add to what is known at the
bedside and must follow the patient between draws. So the procedure tests what PPG adds over `now`, bounds what it adds over the richer bases, reports the gain by the age of the previous value, and reports
tracking within a patient. It evaluates no decision rule, and no result here would establish clinical use.

## Window

PPG, MAP and monitor values will come from the best 5 min in the 10 min before the draw, on windows where one scores at least 0.8. The centred window will be a sensitivity analysis. Pulse-PPG will use windows
with 240 s of continuous signal; the others will count as missing and go to the probe's imputer.

## Candidates

Beat features, PaPaGei and Pulse-PPG (SPEC.md §6), and a small end-to-end CNN trained on the train split. **The CNN** is a 1D ResNet of about 207,000 parameters with one logit per lab's abnormal-value target and heads for age and heart
rate. It will read the `pre` window at 50 Hz, z-scored per window, as random 30 s crops in training and as the mean logit over 10 evenly spaced crops when scoring. The stopping rule uses train only: the checkpoint kept has the
highest mean AUROC on 10% of the fit's training subjects, held out of its gradient steps. Five fits by subject give out-of-fold scores on train, and a final fit on all of train scores every other row. Its logit is the PPG score, with no
probe. Its weights will not be distributed; the score file, identified by its SHA-256, fixes the results.

## Procedure

The bases, candidate models and rules were settled on train and validation rows with validation results in view, and the test split played no part.

1. **Scope.** MIMIC test split only, one evaluation. All models will be fitted on train only. A PPG score will enter a stack on the log-odds scale: out of fold on train (GroupKFold by subject) and a refit on test rows, with each fold centred on the train mean on that scale. For the CNN, these are its out-of-fold logit, centred by the
   file's folds, and its final model's logit. A window with no usable signal takes the CNN's train median, as a missing feature goes to a probe's imputer.

2. **Estimand.** Δ_l = AUROC(stack_l) − max(AUROC(base_l), 0.5), at window level, with the models frozen as fitted. A statement about the recipe retrained on new data is secondary.

3. **Primary base `now`.** Age, sex, the acuity covariates, the 8 monitor values, the treatment and context covariates and the history known at the draw (earlier admissions, their diagnosis groups, and each lab's time since
   its last draw and count of earlier draws, without their values), plus hinges at the train quartiles of age and MAP, female × age, vasopressor × routine draw and vasopressor × (MAP − train mean, missing → 0), in a
   pipeline of median imputation with missing indicators, scaling and logistic regression. PPG reads age, so a base linear in age would let a score with no independent information show a gain. Treatments are those running
   at the draw according to later documentation, since up to about half of at-draw entries were stored after the PPG ended. SpO2 comes from the same oximeter as PLETH, and is kept because a clinician sees it. The other rungs of
   the ladder will be reported beside it.

4. **Family A.** 10 labs, AUROC, k = 10 fixed: BUN, sodium, potassium, bicarbonate, calcium, glucose, haemoglobin, platelets and WBC, each covering at least 70% of non-test windows, plus eGFR. The
   other labs and creatinine, which carries the same information as eGFR, are descriptive. One model per lab: the highest validation Δ̂ over `now` among beat, PaPaGei, Pulse-PPG and the CNN, compared at 3 decimals. A tie will go to the
   first in the order beat, Pulse-PPG, CNN, PaPaGei: the simplest first, the one pretrained on these cohorts last. Each lab tests one pick, so k and the thresholds do not depend on the number of candidates. k will stay fixed at test time:
   a lab with fewer than 20 evaluation subjects in a class will not be tested, will count as not rejected and will keep its share of α.

5. **Tests per lab, in order, one-sided.**
   - **L1, partial association.** On test rows, fit logistic y ~ 1 + Z, where Z is the base with median imputation, one missing indicator per distinct missingness pattern and no constant columns. Let s be
     the frozen score on the log-odds scale, clipped to ±logit(1 − 10⁻⁴), r its residual on the design in the working-weight metric, and U_g = Σ r_i (y_i − p̂_i) per subject.
     z1 = ΣU_g / √(c ΣU_g²), with c = G/(G − 1) · (n − 1)/(n − K), G the subjects, n the rows and K the design's columns.
   - **L2, gain above 0.** z2 = Δ̂ / SE_E, where SE_E is the cluster DeLong (Obuchowski) standard error on test subjects. If the sample base AUROC is below 0.5, the stack's influence terms will be used alone.
   - **L3, practical gain.** z3 = (Δ̂ − 0.01) / SE_T, where SE_T = SE_E · √(1 + r̂).

6. **Named hypotheses.** Lactate and haemoglobin values, T1 over `now`: N1, partial association, by the same score test on the linear model; N2, ΔR² above 0, with a cluster standard error. One model each, the highest validation ΔR²
   among beat, PaPaGei and Pulse-PPG; the CNN has no head for a value.

7. **Multiplicity.** α = 0.025 one-sided in total, by a weighted graphical procedure: 0.02 over Family A, 0.002 per lab, and 0.005 over the named pair, 0.0025 each. The thresholds at the start are Φ⁻¹(1 − 0.002) = 2.878 for a lab and
   Φ⁻¹(1 − 0.0025) = 2.807 for a named hypothesis. Each branch's current level will be tested. A rejection will move it to its next level, and a branch will leave after its last level, L3 or N2. Only leaving passes its α on, to the
   branches still pending in proportion to their shares, so a branch with share w will be tested at Φ⁻¹(1 − 0.025 · w / Σw), the sum over branches still pending: with one lab gone, 2.852 for a lab and 2.780 for a named hypothesis; with both named gone, 2.807 for a lab; with one
   branch left, 1.96. This will repeat until nothing rejects. Family-wise error holds under any dependence if each test is valid.

8. **Headline summary.** The mean Δ̂ over Family A over `now`, with a percentile 95% interval from 2,000 subject resamples shared by all labs, with models fixed. It summarises; step 11 tests its upper side against m.

9. **r̂, to be fixed before opening, on validation only.** B = 50 refits of the picked model on 80% subject subsamples of train. Each will be scored on two random disjoint subject halves of validation, over 30 splits; C_l is the
   mean covariance of the gain across refits. r̂ per base and stratum = 4 · Σ max(C_l, 0) / Σ SE²_val,l. Strata, by sample size first: sparse (validation n < 1,000), mid (1,000-2,000), then rare (prevalence < 0.15 or > 0.85) and common. A
   stratum without an estimate will use the largest r̂ of that base. The CNN has no refits, so a lab that picks it will enter no estimate and will use the largest r̂ of that base.

10. **Claims.** L1: the PPG score carries information beyond the base in test subjects. L2: the stack gains discrimination. L3: practical gain. Simultaneous intervals will be reported as Δ̂ ± 2.878 · SE_T. z2 past
    the threshold without L1 is a flag, not a claim. A PaPaGei lab clearing L1 will be reported with the other candidates beside it. Claims cover these features, encoders and this small CNN, and pulse shape and timing: PLETH is auto-scaled, so pulse amplitude is not used.

11. **Upper-bound chain on the mean.** Over a base b, Δ̄_b is the mean of Δ_l over the 10 Family A labs, with the same picks. H0: the true Δ̄_b ≥ m. The upper bound is U_b = Δ̄_b + √(w² + (1.96 τ)²), where w is the distance from Δ̄_b to the 97.5th percentile of the subject
    resamples of step 8 and τ is the mean over the labs of √r̂_l · SE_E,l. Refits of all labs share one training split, so their errors may move together, and the SD of a mean is at most the mean of the SDs: τ takes them as fully correlated, the conservative end. A step will reject if
    U_b < m. It runs as a fixed sequence, `full` (`now` + previous value, history and the blood-gas previous value), then `now`, `monitor` and `acuity`, at one-sided 0.025, each step only if the one before passed. The hypothesis is about the 10 labs, so a lab that cannot be computed fails the step.
    Every step will report Δ̄_b and U_b, tested or not. Claim: on average over the 10 labs, these models add less than m over that base. Family A with the named hypotheses and the chain together have error ≤ 0.05.

    The per-lab upper bounds Δ̂_l + 1.96 · SE_T over each base, and the largest of them, will be reported as descriptive rows with no claim attached. A mean can hide one lab: that table and L1 to L3 address it.

12. **Margin.** m will be fixed in the plan from train and validation rows. For each Family A lab and each of MAP, pressors and non-invasive pressure: the AUROC of `now`, fitted on train and scored on validation, minus the same with that group's columns held
    constant, floored at 0. m is the median of those 30 gains, rounded up to 0.001, and never below +0.01. No PPG enters. On average PPG is held to adding less than the practical gain of L3, and a bound below a gain already called not practical would say nothing more. On validation that median is
    +0.002, so the floor sets m. What MAP, pressors and non-invasive pressure add together over the rest of `now` is about the same size as m: a mean of +0.013 and a median of +0.008 over the 10 labs.

13. **Size and power.** `ppglabbench.operating` will report these by plasmode on train and validation rows, resampled to the size of the test split: real covariates, subjects and PPG features with simulated outcomes. It will give the size of L1 and L2 over `now`, the
    family-wise error, each step's chance of passing if PPG adds nothing, the size of a step with the true mean at m, with equal gains and with a few labs carrying the mean, and the smallest gain with 80% power at the first step. Fixed scores cannot follow a simulated outcome, so the CNN is not simulated.

14. **Key secondaries,** reported in full, unadjusted, whichever way they fall.
    - **Tracking** (`ppglabbench.tracking`). The gain in a pooled AUROC is mostly between patients. For each candidate: correlation of score and value within and between subjects; the share of patients with both label classes and the shuffle's drop among them, since
      only they can move a pooled AUROC; per-patient AUROC, its mean, SD and share above 0.5; change on change over consecutive windows, against no change, `hist` and the patient's mean of earlier values; change outside exclusion zones of 1.96 · √2 · CV, with CV the desirable imprecision of the biological variation database, 2014 edition, and for eGFR that of creatinine times the equation's slope; and pairs around a transfusion for
      haemoglobin and dialysis for potassium.
    - **Positive controls** (`ppglabbench.positive_control`), from train to validation. A synthetic signal of known strength through beat, PaPaGei and Pulse-PPG. The CNN would need retraining for that, so its control is the R² of its age and heart-rate heads.
    - **The specification list.** Centred window, with a subset excluding labs charted before it started; lactate within ± 30 min; no recent previous value; measured MAP; PTT without a heparin infusion; stable creatinine; gradient boosting on `now` against the same on `now` + beat features; the pulse-rate model on each base.

15. **Other secondaries over `now`, descriptive, not tested for confirmation.** The gain by age of the previous value (≤ 12, 12-24, 24-48, > 48 h, never measured), over `now` and `locf`. A residual arm: PPG features and the base's log-odds, fitted by boosting to the
    base's working residual on the log-odds scale, so that a stack on one score cannot be blamed for missing interactions. Partial R² on the value and Brier skill. Calibration slope and intercept of stack and base. Net benefit of stack, base and acting on everyone at thresholds
    of 0.05, 0.10, 0.20, 0.30 and 0.50. Subgroups, with a Wald statistic of heterogeneity: sex; age band (< 50, 50-64, 65-79, ≥ 80); AF coded for the admission against not coded, which is not a rhythm at the draw; vasopressor; era (MetaVision or CareVue).

16. **Beside each pick, descriptive rows.** The `rate` model, which is pulse rate and its variability: its L1 z and its gain over `now`. And the pick's gain over `now` + the rate score.

17. **Everything else is exploratory:** AUPRC, other bases, the seven descriptive labs and VitalDB. A subject bootstrap of the standard error and the null control will be reported as checks, not decisions.

## What the report will say

| Pattern on the test split | What the report will say |
|---|---|
| No L1 rejection, chain passes `full` and `now` | On average over the 10 labs, these models in this setting add less than m over what is known at the bedside, and no lab shows information beyond it, so there is no support for replacing or triggering a draw. Passing `monitor` and `acuity` extends this to those bases. Not that PPG holds nothing: the positive controls say which strengths the recipe would recover. |
| No L1 rejection, chain passes `full` only | On average less than m over everything listed, previous value included. Over `now`, neither a shown gain nor a bound of m on the mean. |
| No L1 rejection, chain fails at `full` | Inconclusive at m: neither a shown gain nor a bound of m on the mean. U_b gives the mean gains still possible. |
| A lab clears a level over `now`, and the chain passes `now` | On average less than m, with that lab named beside it: a mean can hide one lab. |
| L1 without L2 in a lab | The score holds information beyond the base in that lab. No gain in discrimination is shown. |
| z2 past its threshold without L1 | A flag, not a claim. |
| L2 without L3 in a lab | A gain above 0, not shown to exceed +0.01. No practical claim. |
| L3 in a lab, and the gain is gone over `full` or sits in windows with a recent previous value | A practical gain over `now` that history already supplies. No support for replacing or triggering a draw. |
| L3 in a lab, the gain is largest with an old or no previous value, and tracking beats its shuffle among patients with both labels | A candidate for triggering a draw in that lab, to be tested prospectively. Not a replacement for a draw. |
| A gain without tracking: little shuffle drop among patients with both labels, per-patient AUROC near 0.5 | The score refines who the patient is, not what changed. |
| The gain is gone over `now` + the rate score | Pulse rate and its variability, not morphology. |
| Only PaPaGei picks clear a level | Possibly optimistic: it was pretrained on these cohorts. Reported beside the other candidates. |
| Only CNN picks clear a level | A gain the features and encoders did not show, from a model trained for the task, with no check on VitalDB or the centred window. |
| N1 or N2 for a named lab | The same reading for the value of that lab. |
| The mean gain's interval excludes 0, and no lab clears L2 | A small average gain that no single lab supports. |
| The residual arm gains where the stack does not | A descriptive hint of interactions. Not a claim. |
| Poor calibration, or net benefit no better than the base | Discrimination only: no support for a decision at those thresholds. |
| A specification disagrees with the primary result | The primary result stands, and the disagreement is stated beside it. |

## How it runs

```bash
python -m ppglabbench.features.cnn prepare        # then its fits and assemble: see that module
python -m ppglabbench.confirmatory plan           # model picks, r-hat and the margin, from train and validation rows
python -m ppglabbench.operating                   # size and power, by plasmode on train and validation rows
python -m ppglabbench.confirmatory dry-run        # the whole analysis with validation standing in for test
python -m ppglabbench.confirmatory run --open-test
```

Before opening, the commit that freezes the code and the plan will be time-stamped where a third party can see it, and the plan's SHA-256 will be set as `benchmark.plan_sha256` in `config.yaml`. `plan` and `run` refuse a missing CNN score file. `run` refuses unless the working tree is
clean, the split file matches `splits_sha256`, the plan matches `plan_sha256` and is not provisional (all models, B = 50, a clean tree, the `pre` window), the CNN score file matches the SHA-256 in the plan, `src/` and `config.yaml` have not changed since the plan other than those keys,
a dry run of this plan finished with nothing failed, and no test results exist. It does not read `benchmark.test_opened`. The score file holds test rows, as the embeddings do. They are dropped as it is read, before anything is computed, except by `run` and by the evaluations below.

A descriptive row that raises will be recorded in the results with its error's type and place, and what it sits beside will keep its result. A lab that raises in its own tests will count as not computable: not rejected, and still in k. The results will be in their final folder before anything else is written, and `python -m ppglabbench.confirmatory render` writes the report again from them without reading any data.

After `run`, `benchmark.test_opened` will be set to `true` in a commit. Only then may other models read the split, through `ppglabbench.evaluate --eval test`, `ppglabbench.baselines --eval test --open-test` or
`ppglabbench.tracking --eval test --open-test`. Those evaluations will be exploratory; tracking and the specification list stay the key secondaries named above.

## Limits

- **Only the MIMIC test split is confirmatory.** VitalDB results are exploratory: its validation and test cases were used in cross-validation while the methods were developed.
- **PaPaGei was pretrained on these cohorts,** MIMIC-III and VitalDB, without labels and with no published list of training subjects. A confirmed PaPaGei lab may be optimistic for unseen patients. Pulse-PPG, the
  beat model and the CNN do not share this.
- **The CNN exists for MIMIC's `pre` window only.** VitalDB and the centred window run without it. Its folds are one assignment on the cohort's train rows, not per lab as for the other models. A re-run gives different scores, since training on MPS is not deterministic.
- **Clocks.** The clinical and waveform clocks agree only to about 10 minutes, so "before the draw" is accurate to about that.
- **Missing MAP** does not strictly mean no arterial line, since the numerics may hold an arterial mean. MAP comes from the waveform alone.
- **Test base rates.** Label prevalence pooled over all splits and split counts were computed while the cohort was built, so they are not unknown.
- **Model picks** are made on validation, and validation holds development subjects. More candidates make a pick's validation gain more optimistic, not its test gain.
- **Exclusion zones** rest on a published specification for imprecision, not on what this laboratory achieved, and that stratum selects on the true change, so it is descriptive.
- **Pairs around a treatment** rest on documentation times and cover only what the stored flags allow. Insulin has only an at-draw flag, and creatinine and eGFR leave out dialysis windows, so neither has such pairs.
- **Sensitivity.** A gain near 0 means this recipe extracted nothing, not that PPG holds nothing (SPEC.md §5).
