# PPG-LabBench: results

The MIMIC test split was evaluated once, on 2026-11-03, with the frozen plan (SHA-256 `8b7c5d617da090eb1affb45ce7bc9caf449808a67d2f26ce94274adf49f1be64`, the same value as `benchmark.plan_sha256` in `config.yaml`). The run finished on its first attempt, with no analysis failed. All full results are in [results/](results/).

**Order of events.** The plan and its hash were committed in the development repository before the evaluation. This repository and its archive were made public after the evaluation, so their dates do not show that order.

## In brief

Models: pulse-beat features, two pretrained encoders and a small end-to-end CNN, with one model picked for each lab on the validation split. Gains are in AUROC over a base without PPG. Margin m = +0.01.

- **On average over the 10 labs, these models add less than +0.01 over what is known at the bedside (`now`):** mean gain +0.0053 (95% interval +0.0023, +0.0081), upper bound +0.0095.
- **With previous lab values in the base (`full`), almost nothing remains:** mean gain +0.0002, upper bound +0.0017.
- **Over the smaller `monitor` base, a bound of +0.01 is not shown:** mean gain +0.0106, upper bound +0.0154. The chain stopped there, so `acuity` was not tested.
- **A mean can hide one lab.** Urea nitrogen (BUN) shows a gain above 0 over `now`, +0.0138 (simultaneous interval +0.0032, +0.0245), not shown to exceed +0.01, so this is no practical claim. In bicarbonate and calcium the score holds information beyond the base, with no gain in discrimination shown. The other seven labs clear no level.
- **Named value hypotheses.** Haemoglobin clears the first level only, and with PaPaGei, which was pretrained on these cohorts, so it may be optimistic. The other candidates are beside it [below](#named-hypotheses-the-lab-value-over-now). Lactate clears no level.

These results give no support for replacing or triggering a draw with these models in this setting. They do not show that PPG holds no such information. They are from one ICU database.

## Upper-bound chain on the mean gain

| Base | Tested | Passed | Labs in the mean | Mean gain (95% interval) | Training SD | Upper bound |
|---|---|---|---|---|---|---|
| `full` | yes | yes | 10 of 10 | +0.0002 (-0.0004, +0.0008) | 0.0007 | +0.0017 |
| `now` | yes | yes | 10 of 10 | +0.0053 (+0.0023, +0.0081) | 0.0016 | +0.0095 |
| `monitor` | yes | no | 10 of 10 | +0.0106 (+0.0057, +0.0154) | 0.0000 | +0.0154 |
| `acuity` | no | no | 10 of 10 | +0.0232 (+0.0156, +0.0307) | 0.0045 | +0.0347 |

## Family A, gain over `now`

| Lab | Model | n | Subjects | Prevalence | Base AUROC | Δ | SE (eval) | SE (with training) | Simultaneous interval | z1 | z2 | z3 | Levels cleared |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bun | cnn | 2459 | 725 | 50.1% | 0.749 | +0.0138 | 0.0034 | 0.0037 | +0.0032, +0.0245 | 5.58 | 4.04 | 1.03 | 2 |
| sodium | cnn | 2452 | 725 | 14.1% | 0.646 | +0.0037 | 0.0054 | 0.0059 | -0.0132, +0.0206 | 2.05 | 0.68 | -1.07 | 0 |
| potassium | cnn | 2457 | 725 | 5.5% | 0.674 | +0.0099 | 0.0065 | 0.0070 | -0.0102, +0.0300 | 1.73 | 1.54 | -0.01 | 0 |
| bicarbonate | cnn | 2437 | 725 | 21.8% | 0.701 | +0.0051 | 0.0039 | 0.0043 | -0.0071, +0.0174 | 3.19 | 1.30 | -1.14 | 1 |
| calcium | cnn | 2293 | 709 | 52.3% | 0.679 | +0.0059 | 0.0031 | 0.0033 | -0.0037, +0.0155 | 2.96 | 1.91 | -1.22 | 1 |
| glucose | pulseppg | 2428 | 725 | 12.9% | 0.701 | +0.0002 | 0.0016 | 0.0018 | -0.0049, +0.0052 | 1.26 | 0.12 | -5.59 | 0 |
| haemoglobin | beat | 2002 | 715 | 86.0% | 0.810 | +0.0056 | 0.0034 | 0.0036 | -0.0049, +0.0160 | 2.69 | 1.66 | -1.22 | 0 |
| platelets | cnn | 2005 | 715 | 28.6% | 0.684 | +0.0021 | 0.0046 | 0.0050 | -0.0121, +0.0164 | 2.02 | 0.46 | -1.59 | 0 |
| wbc | cnn | 2003 | 715 | 31.4% | 0.611 | +0.0018 | 0.0033 | 0.0036 | -0.0085, +0.0122 | 1.62 | 0.55 | -2.27 | 0 |
| egfr | cnn | 2175 | 669 | 26.5% | 0.799 | +0.0045 | 0.0033 | 0.0036 | -0.0058, +0.0148 | 2.67 | 1.36 | -1.54 | 0 |

L1: the score carries information beyond the base. L2: the stack gains discrimination. L3: gain above +0.01.

All three labs that clear a level have the CNN as their model. In BUN the other models gain less over `now` and are not tested: beat +0.0097, PaPaGei +0.0082, pulse rate +0.0069, Pulse-PPG +0.0042. Over `now` + the rate score, the CNN keeps +0.0083 (+0.0039, +0.0128).

## Named hypotheses, the lab value over `now`

| Lab | Model | n | Subjects | R² base | ΔR² | SE | Partial R² | Simultaneous interval | z1 | z2 | Levels cleared |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lactate | beat | 373 | 196 | 0.239 | +0.0188 | 0.0127 | +0.0248 | -0.0167, +0.0544 | 2.26 | 1.49 | 0 |
| haemoglobin | papagei | 2002 | 715 | 0.222 | +0.0134 | 0.0054 | +0.0172 | -0.0016, +0.0285 | 4.41 | 2.51 | 1 |

Beside the haemoglobin pick, ΔR² with unadjusted 95% intervals, not tested: PaPaGei +0.0134 (+0.0030, +0.0239), beat +0.0078 (-0.0017, +0.0182), Pulse-PPG +0.0069 (-0.0011, +0.0149), pulse rate +0.0056 (+0.0002, +0.0117). The CNN has no head for the value. The report printed by the confirmatory run leaves these out; they come from the ladder below.

## The full ladder

Exploratory. Mean AUROC of each base over the 10 Family A labs, and the mean gain of each model stacked on that base. PPG alone reaches a mean AUROC of 0.584 with pulse rate, 0.592 with beat morphology, 0.584 with PaPaGei, 0.581 with Pulse-PPG and 0.621 with the CNN.

| Base | Mean AUROC of the base | Pulse rate | Beat morphology | PaPaGei | Pulse-PPG | CNN |
|---|---|---|---|---|---|---|
| `age_sex` | 0.572 | +0.0362 | +0.0387 | +0.0330 | +0.0368 | +0.0590 |
| `acuity` | 0.628 | +0.0178 | +0.0178 | +0.0144 | +0.0142 | +0.0255 |
| `monitor` | 0.648 | +0.0062 | +0.0073 | +0.0048 | +0.0039 | +0.0124 |
| `now` | 0.705 | +0.0022 | +0.0025 | +0.0024 | +0.0020 | +0.0057 |
| `locf` | 0.902 | +0.0005 | +0.0004 | +0.0011 | +0.0011 | +0.0013 |
| `hist` | 0.906 | +0.0005 | +0.0005 | +0.0012 | +0.0011 | +0.0014 |
| `full` | 0.910 | +0.0001 | +0.0000 | +0.0004 | +0.0003 | +0.0003 |

The gain shrinks as the base learns more, and the previous value alone does most of the work. With each subject given another subject's PPG, over 100 permutations, the mean gain over `now` is about -0.0003 for every model. With boosting on both sides, so that no gain can come from the class of model, beat features add +0.0016 on average.

## Within-patient tracking

Secondary and descriptive. Means over the 10 Family A labs, with subject-bootstrap 95% intervals. With no alignment to the patient's course, within-patient r is 0, the share of changes in the same direction 0.500 and the AUROC 0.500.

| Model | Pooled r | Between patients r | Within patients r | Change in same direction | AUROC, pairs within a patient |
|---|---|---|---|---|---|
| CNN | 0.200 (0.175, 0.224) | 0.244 (0.210, 0.280) | 0.079 (0.055, 0.101) | 0.516 (0.507, 0.527) | 0.558 (0.534, 0.583) |
| Beat morphology | 0.164 (0.144, 0.183) | 0.214 (0.184, 0.245) | 0.068 (0.048, 0.089) | 0.520 (0.511, 0.528) | 0.541 (0.519, 0.562) |
| PaPaGei | 0.147 (0.126, 0.167) | 0.188 (0.156, 0.223) | 0.059 (0.039, 0.078) | 0.510 (0.500, 0.519) | 0.537 (0.514, 0.559) |
| Pulse-PPG | 0.137 (0.115, 0.158) | 0.184 (0.151, 0.217) | 0.044 (0.023, 0.062) | 0.501 (0.493, 0.510) | 0.534 (0.510, 0.555) |

Most of the association lies between patients. Within patients it is weak, though above its null. Only 17.2% of subjects have both a normal and an abnormal result of a lab, on average, so few patients can show tracking in AUROC: shuffling the CNN's scores within patients lowers pooled AUROC by 0.015 overall and by 0.037 among those subjects.

As controls, the beat score's within-patient r reaches 0.96 for monitor heart rate and 0.44 for arterial mean pressure.

## Centred window and VitalDB

Exploratory. Mean gain in AUROC over the bedside base. Neither run has the CNN, which exists for MIMIC's `pre` window only. The centred window is the best 5 min around the draw, so part of its signal comes after the sample. VitalDB has its own bedside base and is exploratory because its cases were used during development.

| Run | Labs | Beat morphology | PaPaGei | Pulse-PPG | Pulse rate |
|---|---|---|---|---|---|
| MIMIC, primary window | 10 | +0.0025 | +0.0024 | +0.0020 | +0.0022 |
| MIMIC, centred window | 10 | +0.0034 | +0.0034 | +0.0016 | +0.0021 |
| VitalDB, test split | 12 | +0.0028 | -0.0015 | +0.0014 | +0.0027 |

Neither run changes the reading above. In BUN the centred window agrees, with beat morphology at +0.0112 (+0.0048, +0.0178). VitalDB does not show that gain: its largest in BUN is +0.0026.

## Files

| File | Content |
|---|---|
| [results/report.md](results/report.md) | the full report, with secondary and descriptive rows |
| [results/results.json](results/results.json) | the same numbers, for reading by code |
| [results/confirmatory_plan.json](results/confirmatory_plan.json) | the frozen plan |
| [results/tracking_report.md](results/tracking_report.md) | the within-patient tracking report |
| [results/baselines_report.md](results/baselines_report.md) | the full ladder: every base and model, AUROC, AUPRC, R² and MAE, subgroups and the null control |
| [results/baselines_centered_report.md](results/baselines_centered_report.md) | the same for the centred window, without the CNN |
| [results/vitaldb_report.md](results/vitaldb_report.md) | VitalDB, exploratory |
| [results/operating_report.md](results/operating_report.md) | size and power, by simulation on train and validation rows |

The test split may now be used by other models: `benchmark.test_opened` is `true`.
