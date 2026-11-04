# PPG-LabBench: results

The MIMIC test split was evaluated once, on 2026-11-03, with the frozen plan (SHA-256 `8b7c5d617da090eb1affb45ce7bc9caf449808a67d2f26ce94274adf49f1be64`, the same value as `benchmark.plan_sha256` in `config.yaml`). The run finished on its first attempt, with no analysis failed. All full results are in [results/](results/).

**Order of events.** The plan and its hash were committed in the development repository before the evaluation. This repository and its archive were made public after the evaluation, so their dates do not show that order.

## In brief

Models: pulse-beat features, two pretrained encoders and a small end-to-end CNN, with one model picked for each lab on the validation split. Gains are in AUROC over a base without PPG. Margin m = +0.01.

- **On average over the 10 labs, these models add less than +0.01 over what is known at the bedside (`now`):** mean gain +0.0053 (95% interval +0.0023, +0.0081), upper bound +0.0095.
- **With previous lab values in the base (`full`), almost nothing remains:** mean gain +0.0002, upper bound +0.0017.
- **Over the smaller `monitor` base, a bound of +0.01 is not shown:** mean gain +0.0106, upper bound +0.0154. The chain stopped there, so `acuity` was not tested.
- **A mean can hide one lab.** Urea nitrogen (BUN) shows a gain above 0 over `now`, +0.0138 (simultaneous interval +0.0032, +0.0245), not shown to exceed +0.01, so this is no practical claim. In bicarbonate and calcium the score holds information beyond the base, with no gain in discrimination shown. The other seven labs clear no level.
- **Named value hypotheses.** Haemoglobin clears the first level only, and with PaPaGei, which was pretrained on these cohorts, so it may be optimistic. [PROTOCOL.md](PROTOCOL.md) asks for the other candidates beside such a pick; this report does not print them. Lactate clears no level.

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

## Named hypotheses, the lab value over `now`

| Lab | Model | n | Subjects | R² base | ΔR² | SE | Partial R² | Simultaneous interval | z1 | z2 | Levels cleared |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lactate | beat | 373 | 196 | 0.239 | +0.0188 | 0.0127 | +0.0248 | -0.0167, +0.0544 | 2.26 | 1.49 | 0 |
| haemoglobin | papagei | 2002 | 715 | 0.222 | +0.0134 | 0.0054 | +0.0172 | -0.0016, +0.0285 | 4.41 | 2.51 | 1 |

## Files

| File | Content |
|---|---|
| [results/report.md](results/report.md) | the full report, with secondary and descriptive rows |
| [results/results.json](results/results.json) | the same numbers, for reading by code |
| [results/confirmatory_plan.json](results/confirmatory_plan.json) | the frozen plan |
| [results/operating_report.md](results/operating_report.md) | size and power, by simulation on train and validation rows |

The test split may now be used by other models: `benchmark.test_opened` is `true`.
