# PPG-LabBench

A benchmark of photoplethysmography (PPG) windows paired with blood laboratory values, from MIMIC-III (intensive care) and VitalDB (operating room). It asks how much the pulse waveform adds to what is
already known at the bedside. The code builds the windows and labels, computes PPG features and embeddings, fits reference baselines next to PPG models, and runs an evaluation protocol with a permuted-PPG
null control, a positive control and a test split that is evaluated once.

| Document | What it holds |
|---|---|
| [SPEC.md](SPEC.md) | the benchmark: tasks, labels, windows, bases, models, metrics, required reporting |
| [PROTOCOL.md](PROTOCOL.md) | the confirmatory analysis of the test split and its limits |
| [DATA.md](DATA.md) | access and terms of each data source, and the credentialed release |
| [RESULTS.md](RESULTS.md) | results of the test split |

This repository holds code and documents only, no data.

## Install

Python 3.9 or later. Install in editable mode: the code finds `config.yaml` and `data/` relative to the repository root.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
make test    # unit tests, no restricted data needed
make demo    # evaluates two generated embeddings end to end, no restricted data needed
```

The PaPaGei and Pulse-PPG embeddings and the small end-to-end CNN (`ppglabbench.features.cnn`) need torch, which `pip install -e ".[embed]"` adds to the same environment. That fails where pip can only find torch 2.2, as on
Intel Macs: torch 2.2 needs numpy < 2, and `torch.from_numpy` then raises an error. A second environment on torch 2.2.2 works, and is what the Makefile expects. The CNN fits there; its `prepare` and `assemble` steps run in
`.venv`, as its docstring lists:

```bash
python3 -m venv .venv-torch
.venv-torch/bin/pip install "torch==2.2.2" "numpy<2" pandas scipy
.venv-torch/bin/pip install --no-deps -e .
```

The CNN is trained on the MIMIC train split, and its weights are not distributed.

## Data access

Everything goes under `data/`, which is gitignored. Terms and handling rules are in [DATA.md](DATA.md).

| Source | Access | Where it goes |
|---|---|---|
| [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/) | PhysioNet credentialed access: CITI training and a signed DUA. Not automated | `data/clinical/` |
| [MIMIC-III Waveform Database Matched Subset v1.0](https://physionet.org/content/mimic3wdb-matched/1.0/) | Open on PhysioNet, ODbL v1.0. Streamed by stages S1 and S3 | `data/s3_windows/` (derived windows) |
| VitalDB | Through its API, `api.vitaldb.net`, by `python -m ppglabbench.sources.vitaldb` | `data/external/vitaldb/` |
| PaPaGei-S weights | [zenodo.org/records/13983110](https://zenodo.org/records/13983110) | `data/external/papagei/weights/papagei_s.pt` |
| Pulse-PPG weights | [doi.org/10.5281/zenodo.17270930](https://doi.org/10.5281/zenodo.17270930) | unzip `pulseppg_model_weights.zip` in `data/external/pulseppg/weights/`, to give `data/external/pulseppg/weights/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl` |

Download the clinical tables yourself once you hold PhysioNet access, typing your own password:

```bash
for t in LABEVENTS PATIENTS ADMISSIONS DIAGNOSES_ICD PROCEDURES_ICD INPUTEVENTS_MV INPUTEVENTS_CV; do
  wget -N -c --user <physionet user> --ask-password -P data/clinical https://physionet.org/files/mimiciii/1.4/$t.csv.gz
done
```

**Development subjects.** The subjects used to develop the methods are kept out of the test split. The code reads them from the folder named by `benchmark.dev_exclusion` in `config.yaml`. Where that folder is absent, as
it will be for you, the code reads them from the `dev_set` column of `data/benchmark/mimic_splits.csv`. That file holds subject IDs, so it will come only with the credentialed release, which is not yet published
([DATA.md](DATA.md)). The MIMIC steps stop without it, and the frozen splits, whose hash is `splits_sha256` in `config.yaml`, cannot be reproduced without it.

## Rerun

```bash
make help    # the targets
make all     # labels through dryrun and the tests, in that order
```

| Target | What it runs |
|---|---|
| `labels` | MIMIC cohort S1 to S6 ([stages](src/ppglabbench/pipeline/README.md)), multi-lab labels, frozen splits |
| `features` | beat features, MAP, monitor values, covariates and embeddings for both windows, and the VitalDB download, beat features and embeddings |
| `baselines` | reference baselines on the validation split, for MIMIC and VitalDB, and the positive control |
| `plan` | the confirmatory plan: model picks, training-variance terms and the margin, from train and validation rows |
| `dryrun` | the confirmatory analysis with the validation split standing in for test |
| `test` | unit tests |
| `demo` | `ppglabbench.evaluate` on generated data, to check an install |
| `all` | `labels` through `dryrun` and `test`, up to but not including the test split |

A step runs only if its output is missing; delete the output to run it again. A step stops with a message if something it needs is missing. S1 takes about 40 minutes and S3 about 15 hours at PhysioNet's download
rate; both resume after an interruption, and so do the MAP and monitor downloads. An interrupted VitalDB download leaves its index behind, which `make` takes as done, so finish it by hand with
`python -m ppglabbench.sources.vitaldb`. Choose the environments with `make <target> PYTHON=... PYTHON_TORCH=...`; the defaults are `.venv/bin/python` and `.venv-torch/bin/python`.

Each step is also a module you can run by hand, from the repository root, with the usage in its docstring:

```bash
python -m ppglabbench.pipeline.s1_scan_waveforms --limit 300    # a small test batch
python -m ppglabbench.labels
python -m ppglabbench.features.beat mimic pre
python -m ppglabbench.baselines mimic --window centered
python -m ppglabbench.tune_beat                                 # prints the round count kept in config.yaml
```

VitalDB alone needs no PhysioNet access:

```bash
python -m ppglabbench.sources.vitaldb
python -m ppglabbench.features.beat vitaldb
.venv-torch/bin/python -m ppglabbench.features.papagei vitaldb
.venv-torch/bin/python -m ppglabbench.features.pulseppg vitaldb
python -m ppglabbench.baselines vitaldb
```

## Evaluate your own embeddings

`ppglabbench.evaluate` takes your own PPG embeddings through the benchmark and gives the whole ladder for your model:

```bash
python -m ppglabbench.evaluate --embeddings my_model.npz --name my_model --source {mimic,vitaldb} [--eval val] [--window pre] [--perms 20]
```

To check an install first, `make demo` runs the same path on generated data, with one embedding carrying a planted signal and one of pure noise, and prints both ladders. It reads no restricted data.

**Input.** An `.npz` with `ids` and `emb` [n, d], saved as plain arrays:

```python
np.savez("my_model.npz", ids=np.asarray(ids, dtype=str), emb=np.asarray(emb, dtype=float))
```

`ids` are `draw_id` strings for MIMIC and integer `caseid` for VitalDB. A row of NaN means the window could not be embedded. The run stops with a message on missing keys, wrong dimensions, duplicate ids, ids outside the
cohort (`--drop-extra-ids` ignores those) or fewer than `--min-coverage` (default 50%) of train or evaluation windows embedded. Windows without an embedding go to the probe's imputer, and the share embedded is reported.
`ppglabbench.features.papagei` shows how to read the windows from `data/`.

**Needs.** The cohort, so run `make features` first, the VitalDB commands above, or the commands in [DATA.md](DATA.md) on the credentialed release: a window is in the cohort if it has beat features and a PaPaGei embedding and
meets the source's quality rules.

**Method.** The code of `ppglabbench.baselines`: the probe (imputer, scaler, PCA to 64 dimensions or to d if smaller, then logistic or ridge), the stack on each base, the subject bootstrap and the subject-block null control.
The benchmark's own runs use `--perms 100`; the default of 20 here gives a coarser null range. `--subsets` also computes the sensitivity and subgroup rows.

**Output.** In `--out`, by default `data/evaluate/<name>_<source>_<eval>/`: `results.csv`, `null.csv`, `manifest.json` and `report.md`. The report gives, per lab, the model alone, its increment over each rung with a 95%
subject-bootstrap interval, the null range and, if a results file exists for the same source, split and window, the rows of beat, PaPaGei and Pulse-PPG. It is not corrected for the number of labs or rungs and draws no
conclusions.

**Test split.** `--eval test` refuses until `benchmark.test_opened` is `true` in `config.yaml`. The benchmark's one confirmatory evaluation of the test split comes before any other use of it, and that marker is set in a commit
once the evaluation is done.

## The test split

The confirmatory evaluation of the MIMIC test split has been run once: see [RESULTS.md](RESULTS.md). [PROTOCOL.md](PROTOCOL.md) gives its procedure. `benchmark.test_opened` is now `true`, so other models may use the split.

## Layout

```
src/ppglabbench/        the package
  common.py             paths, config, shared helpers
  pipeline/             MIMIC cohort stages S1 to S6
  sources/vitaldb.py    VitalDB download
  features/             beat features, MAP, monitor values, covariates, PaPaGei and Pulse-PPG embeddings, the small CNN
  labels.py             multi-lab labels, previous values, frozen splits
  baselines.py          reference baselines, PPG increments, permuted-PPG null
  tune_beat.py          boosting rounds of the beat model, on train only
  positive_control.py   recovery of a synthetic signal of known strength
  confirmatory.py       confirmatory analysis: plan, dry run, the one test evaluation
  tracking.py           within-patient and between-patient summaries, a key secondary and descriptive
  operating.py          size and power of the confirmatory procedure, by plasmode on train and validation rows
  evaluate.py           the ladder for your own embeddings, and a demo on generated data
  release.py            builds the credentialed PhysioNet package
  link_extppg.py        optional link table to the segments of MIMIC-III-Ext-PPG
tests/                  unit tests on synthetic data
config.yaml             every parameter of the pipeline and benchmark
data/, logs/            outputs (gitignored, not in the repository)
```

## Citation and licence

Citation details are in [CITATION.cff](CITATION.cff), and the citations the data sources ask for are in [DATA.md](DATA.md). The code is under the [MIT licence](LICENSE). It does not cover the data: each source has its own
terms, listed in [DATA.md](DATA.md). Notices for third-party code are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
