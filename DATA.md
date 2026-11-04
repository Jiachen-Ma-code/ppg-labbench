# PPG-LabBench: data access and terms

This repository holds code and documents only. Everything else goes under `data/`, which is gitignored, and each source has its own terms. The MIT licence of this code does not cover the data.

| Source | Access | Terms | From this project |
|---|---|---|---|
| [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/) | PhysioNet credentialed access: CITI training and a signed data use agreement | PhysioNet Credentialed Health Data License 1.5.0 | labels and covariates, through the credentialed release only |
| [MIMIC-III Waveform Database Matched Subset v1.0](https://physionet.org/content/mimic3wdb-matched/1.0/) | open | Open Data Commons Open Database License v1.0 (ODbL) | PPG windows, MAP and monitor values, in the credentialed release, with the ODbL notice |
| [VitalDB](https://vitaldb.net/dataset/) | open API, `api.vitaldb.net` | stated in two ways, below | **code only**: `ppglabbench.sources.vitaldb` downloads it under its own terms, and nothing derived is redistributed |
| [MIMIC-III-Ext-PPG](https://physionet.org/content/mimic-iii-ext-ppg/) | PhysioNet credentialed access | PhysioNet Credentialed Health Data License 1.5.0 | optional link table, generated locally by `ppglabbench.link_extppg` |
| [PaPaGei-S weights](https://zenodo.org/records/13983110) | open | the record states no licence | not redistributed: download them yourself, and regenerate embeddings with `ppglabbench.features.papagei` |
| [Pulse-PPG weights](https://doi.org/10.5281/zenodo.17270930) | open | MIT | not redistributed: regenerate embeddings with `ppglabbench.features.pulseppg` |

Weights are read with `torch.load(weights_only=True)`, and no third-party code is executed. The weights of the small CNN in `ppglabbench.features.cnn` are not distributed. Licences of third-party code are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**VitalDB terms.** The `api.vitaldb.net` route comes under the terms of vitaldb.net, which states them in two ways: CC BY-NC-SA 4.0 with limits on disclosure, in the data use agreement on its
[dataset page](https://vitaldb.net/dataset/), and CC BY 4.0 in its [documentation](https://vitaldb.net/docs?documentId=General%2FTerms_and_Conditions.md). [PhysioNet's copy](https://physionet.org/content/vitaldb/1.0.0/)
is under CC BY 4.0. Read the source's terms before you use or share the data.

## The credentialed release

The MIMIC part will be released on PhysioNet as MIMIC-III-Ext-LabBench; it is not yet published. It will be one credentialed package, with the ODbL notice for what comes from the open waveforms. It will hold the
windows, the labels with their offsets, previous values, history and flags, the split file whose SHA-256 is `splits_sha256` in `config.yaml`, covariates, a data dictionary and checksums. A window's draw time comes
from LABEVENTS, so a window paired with its draw is credentialed data, although the waveforms are open. It will hold no embedding and no result.

It will have the layout of `data/`, so it can be linked into a checkout; a second command then restores the windows as the code reads them:

```bash
ln -s /path/to/the/package data
python -m ppglabbench.release --restore-windows data
python -m ppglabbench.features.beat mimic pre
.venv-torch/bin/python -m ppglabbench.features.papagei mimic pre
.venv-torch/bin/python -m ppglabbench.features.pulseppg mimic pre
python -m ppglabbench.evaluate --embeddings my_model.npz --name my_model --source mimic
```

Run the modules directly: `make features` also expects the outputs of the cohort stages S1 to S6, which the package will not repeat.

## Handling rules

- Nothing under `data/` or `logs/` is committed. Subject-level rows, IDs and raw MIMIC waveforms never go into a repository, report, manuscript or slide; only aggregates do.
- The split file holds MIMIC subject IDs, so it is distributed only with the credentialed release. This repository records its SHA-256.
- Derived MIMIC data is released through PhysioNet only, with "Ext" in its title, under credentialed access, with dates kept shifted.
- PhysioNet passwords are typed by you, with `wget --ask-password`, and are never stored in a file or script.
- Do not attempt to identify any individual, and do not send MIMIC data to third-party services.

## Citations the sources ask for

- **MIMIC-III Clinical Database:** Johnson A, Pollard T, Mark R. MIMIC-III Clinical Database (version 1.4). PhysioNet, 2016. https://doi.org/10.13026/C2XW26
- **MIMIC-III Waveform Database Matched Subset:** Moody B, Moody G, Villarroel M, Clifford GD, Silva I. MIMIC-III Waveform Database Matched Subset (version 1.0). PhysioNet, 2020. https://doi.org/10.13026/c2294b
- **Both of those also ask for:** Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. Scientific Data 3, 160035 (2016). https://doi.org/10.1038/sdata.2016.35
- **VitalDB:** Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Scientific Data 9, 279 (2022).
  https://doi.org/10.1038/s41597-022-01411-5. PhysioNet's copy also asks for: Lee H, Jung C. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients (version 1.0.0). PhysioNet, 2022.
  https://doi.org/10.13026/czw8-9p62
- **MIMIC-III-Ext-PPG,** if you use the link table: Moulaeifard M, Charlton PH, Strodthoff N. MIMIC-III-Ext-PPG: A PPG Benchmark Dataset for Cardiorespiratory Analysis (version 1.1.0). PhysioNet, 2026.
  https://doi.org/10.13026/r6k1-xt76, and Moulaeifard M, Kutscher M, Aston PJ, et al. MIMIC-III-Ext-PPG, a PPG-based Benchmark Dataset for Cardiovascular and Respiratory Signal Analysis. Scientific Data 13, 668 (2026).
  https://doi.org/10.1038/s41597-026-07335-8
- **PhysioNet,** for any resource from it: Pollard T, Moody BE, Lehman L, et al. PhysioNet as a global platform for biomedical research. Nature Health (2026). https://doi.org/10.1038/s44360-026-00096-z
- **PaPaGei:** Pillai A, Spathis D, Kawsar F, Malekzadeh M. PaPaGei: Open Foundation Models for Optical Physiological Signals. ICLR 2025. https://arxiv.org/abs/2410.20542
- **Pulse-PPG:** Saha M, Xu MA, Mao W, Neupane S, Rehg JM, Kumar S. Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Wearable Applications across Lab and Field Settings. Proceedings of the ACM on
  Interactive, Mobile, Wearable and Ubiquitous Technologies 9(3), 1-35 (2025).
