# Rerun the benchmark in dependency order. `make help` lists the targets.
#
# A step runs only if its output is missing, so nothing restricted is rebuilt or overwritten by accident: delete a step's
# output to run it again. Credentialed downloads are not automated: a step stops with a message if what it needs is missing.
# Nothing here opens the test split: PROTOCOL.md gives that as a separate manual command.
#
# Two environments, because torch 2.2 needs numpy < 2: override with `make <target> PYTHON=... PYTHON_TORCH=...`.
PYTHON       ?= .venv/bin/python
PYTHON_TORCH ?= .venv-torch/bin/python

CLINICAL := data/clinical
TABLES   := LABEVENTS PATIENTS ADMISSIONS DIAGNOSES_ICD PROCEDURES_ICD INPUTEVENTS_MV INPUTEVENTS_CV
DEV_LIST := data/s5_groupB_partial/groupB_draws.csv
SPLITS   := data/benchmark/mimic_splits.csv
FEAT     := data/benchmark/features
RESULTS  := data/benchmark/results

PAPAGEI_WEIGHTS  := data/external/papagei/weights/papagei_s.pt
PULSEPPG_WEIGHTS := data/external/pulseppg/weights/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl

MSG_CLINICAL := MIMIC-III Clinical v1.4 needs PhysioNet credentialed access and its DUA, and is not downloaded here. Put the tables in $(CLINICAL)/: see Data access in README.md.
MSG_DEV      := Either file gives the development subjects kept out of the test split. No stage rebuilds them and neither is distributed yet: see Data access in README.md.
MSG_WEIGHTS  := Download the model weights to this path: see Data access in README.md.
MSG_TORCH    := Create the torch environment or pass PYTHON_TORCH=...: see Install in README.md.

# $(call require,<files>,<message>): stop with the message if a file is missing.
require = @for f in $(1); do [ -e "$$f" ] || { printf 'Missing: %s\n%s\n' "$$f" "$(2)" >&2; exit 1; }; done
need_dev = @[ -e "$(DEV_LIST)" ] || [ -e "$(SPLITS)" ] || { printf 'Missing: %s or %s\n%s\n' "$(DEV_LIST)" "$(SPLITS)" "$(MSG_DEV)" >&2; exit 1; }
need_torch = @$(PYTHON_TORCH) -c "import torch, ppglabbench" || { printf 'Cannot run torch and ppglabbench with %s\n%s\n' "$(PYTHON_TORCH)" "$(MSG_TORCH)" >&2; exit 1; }

S1     := data/s1_catalog/manifest.json
S2     := data/s2_groupA/manifest.json
S3     := data/s3_windows/manifest.json
S4     := data/s4_sqi/manifest.json
S5     := data/s5_groupB/manifest.json
S6     := data/s6_dataset/manifest.json
LABELS := data/benchmark/manifest.json
VDB    := data/external/vitaldb/index.csv
PLAN   := $(RESULTS)/confirmatory_plan.json
DRYRUN := $(RESULTS)/confirmatory_dryrun/manifest.json

WINDOWS := pre centered
FEATURE_FILES := $(FEAT)/mimic_cov/manifest.json \
	$(foreach w,$(WINDOWS),$(FEAT)/$(w)/mimic/manifest.json $(FEAT)/$(w)/mimic_abp/manifest.json $(FEAT)/$(w)/mimic_num/manifest.json \
		$(FEAT)/$(w)/mimic/papagei.npz $(FEAT)/$(w)/mimic/pulseppg.npz) \
	$(FEAT)/vitaldb/manifest.json $(FEAT)/vitaldb/papagei.npz $(FEAT)/vitaldb/pulseppg.npz
BASELINE_FILES := $(RESULTS)/mimic_val/manifest.json $(RESULTS)/mimic_val_centered/manifest.json \
	$(RESULTS)/vitaldb_val/manifest.json $(RESULTS)/mimic_val/positive_control.csv

.DEFAULT_GOAL := help
.NOTPARALLEL:
.PRECIOUS: $(VDB)    # an interrupted download keeps its index, and rerunning the module by hand resumes it
.PHONY: help check-python labels features baselines plan dryrun test demo all

help: ## Show this help
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/^([a-z]+):.*## /\1|/' | awk -F'|' '{printf "  %-10s %s\n", $$1, $$2}'

check-python:
	@$(PYTHON) -c "import ppglabbench" || { printf 'Cannot import ppglabbench with %s\nCreate the environment or pass PYTHON=...: see Install in README.md.\n' "$(PYTHON)" >&2; exit 1; }

labels: check-python $(LABELS) ## MIMIC cohort S1-S6, multi-lab labels and frozen splits

features: check-python $(FEATURE_FILES) ## Beat features, MAP, monitor values, covariates and embeddings for MIMIC and VitalDB

baselines: check-python $(BASELINE_FILES) ## Reference baselines and positive control on the validation split

plan: check-python $(PLAN) ## Confirmatory plan from train and val, kept once written: delete it to plan again

dryrun: check-python $(DRYRUN) ## The confirmatory analysis with val standing in for test

test: check-python ## Unit tests, no restricted data needed
	$(PYTHON) -m unittest

demo: check-python ## Check an install: evaluate on generated data, one planted signal and one pure noise, no restricted data needed
	$(PYTHON) -m ppglabbench.evaluate --demo

all: labels features baselines plan dryrun test ## Labels through dryrun and the unit tests, up to but not including the test split

# ------------------------------------------------------------------------------------------- MIMIC cohort and labels

$(S1):
	$(call require,$(TABLES:%=$(CLINICAL)/%.csv.gz),$(MSG_CLINICAL))
	$(need_dev)
	$(PYTHON) -m ppglabbench.pipeline.s1_scan_waveforms

$(S2): | $(S1)
	$(call require,$(CLINICAL)/LABEVENTS.csv.gz,$(MSG_CLINICAL))
	$(PYTHON) -m ppglabbench.pipeline.s2_match_groupA

$(S3): | $(S2)
	$(PYTHON) -m ppglabbench.pipeline.s3_fetch_windows

$(S4): | $(S3)
	$(PYTHON) -m ppglabbench.pipeline.s4_score_sqi

$(S5): | $(S4)
	$(PYTHON) -m ppglabbench.pipeline.s5_select_groupB

$(S6): | $(S5)
	$(call require,$(TABLES:%=$(CLINICAL)/%.csv.gz),$(MSG_CLINICAL))
	$(PYTHON) -m ppglabbench.pipeline.s6_build_dataset

$(LABELS): | $(S6)
	$(call require,$(CLINICAL)/LABEVENTS.csv.gz,$(MSG_CLINICAL))
	$(need_dev)
	$(PYTHON) -m ppglabbench.labels

# ------------------------------------------------------------------------------------------- features

$(FEAT)/%/mimic/manifest.json: | $(LABELS)
	$(PYTHON) -m ppglabbench.features.beat mimic $*

$(FEAT)/%/mimic_abp/manifest.json: | $(LABELS)
	$(PYTHON) -m ppglabbench.features.abp_map --window $*

$(FEAT)/%/mimic_num/manifest.json: | $(LABELS)
	$(PYTHON) -m ppglabbench.features.numerics --window $*

$(FEAT)/mimic_cov/manifest.json: | $(LABELS)
	$(call require,$(TABLES:%=$(CLINICAL)/%.csv.gz),$(MSG_CLINICAL))
	$(PYTHON) -m ppglabbench.features.covariates

$(FEAT)/%/mimic/papagei.npz: | $(LABELS)
	$(need_torch)
	$(call require,$(PAPAGEI_WEIGHTS),$(MSG_WEIGHTS))
	$(PYTHON_TORCH) -m ppglabbench.features.papagei mimic $*

$(FEAT)/%/mimic/pulseppg.npz: | $(LABELS)
	$(need_torch)
	$(call require,$(PULSEPPG_WEIGHTS),$(MSG_WEIGHTS))
	$(PYTHON_TORCH) -m ppglabbench.features.pulseppg mimic $*

$(VDB):
	$(PYTHON) -m ppglabbench.sources.vitaldb

$(FEAT)/vitaldb/manifest.json: | $(VDB)
	$(PYTHON) -m ppglabbench.features.beat vitaldb

$(FEAT)/vitaldb/papagei.npz: | $(FEAT)/vitaldb/manifest.json
	$(need_torch)
	$(call require,$(PAPAGEI_WEIGHTS),$(MSG_WEIGHTS))
	$(PYTHON_TORCH) -m ppglabbench.features.papagei vitaldb

$(FEAT)/vitaldb/pulseppg.npz: | $(FEAT)/vitaldb/manifest.json
	$(need_torch)
	$(call require,$(PULSEPPG_WEIGHTS),$(MSG_WEIGHTS))
	$(PYTHON_TORCH) -m ppglabbench.features.pulseppg vitaldb

# ------------------------------------------------------------------------------------------- analyses, val only

$(RESULTS)/mimic_val/manifest.json: | $(FEATURE_FILES)
	$(PYTHON) -m ppglabbench.baselines mimic

$(RESULTS)/mimic_val_centered/manifest.json: | $(FEATURE_FILES)
	$(PYTHON) -m ppglabbench.baselines mimic --window centered

$(RESULTS)/vitaldb_val/manifest.json: | $(FEATURE_FILES)
	$(PYTHON) -m ppglabbench.baselines vitaldb

$(RESULTS)/mimic_val/positive_control.csv: | $(RESULTS)/mimic_val/manifest.json
	$(PYTHON) -m ppglabbench.positive_control

$(PLAN): | $(BASELINE_FILES)
	$(PYTHON) -m ppglabbench.confirmatory plan

$(DRYRUN): | $(PLAN)
	$(PYTHON) -m ppglabbench.confirmatory dry-run
