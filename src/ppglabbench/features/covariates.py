"""Benchmark covariates known at the draw, from tables on disk (MIMIC only; never PPG-derived).

The acuity base leaves room for PPG to re-read what the chart already shows, so these feed the `now` base. Treatments:
at the draw (running, or charted in the preceding hour) and in the prior 24 h, from INPUTEVENTS_MV (not cancelled or rewritten)
and INPUTEVENTS_CV (positive rate or amount); ITEMIDs in features/treatment_items.yaml. Pressors: number of agents at the draw
and epinephrine, which raises lactate without hypoperfusion. Context: log hours since admission, MetaVision era, elective
admission, transfer, ethnicity group. Pressor doses are left out: CareVue units are inconsistent.
History: from the subject's admissions that ended before the draw, never the window's own: their number (log1p), days since the last discharge (log1p,
missing without one) and flags for ICD-9 groups coded in any of them (DIAGNOSES_ICD). Codes of the current stay are assigned at discharge, so they are not read.

Usage:  python -m ppglabbench.features.covariates
Output: data/benchmark/features/mimic_cov/{covariates.csv, manifest.json}  (DUA data: not committed)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ppglabbench.common import CFG, CLINICAL, DATA, extract_rows, read_table, write_manifest
from ppglabbench.pipeline.s6_build_dataset import DX

OUT = DATA / "benchmark" / "features" / "mimic_cov"
ITEMS = yaml.safe_load((Path(__file__).parent / "treatment_items.yaml").read_text())
AT_DRAW = ("sedation", "paralytic", "vasodilator", "rate_control", "insulin", "heparin", "potassium")
PRIOR_24H = ("prbc", "ffp", "platelets", "albumin", "dialysis_fluid", "bicarbonate", "calcium")
PRESSORS = {"norepinephrine": ([221906], [30047, 30120]), "epinephrine": ([221289], [30044, 30119, 30309]),
            "phenylephrine": ([221749], [30127, 30128]), "vasopressin": ([222315], [30051]),
            "dopamine": ([221662], [30043, 30307]), "dobutamine": ([221653], [30042, 30306]), "milrinone": ([221986], [30125])}
PRIOR_DX = dict(prior_ckd=DX["dx_ckd"], prior_esrd=DX["dx_esrd"], prior_aki=DX["dx_aki"], prior_af=DX["dx_af"], prior_htn=DX["dx_htn"], prior_dm=DX["dx_dm"],
                prior_chf=DX["dx_chf"], prior_liver=("570", "571", "572", "573"))
HOUR = np.timedelta64(1, "h")


def ethnicity_group(x):
    x = str(x)
    if x.startswith("WHITE") or x == "PORTUGUESE":
        return "white"
    if x.startswith("BLACK") or x == "CARIBBEAN ISLAND":
        return "black"
    if x.startswith(("HISPANIC", "SOUTH AMERICAN")):
        return "hispanic"
    if x.startswith("ASIAN"):
        return "asian"
    return "unknown" if x.startswith(("UNKNOWN", "UNABLE", "PATIENT DECLINED")) else "other"


def admission_history(d, adm, dx):
    """One row per window of d (subject_id, hadm_id, charttime), from adm (subject_id, hadm_id, dischtime) and dx (hadm_id, icd9_code as text): what the earlier
    admissions of the subject say. Earlier means ended before the draw and not the window's own, so the codes of the current stay never enter, even for a draw
    charted after that stay's discharge."""
    past = d[["subject_id", "hadm_id", "charttime"]].assign(row=np.arange(len(d))).merge(adm[["subject_id", "hadm_id", "dischtime"]], on="subject_id", suffixes=("", "_past"))
    past = past[(past.dischtime < past.charttime) & (past.hadm_id_past != past.hadm_id)]
    f = pd.DataFrame(dict(prior_admissions=np.log1p(np.bincount(past.row, minlength=len(d)))), index=d.index)
    last = past.groupby("row").dischtime.max()
    days = np.full(len(d), np.nan)
    days[last.index] = (d.charttime.to_numpy()[last.index] - last.to_numpy()) / np.timedelta64(1, "D")
    f["days_since_discharge"] = np.log1p(days)
    for col, prefixes in PRIOR_DX.items():
        coded = set(dx.hadm_id[dx.icd9_code.str.startswith(prefixes)])
        f[col] = np.isin(np.arange(len(d)), past.row[past.hadm_id_past.isin(coded)]).astype(float)
    return f


def main():
    d = pd.read_csv(DATA / "benchmark" / "mimic_labels.csv", usecols=["draw_id", "subject_id", "hadm_id", "charttime"],
                    parse_dates=["charttime"])
    mv_items = [i for v in ITEMS["metavision"].values() for i in v] + [i for m, _ in PRESSORS.values() for i in m]
    cv_items = [i for v in ITEMS["carevue"].values() for i in v] + [i for _, c in PRESSORS.values() for i in c]
    mv = extract_rows("INPUTEVENTS_MV", "treat_mv_v1",
                      lambda c: c.itemid.isin(mv_items) & (c.cancelreason == 0) & (c.statusdescription != "Rewritten"),
                      {"subject_id", "itemid", "starttime", "endtime", "cancelreason", "statusdescription"},
                      dates=("starttime", "endtime"))
    cv = extract_rows("INPUTEVENTS_CV", "treat_cv_v1", lambda c: c.itemid.isin(cv_items) & ((c.rate > 0) | (c.amount > 0)),
                      {"subject_id", "itemid", "charttime", "rate", "amount"}, dates=("charttime",))
    t, by_subject = d.charttime.to_numpy(), d.groupby("subject_id").indices

    def given(mv_ids, cv_ids, hours):
        """Any such treatment running, or stopped or charted within `hours`, at each draw; never one that starts after it."""
        a = mv[mv.itemid.isin(mv_ids)]
        b = cv[cv.itemid.isin(cv_ids)]
        ev = pd.concat([pd.DataFrame(dict(s=a.subject_id, start=a.starttime, end=a.endtime)),
                        pd.DataFrame(dict(s=b.subject_id, start=b.charttime, end=b.charttime))])
        out = np.zeros(len(d), bool)
        for s, g in ev.groupby("s"):
            ix = by_subject.get(s)
            if ix is not None:
                tt = t[ix][:, None]
                out[ix] = ((g.start.to_numpy() <= tt) & (tt <= g.end.to_numpy() + hours * HOUR)).any(1)
        return out.astype(float)

    f = d[["draw_id"]].copy()
    for g in AT_DRAW:
        f[f"on_{g}"] = given(ITEMS["metavision"][g], ITEMS["carevue"][g], 1)
    for g in PRIOR_24H:
        f[f"{g}_24h"] = given(ITEMS["metavision"][g], ITEMS["carevue"][g], 24)
    agents = {k: given(m, c, 1) for k, (m, c) in PRESSORS.items()}
    f["n_pressors"] = np.sum(list(agents.values()), axis=0)
    f["epi_on"] = agents["epinephrine"]

    admissions = read_table("ADMISSIONS", dates=("admittime", "dischtime"))
    adm = admissions.set_index("hadm_id")
    f["hosp_h"] = np.log1p(((d.charttime - d.hadm_id.map(adm.admittime)).dt.total_seconds() / 3600).clip(lower=0)).to_numpy()
    f["elective"] = (d.hadm_id.map(adm.admission_type) == "ELECTIVE").astype(float).to_numpy()
    f["transfer"] = d.hadm_id.map(adm.admission_location).fillna("").str.startswith("TRANSFER").astype(float).to_numpy()
    eth = d.hadm_id.map(adm.ethnicity).map(ethnicity_group)
    for g in ("black", "hispanic", "asian", "other", "unknown"):
        f[f"eth_{g}"] = (eth == g).astype(float).to_numpy()
    mv_hadm = set()
    for ch in pd.read_csv(CLINICAL / "INPUTEVENTS_MV.csv.gz", chunksize=2_000_000, usecols=["HADM_ID"]):
        mv_hadm |= set(ch.HADM_ID.unique())
    f["era_mv"] = d.hadm_id.isin(mv_hadm).astype(float).to_numpy()
    history = admission_history(d, admissions, read_table("DIAGNOSES_ICD").astype({"icd9_code": str}))
    f = pd.concat([f, history], axis=1)

    OUT.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT / "covariates.csv", index=False)
    write_manifest(OUT, "covariates", dict(at_draw=AT_DRAW, prior_24h=PRIOR_24H, pressors=list(PRESSORS), prior_dx=PRIOR_DX),
                   dict(windows=len(f), columns=[c for c in f.columns if c != "draw_id"]))


if __name__ == "__main__":
    main()
