"""S6: labels, covariates and exclusions for every group B draw.
Subject splits are not made here: the frozen splits come from ppglabbench.labels.

Labels (per draw):
  egfr          CKD-EPI 2021 (race-free), from creatinine + age + sex at the draw
  ckd_egfr      egfr < egfr_ckd_threshold
  aki_kdigo     KDIGO creatinine criterion: rise >= 0.3 mg/dL over the lowest value in the prior 48 h,
                or >= 1.5x the lowest value in the prior 7 days (all MIMIC-III creatinine of the subject)
Covariates: age, sex, Hb, BUN (nearest within ± lab_window_h), vasopressor at the draw, emergency / urgent
  admission (ADMISSIONS.admission_type, known at admission), and
  admission diagnoses (ICD-9): CKD 585, ESRD (5855, 5856, hypertensive stage V or ESRD), AKI 584, AF 42731, hypertension 401-405,
  diabetes 250, heart failure 428.
Exclusion flag (rows are kept, flagged): dialysis = ESRD, dialysis status (V451, V56) or a dialysis
  procedure (39.95 hemodialysis, 54.98 peritoneal) in the admission, or ESRD or dialysis status coded in an
  admission that ended before the draw. Creatinine of dialysed patients
  reflects the dialysis schedule, not kidney function; the primary analysis drops them.
Diagnoses are per admission, not time-stamped: a CKD code is a property of the stay, not the draw.

Needs in data/clinical/: PATIENTS, ADMISSIONS, DIAGNOSES_ICD (required), PROCEDURES_ICD,
INPUTEVENTS_MV, INPUTEVENTS_CV (optional; their covariates are NaN without them).

Usage:
    python -m ppglabbench.pipeline.s6_build_dataset                    # group B from data/s5_groupB
    python -m ppglabbench.pipeline.s6_build_dataset --groupb s5_groupB_s06
Output:
    data/s6_dataset[_<groupB tag>]/dataset.csv   one row per group B draw; x_row = row in groupB_X.npy
"""
import argparse
import sys

import numpy as np
import pandas as pd

from ppglabbench.common import CFG, CLINICAL, DATA, S5, S6, extract_rows, load_creatinine, read_table, write_manifest

C = CFG["s6_dataset"]
REQUIRED = ["PATIENTS", "ADMISSIONS", "DIAGNOSES_ICD"]
OPTIONAL = ["PROCEDURES_ICD", "INPUTEVENTS_MV", "INPUTEVENTS_CV"]
DX = {"dx_ckd": ("585",),
      "dx_esrd": ("5855", "5856", "40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493"), "dx_aki": ("584",), "dx_af": ("42731",),
      "dx_htn": ("401", "402", "403", "404", "405"), "dx_dm": ("250",), "dx_chf": ("428",),
      "dx_dialysis_status": ("V451", "V56")}
DIALYSIS_PROC = {"3995", "5498"}


def egfr_ckdepi2021(scr, age, female):
    """CKD-EPI 2021 creatinine equation (Inker et al., NEJM 2021), mL/min/1.73 m2."""
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    r = scr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.200 * 0.9938 ** age * np.where(female, 1.012, 1.0)


def kdigo_aki(draws, creat):
    """KDIGO creatinine criterion at each draw, using all of the subject's creatinine values."""
    out = np.zeros(len(draws), dtype=bool)
    by_subj = {s: g.sort_values("charttime") for s, g in creat[creat.subject_id.isin(draws.subject_id)].groupby("subject_id")}
    for i, d in enumerate(draws.itertuples()):
        g = by_subj.get(d.subject_id)
        if g is None:
            continue
        prior = g[(g.charttime < d.charttime) & (g.charttime >= d.charttime - pd.Timedelta(days=7))]
        p48 = prior[prior.charttime >= d.charttime - pd.Timedelta(hours=48)]
        out[i] = (len(p48) and d.creat_mgdl - p48.creat_mgdl.min() >= 0.3) or \
                 (len(prior) and d.creat_mgdl >= 1.5 * prior.creat_mgdl.min())
    return out


def nearest_lab(draws, lab, col):
    """Value of the lab nearest in time to each draw, within ± lab_window_h."""
    lab = lab.sort_values("charttime")
    d = draws[["subject_id", "charttime"]].reset_index().sort_values("charttime")
    m = pd.merge_asof(d, lab[["subject_id", "charttime", "valuenum"]], on="charttime", by="subject_id",
                      direction="nearest", tolerance=pd.Timedelta(hours=C["lab_window_h"]))
    return m.set_index("index").valuenum.reindex(draws.index).rename(col)


def vasopressor_on(draws):
    """True / False if the vasopressor tables exist (either), NaN otherwise."""
    have_mv, have_cv = [(CLINICAL / f"{t}.csv.gz").exists() for t in ("INPUTEVENTS_MV", "INPUTEVENTS_CV")]
    if not (have_mv or have_cv):
        return pd.Series(np.nan, index=draws.index)
    on = pd.Series(False, index=draws.index)
    if have_mv:
        mv = extract_rows("INPUTEVENTS_MV", "vaso_mv_v3",
                          lambda c: c.itemid.isin(C["vaso_mv"]) & (c.cancelreason == 0) & (c.statusdescription != "Rewritten"),
                          {"subject_id", "itemid", "starttime", "endtime", "cancelreason", "statusdescription"},
                          dates=("starttime", "endtime"))
        for s, g in mv.groupby("subject_id"):
            ix = draws.index[draws.subject_id == s]
            t = draws.loc[ix, "charttime"].to_numpy()[:, None]
            on[ix] |= ((g.starttime.to_numpy() <= t) & (t <= g.endtime.to_numpy())).any(1)
    if have_cv:
        cv = extract_rows("INPUTEVENTS_CV", "vaso_cv_v3",
                          lambda c: c.itemid.isin(C["vaso_cv"]) & ((c.rate > 0) | (c.amount > 0)),
                          {"subject_id", "itemid", "charttime", "rate", "amount"}, dates=("charttime",))
        w = np.timedelta64(C["vaso_cv_window_h"], "h")
        for s, g in cv.groupby("subject_id"):
            ix = draws.index[draws.subject_id == s]
            t = draws.loc[ix, "charttime"].to_numpy()[:, None]
            dt = t - g.charttime.to_numpy()
            on[ix] |= ((dt >= np.timedelta64(0)) & (dt <= w)).any(1)
    return on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groupb", default=S5.name, help="group B folder under data/")
    args = ap.parse_args()
    missing = [t for t in REQUIRED if not (CLINICAL / f"{t}.csv.gz").exists()]
    if missing:
        sys.exit("missing in data/clinical/: " + ", ".join(missing) + "\ndownload with:\n" + "\n".join(
            f"  wget -N -c --user <physionet user> --ask-password -P data/clinical "
            f"https://physionet.org/files/mimiciii/1.4/{t}.csv.gz" for t in missing + OPTIONAL))
    gb_dir = DATA / args.groupb
    out = S6.with_name(S6.name + args.groupb.removeprefix(S5.name))
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(gb_dir / "groupB_draws.csv", parse_dates=["charttime"])
    d["x_row"] = np.arange(len(d))
    n_in = len(d)

    pat = read_table("PATIENTS", dates=("dob",))
    d = d.merge(pat[["subject_id", "gender", "dob"]], on="subject_id", how="left")
    # MIMIC-III shifts DOB of patients > 89 so that age appears ~300; use the published median 91.4
    age = (d.charttime.dt.year - d.dob.dt.year) - ((d.charttime.dt.month * 100 + d.charttime.dt.day) <
                                                   (d.dob.dt.month * 100 + d.dob.dt.day))
    d["age"] = age.where(age < 90, 91.4).astype(float)
    d["female"] = d.gender == "F"
    d = d.drop(columns=["gender", "dob"])
    d = d[d.age >= 18].reset_index(drop=True)   # matched subset also contains neonates

    d["egfr"] = egfr_ckdepi2021(d.creat_mgdl.to_numpy(), d.age.to_numpy(), d.female.to_numpy()).round(1)
    d["ckd_egfr"] = d.egfr < C["egfr_ckd_threshold"]
    d["aki_kdigo"] = kdigo_aki(d, load_creatinine())

    # all subjects, so the cache does not depend on which group B is being built
    labs = extract_rows("LABEVENTS", "labs_s6_all", lambda c: c.itemid.isin(C["labs"].values()) & c.valuenum.notna(),
                        {"subject_id", "itemid", "charttime", "valuenum"}, dates=("charttime",))
    for col, item in C["labs"].items():
        d[col] = nearest_lab(d, labs[labs.itemid == item], col)

    adm = read_table("ADMISSIONS", dates=("dischtime",))
    d["emergency_admit"] = d.hadm_id.isin(set(adm.hadm_id[adm.admission_type.isin(["EMERGENCY", "URGENT"])]))

    dx_all = read_table("DIAGNOSES_ICD").astype({"icd9_code": str})
    dx = dx_all[dx_all.hadm_id.isin(d.hadm_id)]
    for col, prefixes in DX.items():
        hadm = set(dx.hadm_id[dx.icd9_code.str.startswith(prefixes)])
        d[col] = d.hadm_id.isin(hadm)
    if (CLINICAL / "PROCEDURES_ICD.csv.gz").exists():
        pr = read_table("PROCEDURES_ICD").astype({"icd9_code": str})
        d["proc_dialysis"] = d.hadm_id.isin(set(pr.hadm_id[pr.icd9_code.isin(DIALYSIS_PROC)]))
    else:
        d["proc_dialysis"] = np.nan
    # Chronic dialysis or ESRD coded in an admission that ended before the draw carries forward; a dialysis procedure alone
    # does not, as it may be AKI that recovered. Codes of the current stay have no time, so a window drawn before dialysis
    # started in that stay is excluded too; this cannot be separated here.
    flagged = set(dx_all.hadm_id[dx_all.icd9_code.str.startswith(DX["dx_esrd"] + DX["dx_dialysis_status"])])
    first_end = adm[adm.hadm_id.isin(flagged)].groupby("subject_id").dischtime.min()
    d["prior_dialysis"] = d.charttime > d.subject_id.map(first_end)
    d["exclude_dialysis"] = d.dx_esrd | d.dx_dialysis_status | (d.proc_dialysis == True) | d.prior_dialysis
    d["vasopressor"] = vasopressor_on(d)

    d.to_csv(out / "dataset.csv", index=False)

    p = d[~d.exclude_dialysis]
    write_manifest(out, "s6_dataset", dict(C, groupb=args.groupb,
                                           optional_tables={t: (CLINICAL / f"{t}.csv.gz").exists() for t in OPTIONAL}),
                   dict(draws_in=n_in, draws_adult=len(d),
                        subjects_adult=int(d.subject_id.nunique()),
                        draws_dialysis_excluded=int(d.exclude_dialysis.sum()),
                        subjects_dialysis_excluded=int(d[d.exclude_dialysis].subject_id.nunique()),
                        primary_draws=len(p), primary_subjects=int(p.subject_id.nunique())))


if __name__ == "__main__":
    main()
