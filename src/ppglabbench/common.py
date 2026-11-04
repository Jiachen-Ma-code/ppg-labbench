"""Shared paths, config and helpers for the staged pipeline (S1-S6)."""
import json
import subprocess
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import wfdb
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CLINICAL = DATA / "clinical"
S1 = DATA / "s1_catalog"
S2 = DATA / "s2_groupA"
S3 = DATA / "s3_windows"
S4 = DATA / "s4_sqi"
S5 = DATA / "s5_groupB"
S6 = DATA / "s6_dataset"
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
DB = CFG["wfdb_db"]


def subject_dir(sid):
    s = f"p{sid:06d}"
    return f"{DB}/{s[:3]}/{s}"


def load_creatinine(labevents=CLINICAL / "LABEVENTS.csv.gz"):
    """Creatinine rows from LABEVENTS, cached to clinical/creatinine.csv.

    The cache is keyed on the source file's name and size plus the creatinine config,
    not mtime: wget -N resets mtime to the server's, which would make a stale cache look fresh.
    """
    c = CFG["creatinine"]
    lo, hi = c["valid_range_mgdl"]
    cache, meta = CLINICAL / "creatinine.csv", CLINICAL / "creatinine.meta.json"
    key = {"source": labevents.name, "size": labevents.stat().st_size, "itemid": c["itemid"], "range": [lo, hi]}
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == key:
        return pd.read_csv(cache, parse_dates=["charttime"])
    print(f"extracting creatinine from {labevents} (a few minutes)...")
    chunks = []
    for ch in pd.read_csv(labevents, chunksize=2_000_000, usecols=lambda s: s.lower() in
                          {"subject_id", "hadm_id", "itemid", "charttime", "valuenum"}):
        ch.columns = ch.columns.str.lower()
        chunks.append(ch[(ch.itemid == c["itemid"]) & ch.valuenum.between(lo, hi)])
    lab = pd.concat(chunks).drop(columns="itemid").rename(columns={"valuenum": "creat_mgdl"})
    lab["charttime"] = pd.to_datetime(lab.charttime)
    lab.to_csv(cache, index=False)
    meta.write_text(json.dumps(key))
    return lab


def extract_rows(table, name, keep, usecols, dates=()):
    """Rows of a large MIMIC-III table matching keep(chunk), cached to clinical/<name>.csv.

    Cache key as in load_creatinine; `name` must change whenever `keep` changes.
    """
    src = CLINICAL / f"{table}.csv.gz"
    cache, meta = CLINICAL / f"{name}.csv", CLINICAL / f"{name}.meta.json"
    key = {"source": src.name, "size": src.stat().st_size, "name": name}
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == key:
        return pd.read_csv(cache, parse_dates=list(dates))
    print(f"extracting {name} from {src.name} (a few minutes)...")
    chunks = []
    for ch in pd.read_csv(src, chunksize=2_000_000, usecols=lambda s: s.lower() in usecols):
        ch.columns = ch.columns.str.lower()
        chunks.append(ch[keep(ch)])
    out = pd.concat(chunks)
    for d in dates:
        out[d] = pd.to_datetime(out[d])
    out.to_csv(cache, index=False)
    meta.write_text(json.dumps(key))
    return out


def read_table(table, dates=()):
    """A small MIMIC-III table, lower-cased column names."""
    t = pd.read_csv(CLINICAL / f"{table}.csv.gz")
    t.columns = t.columns.str.lower()
    for d in dates:
        t[d] = pd.to_datetime(t[d])
    return t


@lru_cache(maxsize=4096)
def segment_spans(rec, pn_dir):
    """(segment_name, start_datetime, n_samples, fs) for each real segment of a multi-segment record."""
    h = wfdb.rdheader(rec, pn_dir=pn_dir)
    out, offset = [], 0
    for name, n in zip(h.seg_name, h.seg_len):
        if name != "~" and not name.endswith("_layout"):
            out.append((name, h.base_datetime + timedelta(seconds=offset / h.fs), n, h.fs))
        offset += n
    return tuple(out)


@lru_cache(maxsize=65536)
def segment_signals(seg, pn_dir):
    return tuple(wfdb.rdheader(seg, pn_dir=pn_dir).sig_name)


def git_version():
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "tests", "config.yaml"],
                               capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def write_manifest(stage_dir, stage, params, counts):
    """Record what a stage ran with and what it produced; S5 assembles these into the flow diagram."""
    m = dict(stage=stage, finished=datetime.now().isoformat(timespec="seconds"),
             code_version=git_version(), params=params, counts=counts)
    (stage_dir / "manifest.json").write_text(json.dumps(m, indent=2, default=str))
    print(json.dumps(counts, indent=2, default=str))
