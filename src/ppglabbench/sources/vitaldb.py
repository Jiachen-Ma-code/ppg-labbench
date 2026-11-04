"""External cohort: VitalDB (open, Seoul National University Hospital, 6,388 surgical cases).

For every case with SNUADC/PLETH and a preoperative creatinine, downloads the PLETH track
(500 Hz, whole case, ~4 MB gz) and keeps only the pre-incision segment [casestart, min(opstart, 30 min)],
decimated to 125 Hz to match MIMIC. Note: most of this PPG is under general anaesthesia (MIMIC: ICU, often
sedated), a deliberate domain shift for external validation; 486 of 5,802 cases start recording before `anestart`.
The API's Time column is sparse; sample i is at t = i / 500 s.

Usage:
    python -m ppglabbench.sources.vitaldb [--limit N]
Output: data/external/vitaldb/{cases.csv.gz, trks.csv.gz, pleth/<caseid>.npz, index.csv}
"""
import argparse
import io
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate

ROOT = Path(__file__).resolve().parents[3]
VDB = ROOT / "data" / "external" / "vitaldb"
API = "https://api.vitaldb.net"
FS_SRC, FS = 500, 125
MAX_S = 1800


def fetch(case):
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(f"{API}/{case.tid}", timeout=180).read()
            break
        except Exception as e:
            err = str(e)[:200]
            time.sleep(5 * (attempt + 1))
    else:
        return dict(caseid=case.caseid, n=0, error=err)
    x = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=[1]).iloc[:, 0].to_numpy(np.float64)
    end = int(min(case.opstart if case.opstart > 60 else MAX_S, MAX_S) * FS_SRC)
    x = x[:end]
    # decimate needs finite input: fill short gaps, keep a NaN mask at 125 Hz
    bad = ~np.isfinite(x)
    if bad.all():
        return dict(caseid=case.caseid, n=0, error="all NaN")
    x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    y = decimate(x, FS_SRC // FS, ftype="fir").astype(np.float32)
    y[bad[::FS_SRC // FS][:len(y)]] = np.nan
    np.savez_compressed(VDB / "pleth" / f"{case.caseid}.npz", pleth=y.astype(np.float16), fs=FS)
    return dict(caseid=case.caseid, n=len(y), nan_frac=round(float(np.isnan(y).mean()), 4), error="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    (VDB / "pleth").mkdir(parents=True, exist_ok=True)
    for name in ("cases", "trks"):
        if not (VDB / f"{name}.csv.gz").exists():
            urllib.request.urlretrieve(f"{API}/{name}", VDB / f"{name}.csv.gz")
    cases = pd.read_csv(VDB / "cases.csv.gz")
    trks = pd.read_csv(VDB / "trks.csv.gz")
    cases = cases[cases.preop_cr.notna()].merge(trks[trks.tname == "SNUADC/PLETH"][["caseid", "tid"]], on="caseid")
    index = VDB / "index.csv"
    done = set(pd.read_csv(index).caseid) if index.exists() else set()
    todo = cases[~cases.caseid.isin(done)].head(args.limit)
    print(f"{len(cases)} cases with PLETH + preop creatinine; {len(done)} done; fetching {len(todo)}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, c) for c in todo.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            pd.DataFrame([f.result()]).to_csv(index, mode="a", header=not index.exists(), index=False)
            if i % 100 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"  {i}/{len(futs)}  {el / 60:.1f} min  ETA {(len(futs) - i) * el / i / 60:.0f} min", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
