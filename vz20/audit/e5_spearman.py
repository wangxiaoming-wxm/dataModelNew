#!/usr/bin/env python3
"""E6: how different is each candidate from what has already been submitted?"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/workspace/vz20/audit")
from common import ART  # noqa: E402

FILES = {
    "vz19_SUBMITTED_0.71298": "/workspace/vz20/submission_vz19.csv",
    "W62_SUBMITTED_0.71503": "/tmp/submission_w62.csv",
    "fp_v8_champion": "/workspace/vz20/next_submit/submission_fp_v8_champion.csv",
    "fp_v8_tempered": "/workspace/vz20/next_submit/submission_fp_v8_tempered.csv",
    "fp_v8_aggressive": "/workspace/vz20/next_submit/submission_fp_v8_aggressive.csv",
    "am40": "/workspace/vz20/next_submit/submission_am40.csv",
    "vz20": "/workspace/vz20/submission_vz20.csv",
}


def sha256(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def main():
    extra = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    files = {**FILES, **extra}
    order = pd.read_csv("/workspace/data/submit_sample.csv", dtype={"id": str})["id"].tolist()
    cols = {}
    meta = {}
    for name, path in files.items():
        if not Path(path).is_file():
            print(f"skip {name}: missing {path}")
            continue
        d = pd.read_csv(path, dtype={"id": str}).set_index("id")["label"]
        cols[name] = d.reindex(order).to_numpy(float)
        meta[name] = {"path": path, "sha256": sha256(path)}

    names = list(cols)
    out = {"sha256": meta, "spearman": {}}
    print(f"{'':<26}" + "".join(f"{n[:14]:>16}" for n in names))
    for a in names:
        row = {}
        line = f"{a[:26]:<26}"
        for b in names:
            r = float(spearmanr(cols[a], cols[b]).statistic)
            row[b] = r
            line += f"{r:>16.4f}"
        out["spearman"][a] = row
        print(line)

    (ART / "e5_spearman.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote", ART / "e5_spearman.json")


if __name__ == "__main__":
    main()
