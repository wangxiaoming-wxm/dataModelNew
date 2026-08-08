"""Gather the per-view arm files into one directory for fusion."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SOURCES = [
    ("artifacts/v2main", ["cat_d5", "cat_d6", "lgb_te", "glm"]),
    ("artifacts/v2altmerged", ["cat_alt"]),
    ("artifacts/v2alt2merged", ["cat_alt2"]),
    ("artifacts/v2gap", ["gap"]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/v2"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for d, arms in SOURCES:
        for a in arms:
            src = Path(d) / f"arm_{a}.npz"
            if src.exists():
                shutil.copy(src, args.out / f"arm_{a}.npz")
                print(f"collected {a} from {d}")
            else:
                print(f"missing {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
