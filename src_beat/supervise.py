#!/usr/bin/env python3
"""Independent supervisor for beat-max3 loop.

Only marks a candidate SHIP if gates pass. Never claims LB from nested+gap.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "beat_max3"
LOG = ART / "supervisor_log.jsonl"
MAX3_LB = 0.71222


def run_fuse(extra: list[str], tag: str) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "src_beat" / "fuse_max3_plus.py"),
        "--tag",
        tag,
        "--extra",
        *extra,
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))
    return json.loads((ART / f"report_{tag}.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    report = run_fuse(args.extra, args.tag)
    decision = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "extra": args.extra,
        "passed": report["passed"],
        "delta": report["delta"],
        "nested": report["cand_nested"],
        "spearman": report["spearman_vs_max3"],
        "blocks_plus": report["blocks_plus"],
        "gate": report["gate"],
        "verdict": "SHIP_CANDIDATE" if report["passed"] else "REJECT",
        "lb_claim": None,
        "note": (
            f"Cannot claim LB>{MAX3_LB} without public leaderboard. "
            "Local gate only. Prefer smallest delta-passing recipe if multiple pass."
        ),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")
    (ART / f"supervisor_{args.tag}.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
