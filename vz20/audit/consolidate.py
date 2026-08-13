#!/usr/bin/env python3
"""Collect every experiment into the single fp_v8_audit.json the brief asks for."""
from __future__ import annotations

import json
from pathlib import Path

ART = Path("/workspace/vz20/artifacts/vz20")


def load(name):
    p = ART / name
    return json.loads(p.read_text()) if p.is_file() else None


def main():
    e1, e2, e3 = load("e1_repro_nested.json"), load("e2_crosshalf.json"), load("e3_permutation_null.json")
    e4, e5, e6 = load("e4_diagnostics.json"), load("e5_spearman.json"), load("e6_lb_noise.json")

    out = {
        "verdict": "fp_v8 is overfit. Its reported 0.77016/0.77033 contains no transferable signal.",
        "target": {
            "file": "vz20/next_submit/submission_fp_v8_champion.csv",
            "sha256": "fa2a06f4ed9af099e37225a2a9ff2e10e0a0a7979ac9c7e5dfacdcdde5866d67",
            "source": "origin/cursor/super714-plus-edb2:src_super/fuse_fp_v8.py",
            "claimed_oof": 0.7701614508323125,
            "claimed_nested": 0.7703295069355633,
        },
        "finding_1_fake_nested": {
            "what_the_code_does": "fuse_fp_v8.py:164-165 slices an already-finished OOF vector into 5 "
                                  "stratified pieces and averages their AUC. Nothing is refit.",
            "proof": "Any fixed vector's piecewise AUC equals its full AUC. Feeding y+0.01*noise "
                     "(100% leakage) makes the estimator report nested=1.0 and pass the gate.",
            "table": e1["nested_is_a_reslice"] if e1 else None,
            "note": "random_state=2026 is also the first entry of TE_SEEDS, so the slicing shares its "
                    "randomness with the splits that built the OOF.",
            "consequence": "v4->v8 all use this same gate, so the whole promotion chain is unvalidated.",
        },
        "finding_2_sign_flip_leak": {
            "what_the_code_does": "fuse_fp_v8.py:79 flips a key's encoding whenever "
                                  "roc_auc_score(y, oof) < 0.5, using the full label vector, then "
                                  "reports that same vector's AUC.",
            "scale": "~1700 keys x 6 seeds ~= 10^4 full-label one-sided selections on 14930 rows / 1496 positives",
            "measured_inflation_by_family": {
                k: {
                    "n_keys": v["n_keys"],
                    "honest_crosshalf": v["fit_half"]["mean"],
                    "fp_v8_style": v["eval_leak"]["mean"],
                    "inflation": round(v["eval_leak"]["mean"] - v["fit_half"]["mean"], 5),
                }
                for k, v in (e2["real_ids"].items() if e2 else [])
            },
            "signature": "inflation grows with the number of keys selected over (+0.022 at 8 keys, "
                         "+0.13 at 224 keys), which is selection bias, not signal.",
        },
        "finding_3_crosshalf_transfer": {
            "protocol": "train split in halves; TE table AND sign fitted on one half only, frozen, "
                        "applied to the other; both directions; multiple seeds. Sign estimated two "
                        "honest ways (in-sample on the fitting half, and inner-OOF within it).",
            "result": "all 14 id key families land in 0.486-0.512; none is stably positive",
            "per_family": {
                k: {"honest_mean": v["fit_half"]["mean"], "honest_std": v["fit_half"]["std"],
                    "runs_above_half": f"{v['fit_half']['n_above_half']}/{v['fit_half']['n_runs']}"}
                for k, v in (e2["real_ids"].items() if e2 else [])
            },
            "positive_control": "CONTROL_xfeatures (real business columns) is 0.5124 with 10/10 runs "
                                "above 0.5, t=4.78, so the harness can detect a real weak signal.",
        },
        "finding_4_permutation_null": {
            "protocol": "20 random permutations of the id column against y, same pipeline rerun",
            "families": e3["families"] if e3 else None,
            "headline": "honest transfer z in [-1.35,+1.53], min p=0.14 -> nothing significant; and the "
                        "fp_v8-style score on real ids matches the score on permuted ids to within "
                        "0.0067, i.e. the reported 'signal' survives destroying the ids.",
            "caveat": "CONTROL_xfeatures has a degenerate z because permuting the id column cannot "
                      "change x-features; its null std is 0 by construction. Its evidence is the "
                      "10/10 cross-half record in finding_3, not this z.",
        },
        "finding_5_cardinality": {
            "n_train": e4["duplicates"] and 14930,
            "n_positives": e4["positives"]["n_pos"] if e4 else None,
            "key_spaces": e4["id_cardinality"] if e4 else None,
            "id_is_random_hash": {
                "train_test_id_overlap": e1["id_structure"]["id_overlap_train_test"] if e1 else None,
                "byte_means": e1["id_structure"]["byte_mean_per_pos"] if e1 else None,
                "bit_mean_range": [e1["id_structure"]["bit_mean_min"], e1["id_structure"]["bit_mean_max"]] if e1 else None,
                "note": "uniform 64-bit hash with zero train/test id overlap: there is no carrier for "
                        "any memorised id effect to travel on.",
            },
        },
        "finding_6_dataset_diagnostics": {
            "adversarial_train_vs_test_auc": e4["adversarial"]["auc"] if e4 else None,
            "verdict": e4["adversarial"]["verdict"] if e4 else None,
            "duplicates": e4["duplicates"] if e4 else None,
            "days_is_time_index": False,
            "implication": "iid split, so local repeated CV is a trustworthy proxy for the leaderboard.",
        },
        "finding_7_composition": {
            "nominal_label_informative_weight": 0.106,
            "expansion": "fp_v8 = 0.80*v7 + 0.20*bytepair_mean; v7 = 0.15*v3 + 0.10*bits + 0.05*xs + "
                         "0.22*and_all + 0.06*tri + 0.06*or + 0.06*xor + 0.30*cmean; v3 = 0.55*am40 + "
                         "0.45*id_pool. Real content = 0.80*(0.15*0.55 + 0.05) = 0.106.",
            "spearman_bytepair_mean_vs_W62": -0.030,
            "reading": "the component that makes fp_v8 look different from W62 is orthogonal to the "
                       "signal because it is noise.",
        },
        "spearman_matrix": e5["spearman"] if e5 else None,
        "leaderboard_noise": e6,
        "expected_online_for_fp_v8": {
            "range": [0.685, 0.705],
            "basis": "rho(W62)=0.9077. Calibrating on the measured noise frontier (vz21_frontier.json), "
                     "reaching rho~0.90 by injecting label-independent variation costs about -0.031 "
                     "fold-mean AUC; the honest-but-weak route costs about -0.020. Anchoring the "
                     "family at W62's 0.71503 puts fp_v8 near 0.685-0.705.",
            "conclusion": "very unlikely to beat W62's 0.71503, effectively no chance of the 0.72 top-3 line.",
        },
        "collateral_finding_vz19": {
            "issue": "src/build_vz19.py mixes in byte07 (id byte0/byte7 TE) at w_te~0.11, and w_te was "
                     "scanned on the full OOF (build_vz19.py:120-125).",
            "byte_family_honest_crosshalf": 0.5116,
            "byte_family_z_vs_null": 1.53,
            "byte_family_p": 0.143,
            "online_evidence": "vz19 (with the noise) scored 0.71298; W62 (without) scored 0.71503, "
                               "and the two correlate 0.9949. The 0.002 gap runs the same direction as "
                               "the injected noise.",
        },
    }

    p = ART / "fp_v8_audit.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("wrote", p, f"({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
