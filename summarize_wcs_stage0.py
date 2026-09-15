"""Independently rebuild Stage 0 metrics from saved predictions and evaluate the frozen gates.

The summarizer regenerates the synthetic targets itself (same frozen DGP code path) and
recomputes RMSE from the saved per-sample predictions; it never trusts the aggregate values
written by the runner. Gates are exactly those pre-registered in
`docs/气象条件化机制筛查/04_Stage0协议.md` §5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_wcs_stage0 as stage0

GATES = {
    "P1": "DGP-M / S-shift / H=1: mean >= 5% and >= 16/20 seeds positive",
    "P2": "DGP-M / S-shift / H=6: mean >= 5% and >= 16/20 seeds positive",
    "P3": "DGP-M / S-small / both H: mean >= 5% and >= 16/20 seeds positive",
    "P4": "DGP-M / S-iid / both H: mean >= -1%",
    "N1": "DGP-A / S-shift / both H: mean <= 1% and <= 12/20 seeds positive",
    "N2": "DGP-M / S-shift: wcs_pos mean <= 0 and >= 14/20 seeds negative",
    "N3": "monotonicity + saturation probe passes",
}


def rebuild_targets(seed: int, dgp: str, horizon: int, end_indices: np.ndarray) -> np.ndarray:
    """Regenerate the frozen DGP and read the targets at the recorded absolute window ends."""
    sign = +1.0 if dgp == "M" else -1.0
    state, wind = stage0.generate_series(seed)
    rng = np.random.default_rng(seed + 99991)
    ends = np.arange(stage0.LOOKBACK, state.size)
    noise = rng.normal(0.0, stage0.NOISE_STD, size=(ends.size, 1))
    targets = stage0.true_conditional_mean(state, wind, ends, sign) + noise
    position = {int(end): i for i, end in enumerate(ends)}
    rows = np.array([position[int(end)] for end in end_indices], dtype=int)
    return targets[rows, :horizon]


def reduction_percent(control_pred: np.ndarray, candidate_pred: np.ndarray, targets: np.ndarray) -> float:
    """Paired reduction in percent, computed from residuals (prediction minus target)."""
    control_error = control_pred - targets
    candidate_error = candidate_pred - targets
    n_elements = targets.size
    rmse_c = float(np.sqrt(np.sum(control_error ** 2) / n_elements))
    rmse_k = float(np.sqrt(np.sum(candidate_error ** 2) / n_elements))
    return 100.0 * (rmse_c - rmse_k) / rmse_c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = json.loads((input_dir / "stage0_runs.json").read_text(encoding="utf-8"))

    rows = []
    for record in runs:
        seed, dgp, horizon, setting = record["seed"], record["dgp"], record["horizon"], record["setting"]
        end_indices = np.asarray(record["test_end_indices"], dtype=int)
        targets = rebuild_targets(seed, dgp, horizon, end_indices)
        stored = np.load(input_dir / f"pred_{seed}_{dgp}_H{horizon}_{setting}.npz")
        n_elements = targets.size
        rmse = {}
        for arm in ("uncon_add", "wcs_neg", "wcs_pos"):
            predictions = stored[f"{setting}|{arm}"]
            if predictions.shape != targets.shape:
                raise ValueError(f"prediction shape mismatch for seed={seed} dgp={dgp} H={horizon} {setting} {arm}")
            rmse[arm] = float(np.sqrt(np.sum((predictions - targets) ** 2) / n_elements))
            recorded = record["arms"][arm]["rmse"]
            if abs(rmse[arm] - recorded) > 1e-6:
                raise ValueError(f"runner metric disagrees with rebuild: {rmse[arm]} vs {recorded}")
        rows.append({
            "seed": seed, "dgp": dgp, "horizon": horizon, "setting": setting,
            "n_train": record["n_train"], "n_test": record["n_test"],
            "params_uncon": record["arms"]["uncon_add"]["parameters"],
            "params_wcs": record["arms"]["wcs_neg"]["parameters"],
            "rmse_uncon": rmse["uncon_add"], "rmse_wcs_neg": rmse["wcs_neg"], "rmse_wcs_pos": rmse["wcs_pos"],
            "delta_neg_vs_uncon": reduction_percent(stored[f"{setting}|uncon_add"], stored[f"{setting}|wcs_neg"], targets),
            "delta_pos_vs_uncon": reduction_percent(stored[f"{setting}|uncon_add"], stored[f"{setting}|wcs_pos"], targets),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "stage0_paired.csv", index=False, float_format="%.6f")

    def stats(dgp: str, setting: str, horizon: int, column: str) -> tuple[float, int, int]:
        sel = frame[(frame.dgp == dgp) & (frame.setting == setting) & (frame.horizon == horizon)]
        return float(sel[column].mean()), int((sel[column] > 0).sum()), int((sel[column] < 0).sum())

    gates: dict[str, dict] = {}
    p1 = stats("M", "S-shift", 1, "delta_neg_vs_uncon")
    p2 = stats("M", "S-shift", 6, "delta_neg_vs_uncon")
    gates["P1"] = {"mean": p1[0], "positive": p1[1], "pass": bool(p1[0] >= 5.0 and p1[1] >= 16)}
    gates["P2"] = {"mean": p2[0], "positive": p2[1], "pass": bool(p2[0] >= 5.0 and p2[1] >= 16)}
    small = [stats("M", "S-small", h, "delta_neg_vs_uncon") for h in stage0.HORIZONS]
    gates["P3"] = {"detail": [{"horizon": h, "mean": s[0], "positive": s[1]} for h, s in zip(stage0.HORIZONS, small)],
                   "pass": bool(all(s[0] >= 5.0 and s[1] >= 16 for s in small))}
    iid = [stats("M", "S-iid", h, "delta_neg_vs_uncon") for h in stage0.HORIZONS]
    gates["P4"] = {"detail": [{"horizon": h, "mean": s[0]} for h, s in zip(stage0.HORIZONS, iid)],
                   "pass": bool(all(s[0] >= -1.0 for s in iid))}
    neg = [stats("A", "S-shift", h, "delta_neg_vs_uncon") for h in stage0.HORIZONS]
    gates["N1"] = {"detail": [{"horizon": h, "mean": s[0], "positive": s[1]} for h, s in zip(stage0.HORIZONS, neg)],
                   "pass": bool(all(s[0] <= 1.0 and s[1] <= 12 for s in neg))}
    wrong = [stats("M", "S-shift", h, "delta_pos_vs_uncon") for h in stage0.HORIZONS]
    gates["N2"] = {"detail": [{"horizon": h, "mean": s[0], "negative": s[2]} for h, s in zip(stage0.HORIZONS, wrong)],
                   "pass": bool(all(s[0] <= 0.0 and s[2] >= 14 for s in wrong))}
    probe = json.loads((input_dir / "monotonicity_probe.json").read_text(encoding="utf-8"))
    gates["N3"] = {"probe": {k: v for k, v in probe.items() if k != "differences"},
                   "pass": bool(probe["strictly_decreasing"] and probe["differences_abs_decreasing"])}

    parameter_fairness = {
        "uncon_add": int(frame.params_uncon.iloc[0]),
        "wcs_neg": int(frame.params_wcs.iloc[0]),
        "control_not_smaller": bool(int(frame.params_uncon.iloc[0]) >= int(frame.params_wcs.iloc[0])),
    }
    verdict = {
        "gates": gates,
        "gate_definitions": GATES,
        "parameter_fairness": parameter_fairness,
        "all_pass": bool(all(g["pass"] for g in gates.values()) and parameter_fairness["control_not_smaller"]),
        "status": "STAGE0_PASS" if all(g["pass"] for g in gates.values()) and parameter_fairness["control_not_smaller"] else "STAGE0_FAIL",
    }
    (output_dir / "stage0_gate_summary.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
