import ast
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

import run_cross_city_generalization as runner
from summarize_cross_city_generalization import (
    ArtifactValidationError,
    protocol_status,
    summarize,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
AST_EXCLUDED_DIRECTORY_NAMES = {
    ".git", ".venv", "venv", "env", "__pycache__", "data", "results",
    "site-packages", "dist-packages",
}

# 稳定 main 上的既有实现，超出 round14 工作单授权（工作单明令不得改动既有文件）。
# 此例外仅用于隔离历史代码：若未来单独开轮次修复该文件，删除本条目即可。
LEGACY_BACKWARD_FILL_EXCEPTIONS = {
    "run_st_patchtst_ablation.py",
}


def backward_fill_calls(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"bfill", "backfill"}:
            violations.append((node.lineno, node.func.attr))
        if node.func.attr != "fillna":
            continue
        method_nodes = [
            keyword.value for keyword in node.keywords if keyword.arg == "method"
        ]
        if len(node.args) >= 2:
            method_nodes.append(node.args[1])
        for method_node in method_nodes:
            if (
                isinstance(method_node, ast.Constant)
                and str(method_node.value).lower() in {"bfill", "backfill"}
            ):
                violations.append((node.lineno, f"fillna(method={method_node.value!r})"))
    return violations


def independent_block_labels(length: int) -> np.ndarray:
    quotient, remainder = divmod(length, 5)
    return np.concatenate([
        np.full(quotient + (block < remainder), block + 1, dtype=np.int16)
        for block in range(5)
    ])


def rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {name: np.asarray(artifact[name]) for name in artifact.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)


def write_summary_fixture(root: Path) -> tuple[Path, Path]:
    config = runner.RunConfig(
        24, 1, 32, 1, 1,
        neighbor_hidden_dim=8, n_layers=1, n_heads=2, d_model=8, d_ff=16,
    )
    _, metadata = runner.prepare_shared_data(runner.synthetic_frame(), 100, config)
    metadata.update({
        "config": asdict(config),
        "center_station_id": 100,
        "seeds": [1],
        "arms": list(runner.ARMS),
        "opened_station_ids": [],
        "smoke_test": True,
        "code_commit": "test",
        "device": "cpu",
    })
    payload = runner.build_frozen_run_payload(metadata)
    metadata["frozen_run_payload"] = payload
    metadata["config_fingerprint"] = runner.config_fingerprint(payload)

    task = root / "input" / "24h_1h" / "station_100"
    predictions = task / "predictions"
    predictions.mkdir(parents=True)
    (task / "run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    count = len(metadata["confirm_scalar_timestamps_ns"])
    target = np.linspace(1.0, 20.0, count)
    timestamps = np.asarray(metadata["confirm_scalar_timestamps_ns"], dtype=np.int64)
    leads = np.asarray(metadata["confirm_scalar_leads"], dtype=np.int16)
    blocks = np.asarray(metadata["confirm_scalar_block_labels"], dtype=np.int16)
    rows = []
    for arm, offset in zip(runner.ARMS, (1.0, 0.8)):
        name = f"{arm}_seed1.npz"
        runner.save_prediction_artifact(
            predictions / name, target + offset, target, timestamps, leads, blocks
        )
        rows.append({
            "arm": arm,
            "seed": 1,
            "artifact": f"predictions/{name}",
            "config_fingerprint": metadata["config_fingerprint"],
            "trained_epochs": 1,
            "initialized_from_degraded": arm == runner.ARMS[1],
            "backbone_frozen": arm == runner.ARMS[1],
        })
    pd.DataFrame(rows).to_csv(task / "run_manifest.csv", index=False)
    return root / "input", task


class CrossCityProtocolTest(unittest.TestCase):
    def test_whitelist_fail_closed_including_consumed_b2_b3(self):
        authorized = runner.authorized_station_ids(9026)
        for station_id in runner.FORBIDDEN_STATIONS:
            with self.subTest(station_id=station_id):
                with self.assertRaises(runner.ProtocolViolation):
                    runner.guarded_read_station("unused", station_id, authorized, set())
        with self.assertRaises(runner.ProtocolViolation):
            runner.validate_authorized_station_set(9026, (*authorized[:-1], 9022))

    def test_read_guard_is_fail_closed_before_read_csv(self):
        with mock.patch.object(pd, "read_csv") as read_csv:
            with self.assertRaises(runner.ProtocolViolation):
                runner.guarded_read_station(
                    "unused", 9027, runner.authorized_station_ids(9026), set()
                )
            read_csv.assert_not_called()

    def test_no_backward_fill_calls_outside_registered_legacy_exception(self):
        violations = []
        scanned = []
        legacy_hits = []
        for path in sorted(REPOSITORY_ROOT.rglob("*.py")):
            if any(part in AST_EXCLUDED_DIRECTORY_NAMES for part in path.parts):
                continue
            scanned.append(path)
            relative = str(path.relative_to(REPOSITORY_ROOT))
            found = backward_fill_calls(path)
            if relative in LEGACY_BACKWARD_FILL_EXCEPTIONS:
                legacy_hits.extend((relative, line, call) for line, call in found)
                continue
            violations.extend(
                (relative, line, call) for line, call in found
            )
        self.assertGreater(len(scanned), 0)
        self.assertEqual(violations, [])
        # 例外只允许出现在已登记的历史文件里，不得因此掩盖其他文件的违规。
        for relative, _line, _call in legacy_hits:
            self.assertIn(relative, LEGACY_BACKWARD_FILL_EXCEPTIONS)

    def test_c_fit_only_station_selection_and_tie_rule(self):
        index = pd.date_range("2020-01-01", periods=100, freq="h")
        base = np.linspace(0, 1, 100)
        frame = pd.DataFrame({0: base}, index=index)
        for station in (7, 3, 9, 2, 8, 1):
            frame[station] = base
        before, correlations_before = runner.select_top5_neighbors(
            frame.iloc[:70], 0, [7, 3, 9, 2, 8, 1]
        )
        frame.iloc[70:, 1:] = np.arange(30)[:, None] * np.asarray([1, -1, 2, -2, 3, -3])
        after, correlations_after = runner.select_top5_neighbors(
            frame.iloc[:70], 0, [7, 3, 9, 2, 8, 1]
        )
        self.assertEqual(before, [1, 2, 3, 7, 8])
        self.assertEqual(before, after)
        self.assertEqual(correlations_before, correlations_after)

    def test_run_station_task_shares_datasets_and_orders_baseline_initialization(self):
        config = runner.RunConfig(
            24, 1, 32, 1, 1,
            neighbor_hidden_dim=8, n_layers=1, n_heads=2, d_model=8, d_ff=16,
        )
        train_calls = []

        def fake_train_arm(
            passed_config, datasets, arm, seed, device, checkpoint_path,
            initialization_checkpoint=None,
        ):
            train_calls.append({
                "datasets": datasets,
                "arm": arm,
                "seed": seed,
                "checkpoint": checkpoint_path,
                "initialization": initialization_checkpoint,
            })
            checkpoint_path.touch()
            return object(), [{"epoch": 1, "train_loss": 1.0, "valid_loss": 1.0}], 1, 1.0

        def fake_predict(model, loader, device):
            count = len(loader.dataset)
            return np.zeros((count, 1, config.horizon)), np.zeros((count, 1, config.horizon))

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runner, "train_arm", side_effect=fake_train_arm
        ), mock.patch.object(runner, "predict", side_effect=fake_predict):
            runner.run_station_task(
                runner.synthetic_frame(), (), 100, config, (777,),
                Path(temporary), torch.device("cpu"), smoke=True,
            )
        self.assertEqual([call["arm"] for call in train_calls], list(runner.ARMS))
        self.assertIs(train_calls[0]["datasets"], train_calls[1]["datasets"])
        for split in ("fit", "val", "confirm"):
            self.assertIs(
                train_calls[0]["datasets"][split], train_calls[1]["datasets"][split]
            )
        self.assertIsNone(train_calls[0]["initialization"])
        self.assertEqual(train_calls[1]["initialization"], train_calls[0]["checkpoint"])
        self.assertEqual(train_calls[0]["seed"], train_calls[1]["seed"])

    def test_horizon_six_prepare_shared_data_scalar_block_assignment(self):
        config = runner.RunConfig(
            24, 6, 32, 1, 1,
            neighbor_hidden_dim=8, n_layers=1, n_heads=2, d_model=8, d_ff=16,
        )
        _, metadata = runner.prepare_shared_data(runner.synthetic_frame(), 100, config)
        full_timestamps = np.asarray(metadata["confirm_timestamps_ns"], dtype=np.int64)
        full_blocks = independent_block_labels(len(full_timestamps))
        confirm_start = metadata["bounds"]["confirm"][0]
        expected_timestamps, expected_leads, expected_blocks = [], [], []
        for start in metadata["sample_indices"]["confirm"]:
            local = start + config.history - confirm_start
            expected_timestamps.extend(full_timestamps[local:local + config.horizon])
            expected_leads.extend(range(1, config.horizon + 1))
            expected_blocks.extend(full_blocks[local:local + config.horizon])
        self.assertEqual(metadata["confirm_block_labels"], full_blocks.tolist())
        self.assertEqual(metadata["confirm_scalar_timestamps_ns"], list(expected_timestamps))
        self.assertEqual(metadata["confirm_scalar_leads"], expected_leads)
        self.assertEqual(metadata["confirm_scalar_block_labels"], list(expected_blocks))

    def test_segment_bounds_and_early_remainder_blocks(self):
        self.assertEqual(
            runner.segment_bounds(101),
            {"fit": (0, 70), "val": (70, 80), "confirm": (80, 101)},
        )
        self.assertEqual(
            [int((independent_block_labels(12) == block).sum()) for block in range(1, 6)],
            [3, 3, 2, 2, 2],
        )

    def test_sse_independent_recalculation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.npz"
            target = np.arange(10, dtype=float)
            prediction = target + np.linspace(-1, 1, 10)
            blocks = np.repeat(np.arange(1, 6), 2)
            runner.save_prediction_artifact(
                path, prediction, target, np.arange(10), np.ones(10), blocks
            )
            with np.load(path) as artifact:
                rebuilt = [
                    np.square((prediction - target)[blocks == block]).sum()
                    for block in range(1, 6)
                ]
                np.testing.assert_allclose(artifact["sse_by_block"], rebuilt)
                np.testing.assert_array_equal(
                    artifact["element_count_by_block"], np.full(5, 2)
                )

    def test_top5_shortage_stops(self):
        index = pd.date_range("2020-01-01", periods=20, freq="h")
        base = np.arange(20, dtype=float)
        frame = pd.DataFrame(
            {0: base, 1: base, 2: base, 3: 1.0, 4: np.nan, 5: np.nan},
            index=index,
        )
        with self.assertRaises(runner.InsufficientNeighbors):
            runner.select_top5_neighbors(frame, 0, [1, 2, 3, 4, 5])

    def test_formal_git_preflight_rejects_dirty_or_untracked_code(self):
        clean = SimpleNamespace(returncode=0, stdout="", stderr="")
        dirty = SimpleNamespace(returncode=0, stdout="?? test_cross_city_generalization.py\n", stderr="")
        with mock.patch.object(runner, "git_commit", return_value="a" * 40), mock.patch.object(
            runner.subprocess, "run", side_effect=[clean, clean, clean, dirty]
        ):
            with self.assertRaisesRegex(runner.ProtocolViolation, "status"):
                runner.verify_formal_git_state()
        missing = SimpleNamespace(returncode=1, stdout="", stderr="missing")
        with mock.patch.object(runner, "git_commit", return_value="a" * 40), mock.patch.object(
            runner.subprocess, "run", return_value=missing
        ):
            with self.assertRaisesRegex(runner.ProtocolViolation, "HEAD"):
                runner.verify_formal_git_state()

    def _assert_fixture_tamper_rejected(self, mutate):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir, task = write_summary_fixture(root)
            summarize(input_dir, root / "valid-summary")
            mutate(task)
            with self.assertRaises(ArtifactValidationError):
                summarize(input_dir, root / "tampered-summary")

    def test_summarizer_rejects_prediction_tamper_with_stale_sse(self):
        def mutate(task):
            path = task / "predictions" / f"{runner.ARMS[1]}_seed1.npz"
            def change(arrays):
                arrays["prediction_ugm3"][0] += 10
            rewrite_npz(path, change)

        self._assert_fixture_tamper_rejected(mutate)

    def test_summarizer_rejects_coordinated_block_and_sse_tamper(self):
        def mutate(task):
            for arm in runner.ARMS:
                path = task / "predictions" / f"{arm}_seed1.npz"
                def change(arrays):
                    changed = (arrays["block_label"].astype(int) % 5 + 1).astype(np.int16)
                    error = arrays["prediction_ugm3"] - arrays["target_ugm3"]
                    arrays["block_label"] = changed
                    arrays["sse_by_block"] = np.asarray([
                        np.square(error[changed == block]).sum() for block in range(1, 6)
                    ])
                    arrays["element_count_by_block"] = np.asarray([
                        (changed == block).sum() for block in range(1, 6)
                    ])
                rewrite_npz(path, change)

        self._assert_fixture_tamper_rejected(mutate)

    def test_summarizer_rejects_timestamp_lead_metadata_and_fingerprint_tamper(self):
        def artifact_mutator(field, value):
            def mutate(task):
                path = task / "predictions" / f"{runner.ARMS[0]}_seed1.npz"
                def change(arrays):
                    arrays[field][0] = value(arrays[field][0])
                rewrite_npz(path, change)
            return mutate

        mutations = {
            "timestamp": artifact_mutator("target_timestamp_ns", lambda value: value + 1),
            "lead": artifact_mutator("lead", lambda value: value + 1),
            "metadata_support": lambda task: self._mutate_metadata(task, "confirm_scalar_timestamps_ns"),
            "fingerprint": self._mutate_fingerprint,
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                self._assert_fixture_tamper_rejected(mutation)

    @staticmethod
    def _mutate_metadata(task: Path, field: str):
        path = task / "run_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata[field][0] += 1
        path.write_text(json.dumps(metadata), encoding="utf-8")

    @staticmethod
    def _mutate_fingerprint(task: Path):
        path = task / "run_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["config_fingerprint"] = "0" * 64
        path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_protocol_status_semantics(self):
        self.assertEqual(protocol_status(False, True, True), "PASS")
        self.assertEqual(protocol_status(False, True, False), "STOP")
        self.assertEqual(protocol_status(False, False, False), "INCOMPLETE")
        self.assertEqual(protocol_status(True, False, False), "SMOKE_NOT_ELIGIBLE")


if __name__ == "__main__":
    unittest.main()
