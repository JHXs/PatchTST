"""Formal and development-fixture entry point for the CCSPRF synthetic L0."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ccsprf_l0 import ARMS, CONDITIONS, FORMAL_SEEDS, ModelConfig, TASKS, repository_root, run_experiment


def _tasks(value: str) -> list[tuple[int, int]]:
    return [(int(history), int(horizon)) for history, horizon in (item.split(":") for item in value.split(","))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--formal", action="store_true", help="Run frozen 30-seed L0.")
    modes.add_argument("--fixture", action="store_true", help="Run a non-evidence development fixture.")
    modes.add_argument("--pilot", action="store_true", help="Run non-formal pilot with the exact formal dimensions/configuration.")
    parser.add_argument("--resume", action="store_true", help="Resume only a matching existing run.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fixture-seed", type=int, default=991104)
    parser.add_argument("--fixture-tasks", default="24:1")
    parser.add_argument("--fixture-conditions", default="calendar_on")
    parser.add_argument("--fixture-arms", default="BASE")
    parser.add_argument("--fixture-points", type=int, default=24 * 7 * 8)
    parser.add_argument("--fixture-epochs", type=int, default=2)
    parser.add_argument("--fixture-patience", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.formal:
        if args.resume and args.output_dir is None:
            raise SystemExit("--resume 必须显式指定 --output-dir")
        output_dir = args.output_dir or repository_root() / "experiments/results/round9/l0_formal"
        seeds, tasks, conditions, arms = FORMAL_SEEDS, TASKS, CONDITIONS, ARMS
        config, points = ModelConfig(), 8736
        print("FORMAL L0: 只使用固定 30 seeds；本命令不会按结果选择种子")
    elif args.fixture:
        if args.resume:
            raise SystemExit("fixture 不支持 --resume；恢复门仅用于 formal")
        output_dir = args.output_dir or Path("/tmp/ccsprf-l0-fixture")
        seeds, tasks = [args.fixture_seed], _tasks(args.fixture_tasks)
        conditions = [value.strip() for value in args.fixture_conditions.split(",") if value.strip()]
        arms = [value.strip() for value in args.fixture_arms.split(",") if value.strip()]
        config = ModelConfig(n_layers=1, d_ff=32, epochs=args.fixture_epochs, patience=args.fixture_patience, batch_size=64)
        points = args.fixture_points
        print("FIXTURE ONLY: 非正式小实验，不构成 L0 证据")
    else:
        if args.resume:
            raise SystemExit("pilot 不支持 --resume；恢复门仅用于 formal")
        output_dir = args.output_dir or Path("/tmp/ccsprf-l0-pilot-formal-shape")
        seeds, tasks, conditions, arms = [args.fixture_seed], TASKS, CONDITIONS, ARMS
        config, points = ModelConfig(), 8736
        print("PILOT ONLY: 非正式种子、正式维度/配置；不执行 formal gate")
    rows = run_experiment(output_dir, seeds, tasks, conditions, arms, config, points, args.device, formal=args.formal, resume=args.resume)
    print(f"完成 {len(rows)} 个 key，输出: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
