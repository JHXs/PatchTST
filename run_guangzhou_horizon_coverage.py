"""阶段 C：广州 8 个 b1 中心站的跨预测步覆盖（20 个配置）。

复用 `run_cross_city_generalization.py`（round14 确认运行器）的全部安全组件：
显式站点白名单与 fail-closed 读取、因果填充（禁 bfill）、训练段筛站、两臂共享样本支持、
逐标量预测与块标签落盘。

任务网格：L ∈ {24,48,72,168} × H ∈ {1,3,6,12,24}；训练预算按历史长度分档
（L ≤ 48 → 40/8/256；L > 48 → 30/6/512），与北京覆盖实验一致。

用法：
    python run_guangzhou_horizon_coverage.py --output-root experiments/results/guangzhou_horizon_coverage
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch

import run_cross_city_generalization as gz

TASK_GRID = [(24, 1), (24, 3), (24, 6), (24, 12), (24, 24),
             (48, 1), (48, 3), (48, 6), (48, 12), (48, 24),
             (72, 1), (72, 3), (72, 6), (72, 12), (72, 24),
             (168, 1), (168, 3), (168, 6), (168, 12), (168, 24)]
SEEDS = (7001, 7002, 7003, 7004, 7005)


def verify_code_state(output_root: Path) -> str:
    """要求代码/数据无未提交改动，但允许本实验自身的未跟踪输出目录。

    与 round14 的 `verify_formal_git_state` 的差别：那次运行的输出目录在运行前不存在，
    因此可以直接要求 porcelain 为空；本阶段是"可续跑"的长任务，输出目录会持续增长。
    """
    head = gz.git_commit()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise gz.ProtocolViolation("HEAD 必须是 40 位小写十六进制 commit")
    for relative in ("run_guangzhou_horizon_coverage.py", "run_cross_city_generalization.py",
                     "frequency_residual_adapter.py"):
        tracked = subprocess.run(["git", "cat-file", "-e", f"HEAD:{relative}"],
                                 check=False, capture_output=True, text=True)
        if tracked.returncode != 0:
            raise gz.ProtocolViolation(f"实现文件必须已被 HEAD 跟踪: {relative}")
    status = subprocess.run(["git", "status", "--porcelain"], check=False,
                            capture_output=True, text=True).stdout
    allowed_prefix = f"?? {output_root.as_posix().rstrip('/')}"
    dirty = [line for line in status.splitlines()
             if not line.startswith(allowed_prefix)]
    if dirty:
        raise gz.ProtocolViolation(f"存在未提交的源码/数据改动，拒绝运行: {dirty[:5]}")
    return head


def stage_config(history: int, horizon: int) -> gz.RunConfig:
    if history <= 48:
        return gz.RunConfig(history, horizon, batch_size=256, epochs=40, patience=8)
    return gz.RunConfig(history, horizon, batch_size=512, epochs=30, patience=6)


def task_dir_for(output_root: Path, history: int, horizon: int, center: int) -> Path:
    return output_root / f"{history}h_{horizon}h" / f"station_{center}"


def expected_artifacts(seeds) -> set[str]:
    return {f"{arm}_seed{seed}.npz" for arm in gz.ARMS for seed in seeds}


def is_complete(task_dir: Path, seeds) -> bool:
    manifest = task_dir / "run_manifest.csv"
    if not manifest.is_file():
        return False
    predictions = task_dir / "predictions"
    if not predictions.is_dir():
        return False
    return expected_artifacts(seeds).issubset({p.name for p in predictions.glob('*.npz')})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="experiments/results/guangzhou_horizon_coverage")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--configs", default=None, help="可选：如 24x24,168x6")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="仅用合成数据跑一个极小的冒烟任务")
    parser.add_argument("--list-plan", action="store_true", help="只打印计划访问的站点，不读数据")
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(",") if s)
    output_root = Path(args.output_root)
    device = torch.device(args.device)
    grid = ([(int(v.split("x")[0]), int(v.split("x")[1])) for v in args.configs.split(",")]
            if args.configs else TASK_GRID)

    access_plan = {str(center): list(gz.authorized_station_ids(center)) for center in gz.B1_STATIONS}
    for center, ids in access_plan.items():
        gz.validate_authorized_station_set(int(center), ids)
    print(json.dumps({"planned_station_access": access_plan, "forbidden": sorted(gz.FORBIDDEN_STATIONS),
                      "configs": [f"{l}h_{h}h" for l, h in grid], "seeds": list(seeds)},
                     ensure_ascii=False))
    if args.list_plan:
        return

    if args.smoke:
        config = replace(stage_config(24, 1), batch_size=32, epochs=1, patience=1,
                         neighbor_hidden_dim=8, n_layers=1, d_model=8, n_heads=2, d_ff=16, dropout=0.0)
        frame = gz.synthetic_frame()
        gz.run_station_task(frame, (), 100, config, (14001,), output_root, device, smoke=True)
        print(f"合成冒烟产物: {output_root.resolve()}")
        return

    verify_code_state(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for center in gz.B1_STATIONS:
        frame, opened = gz.load_center_candidate_frame(center)
        for history, horizon in grid:
            task_dir = task_dir_for(output_root, history, horizon, center)
            if task_dir.exists() and is_complete(task_dir, seeds):
                print(f"[跳过] 已完成 {task_dir.relative_to(output_root)}")
                continue
            if task_dir.exists():
                stash = task_dir.with_name(task_dir.name + f"_partial_{int(time.time())}")
                shutil.move(str(task_dir), str(stash))
                print(f"[提示] 不完整结果已移开: {stash.name}")
            config = stage_config(history, horizon)
            print(f"=== {history}h→{horizon}h station={center} "
                  f"(epochs={config.epochs} patience={config.patience} batch={config.batch_size}) ===")
            gz.run_station_task(frame, opened, center, config, seeds, output_root, device, smoke=False)
            print(f"[完成] {history}h_{horizon}h station={center} "
                  f"(累计 {time.perf_counter() - started:.0f}s)")
    print("STAGE_C_DONE")


if __name__ == "__main__":
    main()
