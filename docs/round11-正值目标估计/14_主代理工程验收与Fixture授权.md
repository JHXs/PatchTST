# Round 11 主代理工程验收与 registered fixture 授权

日期：2026-09-10。

## 验收对象

- 分支：`experiment/round11-cf-cv-smearing-fixture`
- 源提交：`d608c34afb24cb3631546f230c2a67eff9881410`
- 未读取站点 CSV、旧 checkpoint 或旧性能结果。

主代理复跑：

- `test_round11_cf_cv_fixture.py`：26/26 PASS；
- `test_st_patchtst.py`：10/10 PASS；
- `py_compile` 与 `git diff --check`：PASS；
- 独立 code reviewer：`13_工程终审.md`，结论 PASS。

## clean-commit smoke

在 `/tmp/round11_cf_cv_main_d608c34_smoke` 运行单 seed 4001、两任务、每 fit 1 epoch、N=2048 的新 smoke。结果只用于完整性，不查看或解释性能指标。

- 状态：`SMOKE_COMPLETE`；
- fit keys：14/14；task：2；
- independent summarizer：`verified=true`；
- flags：`registered_fixture=false`、`formal_l0=false`、`formal=false`、`non_formal=true`、`smoke=true`；
- `rho_gate_applied=false`、`rho_stop=false`；
- preflight：clean，绑定上述分支与 `d608c34`；
- runner wall time：32.1296 秒；task elapsed sum：31.6886 秒；`ru_maxrss=1219556` KiB（Linux 口径）。

连续执行 summarizer 两次后，下列三个 SHA-256 均保持不变：

```text
manifest.json                 0f20fb6bb3243489dcf695e4965b4203b6951e0d932866408779f9e6224b1ac2
verification/summary.json     143a484984dbc076526f7eec9056d8539c23468c62628c3bdec268f0418aa302
verification/manifest.json    a865a359dd3e3eedc7e86b607cf6cc8f2035dfb868c0003a2278bb7ac07cdca9
```

## 阶段决定

工程验收 PASS。授权在本文件提交并推送后的 clean commit 上运行唯一 registered fixture：

```bash
python run_round11_cf_cv_fixture.py \
  --device cuda:0 \
  --output-dir experiments/results/round11/cf_cv_registered_fixture
```

该运行固定 seeds 4001–4003、N=4096、burn-in=512、40 epoch 上限、patience=8、任务 24→1 与 168→6，共 42 个真实 PatchTST fits。AMD ROCm 仍通过 `cuda:0` API 使用。

该授权仅检查完整 42-key 契约、M/S/算子重建、rho 路径是否近零和资源可执行性。输出必须保持 `registered_fixture=true`、`formal_l0=false`、`formal=false`、`non_formal=true`。即使状态 COMPLETE，也不能自动运行 50-seed 正式 L0 或真实站点数据；若 rho STOP，则本机制直接止损，不用主效应指标挽救。
