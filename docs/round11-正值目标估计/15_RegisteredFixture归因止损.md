# Round 11 registered fixture 归因止损

日期：2026-09-10。结论：**工程完整性 PASS；控制变量路径归因门 STOP。**

## 执行绑定

- 分支：`experiment/round11-cf-cv-smearing-fixture`
- preflight commit：`59d2c11f59d01086907c25902c632df95b00792f`
- 输出：`experiments/results/round11/cf_cv_registered_fixture`
- 设备：AMD ROCm，通过 `cuda:0` API
- 规模：3 seeds（4001–4003）× 2 tasks × 7 fits = 42/42
- 标志：`registered_fixture=true`、`formal_l0=false`、`formal=false`、`non_formal=true`

runner 完成 42 fits 后写出完整 metadata/manifest/status；independent summarizer 从 DGP、split、OOF plan、M/S、B/H、逐 lead 控制变量和各臂公式重建，得到 `verified=true`。连续两次独立汇总的 runner manifest、verification summary 和 verification manifest 哈希完全一致。

## 冻结 rho 门

控制变量路径诊断为

\[
\rho_{E,h}=\frac{\lVert H_{E,h}\widetilde v_{E,h}\rVert_2}
{\lVert H_{E,h}u_{F_{E,h}}\rVert_2+10^{-12}}.
\]

冻结规则：H=1 的 1/1 lead 或 H=6 的至少 4/6 lead 满足 `rho<=0.05`，即归因 STOP。registered fixture 的独立复算如下；H6 列报告六个 lead 中的最小 rho，完整逐 lead 值保存在 verification summary。

| seed | H1 valid | H1 test | H6 valid min | H6 test min | STOP 单元 |
|---:|---:|---:|---:|---:|---|
| 4001 | 0.0681556760 | 0.1419822361 | 0.1788985988 | 0.2878486546 | 无 |
| 4002 | 0.3217424355 | 0.1396382920 | 0.1689821669 | 0.2542473846 | 无 |
| 4003 | **0.0461967350** | 0.1042937197 | 0.0616896960 | 0.1921720413 | **H1 valid：1/1 近零** |

seed 4003 的 24→1 valid 路径低于 0.05，因此 runner `rho_diagnostic_stop=true/rho_stop=true/status=STOP`；independent summarizer 得到同一 STOP。该判定在查看任何 RMSE/MAE 主效应之前成立。

## 完整性与资源

- runner invocation wall time：76.6307 秒；
- task elapsed sum：66.1624 秒；
- `ru_maxrss=4186452` KiB（Linux 口径）；
- 输出约 491 MiB；
- runner manifest SHA-256：`a0eef3fca79eb2fc59dfd7cc0a95a93df8fc933f81225ea668326719c474ab6c`；
- verification summary SHA-256：`7e4d7edb7ff93fbc3cc6593f4a787195e6f1b5606e7baee842bdc865781d8ed0`；
- verification manifest SHA-256：`dbfe71c1210422deb673a5ca174ed4ada1c6050739bf87f8c807bb28ae0feeab`。

完整 491 MiB 产物以 zstd 归档保存于：

```text
/home/hansel/.herdr/artifacts/PatchTST/round11-cf-cv-registered-fixture-59d2c11.tar.zst
SHA-256: 04b82e27dd6899e176bf2f1659798a6b91cbccc9a185f768a418d779aad16dca
size: 9520972 bytes
```

任务 JSON 中可重建的逐行支持索引占据大部分未压缩体积，因此 Git 只保存小型 manifest/metadata/status/verification 和 `artifact_archive.json` 指针；完整 sidecar/NPZ 保留在上述归档中，原 runner manifest 未被改写。

## 阶段决定

Round 11 在 registered non-formal fixture 止损：**不授权 50-seed 正式 synthetic L0，不运行负对照神经训练，不读取真实站点 CSV，不进入 ST 消融或 confirmation。**

禁止用以下操作挽救：更换 4001–4003、放宽 rho 阈值、只看 test 不看 valid、调 bandwidth/support、改变条件变量、删掉 H1、先查看 RMSE 后重定义门，或把 ordinary conditional smearing/log1p 的潜在收益包装成控制变量贡献。

本轮仅能作为可审计的负结果表述：CF-CV-CS 的工程实现和信息边界可重建，但预注册的控制变量路径非近零要求未在全部 registered 单元中成立，因此没有资格进入正式性能筛查。
