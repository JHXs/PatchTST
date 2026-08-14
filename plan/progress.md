# 进度记录

## 当前阶段

- 阶段：S3 Experiments
- 状态：alpha 搜索实验记录与真实数据图已完成核验
- 约束：只写独立实验档案，不扩展为完整论文

## 已确认信息

- 中文硕士毕业论文
- 新一代电子信息技术
- 暂定题目：《基于时空特征融合与 PatchTST 的多站点空气质量预测研究》
- Markdown 输出，配 PNG/SVG 科研图

## 执行记录

- 已建立实验协议、结果表结构、证据追踪表与图表数据清单。
- 已将三轮主搜索、双随机种子复核和 O3 补充消融整理为三个真实数据 CSV。
- 已生成 3 张论文风格图，每张均包含 450 DPI PNG 和可编辑 SVG。
- 已扩写 `docs/ST-PatchTST改进/alpha搜索结果.md`，包含方法公式、完整过程、结果解释、证据边界和复现入口。
- 已完成规格审查与写作质量审查，记录见 `plan/review/`。
- 最终核验：3 个 PNG 均为 450 DPI，3 个 SVG 可解析，Markdown 图片链接 3/3 有效；图表 CSV 与原始实验 CSV 在 `1e-9` 容差内一致；Python 语法检查和 `git diff --check` 通过。

## Capability-use audit

- 研究写作编排：采用任务包、项目概览、提纲和进度记录控制范围，未扩写为整篇论文。
- 实验结果规划：所有主张均绑定真实 CSV 或本地实验产物；未使用模拟数据补齐结果。
- 科研制图：图表从 CSV 独立生成，使用中文字体、色盲友好配色，并同时导出 PNG/SVG。
- 核验：执行图片尺寸/DPI、SVG 解析、CSV 溯源、Markdown 图片链接、Python 语法与 Git 空白错误检查。

## 补强实验运行记录

- 预注册矩阵：4 种 alpha 策略 × 3 个随机种子，30 epoch，batch size=16，学习率 `1e-4`。
- 设备核查：RX 7600 的 ROCm 训练烟雾测试通过；第二块集成显卡报 `device kernel image is invalid`，未纳入正式实验。
- `alpha_zero` 与 `learnable_alpha` 的单 epoch 烟雾测试、模型导出和 CSV 写入均通过。
- 正式实验仅在 RX 7600 顺序运行，日志为 `plan/logs/alpha_ablation_3seed_4condition.log`。
- 正式矩阵已完成：12/12 个条件—种子组合，每组 30 epoch；压缩后的可读训练记录为 `plan/logs/alpha_ablation_training_summary.log`。
- 三 seed 均值：alpha=0 为 0.716219，统一 alpha 为 0.702654，逐污染物 alpha 为 0.702658，可学习 alpha 为 0.704894。
- 统一与逐污染物策略的均值差约 0.000005；研究主张已从“逐污染物更准确”收窄为“空间融合稳定有效，逐污染物门控提供通道解释与负迁移抑制”。
- 已生成补强图 4—5 的 450 DPI PNG/SVG，并将统计结果和结论写入实验记录第 7 节。
- 最终核验通过：12/12 实验矩阵与统计表重算一致；模型导出可加载；可学习 alpha 保持在 `[0,0.4]`；alpha=0 输出与中心 PatchTST 分支逐元素完全一致；五张图均为 450 DPI 且 SVG 可解析；Markdown 图片链接 5/5 有效；补强脚本未索引测试集；语法与 `git diff --check` 通过。

## 严格 PatchTST 基线运行记录

- 未运行会覆盖资产且硬编码广州 9022 的 `data_preparation_single.py`；直接从北京多站点数组隔离抽取中心站 1013。
- 三个正式 seed 均完成 30 epoch，训练日志包含 90 条 epoch 汇总；严格基线 MSE 为 0.714366、0.716932、0.715060。
- 严格 PatchTST valid MSE 为 0.715453 ± 0.001327；统一/逐污染物空间策略均平均改善约 1.79%，可学习策略改善约 1.48%。
- 原始运行进程在验证推理后读取 recorder，导致 CSV 的 `epochs_completed` 初始误记为 1；日志确认每个 seed 均为 30 epoch，CSV 已校正，脚本已改为在验证推理前捕获该字段。
- 已生成图 6 的 450 DPI PNG/SVG、四个严格基线汇总 CSV 和审查记录；训练脚本未索引测试集。
- 严格基线最终核验通过：3/3 seed、90 条 epoch 日志、汇总与配对改善重算一致；模型导出可重新加载并输出 `(2, 12, 24)`；六张图及 SVG、Markdown 6/6 图片链接、无测试集/数据写入检查、`uv run` 语法检查和 `git diff --check` 全部通过。
