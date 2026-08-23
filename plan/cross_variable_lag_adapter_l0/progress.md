# L0 进度记录

日期：2026-08-23。

当前阶段：S5 Review，L0 权威运行与独立复算完成，唯一状态为 PASS。

- [x] 完整读取工作流、最终推荐方案、实验协议、数据语义签字与阶段门记录。
- [x] 建立 L0 专用任务包、结果契约和方法—实验追踪。
- [x] 实现数据模块、运行器与独立复算器。
- [x] 单元测试先行并通过（最终11项）。
- [x] 运行北京 1013 的 168→6 与 24→1。
- [x] 生成唯一 PASS、中文报告和决策记录。
- [x] 独立复算逐预测指标、配对差、alpha、34个边界、fit范围和gate。
- [x] 记录并隔离首次7007浮点边界无效运行；以精确6132/876/1752边界从头重跑。
- [x] 完成规范审查、质量审查与最终验证命令。

## 权威 L0 结果

- 168→6：`R_self=42.092536`、`R_recent=42.027657`、`R_lag=41.175590`；lag相对self/recent改善2.178407%/2.027396%，方向4/5与5/5。
- 24→1（支持性）：`R_self=18.179102`、`R_recent=17.861815`、`R_lag=17.801861`；lag相对self/recent改善2.075136%/0.335655%，方向5/5与3/5。
- 唯一阶段门：**PASS**。只允许下一阶段按冻结协议进入L1；当前任务不实现L1。
- 数据访问：全量只读时间戳8760行；选择特征7008行；test特征/样本/预测/指标均为0。

## 执行命令

工作环境已有完整只读`uv`环境，本次在受限容器中用`UV_PROJECT_ENVIRONMENT`指向该环境，并把`UV_CACHE_DIR`放到`/tmp`；核心命令为：

```bash
uv run python -m unittest -v test_cross_variable_lag_l0.py
uv run python run_cross_variable_lag_l0.py \
  --tasks 168:6,24:1 \
  --output-dir experiments/results/cross_variable_lag_adapter/l0
uv run python summarize_cross_variable_lag_l0.py \
  --output-dir experiments/results/cross_variable_lag_adapter/l0
```

## 最终验证（2026-08-24）

- `uv run python -m py_compile *.py`：通过。
- `uv run python -m unittest -v test_cross_variable_lag_l0.py`：11/11通过。
- 独立结果复算：PASS；34个目标时间边界、30个alpha组、全部fit范围和13项完整性检查通过，最大绝对数值误差`2.14e-14`。
- `git diff --check`：通过；新增未跟踪源码/文档另以`git diff --no-index --check`检查通过。
- 内部相对链接、gate/integrity一致性和源CSV工作区状态：通过；原始CSV无修改。
- 未commit、未push。

### Capability-use audit

- Required skills：using-research-writing、paper-orchestration、experiment-results-planning、peer-review、verification。
- Skills actually used：全部使用；分别约束阶段路由、任务包/双审、结果契约、偏倚与表述审查、最终机械验证。
- Inputs consumed：用户AGENTS指令、研究工作流、最终推荐方案、阶段门、数据语义签字、实验交接协议、北京1013全量时间戳与选择区间允许列。
- Inputs not used and why：正式test特征/目标/样本/预测/指标、MC-CVLRA/ST代码路径、禁用四列均因L0范围和协议禁止未使用。
- Artifacts produced：3个实现/复算模块、11项单测、L0专用plan与双审、方向文档回填、19个权威结果文件和1份无效运行说明。
- Verification run：见“最终验证”各项；全部通过。
- Remaining risk：L0只有北京1013单年度选择区间线性证据；五折不是独立随机重复；L1、2×2与独立确认均尚未执行。
