# Progress log

日期：2026-08-23。

当前阶段：L1实现与smoke审查完成；正式L1三种子选择尚未运行，第二创新点仍未成立。

## 已完成

- 阅读仓库 AGENTS 指令与研究探索/消融/确认工作流。
- 审阅当前 ST 模型、消融运行器、数据准备、测试、lag 诊断、稳定性报告和原始 CSV/JSON。
- 审阅频域分支两次提交、协议、P0、P1、P1A、独立复核和止损结论。
- 审阅 pollutant-specific-alpha 分支代码、校准和三种子验证结论。
- 使用实时网络检索并深核验 17 组论文/官方代码/许可证，补充 TFT 与 Lag-FLSTM 直接对比并精确核验 iScience 计数/许可。
- 建立证据—主张映射，提出四个实质不同候选并统一打分。
- 保留主候选 MC-CVLRA、备选 RES-Peak；按独立reviewer意见重写2×2、无ST残差L0、L1、M-shuffle和独立确认门。
- 编写交付文档。
- 只读审计北京1013与广州9022原始CSV：各8760×14，时间戳唯一连续，原九候选解析缺失/非数值/非有限均0；另识别PM10平台、pressure双量级、humidity/污染物零值及wind_speed编码风险。
- 落实第二轮复审：为L0所有outer/inner边界加入`H−1`起点embargo与目标时间空交集断言；冻结M/B-flat逐层架构和3.0%参数匹配门；把M-shuffle改为三个训练种子分别执行5个置换并逐种子过门。
- 核验可能的Microsoft Urban Air/KDD 2015来源、NIST压力换算、WMO单位与HJ 633时间发布边界；仓库无数据字典/哈希，故只写“可能来源”。
- 研究负责人代理签字：冻结`P=7`，排除PM10/wind_speed，固定pressure换算、零值缺失、6 h因果ffill与历史回放定义；同步重算L0、M/B-flat和支持文档。

## 未执行（按任务禁止）

- 未修改模型、数据脚本、运行器或测试。
- 未运行训练、正式实验、验证集诊断或测试集评估。
- 未创建提交、未 push。

## 能力使用审计

| 能力 | 使用情况 | 证据 |
|---|---|---|
| 仓库检索/代码审计 | 已用 | 01 文档及相对文件链接 |
| 分支/提交审计 | 已用 | 频域与 alpha commit 链接 |
| 实时网络检索 | 已用 | 02 的论文/代码/许可证表 |
| 学术来源核验 | 已用 | 正式 proceedings、DOI、PMC、作者仓库优先 |
| 统一候选评分 | 已用 | 02 第 4 节 |
| 实验规划（不执行） | 已用 | 03 第 6–11 节 |
| 引用与内部一致性复核 | 已用 | 21个Markdown；相对链接错误0、空白错误0、17组深核验论文来源及数据语义权威来源、61个唯一外部URL |
| 模型实现/训练 | 未用，任务禁止 | 无代码变更、无结果产物 |

## 当前状态

独立reviewer大修与本轮数据语义审计已落实。当前结论为：研究负责人代理签字后，按冻结七变量历史回放协议条件放行进入L0；不是性能放行，第二创新点尚未成立。

## 实验分支 L0 回填（2026-08-23）

- 分支：`experiment/cross-variable-lag-adapter-ablation`；范围严格停在L0，没有实现MC-CVLRA神经模块。
- 精确70/10/20行边界为6132/876/1752；只全量读取时间戳，特征/目标只加载前7008行。test特征、样本、预测和指标构造数均为0。
- 11项L0单元测试通过；覆盖P=7清洗、pressure换算、6h因果ffill、test不构造、边界目标集合、embargo、训练期scaler、特征/lag0、门与复算。
- 168→6主任务：`R_lag`五折平均RMSE为41.175590，相对`R_self`改善2.178407%（4/5），相对`R_recent`改善2.027396%（5/5）。
- 24→1支持任务：`R_lag`五折平均RMSE为17.801861，相对self/recent改善2.075136%/0.335655%，方向5/5与3/5；不参与主门。
- 唯一阶段结论：**PASS**；只允许下一阶段按冻结协议实现L1，仍不是神经性能或正式test证据。
- 首轮7007止点运行因浮点截断判无效；改为精确整数比例并新增回归测试后从头重跑，没有改变任何预注册实验项。
- 独立完整性复算通过：34个目标时间边界、30个alpha组、全部ledger和逐预测指标一致；最大绝对误差`2.14e-14`。

### Capability-use audit（L0）

- Required skills：using-research-writing、paper-orchestration、experiment-results-planning、verification。
- Skills actually used：前三项用于阶段识别、任务包、结果契约和报告边界；verification用于最终语法、测试、复算、diff和审查门。
- Inputs consumed：AGENTS指令、研究工作流、03最终方案、04阶段门、05签字、实验交接协议、北京1013允许列前80%行及全序列时间戳。
- Inputs not used and why：正式test的特征、目标、窗口、预测和指标均未构造；MC-CVLRA/ST模型、PM10、wind_speed、weather、wind_direction均因L0范围或协议禁止未使用。
- Artifacts produced：独立数据模块、L0运行器、复算器、11项单测、逐预测/逐折/alpha/边界/ledger产物、唯一gate JSON和中文报告。
- Verification run：见本分支`plan/cross_variable_lag_adapter_l0/progress.md`最终命令记录。
- Remaining risk：L0只证明单站选择区间线性关联；精确数据血缘、在线报送时刻和原始缺失掩码仍不可证；L1和独立确认尚未执行。

### Capability-use audit（第二轮复审）

- Required skills：using-research-writing、paper-orchestration、experiment-results-planning、writing-core、peer-review、verification。
- Skills actually used：以上技能均用于协议冻结、跨文档追踪和机械验收。
- Inputs consumed：用户三项复审意见、`03`/`04`、traceability、review、progress、notes、experiment protocol与任务包。
- Inputs not used and why：未读取测试结果、未调用训练或模型实现；任务明确禁止。
- Artifacts produced：L0 embargo/断言、M/B-flat固定结构与参数规则、逐训练种子M-shuffle协议及同步审查记录。
- Verification run：参数公式/最优宽度复算与embargo目标集合断言通过；21个Markdown相对链接/空白问题0，逐文件no-index空白错误0，范围仅指定docs目录，HEAD仍为`1a26492`。
- Remaining risk：精确数据血缘/真实报送延迟不可证、预登记种子未消费核验，以及未来实现参数计数与理论公式一致性。

### Capability-use audit（数据语义审计）

- Skills used：academic-search、evidence-driven-writing、paper-orchestration、verification。
- Inputs consumed：北京1013/广州9022全部特征值与时间戳、仓库历史/来源线索、Microsoft/NIST/WMO/MEE官方页面；没有消费模型预测或测试指标。
- Decision：`P=7`，固定历史回放清洗规则，代理签字并条件放行L0。
- Scope：只新增/修改`docs/第二创新点理论探索/` Markdown；未改CSV、代码、结果或Git历史。
- Final verification：21个Markdown、61个唯一外部URL；相对链接/空白错误0；旧九变量张量、旧参数公式和旧未决阶段门冲突0；参数重算与embargo玩具断言通过；Git状态仅列出该文档目录，HEAD保持`1a26492`。

## 实验分支 L1 实现与 Smoke 回填（2026-08-24）

- 起点`c3da238`已由独立reviewer接受L0 PASS；本轮严格停在L1实现、测试和smoke，没有运行正式`2060–2062`三种子选择。
- 新管线只全量读取时间戳，特征限前7008行；test特征、目标值、样本、预测、缓存和指标计数均为0。train/validation/保留test时间戳目标集合两两无交。
- 北京池修正为1001–1036且排除1013；完整性和Pearson只使用原始训练片，训练期筛出17邻站。P=7清洗和顺序直接复用L0，无bfill。
- 实现F00/F01/F10/F11、B1/B2/B-flat/B3；F10从同种子F00最佳checkpoint初始化并冻结PatchTST，适配器共享配对基础checkpoint且前后state hash不变。
- M参数数为1532/6547；B-flat宽15/47、参数1532/6540。首次/后续梯度、显式旁路、完整基础eval/no-grad、组质量和输入替换均有单测与运行时审计。
- 固定M-shuffle置换完整七变量历史窗口，块大小24、种子3101–3105，并在同一删尾子集复算。
- 相关测试最终34/34通过；24→1和168→6均完成1 seed/1 epoch/64 train/48 validation的CPU smoke。联合复算完整性PASS，但gate强制为`NOT_EVALUABLE_SMOKE`，不报告或使用模型优劣。
- 中文审计见[07_L1实现与Smoke审计报告](../07_L1实现与Smoke审计报告.md)，独立计划与双审见`plan/cross_variable_lag_adapter_l1/`。
- 正式输出目录已预登记但未创建；本轮未commit、未push，也未修改原始CSV。

### L1 reviewer P1 修复回填（2026-08-24）

- 独立reviewer确认首轮F10在整个`model.train()`后未重锁PatchTST eval，导致两个旧smoke各有18个持久BatchNorm buffer变化；旧两个CPU smoke及联合汇总全部作废并移至`/tmp/l1_bn_invalid_smoke_20260824_7Gq22k/`。
- F10现于每轮train后强制整个PatchTST子模块eval，只训练空间分支；F00来源checkpoint、F10初始化现场、逐epoch、最佳checkpoint与最终加载均独立计算hash并逐元素断言全部参数/buffer一致。
- 新增真实PatchTST BatchNorm回归测试：一步F10训练后权重、`running_mean`、`running_var`、`num_batches_tracked`不变，空间分支实际更新。相关ST/L0/L1测试现为36/36通过。
- 汇总器直接加载F00/F10 checkpoint核对`patch_tst.*`子state；runner manifest记录HEAD、dirty明细和六个关键L1源码SHA256，正式入口在读数据/建目录前拒绝dirty tree。
- 修复后两个CPU smoke从空目录重跑，F10四阶段持久与完整状态变化数均为0；联合复算CPU部分通过，gate仍为`NOT_EVALUABLE_SMOKE`。本轮产物明确是未提交代码上的预提交工程smoke。
- 已实现168→6、F11、真实512样本完整batch的GPU资源验收脚本。当前容器无`/dev/kfd`/`/dev/dri`且PyTorch设备数为0，实际运行按设计中止，无伪GPU产物；联合复算记录`BLOCKED_NO_EXPOSED_GPU_DEVICE`并保持integrity=false。
- 当前结论修正为：P1工程修复通过，旧CPU smoke已替换；GPU资源验收仍阻断，正式2060–2062不得启动。三个正式目录不存在，未commit/push，原始CSV未改。
