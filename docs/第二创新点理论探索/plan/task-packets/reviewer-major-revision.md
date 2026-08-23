# Reviewer major-revision task packet

日期：2026-08-23。

## Task Packet

- Scope：只修订理论协议文档，落实两轮复审，不实现、不训练、不读取测试形成方案。
- Files to read：`03_最终推荐方案.md`、`04_阶段门决策记录.md`及`plan/review/`、`plan/progress.md`、`plan/notes.md`。
- Files allowed to edit：仅`docs/第二创新点理论探索/`下Markdown。
- Required skills：using-research-writing、paper-orchestration、experiment-results-planning、writing-core、peer-review、verification。
- Evidence/data inputs：用户复审条目、已有理论文档和只读仓库审计事实；不消费新实验数据。
- Required artifacts：主协议、阶段门、traceability、quality/spec review、progress/notes和实验交接同步。
- Rejection checks：任一文档保留起点块自然互斥、未冻结架构、性能选宽度、shuffle跨训练种子合并或旧阶段门矛盾即拒绝。
- Validation commands：参数公式与embargo断言、相对链接、空白/style、关键词、`git diff --check`、逐文件no-index和Git范围。

| Reviewer要求 | 落点 |
|---|---|
| 1 轴/接口区分、2×2、beta边界 | `03`第2、4、5节；`04` |
| 2 L0 stacking/泄漏修复 | `03`第8节 |
| 3 B1硬门、普通MLP、B3降级 | `03`第6、9节 |
| 4 广州完整预注册 | `03`第11.2节 |
| 5 TFT/Lag-FLSTM与原创性 | `02`第2、4.1节；`03`第1、13节 |
| 6 外部/逻辑形状、唯一组门、首步梯度 | `03`第3–4节 |
| 7 直接数据审计与语义风险 | `01`第7节；`03`第7节 |
| 8 iScience计数/许可 | `02` S11 |
| 9 自动无泄漏条件与候选池 | `03`第7.2节 |
| 10 固定M-shuffle与1.0%门 | `03`第6、9、10节 |
| 11 评分锚点和证据降级 | `02`第4节 |
| 12 README/gate/evidence/traceability/review同步 | README、`04`、`plan/` |

## 第二轮复审追加验收

| 要求 | 落点 |
|---|---|
| L0 outer/inner `H−1`起点embargo、目标时间空交集断言 | `03`第7.2、8.1、8.2节；`04`；traceability |
| M/B-flat逐层架构、参数公式、候选宽度和失败上限 | `03`第4、6节；`04`；traceability |
| 每训练种子同一删尾子集的5次M-shuffle与逐种子50%门 | `03`第6、9节；`04`；traceability |

当时验收结论要求“人工前置项后仅条件放行L0”。后续用户已授权研究负责人代理签字，并在[数据语义审计任务包](data-semantics-audit.md)与[签字记录](../../05_数据语义审计与代理签字.md)中完成前置项；当前结论更新为按冻结`P=7`协议条件放行L0，仍不是性能放行或已成立第二创新点。
