# Data semantics audit task packet

日期：2026-08-23。

## Task Packet

- Scope：只读审计北京1013与广州9022候选变量，核验可能来源和常见单位，形成唯一输入冻结与代理签字决定。
- Files to read：两个中心站原始CSV、仓库数据说明/历史、现有理论文档、权威来源页面。
- Files allowed to edit：仅`docs/第二创新点理论探索/`下Markdown。
- Required skills：academic-search、evidence-driven-writing、paper-orchestration、peer-review、verification。
- Evidence/data inputs：逐列分布、异常值、长平台、重复/采样模式、时间戳；仓库来源线索；官方单位/换算依据。
- Required artifacts：数据语义签字/否决记录；README、主方案、阶段门、L0、M/B-flat参数与全部支持文档同步。
- Rejection checks：不得改CSV、不得凭猜测修值、不得使用测试指标、不得保留旧变量数、不得把历史回放假设写成真实在线可用性。
- Validation commands：审计脚本复算、参数公式复算、变量数/名称关键词一致性、相对链接、空白/style、Git范围和HEAD检查。

## 决策问题

- PM10是否全局排除；pressure是否固定换算或排除；humidity零值是否按缺失处理。
- 其余候选变量是否存在范围、平台、重复或采样异常。
- 在来源与报送延迟不能直接证明时，是否仍能以历史回放操作定义条件放行L0。

## 完成状态

- 唯一决定：代理签字并条件放行L0。
- 输入冻结：`P=7`；排除PM10、wind_speed，固定pressure/humidity/辅助污染物处理。
- 权威记录：[数据语义审计与代理签字](../../05_数据语义审计与代理签字.md)。
- 表述边界：历史回放预测，不宣称真实在线无延迟。
