# L0 规范符合性审查

审查日期：2026-08-24。结论：**通过**。

## 范围与协议

- 当前分支只新增独立L0数据模块、ridge运行器、复算器、单测、结果和研究记录；没有MC-CVLRA/ST神经模块或torch依赖。
- 北京1013、P=7顺序、六lag、三个ridge、alpha网格、inner 85/15、五个outer折与硬门均与签字协议一致。
- 源CSV在git状态中无修改；禁用列只出现在禁止名单常量和测试断言中，没有进入`read_csv`允许列。

## 数据与边界

- 全量只读8760个时间戳；允许特征/目标只加载前7008行，精确边界为6132/876/1752。
- train/validation独立最多6h因果ffill；代码无`bfill`。
- 168→6与24→1分别保留6800/6984个共同完整样本；残余缺失窗口均为0。
- 7个Q边界、5个outer边界、5个inner边界每任务共17个；34个目标时间集合交集均为0。168→6所有边界embargo=5，24→1=0。
- test特征、样本、预测和指标构造计数均为0；结果scope统一为`selection_interval_L0_not_test_not_neural`。

## 拟合与结果契约

- scaler、inner ridge、alpha selector、outer scaler和outer ridge均有fit_start、fit_end、row_count、config_hash和state_hash。
- 独立复算确认inner/outer scaler与ridge的fit_end严格早于对应验证目标起点，且所有拟合/处理对象早于test起点。
- 30个“任务×折×模型”alpha选择组均能从inner分数复算；数值平局取更大alpha。
- 逐预测可重建逐折逐horizon、RMSE_all、配对差和唯一gate；最大绝对误差`2.14e-14`。

## 工程异常

首次运行的7007止点违反精确70/10/20，已判无效、移出权威目录并保留说明。修复只改变边界算术实现，未改变任何冻结实验项；回归测试锁定6132/876/1752。

## 验收结论

用户任务1–9的L0范围、产物与最终验证均满足。没有发现需要改变结果或协议的问题。
