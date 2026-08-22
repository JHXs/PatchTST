## Task Packet

- Scope: 改进 ST-PatchTST 空间门控，并完成退化 PatchTST、旧式中心门控、成对门控三组实验。
- Files to read: `ST_PatchTST_model.py`, `PatchTST.py`, `data_preparation.py`, `docs/ST-PatchTST改进/ST-PatchTST问题诊断与改进方案.md`。
- Files allowed to edit: `ST_PatchTST_model.py`, 新增根目录实验脚本，`plan/`, `tables/`, `figures/`, `docs/实验报告/`。
- Required skills: paper-orchestration, experiment-results-planning, verification。
- Evidence/data inputs: `tsai/data/stations_data/df_station_*.csv` 以及实验生成的 CSV/JSON/NPZ 日志。
- Required artifacts: 配置、逐种子指标、聚合指标、训练历史、预测、空间诊断、中文报告。
- Rejection checks: 基线不是同类退化；模型训练设置不一致；把规划数据写成真实结果；缺失逐种子数据；只报告总体指标而无空间真实性检查。
- Validation commands: `uv run python -m py_compile *.py`; 快速烟雾实验；正式实验；结果文件完整性检查。
