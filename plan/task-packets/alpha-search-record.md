# Task Packet: alpha 搜索实验记录
- Scope：将已经完成的 alpha 搜索整理为一份独立、可追溯的中文 Markdown 实验记录。
- Files to read：`ST_PatchTST_model.py`、`alpha_search.py`、搜索 CSV/JSON、原改进指标记录。
- Files allowed to edit：本任务的计划文件、`figures/`、`tables/`、目标 Markdown 文档。
- Required skills：paper-orchestration、experiment-results-planning、figures-python、verification。
- Evidence/data inputs：真实的三轮搜索、双 seed 复核、O3 补充消融、既有基线/v10 指标。
- Required artifacts：数据 CSV、数据清单、绘图脚本、PNG/SVG、嵌图实验记录。
- Rejection checks：不得把 test 用于 alpha 选择；不得把单 seed 结果写成普适最优；不得使用 mock 数据；必须说明尺度不可辨识和 O3 补测时序。
- Validation commands：运行绘图脚本、检查 PNG/SVG、核对 CSV 与 Markdown 数值、`git diff --check`。
