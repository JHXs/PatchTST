 # ST-PatchTST 改模方案 v1

  ## Summary

  目标是先让多站点模型稳定超过“中心站单站点 PatchTST”基线，不追求一次到位做最复杂的时空建模。
  默认方案改成“中心站主干 + 邻站辅助”架构：中心站历史序列直接交给 PatchTST 主干建模，邻站信息只做可控增强，不再在输入端把所有站点压成一
  个虚拟站点。

  ## Key Changes

  1. 替换现有 ST_PatchTST 的前端融合逻辑。
      - 删除当前 Conv2d(num_stations -> 64 -> 1) 的早期空间压缩。
      - 将输入重排为 [B, num_stations, feat_size, seq_len] 后，明确拆出：
          - center_x: 中心站特征，形状 [B, feat_size, seq_len]
          - neighbor_x: 其他站点特征，形状 [B, num_neighbors, feat_size, seq_len]
  2. 新模型采用“双分支”。
      - 主分支：center_x 直接送入 PatchTST，作为主预测路径。
      - 辅助分支：对 neighbor_x 做轻量聚合，输出与中心站同形状的增强张量 [B, feat_size, seq_len]。
      - 聚合方式固定为：
          - 先在每个邻站上做 Linear/1x1 Conv 的特征投影。
          - 再用一个基于中心站上下文的门控权重，对邻站做加权求和。
          - 输出 neighbor_context 后，与 center_x 做残差相加：enhanced_x = center_x + alpha * neighbor_context。
      - alpha 设为可学习标量，初始化为较小值，如 0.1，防止邻站噪声一开始压过中心站主信号。
  3. PatchTST 只处理增强后的中心站表示。
      - PatchTST 的 c_in 和 c_out 都保持为 feat_size。
      - 训练目标仍为中心站未来 feat_size x pred_len，不改现有标签定义。
  4. 数据准备只做一项关键修正。
      - 保留当前“多站点输入 + 中心站输出”的总体设定。
      - 在 model_params.npz 中额外保存 center_station_id 对应的站点索引，避免模型靠硬编码假设中心站位置。
      - 不改样本切分、不改预测窗口，先保证和现有实验口径一致。
  5. 评估链路统一。
      - main.py 训练 ST 模型时，必须调用 evaluate_st_patchtst(...)，不要再混用 evaluate_patchtst(...)。
      - 输出统一记录 valid/test 的 MSE、MAE，并和单站点 PatchTST 同口径比较。

  ## Model Interface Changes

  - ST_PatchTST.__init__ 增加或明确以下配置：
      - center_station_idx
      - num_stations
      - feat_size
      - neighbor_hidden_dim
      - neighbor_dropout
  - forward(x) 输入输出保持不变：
      - 输入 [B, num_stations * feat_size, seq_len]
      - 输出 [B, feat_size, pred_len]

  ## Test Plan

  1. 静态检查
      - py_compile 通过。
      - 前向传播 shape 检查通过，覆盖 batch size 1 和正常 batch。
  2. 单元级行为检查
      - 验证 center_x 与 neighbor_x 的切分索引正确。
      - 验证 neighbor_context、enhanced_x、模型输出 shape 全部匹配。
      - 验证 alpha 初值生效，且邻站分支关闭时模型退化为接近单站点主干。
  3. 对照实验
      - Baseline A：单站点 PatchTST。
      - Baseline B：当前旧版 ST_PatchTST。
      - Candidate C：新“双分支 ST-PatchTST”。
      - 保持相同数据划分、相同 epoch、相同 lr_find 流程。
      - 以 valid MSE 为主排序，test MSE/MAE 为最终确认。
  4. 消融实验
      - alpha = 0 的退化版本，确认主干本身没有被新代码破坏。
      - 去掉门控、仅邻站平均聚合的版本，确认门控是否有实际收益。

  ## Assumptions

  - 当前优先目标是“稳定超过单站点 PatchTST”，不是最大化上限。
  - 中心站自己的历史是最强信号，邻站只能做辅助，不能替代主干。
  - 这轮不引入图网络、双重 Transformer、全站点联合输出等高复杂度方案。
  - 这轮不改数据筛站规则；如果新架构仍无提升，再单独处理“按 PM2.5 相关性筛站却预测全部变量”的数据问题。

  ## 后续执行顺序

  1. 第一步：保存方案
  2. 第二步：检查当前 ST_PatchTST 实现
  3. 第三步：列出需要修改的文件
  4. 第四步：再开始编码
