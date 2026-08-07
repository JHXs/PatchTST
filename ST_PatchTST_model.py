import torch
import torch.nn as nn
import numpy as np
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.training import GradientClip
from tsai.models.PatchTST import PatchTST
from tsai.all import *
from evaluation_utils import (
    build_results_dataframes,
    build_split_evaluation,
    infer_num_metrics,
    load_target_names,
    print_evaluation_results,
)


class ST_PatchTST(nn.Module):
    """
    ST-PatchTST (Spatial-Temporal PatchTST) v9 最终版

    架构（短梯度残差预测）:
      center_x ──→ PatchTST ──────────→ output_center  [B, C, H]
      neighbor_x ─→ Cross-Attention ──→ neighbor_pred_head ─→ output_neighbor
      output = output_center + alpha * output_neighbor

    关键设计:
      - 邻站分支做独立预测，梯度不经过 PatchTST 主干（短梯度路径）
      - 空间差分 (neighbor - center) 代替原始值，捕获空间梯度
      - 7h 时间卷积核，捕获污染物传输延迟
      - 固定 alpha，仅作用于污染物特征（索引 0-5）

    输入: [Batch, num_stations * feat_size, seq_len]
    输出: [Batch, feat_size, pred_len]
    """

    def __init__(
        self,
        c_in,                     # 输入通道数 = num_stations * feat_size（TSForecaster 自动传入）
        c_out,                    # 输出通道数（TSForecaster 自动传入）
        seq_len,                  # 输入序列长度（TSForecaster 自动传入）
        pred_dim,                 # 预测长度，可能是 [feat_size, horizon]（TSForecaster 自动传入）
        num_stations,             # 站点总数
        feat_size,                # 每个站点的特征数（默认 12）
        center_station_idx,       # 中心站在站点列表中的索引
        # --- 邻站分支超参 ---
        neighbor_d_model=24,      # cross-attention 隐层维度
        neighbor_n_heads=4,       # attention 头数
        neighbor_dropout=0.3,     # 投影层 dropout
        neighbor_attn_dropout=0.1,# attention 内部 dropout
        neighbor_temporal_kernel=7,# 时序卷积核大小（奇数，±kernel//2 小时窗口）
        alpha=0.15,               # 邻站贡献度: float 统一值 或 [PM2.5,PM10,NO2,CO,O3,SO2]。0.0=基线
        # --- PatchTST 骨干超参 ---
        n_layers=3, n_heads=4, d_model=16, d_ff=128,
        attn_dropout=0.0, dropout=0.2,
        patch_len=4, stride=2, padding_patch=True,
        **kwargs
    ):
        super().__init__()

        # ---- 基础属性 ----
        self.num_stations = int(num_stations)
        self.feat_size = int(feat_size)
        self.seq_len = seq_len
        self.pred_len = pred_dim
        self.pred_horizon = int(pred_dim) if isinstance(pred_dim, (int, float)) else int(pred_dim[-1])
        self.center_station_idx = int(center_station_idx)
        self.num_neighbors = self.num_stations - 1
        self.neighbor_d_model = int(neighbor_d_model)
        self.neighbor_n_heads = int(neighbor_n_heads)

        # ---- 输入校验 ----
        if c_in != self.num_stations * self.feat_size:
            raise ValueError(f"c_in={c_in}，期望 {self.num_stations * self.feat_size}")
        if not 0 <= self.center_station_idx < self.num_stations:
            raise ValueError(f"center_station_idx={self.center_station_idx} 越界 [0, {self.num_stations-1}]")
        if self.num_neighbors < 1:
            raise ValueError("ST_PatchTST 需要 num_stations >= 2")
        if self.neighbor_d_model % self.neighbor_n_heads != 0:
            raise ValueError(f"neighbor_d_model ({self.neighbor_d_model}) 必须整除 n_heads ({self.neighbor_n_heads})")

        # ---- 邻站索引 ----
        neighbor_indices = [i for i in range(self.num_stations) if i != self.center_station_idx]
        self.register_buffer("neighbor_indices", torch.tensor(neighbor_indices, dtype=torch.long))

        # ---- 时间卷积核（确保奇数，用于 'same' padding）----
        kernel = int(neighbor_temporal_kernel)
        if kernel % 2 == 0:
            kernel += 1

        # ============ 邻站分支：空间差分 → Cross-Attention → 残差预测 ============

        # 中心站投影：feat_size → d_model，kernel>1 捕获局部时序
        self.center_proj = nn.Sequential(
            nn.Conv1d(self.feat_size, self.neighbor_d_model, kernel_size=kernel, padding='same'),
            nn.GELU(),
            nn.Dropout(neighbor_dropout),
        )

        # 邻站投影（所有邻站共享权重），同样有时序卷积
        self.neighbor_proj = nn.Sequential(
            nn.Conv1d(self.feat_size, self.neighbor_d_model, kernel_size=kernel, padding='same'),
            nn.GELU(),
            nn.Dropout(neighbor_dropout),
        )

        # 逐时间步 Cross-Attention：中心站(t) 作 Query，所有邻站(t) 作 Key/Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.neighbor_d_model, num_heads=self.neighbor_n_heads,
            dropout=neighbor_attn_dropout, batch_first=True,
        )
        self.neighbor_norm = nn.LayerNorm(self.neighbor_d_model)

        # 时域平滑 + 投影回 feat_size
        self.temporal_smooth = nn.Sequential(
            nn.Conv1d(self.neighbor_d_model, self.neighbor_d_model, kernel_size=kernel, padding='same'),
            nn.GELU(),
        )
        self.neighbor_out = nn.Conv1d(self.neighbor_d_model, self.feat_size, kernel_size=1)

        # 邻站 → 残差预测头：[B, feat_size, seq_len] → [B, feat_size, pred_horizon]
        self.neighbor_pred_head = nn.Sequential(
            nn.Conv1d(self.feat_size, 32, kernel_size=kernel, padding='same'),
            nn.GELU(), nn.Dropout(neighbor_dropout),
            nn.Conv1d(32, self.feat_size, kernel_size=kernel, padding='same'),
            nn.AdaptiveAvgPool1d(self.pred_horizon),
        )
        # 零初始化输出层 → 训练初期邻站分支输出为 0，从纯中心站开始
        nn.init.zeros_(self.neighbor_pred_head[-2].weight)
        if self.neighbor_pred_head[-2].bias is not None:
            nn.init.zeros_(self.neighbor_pred_head[-2].bias)

        # ============ Alpha 门控（逐污染物独立）============
        # 污染物特征（索引 0-5）: PM2.5/PM10/NO2/CO/O3/SO2 → 有传输时延，用邻站
        # 气象特征（索引 6-11）: 空间均匀，alpha=0 纯中心站
        # alpha 可以是单一 float（所有污染物相同）或 6 元素列表（逐污染物独立）
        self.pollutant_idx = [0, 1, 2, 3, 4, 5]
        if isinstance(alpha, (int, float)):
            alpha_vals = [float(alpha)] * 6
        else:
            alpha_vals = [float(a) for a in alpha]
            if len(alpha_vals) != 6:
                raise ValueError(f"alpha 列表长度应为 6（对应 6 种污染物），当前为 {len(alpha_vals)}")
        self.register_buffer("alpha", torch.tensor(alpha_vals))  # [6]

        # ============ PatchTST 主干（仅处理中心站）============
        self.patch_tst = PatchTST(
            c_in=feat_size, c_out=feat_size,
            seq_len=seq_len, pred_dim=self.pred_len,
            n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff,
            dropout=dropout, attn_dropout=attn_dropout,
            patch_len=patch_len, stride=stride, padding_patch=padding_patch,
        )

    # ========== 前向传播 ==========

    def _reshape_input(self, x):
        """[B, N*C, T] → [B, N, C, T]"""
        return x.reshape(x.shape[0], self.num_stations, self.feat_size, -1)

    def _split_center_and_neighbors(self, x):
        """拆分中心站 [B, C, T] 与邻站 [B, N-1, C, T]"""
        return x[:, self.center_station_idx], x.index_select(1, self.neighbor_indices)

    def _build_neighbor_context(self, center_x, neighbor_x):
        """
        Cross-Attention 邻站特征提取。

        流程:
          1. 计算空间差分 (neighbor - center)，捕获空间梯度
          2. 7h 时间卷积投影，融入 ±3h 时序上下文
          3. 逐时间步 Cross-Attention: 每个时刻独立地对邻站做注意力
          4. 残差 + LayerNorm → 时域平滑 → 投影回 feat_size

        输入: center_x [B, C, T], neighbor_x [B, N-1, C, T]
        输出: neighbor_context [B, C, T]
        """
        B, Nn, _, T = neighbor_x.shape

        # 空间差分：突显邻站与中心站的差异（如污染梯度方向）
        spatial_diff = neighbor_x - center_x.unsqueeze(1)

        # Q: 中心站 [B*T, 1, d_model]
        q = self.center_proj(center_x).permute(0, 2, 1)
        q = q.reshape(B * T, 1, self.neighbor_d_model)

        # K/V: 空间差分 → 逐邻站时序投影 → [B*T, Nn, d_model]
        diff_flat = spatial_diff.reshape(B * Nn, self.feat_size, T)
        kv = self.neighbor_proj(diff_flat)
        kv = kv.reshape(B, Nn, self.neighbor_d_model, T).permute(0, 3, 1, 2)
        kv = kv.reshape(B * T, Nn, self.neighbor_d_model)

        # Cross-Attention + 残差 + LayerNorm
        attn_out, _ = self.cross_attn(q, kv, kv)           # [B*T, 1, d_model]
        attn_out = attn_out.squeeze(1).reshape(B, T, self.neighbor_d_model)
        attn_out = self.neighbor_norm(q.reshape(B, T, self.neighbor_d_model) + attn_out)

        # 时域平滑 → 投影回 C
        attn_out = attn_out.transpose(1, 2)                # [B, d_model, T]
        attn_out = self.temporal_smooth(attn_out)
        return self.neighbor_out(attn_out)                 # [B, C, T]

    def forward(self, x):
        """
        x: [B, num_stations * feat_size, seq_len]
        返回: [B, feat_size, pred_len]

        两步预测:
          1. PatchTST 纯中心站 → output_center
          2. 邻站分支 → output_neighbor（残差）
          融合: output = output_center + alpha * output_neighbor（仅污染物）
        """
        x = self._reshape_input(x)
        center_x, neighbor_x = self._split_center_and_neighbors(x)

        output_center = self.patch_tst(center_x)

        neighbor_ctx = self._build_neighbor_context(center_x, neighbor_x)
        output_neighbor = self.neighbor_pred_head(neighbor_ctx)

        # 仅污染物特征融合邻站残差
        alpha_vec = torch.zeros(self.feat_size, device=center_x.device)
        alpha_vec[self.pollutant_idx] = self.alpha
        return output_center + alpha_vec.view(1, -1, 1) * output_neighbor


# =====================================================================
# 训练 & 评估
# =====================================================================

def train_st_patchtst(X, y, splits, preproc_pipe, exp_pipe):
    """训练 ST_PatchTST。alpha=0 时等价于纯 PatchTST 基线。"""

    # 读取 data_preparation.py 保存的参数
    params = np.load("tsai/data/model_params.npz")
    for key in ["num_stations", "feat_size", "n_vars_total",
                "fcst_history", "fcst_horizon", "center_station_idx"]:
        if key not in params.files:
            raise KeyError(f"model_params.npz 缺少字段: {key}，请先运行 data_preparation.py")

    arch_config = {
        # PatchTST 骨干
        "n_layers": 3, "n_heads": 4, "d_model": 16, "d_ff": 128,
        "attn_dropout": 0.0, "dropout": 0.2,
        "patch_len": 4, "stride": 2, "padding_patch": True,
        # 站点信息
        "num_stations": int(params["num_stations"]),
        "feat_size": int(params["feat_size"]),
        "center_station_idx": int(params["center_station_idx"]),
        # 邻站分支（v9 最终参数）
        "neighbor_d_model": 24, "neighbor_n_heads": 4,
        "neighbor_dropout": 0.3, "neighbor_attn_dropout": 0.1,
        "neighbor_temporal_kernel": 7,
        # 逐污染物独立 alpha: PM2.5高 SO2中高 PM10/NO2中 CO/O3低，设为 0 即基线
        "alpha": [0.20, 0.10, 0.10, 0.05, 0.02, 0.15],
        #         PM2.5 PM10  NO2   CO    O3   SO2
    }

    print("\n========== ST-PatchTST 配置 ==========")
    for k, v in arch_config.items():
        print(f"  {k}: {v}")

    cbs = [
        GradientClip(1.0),
        SaveModelCallback(monitor="valid_loss", fname="ST_PatchTST_best"),
        EarlyStoppingCallback(monitor="valid_loss", patience=25),
    ]

    learn = TSForecaster(
        X, y, splits=splits, batch_size=16, path="tsai/models",
        pipelines=[preproc_pipe, exp_pipe],
        arch=ST_PatchTST, arch_config=arch_config,
        metrics=[rmse, mse, mae], cbs=cbs,
    )

    lr = min(learn.lr_find().valley, 1e-4)
    print(f"[OK] 学习率: {lr:.2e}")

    learn.fit_one_cycle(50, lr_max=lr)
    learn.export("ST_PatchTST.pt")
    print("[OK] 训练完成，模型已导出")
    return learn


def evaluate_st_patchtst(learn, X, y, splits):
    """评估 ST_PatchTST 模型"""
    target_names = load_target_names(expected_count=infer_num_metrics(y))
    summary_rows, per_metric_rows = [], []

    for split_name, split_idxs in (("valid", splits[1]), ("test", splits[2])):
        preds = to_np(learn.get_X_preds(X[split_idxs])[0])
        print(f"\n{split_name} 预测形状: {preds.shape}")

        summary_row, rows = build_split_evaluation(y[split_idxs], preds, split_name, target_names)
        summary_rows.append(summary_row)
        per_metric_rows.extend(rows)

    summary_df, per_metric_df = build_results_dataframes(summary_rows, per_metric_rows)
    print_evaluation_results(summary_df, per_metric_df)
    return {"summary": summary_df, "per_metric": per_metric_df}


if __name__ == "__main__":
    print("ST-PatchTST v9 模型定义完成")
    print("  训练: train_st_patchtst(X, y, splits, preproc_pipe, exp_pipe)")
    print("  评估: evaluate_st_patchtst(learn, X, y, splits)")
    print("  基线: arch_config['alpha'] = 0.0 → 等价于纯 PatchTST")
