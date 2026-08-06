import torch
import torch.nn as nn
import numpy as np
# import pandas as pd
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.training import GradientClip
from tsai.models.PatchTST import PatchTST
# import sklearn.pipeline
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
    时空PatchTST模型

    架构:
    1. 主分支: 中心站历史序列直接送入 PatchTST 主干
    2. 辅助分支: 邻站信息经过轻量投影 + 门控聚合后形成残差增强
    3. 时间预测层: PatchTST 只处理增强后的中心站表示

    输入: [Batch, num_stations * feat_size, seq_len]
    输出: [Batch, feat_size, pred_len]
    """

    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        pred_dim,
        num_stations,
        feat_size,
        center_station_idx,
        neighbor_d_model=32,
        neighbor_n_heads=4,
        neighbor_dropout=0.1,
        neighbor_attn_dropout=0.0,
        neighbor_temporal_kernel=7,
        alpha_max=0.2,
        alpha_init=0.1,
        # PatchTST 骨干网络参数（由 TSForecaster 通过 **arch_config 展开后直接传入）
        n_layers=3,
        n_heads=4,
        d_model=16,
        d_ff=128,
        attn_dropout=0.0,
        dropout=0.2,
        patch_len=24,
        stride=2,
        padding_patch=True,
        **kwargs
    ):
        """
        参数:
        - c_in: 输入通道数 (num_stations * feat_size) - 由TSForecaster自动传递
        - c_out: 输出通道数 (feat_size) - 由TSForecaster自动传递
        - seq_len: 输入序列长度 - 由TSForecaster自动传递
        - pred_dim: 预测序列长度 - 由TSForecaster自动传递
        - num_stations: 站点数量
        - feat_size: 每个站点的特征数量
        - center_station_idx: 中心站在站点维度中的索引
        - neighbor_d_model: 邻站 cross-attention 的模型维度
        - neighbor_n_heads: cross-attention 头数
        - neighbor_dropout: 邻站辅助分支投影层 dropout
        - neighbor_attn_dropout: cross-attention 内部 dropout
        - neighbor_temporal_kernel: 时域平滑卷积核大小（奇数）
        - alpha_max: 邻站残差系数上限，设为 0 时退化为纯中心站主干
        - alpha_init: 邻站残差系数初值

        PatchTST 骨干网络参数（通过 arch_config 展开后直接传入，与 PatchTST.py 对标）：
        - n_layers: 编码器层数
        - n_heads: attention 头数
        - d_model: 模型维度
        - d_ff: 前馈网络维度
        - attn_dropout: attention dropout
        - dropout: dropout 比率
        - patch_len: patch 长度
        - stride: patch 步长
        - padding_patch: 是否对最后一个 patch 做 padding
        """
        super().__init__()

        self.num_stations = int(num_stations)
        self.feat_size = int(feat_size)
        self.seq_len = seq_len
        # TSForecaster 传入的 pred_dim 可能是 [feat_size, horizon] 列表
        self.pred_len = pred_dim
        self.pred_horizon = int(pred_dim) if isinstance(pred_dim, (int, float)) else int(pred_dim[-1])
        self.center_station_idx = int(center_station_idx)
        self.num_neighbors = self.num_stations - 1
        self.neighbor_d_model = int(neighbor_d_model)
        self.neighbor_n_heads = int(neighbor_n_heads)
        self.max_alpha = float(alpha_max)
        self.alpha_init = float(alpha_init)

        # 验证输入维度
        expected_c_in = self.num_stations * self.feat_size
        if c_in != expected_c_in:
            raise ValueError(f"c_in={c_in}, 但期望输入通道数为 {expected_c_in}")
        self.tsai_c_out = c_out
        if not 0 <= self.center_station_idx < self.num_stations:
            raise ValueError(
                f"center_station_idx={self.center_station_idx} 超出有效范围 [0, {self.num_stations - 1}]"
            )
        if self.num_neighbors < 1:
            raise ValueError("ST_PatchTST 需要至少 1 个邻站，当前 num_stations 必须大于 1")
        if self.neighbor_d_model < 1:
            raise ValueError(f"neighbor_d_model={self.neighbor_d_model} 必须大于 0")
        if self.neighbor_d_model % self.neighbor_n_heads != 0:
            raise ValueError(
                f"neighbor_d_model ({self.neighbor_d_model}) 必须能被 neighbor_n_heads ({self.neighbor_n_heads}) 整除"
            )

        neighbor_indices = [idx for idx in range(self.num_stations) if idx != self.center_station_idx]
        self.register_buffer(
            "neighbor_indices",
            torch.tensor(neighbor_indices, dtype=torch.long),
            persistent=False
        )

        # ===== Cross-Attention 邻站分支（时延感知） =====
        # 中心站投影: feat_size → neighbor_d_model
        # kernel_size=7: ±3小时时序上下文，让 Query 感知局部时序模式
        temporal_window = int(neighbor_temporal_kernel)
        if temporal_window % 2 == 0:
            temporal_window += 1
        self.center_proj = nn.Sequential(
            nn.Conv1d(self.feat_size, self.neighbor_d_model, kernel_size=temporal_window, padding='same'),
            nn.GELU(),
            nn.Dropout(neighbor_dropout),
        )
        # 邻站投影: feat_size → neighbor_d_model（所有邻站共享权重）
        # kernel_size=7: 邻站 ±3h 时序上下文，捕获污染物传输延迟
        self.neighbor_proj = nn.Sequential(
            nn.Conv1d(self.feat_size, self.neighbor_d_model, kernel_size=temporal_window, padding='same'),
            nn.GELU(),
            nn.Dropout(neighbor_dropout),
        )
        # Multi-head Cross-Attention: Q(中心站) 逐时间步 attention 到 K/V(所有邻站)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.neighbor_d_model,
            num_heads=self.neighbor_n_heads,
            dropout=neighbor_attn_dropout,
            batch_first=True,
        )
        self.neighbor_norm = nn.LayerNorm(self.neighbor_d_model)
        # 时域平滑: 小卷积核捕获邻站上下文的局部时序连续性
        temporal_kernel = int(neighbor_temporal_kernel)
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1  # 确保奇数
        self.temporal_smooth = nn.Sequential(
            nn.Conv1d(
                self.neighbor_d_model, self.neighbor_d_model,
                kernel_size=temporal_kernel, padding='same',
            ),
            nn.GELU(),
        )
        self.neighbor_out = nn.Conv1d(self.neighbor_d_model, self.feat_size, kernel_size=1)
        # Kaiming 初始化（非零），配合 neighbor_pred_head 零初始化

        # 邻站预测头: 将邻站时序特征 [B, feat_size, seq_len] 映射为预测残差 [B, feat_size, pred_len]
        # 短梯度路径: loss → output → neighbor_pred → neighbor_features
        self.neighbor_pred_head = nn.Sequential(
            nn.Conv1d(self.feat_size, 32, kernel_size=temporal_kernel, padding='same'),
            nn.GELU(),
            nn.Dropout(neighbor_dropout),
            nn.Conv1d(32, self.feat_size, kernel_size=temporal_kernel, padding='same'),
            nn.AdaptiveAvgPool1d(self.pred_horizon),  # 压缩到预测长度
        )
        # 零初始化输出层，让训练从纯中心站预测开始
        nn.init.zeros_(self.neighbor_pred_head[-2].weight)
        if self.neighbor_pred_head[-2].bias is not None:
            nn.init.zeros_(self.neighbor_pred_head[-2].bias)

        # ===== Alpha 固定门控（仅作用于污染物特征） =====
        # 邻站分支做独立预测，alpha 控制残差贡献度
        self.pollutant_idx = [0, 1, 2, 3, 4, 5]
        self.meteo_idx = [6, 7, 8, 9, 10, 11]
        self.register_buffer(
            'alpha_fixed',
            torch.tensor(float(alpha_init), dtype=torch.float32),
            persistent=True,
        )

        # ========== 2. 时间预测层 (PatchTST Backbone) ==========
        # 只对增强后的中心站表示进行时间序列预测
        # 输入: [Batch, feat_size, seq_len]
        # 输出: [Batch, feat_size, pred_len]

        # PatchTST 骨干网络参数已由 __init__ 签名直接接收，无需从字典中提取

        self.patch_tst = PatchTST(
            c_in=feat_size,       # 融合后的特征数
            c_out=feat_size,      # 输出维度
            seq_len=seq_len,
            pred_dim=self.pred_len,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
        )

    def _reshape_input(self, x):
        """将 [B, num_stations * feat_size, seq_len] 重排为 [B, num_stations, feat_size, seq_len]。"""
        if x.ndim != 3:
            raise ValueError(f"输入张量维度错误: 期望 3 维，实际为 {x.ndim} 维")

        b, c, t = x.shape
        expected_c = self.num_stations * self.feat_size
        if c != expected_c:
            raise ValueError(f"输入通道数错误: 期望 {expected_c}，实际为 {c}")
        return x.reshape(b, self.num_stations, self.feat_size, t)

    def _split_center_and_neighbors(self, x):
        """按中心站索引拆分中心站与邻站张量。"""
        center_x = x[:, self.center_station_idx]
        neighbor_x = x.index_select(1, self.neighbor_indices)
        return center_x, neighbor_x

    def _build_neighbor_context(self, center_x, neighbor_x):
        """
        Cross-Attention 邻站上下文聚合（空间差分模式）。

        关键改进：不使用邻站原始值，而是使用"邻站 - 中心站"的空间差分。
        这样模型学到的是空间梯度（污染物传输方向、风速影响等），
        而非与中心站高度相关的冗余原始值。

        - 每个时间步独立：中心站时刻 t 作为 Query，邻站空间差分作为 Key/Value
        - 输出: [B, feat_size, seq_len]
        """
        b, num_neighbors, _, t = neighbor_x.shape

        # 0. 计算空间差分: 邻站 - 中心站（捕获空间梯度而非冗余绝对值）
        spatial_diff = neighbor_x - center_x.unsqueeze(1)  # [B, N, feat_size, T]

        # 1. 投影中心站 → Q: [B, d_model, T] → [B*T, 1, d_model]
        q = self.center_proj(center_x)                 # [B, d_model, T]
        q = q.permute(0, 2, 1)                         # [B, T, d_model]
        q = q.reshape(b * t, 1, self.neighbor_d_model) # [B*T, 1, d_model]

        # 2. 投影空间差分 → K/V: [B*N, d_model, T] → [B*T, N, d_model]
        diff_flat = spatial_diff.reshape(b * num_neighbors, self.feat_size, t)
        kv = self.neighbor_proj(diff_flat)             # [B*N, d_model, T]
        kv = kv.reshape(b, num_neighbors, self.neighbor_d_model, t)
        kv = kv.permute(0, 3, 1, 2)                    # [B, T, N, d_model]
        kv = kv.reshape(b * t, num_neighbors, self.neighbor_d_model)  # [B*T, N, d_model]

        # 3. Cross-Attention: 每个时间步独立地对邻站空间差分做注意力
        attn_out, _ = self.cross_attn(q, kv, kv)       # [B*T, 1, d_model]
        attn_out = attn_out.squeeze(1)                  # [B*T, d_model]
        attn_out = attn_out.reshape(b, t, self.neighbor_d_model)  # [B, T, d_model]

        # 4. 残差连接 + LayerNorm
        q_residual = q.reshape(b, t, self.neighbor_d_model)  # [B, T, d_model]
        attn_out = self.neighbor_norm(q_residual + attn_out)

        # 5. 时域平滑 + 投影回 feat_size
        attn_out = attn_out.transpose(1, 2)             # [B, d_model, T]
        attn_out = self.temporal_smooth(attn_out)       # [B, d_model, T]
        neighbor_context = self.neighbor_out(attn_out)  # [B, feat_size, T]

        return neighbor_context

    def forward(self, x):
        """
        前向传播

        输入 x: [Batch, num_stations * feat_size, seq_len]
        输出: [Batch, feat_size, pred_len]

        架构: PatchTST 只处理中心站 → 邻站分支独立预测残差 → 加和
        优势: 邻站分支梯度路径短，不受 PatchTST 主干衰减影响
        """
        reshaped_x = self._reshape_input(x)
        center_x, neighbor_x = self._split_center_and_neighbors(reshaped_x)

        # 1. PatchTST 纯中心站预测（主干）
        output_center = self.patch_tst(center_x)  # [B, feat_size, pred_len]

        # 2. 邻站分支独立预测残差
        neighbor_context = self._build_neighbor_context(center_x, neighbor_x)  # [B, feat_size, seq_len]
        output_neighbor = self.neighbor_pred_head(neighbor_context)  # [B, feat_size, pred_len]

        # 3. 融合: 仅污染物特征使用邻站残差
        alpha = torch.zeros(self.feat_size, device=center_x.device)
        alpha[self.pollutant_idx] = self.alpha_fixed
        alpha = alpha.view(1, -1, 1)  # [1, feat_size, 1]

        return output_center + alpha * output_neighbor


# ========== 模型训练代码 ==========

def train_st_patchtst(X, y, splits, preproc_pipe, exp_pipe):
    """
    训练ST_PatchTST模型
    """
    # 加载保存的参数
    params = np.load('tsai/data/model_params.npz')
    required_keys = [
        'num_stations',
        'feat_size',
        'n_vars_total',
        'fcst_history',
        'fcst_horizon',
        'center_station_idx',
    ]
    missing_keys = [key for key in required_keys if key not in params.files]
    if missing_keys:
        raise KeyError(
            f"tsai/data/model_params.npz 缺少字段: {missing_keys}。"
            "请先运行 data_preparation.py 生成多站点参数文件。"
        )

    num_stations = int(params['num_stations'])
    feat_size = int(params['feat_size'])
    n_vars_total = int(params['n_vars_total'])
    fcst_history = int(params['fcst_history'])
    fcst_horizon = int(params['fcst_horizon'])
    center_station_idx = int(params['center_station_idx'])
    neighbor_d_model = 24       # 降维减少过拟合（原 32）
    neighbor_n_heads = 4
    neighbor_dropout = 0.3      # 增大 dropout 正则化（原 0.1）
    neighbor_attn_dropout = 0.1 # 新增 attention dropout（原 0.0）
    neighbor_temporal_kernel = 7   # ±3h 时间窗口，捕捉污染物传输延迟（原 3）
    alpha_max = 0.0             # 不再使用（固定 alpha 模式）
    alpha_init = 0.15           # 固定 alpha 值，仅作用于污染物特征（经验值）

    # ========== 模型配置 ==========
    # TSForecaster 会将 arch_config 以 **arch_config 展开后传给 ST_PatchTST.__init__，
    # 因此所有参数（PatchTST 骨干 + ST 特有）统一放在同一个扁平字典中。
    arch_config = {
        # PatchTST 骨干网络参数
        'n_layers': 3,            # 编码器层数
        'n_heads': 4,             # attention头数
        'd_model': 16,            # 模型维度
        'd_ff': 128,              # 前馈网络维度
        'attn_dropout': 0.0,      # attention dropout
        'dropout': 0.2,           # dropout比率
        'patch_len': 4,           # patch长度
        'stride': 2,              # patch步长
        'padding_patch': True,    # 是否padding patch
        # ST_PatchTST 特有参数（Cross-Attention 邻站分支）
        'num_stations': num_stations,
        'feat_size': feat_size,
        'center_station_idx': center_station_idx,
        'neighbor_d_model': neighbor_d_model,
        'neighbor_n_heads': neighbor_n_heads,
        'neighbor_dropout': neighbor_dropout,
        'neighbor_attn_dropout': neighbor_attn_dropout,
        'neighbor_temporal_kernel': neighbor_temporal_kernel,
        'alpha_max': alpha_max,
        'alpha_init': alpha_init,
    }

    print("\n========== ST_PatchTST 模型配置 ==========")
    print("\nPatchTST 骨干网络参数（对标PatchTST.py）:")
    for key in ['n_layers', 'n_heads', 'd_model', 'd_ff', 'attn_dropout', 'dropout', 'patch_len', 'stride', 'padding_patch']:
        print(f"  {key}: {arch_config[key]}")

    print("\nST_PatchTST 特有参数（Cross-Attention 邻站分支）:")
    print(f"  num_stations: {num_stations}")
    print(f"  feat_size: {feat_size}")
    print(f"  center_station_idx: {center_station_idx}")
    print(f"  neighbor_d_model: {neighbor_d_model}")
    print(f"  neighbor_n_heads: {neighbor_n_heads}")
    print(f"  neighbor_dropout: {neighbor_dropout}")
    print(f"  neighbor_attn_dropout: {neighbor_attn_dropout}")
    print(f"  neighbor_temporal_kernel: {neighbor_temporal_kernel}")
    print(f"  n_vars_total: {n_vars_total}")
    print(f"  fcst_history: {fcst_history}")
    print(f"  fcst_horizon: {fcst_horizon}")
    print(f"  alpha_max: {alpha_max}")
    print(f"  alpha_init: {alpha_init}")

    cbs = [
        GradientClip(1.0),  # 限制梯度范数，防止梯度爆炸
        SaveModelCallback(monitor='valid_loss', fname='ST_PatchTST_best'), # 训练过程中保存验证集 valid_loss 最好的模型，而不是只保留最后一个 epoch 的模型
        EarlyStoppingCallback(monitor='valid_loss', patience=25),  # 更长的耐心，给邻站分支更多学习时间（原 15）
    ]

    # 实例化TSForecaster
    learn = TSForecaster(
        X, y,
        splits=splits,
        batch_size=16,
        path="tsai/models",
        pipelines=[preproc_pipe, exp_pipe],
        arch=ST_PatchTST,
        arch_config=arch_config,
        metrics=[rmse, mse, mae],
        cbs=cbs  # 传入回调函数列表
    )

    print("[OK] TSForecaster实例化成功")

    # 查找学习率
    lr_max = learn.lr_find().valley
    lr = min(lr_max, 1e-4)  # 设置学习率上限，避免过大
    print(f"[OK] 最优学习率: {lr_max}", f"（实际使用学习率: {lr}）")

    # 训练模型
    n_epochs = 50
    learn.fit_one_cycle(n_epochs, lr_max=lr)

    # 导出模型
    learn.export('ST_PatchTST.pt')
    print("[OK] 模型训练完成并已导出")

    return learn


# ========== 模型评估代码 ==========

def evaluate_st_patchtst(learn, X, y, splits):
    """
    评估ST_PatchTST模型
    """
    target_names = load_target_names(expected_count=infer_num_metrics(y))
    summary_rows = []
    per_metric_rows = []

    for split_name, split_idxs in (("valid", splits[1]), ("test", splits[2])):
        preds, *_ = learn.get_X_preds(X[split_idxs])
        preds = to_np(preds)
        print(f"\n{split_name} 预测形状: {preds.shape}")

        y_true = y[split_idxs]
        summary_row, split_metric_rows = build_split_evaluation(y_true, preds, split_name, target_names)
        summary_rows.append(summary_row)
        per_metric_rows.extend(split_metric_rows)

    summary_df, per_metric_df = build_results_dataframes(summary_rows, per_metric_rows)
    print_evaluation_results(summary_df, per_metric_df)
    return {"summary": summary_df, "per_metric": per_metric_df}


if __name__ == "__main__":
    print("ST_PatchTST模型定义完成")
    print("\n使用说明:")
    print("1. 先运行 data_preparation.py 进行数据准备")
    print("2. 然后运行 train_st_patchtst(X, y, splits, preproc_pipe, exp_pipe) 进行训练")
    print("3. 最后运行 evaluate_st_patchtst(learn, X, y, splits) 进行评估")
