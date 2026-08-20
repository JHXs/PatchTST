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
        neighbor_hidden_dim=32,
        neighbor_dropout=0.1,
        alpha_max=0.2,
        alpha_init=0.1,
        gate_mode="pairwise_summary",
        use_null_neighbor=True,
        neighbor_value_mode="raw",
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
        - neighbor_hidden_dim: 邻站辅助分支的隐藏维度
        - neighbor_dropout: 邻站辅助分支的 dropout
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
        self.pred_len = pred_dim
        self.center_station_idx = int(center_station_idx)
        self.num_neighbors = self.num_stations - 1
        self.neighbor_hidden_dim = int(neighbor_hidden_dim)
        self.max_alpha = float(alpha_max)
        self.alpha_init = float(alpha_init)
        self.gate_mode = str(gate_mode)
        self.use_null_neighbor = bool(use_null_neighbor)
        self.neighbor_value_mode = str(neighbor_value_mode)

        # 验证输入维度
        expected_c_in = self.num_stations * self.feat_size
        if c_in != expected_c_in:
            raise ValueError(f"c_in={c_in}, 但期望输入通道数为 {expected_c_in}")
        # tsai 的 forecasting 链路里 c_out 可能来自 dls.c，通常不等于多变量预测的通道数。
        self.tsai_c_out = c_out
        if not 0 <= self.center_station_idx < self.num_stations:
            raise ValueError(
                f"center_station_idx={self.center_station_idx} 超出有效范围 [0, {self.num_stations - 1}]"
            )
        if self.num_neighbors < 1:
            raise ValueError("ST_PatchTST 需要至少 1 个邻站，当前 num_stations 必须大于 1")
        if self.neighbor_hidden_dim < 1:
            raise ValueError(f"neighbor_hidden_dim={self.neighbor_hidden_dim} 必须大于 0")
        if self.gate_mode not in {"center_only", "pairwise_summary"}:
            raise ValueError(
                f"gate_mode={self.gate_mode!r} 无效，"
                "可选值为 'center_only' 或 'pairwise_summary'"
            )
        if self.neighbor_value_mode not in {"raw", "difference"}:
            raise ValueError(
                f"neighbor_value_mode={self.neighbor_value_mode!r} 无效，"
                "可选值为 'raw' 或 'difference'"
            )

        neighbor_indices = [idx for idx in range(self.num_stations) if idx != self.center_station_idx]
        self.register_buffer(
            "neighbor_indices",
            torch.tensor(neighbor_indices, dtype=torch.long),
            persistent=False
        )

        # 邻站先共享一个 1x1 Conv 投影到隐藏维度，再由中心站上下文生成门控权重。
        self.neighbor_proj = nn.Sequential(
            nn.Conv1d(self.feat_size, self.neighbor_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(neighbor_dropout),
        )
        self.center_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.feat_size, self.neighbor_hidden_dim),
            nn.GELU(),
            nn.Dropout(neighbor_dropout),
            nn.Linear(self.neighbor_hidden_dim, self.num_neighbors),
        )
        # 成对门控同时读取中心站和每个邻站。每个站点以窗口均值、标准差、
        # 最近值和首尾斜率概括，随后构造 [center, neighbor, difference, product]
        # 的成对特征。该设计比只读取中心站的全局门控更能表达样本级邻站价值。
        summary_dim = self.feat_size * 4
        pair_dim = summary_dim * 4
        self.pairwise_gate = nn.Sequential(
            nn.Linear(pair_dim, self.neighbor_hidden_dim),
            nn.GELU(),
            nn.Dropout(neighbor_dropout),
            nn.Linear(self.neighbor_hidden_dim, 1),
        )
        if self.use_null_neighbor:
            # 空邻站允许模型在当前样本没有可靠空间信息时拒绝全部邻站。
            self.null_neighbor_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("null_neighbor_logit", None)
        self.neighbor_out = nn.Conv1d(self.neighbor_hidden_dim, self.feat_size, kernel_size=1)
        # 让邻站分支从“几乎不影响主干”开始训练，避免一开始就把中心站主信号打乱。
        nn.init.zeros_(self.neighbor_out.weight)
        if self.neighbor_out.bias is not None:
            nn.init.zeros_(self.neighbor_out.bias)
        if self.max_alpha < 0:
            raise ValueError(f"alpha_max={self.max_alpha} 必须大于等于 0")
        if self.max_alpha == 0:
            if self.alpha_init != 0:
                raise ValueError("当 alpha_max=0 时，alpha_init 必须为 0")
            self.alpha_logit = None
        else:
            if not 0 <= self.alpha_init <= self.max_alpha:
                raise ValueError(
                    f"alpha_init={self.alpha_init} 必须落在 [0, alpha_max={self.max_alpha}] 内"
                )
            alpha_ratio = self.alpha_init / self.max_alpha
            eps = 1e-6
            alpha_ratio = min(max(alpha_ratio, eps), 1 - eps)
            alpha_logit_init = np.log(alpha_ratio / (1 - alpha_ratio))
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init, dtype=torch.float32))

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
        """利用中心站上下文对邻站做门控聚合。"""
        b, num_neighbors, _, t = neighbor_x.shape

        if self.neighbor_value_mode == "difference":
            neighbor_values = neighbor_x - center_x.unsqueeze(1)
        else:
            neighbor_values = neighbor_x
        projected_neighbors = neighbor_values.reshape(b * num_neighbors, self.feat_size, t)
        projected_neighbors = self.neighbor_proj(projected_neighbors)
        projected_neighbors = projected_neighbors.reshape(
            b, num_neighbors, self.neighbor_hidden_dim, t
        )

        if self.gate_mode == "center_only":
            gate_logits = self.center_gate(center_x)
        else:
            center_summary = self._summarize_station_window(center_x.unsqueeze(1))
            neighbor_summary = self._summarize_station_window(neighbor_x)
            center_summary = center_summary.expand(-1, num_neighbors, -1)
            pair_features = torch.cat(
                [
                    center_summary,
                    neighbor_summary,
                    center_summary - neighbor_summary,
                    center_summary * neighbor_summary,
                ],
                dim=-1,
            )
            gate_logits = self.pairwise_gate(pair_features).squeeze(-1)

        if self.null_neighbor_logit is not None:
            null_logit = self.null_neighbor_logit.expand(b, 1)
            all_gate_weights = torch.softmax(
                torch.cat([gate_logits, null_logit], dim=1), dim=1
            )
            gate_weights = all_gate_weights[:, :num_neighbors]
            null_weight = all_gate_weights[:, -1]
        else:
            gate_weights = torch.softmax(gate_logits, dim=1)
            null_weight = center_x.new_zeros(b)

        expanded_gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)

        weighted_neighbors = projected_neighbors * expanded_gate_weights
        neighbor_context_hidden = weighted_neighbors.sum(dim=1)
        neighbor_context = self.neighbor_out(neighbor_context_hidden)
        return neighbor_context, gate_weights, null_weight

    @staticmethod
    def _summarize_station_window(x):
        """将 [B, S, F, L] 汇总为保留近期状态和趋势的 [B, S, 4F]。"""
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        last = x[..., -1]
        slope = (x[..., -1] - x[..., 0]) / max(x.shape[-1] - 1, 1)
        return torch.cat([mean, std, last, slope], dim=-1)

    def spatial_components(self, x):
        """返回空间分支中间量，供消融实验与诊断报告使用。"""
        reshaped_x = self._reshape_input(x)
        center_x, neighbor_x = self._split_center_and_neighbors(reshaped_x)
        neighbor_context, gate_weights, null_weight = self._build_neighbor_context(
            center_x, neighbor_x
        )
        if self.alpha_logit is None:
            alpha = center_x.new_tensor(0.0)
        else:
            alpha = self.max_alpha * torch.sigmoid(self.alpha_logit)
        return {
            "center_x": center_x,
            "neighbor_context": neighbor_context,
            "gate_weights": gate_weights,
            "null_weight": null_weight,
            "alpha": alpha,
        }

    def forward(self, x):
        """
        前向传播

        输入 x: [Batch, num_stations * feat_size, seq_len]
        输出: [Batch, feat_size, pred_len]
        """
        # 中心站走主干，邻站只作为可控增强。
        components = self.spatial_components(x)
        center_x = components["center_x"]
        neighbor_context = components["neighbor_context"]
        alpha = components["alpha"]
        enhanced_x = center_x + alpha * neighbor_context

        # PatchTST 期望输入 [Batch, vars, seq_len]
        temporal_out = self.patch_tst(enhanced_x)

        return temporal_out


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
    neighbor_hidden_dim = 32
    neighbor_dropout = 0.1
    alpha_max = 0.5 # 经验值，允许邻站有一定影响力但不过度干扰中心站
    alpha_init = 0.05
    gate_mode = "pairwise_summary"
    use_null_neighbor = True

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
        # ST_PatchTST 特有参数
        'num_stations': num_stations,
        'feat_size': feat_size,
        'center_station_idx': center_station_idx,
        'neighbor_hidden_dim': neighbor_hidden_dim,
        'neighbor_dropout': neighbor_dropout,
        'alpha_max': alpha_max,
        'alpha_init': alpha_init,
        'gate_mode': gate_mode,
        'use_null_neighbor': use_null_neighbor,
    }

    print("\n========== ST_PatchTST 模型配置 ==========")
    print("\nPatchTST 骨干网络参数（对标PatchTST.py）:")
    for key in ['n_layers', 'n_heads', 'd_model', 'd_ff', 'attn_dropout', 'dropout', 'patch_len', 'stride', 'padding_patch']:
        print(f"  {key}: {arch_config[key]}")

    print("\nST_PatchTST 特有参数:")
    print(f"  num_stations: {num_stations}")
    print(f"  feat_size: {feat_size}")
    print(f"  center_station_idx: {center_station_idx}")
    print(f"  neighbor_hidden_dim: {neighbor_hidden_dim}")
    print(f"  neighbor_dropout: {neighbor_dropout}")
    print(f"  n_vars_total: {n_vars_total}")
    print(f"  fcst_history: {fcst_history}")
    print(f"  fcst_horizon: {fcst_horizon}")
    print(f"  alpha_max: {alpha_max}")
    print(f"  alpha_init: {alpha_init}")
    print(f"  gate_mode: {gate_mode}")
    print(f"  use_null_neighbor: {use_null_neighbor}")

    cbs = [
        GradientClip(1.0),  # 限制梯度范数，防止梯度爆炸
        SaveModelCallback(monitor='valid_loss', fname='ST_PatchTST_best'), # 训练过程中保存验证集 valid_loss 最好的模型，而不是只保留最后一个 epoch 的模型
        EarlyStoppingCallback(monitor='valid_loss', patience=15),  # loss 连续 15 个 epoch 没改善，提前停止训练
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
