import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.training import GradientClip
from tsai.models.PatchTST import PatchTST
import sklearn.pipeline
from tsai.all import *

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
        arch_config=None,
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
        - arch_config: PatchTST骨干网络配置 (n_layers, n_heads, d_model, d_ff, dropout等)
        
        arch_config 参数（与PatchTST对标）：
        - n_layers: 编码器层数
        - n_heads: attention头数  
        - d_model: 模型维度
        - d_ff: 前馈网络维度
        - dropout: dropout比率
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

        if arch_config is None:
            arch_config = {}

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

        # 从arch_config中提取PatchTST参数（扁平结构，对标PatchTST.py）
        n_layers = arch_config.get('n_layers', 3)
        n_heads = arch_config.get('n_heads', 4)
        d_model = arch_config.get('d_model', 16)
        d_ff = arch_config.get('d_ff', 128)
        dropout = arch_config.get('dropout', 0.2)
        attn_dropout = arch_config.get('attn_dropout', 0.0)
        patch_len = arch_config.get('patch_len', 24)
        stride = arch_config.get('stride', 2)
        padding_patch = arch_config.get('padding_patch', True)

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
        """利用中心站上下文对邻站做门控聚合，输出 [B, feat_size, seq_len]。"""
        b, num_neighbors, _, t = neighbor_x.shape

        projected_neighbors = neighbor_x.reshape(b * num_neighbors, self.feat_size, t)
        projected_neighbors = self.neighbor_proj(projected_neighbors)
        projected_neighbors = projected_neighbors.reshape(
            b, num_neighbors, self.neighbor_hidden_dim, t
        )

        gate_logits = self.center_gate(center_x)
        gate_weights = torch.softmax(gate_logits, dim=1).unsqueeze(-1).unsqueeze(-1)

        weighted_neighbors = projected_neighbors * gate_weights
        neighbor_context_hidden = weighted_neighbors.sum(dim=1)
        neighbor_context = self.neighbor_out(neighbor_context_hidden)
        return neighbor_context

    def forward(self, x):
        """
        前向传播

        输入 x: [Batch, num_stations * feat_size, seq_len]
        输出: [Batch, feat_size, pred_len]
        """
        reshaped_x = self._reshape_input(x)

        # 中心站走主干，邻站只作为可控增强。
        center_x, neighbor_x = self._split_center_and_neighbors(reshaped_x)
        neighbor_context = self._build_neighbor_context(center_x, neighbor_x)
        if self.alpha_logit is None:
            alpha = center_x.new_tensor(0.0)
        else:
            alpha = self.max_alpha * torch.sigmoid(self.alpha_logit)
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
    alpha_max = 0.7
    alpha_init = 0.0

    # ========== 模型配置 ==========
    # PatchTST 骨干网络配置（扁平结构，对标 PatchTST.py）
    arch_config = {
        # PatchTST 参数
        'n_layers': 3,            # 编码器层数
        'n_heads': 4,             # attention头数
        'd_model': 16,            # 模型维度  
        'd_ff': 128,              # 前馈网络维度
        'attn_dropout': 0.0,      # attention dropout
        'dropout': 0.2,           # dropout比率
        'patch_len': 24,          # patch长度
        'stride': 2,              # patch步长
        'padding_patch': True,    # 是否padding patch
    }
    
    # ST_PatchTST 特有参数（传递给TSForecaster）
    st_config = {
        'num_stations': num_stations,
        'feat_size': feat_size,
        'center_station_idx': center_station_idx,
        'neighbor_hidden_dim': neighbor_hidden_dim,
        'neighbor_dropout': neighbor_dropout,
        'alpha_max': alpha_max,
        'alpha_init': alpha_init,
    }
    
    # 合并配置：arch_config是PatchTST参数，st_config通过kwargs传递
    arch_config.update(st_config)

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

    cbs = [
        GradientClip(1.0),
        SaveModelCallback(monitor='valid_loss', fname='ST_PatchTST_best'),
        EarlyStoppingCallback(monitor='valid_loss', patience=5),
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
        metrics=[mse, mae],
        cbs=[]
    )

    print("✓ TSForecaster实例化成功")

    # 查找学习率
    lr_max = learn.lr_find().valley
    lr = min(lr_max, 1e-4)  # 设置学习率上限，避免过大
    print(f"✓ 最优学习率: {lr_max}", f"（实际使用学习率: {lr}）")

    # 训练模型
    n_epochs = 50
    learn.fit_one_cycle(n_epochs, lr_max=lr)

    # 导出模型
    learn.export('ST_PatchTST.pt')
    print("✓ 模型训练完成并已导出")

    return learn


# ========== 模型评估代码 ==========

def evaluate_st_patchtst(learn, X, y, splits):
    """
    评估ST_PatchTST模型
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    results_df = pd.DataFrame(columns=["mse", "mae"])

    # 验证集评估
    print("\n验证集评估...")
    scaled_preds, *_ = learn.get_X_preds(X[splits[1]])
    scaled_preds = to_np(scaled_preds)
    print(f"验证集预测形状: {scaled_preds.shape}")

    scaled_y_true = y[splits[1]]
    results_df.loc["valid", "mse"] = mean_squared_error(scaled_y_true.flatten(), scaled_preds.flatten())
    results_df.loc["valid", "mae"] = mean_absolute_error(scaled_y_true.flatten(), scaled_preds.flatten())

    # 测试集评估
    print("\n测试集评估...")
    y_test_preds, *_ = learn.get_X_preds(X[splits[2]])
    y_test_preds = to_np(y_test_preds)
    print(f"测试集预测形状: {y_test_preds.shape}")

    y_test = y[splits[2]]
    results_df.loc["test", "mse"] = mean_squared_error(y_test.flatten(), y_test_preds.flatten())
    results_df.loc["test", "mae"] = mean_absolute_error(y_test.flatten(), y_test_preds.flatten())

    print("\n评估结果:")
    print(results_df)

    return results_df


if __name__ == "__main__":
    print("ST_PatchTST模型定义完成")
    print("\n使用说明:")
    print("1. 先运行 data_preparation.py 进行数据准备")
    print("2. 然后运行 train_st_patchtst(X, y, splits, preproc_pipe, exp_pipe) 进行训练")
    print("3. 最后运行 evaluate_st_patchtst(learn, X, y, splits) 进行评估")
