import torch
import torch.nn as nn
from tsai.models.PatchTST import PatchTST
import sklearn.pipeline
from tsai.all import *

class ST_PatchTST(nn.Module):
    """
    时空PatchTST模型

    架构:
    1. 空间融合层: 使用Conv2d将多个站点的数据融合为一个"虚拟超级站点"
    2. 时间预测层: 使用PatchTST进行时间序列预测

    输入: [Batch, num_stations * feat_size, seq_len]
    输出: [Batch, feat_size, pred_len]
    """

    def __init__(self, c_in, c_out, seq_len, pred_dim, num_stations, feat_size, arch_config={}, **kwargs):
        """
        参数:
        - c_in: 输入通道数 (num_stations * feat_size) - 由TSForecaster自动传递
        - c_out: 输出通道数 (feat_size) - 由TSForecaster自动传递
        - seq_len: 输入序列长度 - 由TSForecaster自动传递
        - pred_dim: 预测序列长度 - 由TSForecaster自动传递
        - num_stations: 站点数量 - 需要通过arch_config传递
        - feat_size: 每个站点的特征数量 - 需要通过arch_config传递
        - arch_config: 内部PatchTST的配置
        """
        super().__init__()

        self.num_stations = num_stations
        self.feat_size = feat_size
        self.seq_len = seq_len
        self.pred_len = pred_dim

        # 验证输入维度
        expected_c_in = num_stations * feat_size
        if c_in != expected_c_in:
            print(f"警告: c_in={c_in}, 期望值={expected_c_in}")

        # ========== 1. 空间融合层 (Spatial Convolution) ==========
        # 将多个站点的数据融合为一个"虚拟超级站点"
        # 输入: [Batch, num_stations, seq_len, feat_size]
        # 输出: [Batch, 1, seq_len, feat_size]

        self.spatial_conv = nn.Sequential(
            # 第一层: num_stations -> 64
            # 在站点间进行线性组合,保持时间和特征维度不变
            nn.Conv2d(in_channels=num_stations, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 第二层: 64 -> 1
            # 融合为一个虚拟站点
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            # nn.ReLU()
        )

        # ========== 2. 时间预测层 (PatchTST Backbone) ==========
        # 对融合后的数据进行时间序列预测
        # 输入: [Batch, feat_size, seq_len]
        # 输出: [Batch, feat_size, pred_len]

        # 从arch_config中提取PatchTST参数
        n_layers = arch_config.get('n_layers', 3)
        n_heads = arch_config.get('n_heads', 4)
        d_model = arch_config.get('d_model', 128)
        d_ff = arch_config.get('d_ff', 512)
        dropout = arch_config.get('dropout', 0.2)

        self.patch_tst = PatchTST(
            c_in=feat_size,       # 融合后的特征数
            c_out=feat_size,      # 输出维度 (虚拟站点)
            seq_len=seq_len,
            pred_dim=self.pred_len,  # 使用pred_dim而不是pred_len
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, x):
        """
        前向传播

        输入 x: [Batch, num_stations * feat_size, seq_len]
        输出: [Batch, feat_size, pred_len]
        """
        # 1. Reshape: [Batch, num_stations * feat_size, seq_len] -> [Batch, num_stations, feat_size, seq_len]
        b, _, t = x.shape
        x = x.view(b, self.num_stations, self.feat_size, t)

        # 2. Permute: [Batch, num_stations, feat_size, seq_len] -> [Batch, num_stations, seq_len, feat_size]
        # 调整维度以适配Conv2d (N, C, H, W), 其中:
        # - N = Batch
        # - C = num_stations (站点作为通道)
        # - H = seq_len (时间作为高度)
        # - W = feat_size (特征作为宽度)
        x = x.permute(0, 1, 3, 2)

        # 3. 空间融合: [Batch, num_stations, seq_len, feat_size] -> [Batch, 1, seq_len, feat_size]
        spatial_out = self.spatial_conv(x)

        # 4. Squeeze: [Batch, 1, seq_len, feat_size] -> [Batch, seq_len, feat_size]
        spatial_out = spatial_out.squeeze(1)

        # 5. Permute: [Batch, seq_len, feat_size] -> [Batch, feat_size, seq_len]
        # PatchTST期望输入 [Batch, vars, seq_len]
        spatial_out = spatial_out.permute(0, 2, 1)

        # 6. 时间预测: [Batch, feat_size, seq_len] -> [Batch, feat_size, pred_len]
        temporal_out = self.patch_tst(spatial_out)

        return temporal_out


# ========== 模型训练代码 ==========

def train_st_patchtst(X, y, splits, preproc_pipe, exp_pipe):
    """
    训练ST_PatchTST模型
    """
    # 加载保存的参数
    params = np.load('tsai/data/model_params.npz')
    num_stations = int(params['num_stations'])
    feat_size = int(params['feat_size'])
    n_vars_total = int(params['n_vars_total'])
    fcst_history = int(params['fcst_history'])
    fcst_horizon = int(params['fcst_horizon'])
    

    # 模型配置
    arch_config = {
        'num_stations': num_stations,
        'feat_size': feat_size,
        'arch_config': {
            'n_layers': 3,
            'n_heads': 4,
            'd_model': 128,
            'd_ff': 512,
            'dropout': 0.2
        }
    }

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
    print(f"✓ 最优学习率: {lr_max}")

    # 训练模型
    n_epochs = 50
    learn.fit_one_cycle(n_epochs, lr_max=lr_max)

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

    return results_df, y_test_preds


if __name__ == "__main__":
    print("ST_PatchTST模型定义完成")
    print("\n使用说明:")
    print("1. 先运行 ST_PatchTST_data_preparation_fixed.py 进行数据准备")
    print("2. 然后运行 train_st_patchtst(X, y, splits, preproc_pipe, exp_pipe) 进行训练")
    print("3. 最后运行 evaluate_st_patchtst(learn, X, y, splits) 进行评估")