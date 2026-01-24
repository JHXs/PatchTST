import torch
import torch.nn as nn
import numpy as np
from tsai.all import *

# ========== 1. 基础组件: RevIN (Reversible Instance Normalization) ==========
# 论文中明确提到需要使用 RevIN 来处理分布偏移 [cite: 105]
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(x.ndim - 1))
        if x.ndim == 3: # [B, C, L]
            self.mean = torch.mean(x, dim=-1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
        elif x.ndim == 4: # [B, N, C, D]
             # 针对Patch后的处理，这里简化为只处理原始输入
             pass

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight.unsqueeze(-1) + self.affine_bias.unsqueeze(-1)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias.unsqueeze(-1)) / (self.affine_weight.unsqueeze(-1) + 1e-10)
        x = x * self.stdev
        x = x + self.mean
        return x

# ========== 2. 核心组件: Channel-Time Attention Encoder ==========
# 这是论文图2和图5的核心实现 [cite: 139, 490]
class CT_EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads_channel, n_heads_time, d_ff, dropout=0.1):
        super().__init__()
        
        # 1. Channel Attention (捕捉站点/变量间关系)
        # Head数通常设为 1 [cite: 283]
        self.channel_attn = nn.MultiheadAttention(d_model, n_heads_channel, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Channel Feed Forward
        self.ff_channel = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 2. Time Attention (捕捉时间依赖)
        # Head数通常设为 16 [cite: 283]
        self.time_attn = nn.MultiheadAttention(d_model, n_heads_time, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        # Time Feed Forward
        self.ff_time = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [Batch, N_patches, M_channels, d_model]
        B, N, M, D = x.shape

        # --- Part A: Channel Attention ---
        # 我们需要在 M (Channel) 维度上做 Attention
        # Reshape: [Batch * N, M, D]
        x_chan = x.reshape(B * N, M, D)
        
        # Self-Attention over Channels
        attn_out, _ = self.channel_attn(x_chan, x_chan, x_chan)
        x_chan = self.norm1(x_chan + self.dropout1(attn_out))
        
        # Feed Forward
        ff_out = self.ff_channel(x_chan)
        x_chan = self.norm2(x_chan + self.dropout2(ff_out))
        
        # 恢复形状: [Batch, N, M, D]
        x = x_chan.reshape(B, N, M, D)

        # --- Part B: Time Attention ---
        # 我们需要在 N (Patches) 维度上做 Attention
        # Permute to: [Batch, M, N, D] -> Reshape to [Batch * M, N, D]
        x_time = x.permute(0, 2, 1, 3).reshape(B * M, N, D)
        
        # Self-Attention over Time (Patches)
        attn_out, _ = self.time_attn(x_time, x_time, x_time)
        x_time = self.norm3(x_time + self.dropout3(attn_out))
        
        # Feed Forward
        ff_out = self.ff_time(x_time)
        x_time = self.norm4(x_time + self.dropout4(ff_out))
        
        # 恢复形状并转回: [Batch * M, N, D] -> [Batch, M, N, D] -> [Batch, N, M, D]
        x = x_time.reshape(B, M, N, D).permute(0, 2, 1, 3)
        
        return x

# ========== 3. 主模型: CT-PatchTST ==========
class CT_PatchTST(nn.Module):
    """
    CT-PatchTST: Channel-Time Patch Time-Series Transformer
    论文复现版本
    """
    def __init__(self, c_in, c_out, seq_len, pred_dim, arch_config={}, **kwargs):
        super().__init__()
        
        # 参数配置 (默认值基于论文 Table 1 [cite: 283])
        self.c_in = c_in            # 输入通道数 M (num_stations * feat_size)
        self.seq_len = seq_len      # 历史窗口 L
        
        # pred_dim 可能是整数或列表 [c_out, pred_len]
        print(f"初始化 CT_PatchTST 模型: c_in={c_in}, c_out={c_out}, seq_len={seq_len}, pred_dim={pred_dim}")
        if isinstance(pred_dim, (list, tuple)):
            self.c_out = pred_dim[0]  # 实际输出通道数
            self.pred_len = pred_dim[-1]  # 取最后一个值作为预测长度
        else:
            self.c_out = c_out          # 输出通道数 (预测变量数)
            self.pred_len = pred_dim
        
        # 超参数
        patch_len = arch_config.get('patch_len', 16)
        stride = arch_config.get('stride', 8)
        d_model = arch_config.get('d_model', 256)
        d_ff = arch_config.get('d_ff', 512)
        n_layers = arch_config.get('n_layers', 4) # 论文推荐 4层
        n_heads_channel = arch_config.get('n_heads_channel', 1) # 论文推荐 1
        n_heads_time = arch_config.get('n_heads_time', 16)      # 论文推荐 16
        dropout = arch_config.get('dropout', 0.1)
        
        # Patch 参数
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model  # Store d_model as instance variable
        
        # 计算实际的 num_patches
        # 考虑 padding: remaining = (seq_len - patch_len) % stride
        # 如果 remaining != 0, 会 pad stride - remaining
        remaining = (seq_len - patch_len) % stride
        if remaining != 0:
            padded_len = seq_len + (stride - remaining)
        else:
            padded_len = seq_len
        # unfold 后的 patch 数量
        self.num_patches = (padded_len - patch_len) // stride + 1
        
        # 1. RevIN
        self.revin = RevIN(self.c_in, affine=True)
        
        # 2. Embedding (Patching + Projection)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        
        # Positional Embedding - 使用可学习的参数
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        # 3. Channel-Time Encoders
        self.encoder_layers = nn.ModuleList([
            CT_EncoderLayer(d_model, n_heads_channel, n_heads_time, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 4. Output Head
        # 输出形状 [B, c_out, T]
        # Transformer输出是 [B, N, M, D] -> 转置为 [B, M, N, D] -> Flatten -> [B, M, N*D]
        # 然后需要从 M 个通道投影到 c_out 个输出通道
        head_in_features = int(self.num_patches * self.d_model)
        head_out_features = int(self.pred_len)
        self.head = nn.Linear(head_in_features, head_out_features)
        self.channel_projection = nn.Linear(self.c_in, self.c_out)  # 通道投影层

    def forward(self, x):
        # Input x: [Batch, c_in, seq_len]
        
        # 1. Instance Normalization [cite: 105]
        x = self.revin(x, 'norm') # [Batch, c_in, seq_len]
        
        # 2. Patching Process [cite: 144-152]
        # Permute for unfolding: [B, c_in, L] -> [B, c_in, L]
        # 我们需要在 seq_len 维度上进行 unfold
        # x_patch shape 目标: [B, N, M, P]
        
        B, M, L = x.shape
        
        # 简单的 Patching 实现
        # Unfold input: [B, M, L] -> [B, M, N, P]
        # 首先 padding 使得最后一个 patch 完整
        remaining = (L - self.patch_len) % self.stride
        if remaining != 0:
             pad_len = self.stride - remaining
             x = torch.nn.functional.pad(x, (0, pad_len), mode='replicate')
        
        # Unfold: [B, M, L_pad] -> [B, M, N, P]
        x_patched = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Permute to [B, N, M, P] 以符合论文逻辑 (Batch, Time-Patches, Channels, Patch-Size)
        x_patched = x_patched.permute(0, 2, 1, 3) 
        
        # 3. Projection [cite: 149]
        # [B, N, M, P] -> [B, N, M, d_model]
        x_emb = self.patch_embedding(x_patched)
        
        # Add Positional Embedding [cite: 168]
        # Pos Emb shape: [1, N, 1, d_model] -> 广播加到 [B, N, M, D]
        x_emb = x_emb + self.position_embedding
        x_emb = self.dropout(x_emb)
        
        # 4. Dual-Attention Encoder [cite: 154]
        # Loop through layers
        z = x_emb
        for layer in self.encoder_layers:
            z = layer(z)
            
        # z shape: [B, N, M, D]
            
        # 5. Output Head
        # 我们需要预测 [B, c_out, pred_len]
        # Reshape: [B, N, M, D] -> [B, M, N, D]
        z = z.permute(0, 2, 1, 3)
        # Flatten: [B, M, N*D]
        z = z.reshape(B, M, -1)
        # Linear Projection to time dimension
        out = self.head(z) # -> [B, M, pred_len]
        
        # Channel projection: [B, M, pred_len] -> [B, c_out, pred_len]
        # Permute to [B, pred_len, M] for linear layer
        out = out.permute(0, 2, 1)  # [B, pred_len, M]
        out = self.channel_projection(out)  # [B, pred_len, c_out]
        out = out.permute(0, 2, 1)  # [B, c_out, pred_len]
        
        # 注意: 我们不对输出进行 denorm,因为输出的通道数 (c_out) 与输入 (c_in) 不同
        # RevIN 的统计信息是基于 c_in 个通道计算的,无法直接应用到 c_out 个通道
        # 如果需要反归一化,应该在数据pipeline中处理
        
        return out


# ========== 模型训练代码 ==========

def train_ct_patchtst(X, y, splits, preproc_pipe, exp_pipe):
    # 加载保存的参数
    params = np.load('tsai/data/model_params.npz')
    num_stations = int(params['num_stations'])
    feat_size = int(params['feat_size'])
    n_vars_total = int(params['n_vars_total'])
    fcst_history = int(params['fcst_history'])
    fcst_horizon = int(params['fcst_horizon'])

    # CT-PatchTST 的配置 [cite: 283]
    # 进一步调整为适合8GB GPU的配置
    arch_config = {
        'patch_len': 16,
        'stride': 8,          # 通常设为 patch_len / 2
        'd_model': 64,        # 进一步减小以适应GPU内存
        'd_ff': 128,          # 进一步减小以适应GPU内存
        'n_layers': 1,        # 减少到1层以适应GPU内存
        'n_heads_channel': 1, # 论文 Table 1
        'n_heads_time': 4,    # 进一步减少头数以适应GPU内存
        'dropout': 0.1
    }

    learn = TSForecaster(
        X, y,
        splits=splits,
        batch_size=16,          # 进一步减小批次大小以适应GPU内存
        pipelines=[preproc_pipe, exp_pipe],
        arch=CT_PatchTST,       # 使用新的类
        arch_config=arch_config,
        metrics=[mse, mae],
    )
    
    print("✓ TSForecaster实例化成功")

    # 查找学习率
    lr_max = learn.lr_find().valley
    print(f"✓ 最优学习率: {lr_max}")

    # 训练模型
    n_epochs = 50
    learn.fit_one_cycle(n_epochs, lr_max=lr_max)

    # 导出模型
    learn.export('tsai/models/ct_patchTST.pt')
    print("✓ 模型训练完成并已导出")

    return learn

# ========== 模型评估代码 ==========

def evaluate_ct_patchtst(learn, X, y, splits):
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
    print("CT_PatchTST模型定义完成")
    print("\n使用说明:")
    print("1. 先运行 ST_PatchTST_data_preparation.py 进行数据准备")
    print("2. 然后运行 train_ct_patchtst(X, y, splits, preproc_pipe, exp_pipe) 进行训练")
    print("3. 最后运行 evaluate_ct_patchtst(learn, X, y, splits) 进行评估")