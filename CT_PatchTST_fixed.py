import torch
import torch.nn as nn
import numpy as np
from tsai.all import *

# ========== 精确修正版: CT-PatchTST 模型 ==========
class CT_PatchTST_Final(nn.Module):
    """
    最终修正版 CT-PatchTST
    """
    def __init__(self, c_in, c_out, seq_len, pred_dim, arch_config={}, **kwargs):
        super().__init__()
        
        # 参数配置
        self.c_in = c_in            # 输入通道数 (所有站点的特征总数)
        self.c_out = c_out          # 输出通道数 (目标站点的特征数)
        self.seq_len = seq_len      # 历史窗口
        
        if isinstance(pred_dim, (list, tuple)):
            self.pred_len = pred_dim[-1]  # 预测长度
        else:
            self.pred_len = pred_dim
        
        # 超参数
        patch_len = arch_config.get('patch_len', 16)
        stride = arch_config.get('stride', 8)
        d_model = arch_config.get('d_model', 64)
        d_ff = arch_config.get('d_ff', 128)
        n_layers = arch_config.get('n_layers', 1)
        n_heads_channel = arch_config.get('n_heads_channel', 1)
        n_heads_time = arch_config.get('n_heads_time', 4)
        dropout = arch_config.get('dropout', 0.1)
        
        # Patch 参数
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # 计算实际的 num_patches
        remaining = (seq_len - patch_len) % stride
        if remaining != 0:
            padded_len = seq_len + (stride - remaining)
        else:
            padded_len = seq_len
        self.num_patches = (padded_len - patch_len) // stride + 1
        
        # 1. RevIN
        self.revin = RevIN(self.c_in, affine=True)
        
        # 2. Embedding (Patching + Projection)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        
        # Positional Embedding
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        # 3. Channel-Time Encoders
        self.encoder_layers = nn.ModuleList([
            CT_EncoderLayer(d_model, n_heads_channel, n_heads_time, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 4. 最终输出层
        # 为所有输入通道计算时序预测，然后选择目标通道
        head_in_features = int(self.num_patches * self.d_model)
        
        # 首先对所有通道进行时间预测
        self.temporal_projection = nn.Linear(head_in_features, self.pred_len)
        
        # 然后创建一个从所有输入通道到目标输出通道的映射
        # 重要：我们需要将 c_in 个输入通道转换为 c_out 个输出通道
        self.channel_mapping = nn.Linear(c_in, c_out)

    def forward(self, x):
        # Input x: [Batch, c_in, seq_len]
        B, M, L = x.shape  # M 是 c_in
        
        # 1. Instance Normalization
        x = self.revin(x, 'norm') # [Batch, c_in, seq_len]
        
        # 2. Patching Process
        # Padding to ensure complete patches
        remaining = (L - self.patch_len) % self.stride
        if remaining != 0:
            pad_len = self.stride - remaining
            x = torch.nn.functional.pad(x, (0, pad_len), mode='replicate')
        
        # Unfold: [B, M, L_pad] -> [B, M, N, P]
        x_patched = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Permute to [B, N, M, P]
        x_patched = x_patched.permute(0, 2, 1, 3) 
        
        # 3. Projection
        x_emb = self.patch_embedding(x_patched) # [B, N, M, d_model]
        
        # Add Positional Embedding
        x_emb = x_emb + self.position_embedding
        x_emb = self.dropout(x_emb)
        
        # 4. Dual-Attention Encoder
        z = x_emb
        for layer in self.encoder_layers:
            z = layer(z)
        # z shape: [B, N, M, d_model]
            
        # 5. 输出处理 - 关键步骤
        # Reshape: [B, N, M, d_model] -> [B, M, N, d_model]
        z = z.permute(0, 2, 1, 3)  # [B, M, N, d_model]
        # Flatten: [B, M, N*d_model]
        z = z.reshape(B, M, -1)      # [B, c_in, N*d_model]
        
        # Apply temporal projection to all channels: [B, c_in, N*d_model] -> [B, c_in, pred_len]
        temp_pred = self.temporal_projection(z)  # [B, c_in, pred_len]
        
        # 通过通道映射层: [B, c_in, pred_len] -> [B, c_out, pred_len]
        # 首先转置以适配线性层: [B, c_in, pred_len] -> [B, pred_len, c_in]
        temp_pred_transposed = temp_pred.transpose(1, 2)  # [B, pred_len, c_in]
        mapped_pred = self.channel_mapping(temp_pred_transposed)  # [B, pred_len, c_out]
        out = mapped_pred.transpose(1, 2)  # [B, c_out, pred_len]
        
        return out

# 辅助类保持不变
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

class CT_EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads_channel, n_heads_time, d_ff, dropout=0.1):
        super().__init__()
        
        # 1. Channel Attention (捕捉站点/变量间关系)
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

def train_ct_patchtst_final(X, y, splits, preproc_pipe, exp_pipe):
    # 加载保存的参数
    params = np.load('tsai/data/model_params.npz')
    num_stations = int(params['num_stations'])
    feat_size = int(params['feat_size'])
    n_vars_total = int(params['n_vars_total'])
    fcst_history = int(params['fcst_history'])
    fcst_horizon = int(params['fcst_horizon'])

    print(f"训练参数: c_in={X.shape[1]}, c_out={y.shape[1]}, seq_len={X.shape[2]}, pred_len={y.shape[2]}")

    # CT-PatchTST 的配置
    arch_config = {
        'patch_len': 16,
        'stride': 8,
        'd_model': 64,        
        'd_ff': 128,          
        'n_layers': 1,        
        'n_heads_channel': 1,
        'n_heads_time': 4,    
        'dropout': 0.1
    }

    learn = TSForecaster(
        X, y,
        splits=splits,
        batch_size=16,
        pipelines=[preproc_pipe, exp_pipe],
        arch=CT_PatchTST_Final,  # 使用最终修正版本
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
    learn.export('tsai/models/ct_patchTST_final.pt')
    print("✓ 模型训练完成并已导出")

    return learn