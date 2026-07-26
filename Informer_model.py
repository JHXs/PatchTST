import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tsai.all import *

from evaluation_utils import (
    build_results_dataframes,
    build_split_evaluation,
    infer_num_metrics,
    load_target_names,
    print_evaluation_results,
)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.token_conv(x.transpose(1, 2)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super().__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        b, query_len, _, head_dim = queries.shape
        _, key_len, _, _ = keys.shape
        scale = self.scale or 1.0 / math.sqrt(head_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = _causal_mask(query_len, key_len, scores.device)
            scores = scores.masked_fill(attn_mask, -torch.finfo(scores.dtype).max)

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attn, values)
        return values.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_qk(self, q, k, sample_k, n_top):
        b, h, key_len, head_dim = k.shape
        _, _, query_len, _ = q.shape

        k_expand = k.unsqueeze(-3).expand(b, h, query_len, key_len, head_dim)
        index_sample = torch.randint(key_len, (query_len, sample_k), device=k.device)
        query_index = torch.arange(query_len, device=k.device).unsqueeze(1)
        k_sample = k_expand[:, :, query_index, index_sample, :]
        qk_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze(-2)

        sparsity = qk_sample.max(dim=-1).values - qk_sample.mean(dim=-1)
        top_index = sparsity.topk(n_top, sorted=False).indices

        batch_index = torch.arange(b, device=q.device)[:, None, None]
        head_index = torch.arange(h, device=q.device)[None, :, None]
        q_reduce = q[batch_index, head_index, top_index, :]
        qk = torch.matmul(q_reduce, k.transpose(-2, -1))
        return qk, top_index

    def _initial_context(self, values, query_len):
        if not self.mask_flag:
            context = values.mean(dim=-2).unsqueeze(-2).expand(
                *values.shape[:2], query_len, values.shape[-1]
            )
            return context.clone()

        if query_len != values.shape[-2]:
            raise ValueError("ProbSparse causal self-attention requires query_len == value_len")
        return values.cumsum(dim=-2)

    def _update_context(self, context, values, scores, index):
        b, h, value_len, _ = values.shape
        if self.mask_flag:
            query_len = context.shape[-2]
            mask = _prob_mask(b, h, query_len, value_len, index, scores.device)
            scores = scores.masked_fill(mask, -torch.finfo(scores.dtype).max)

        attn = self.dropout(torch.softmax(scores, dim=-1))
        batch_index = torch.arange(b, device=values.device)[:, None, None]
        head_index = torch.arange(h, device=values.device)[None, :, None]
        context[batch_index, head_index, index, :] = torch.matmul(attn, values).type_as(context)
        return context

    def forward(self, queries, keys, values, attn_mask=None):
        b, query_len, h, head_dim = queries.shape
        _, key_len, _, _ = keys.shape

        q = queries.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)

        sample_k = min(_log_sample_size(key_len, self.factor), key_len)
        n_top = min(_log_sample_size(query_len, self.factor), query_len)

        scores_top, top_index = self._prob_qk(q, k, sample_k=sample_k, n_top=n_top)
        scale = self.scale or 1.0 / math.sqrt(head_dim)
        scores_top = scores_top * scale

        context = self._initial_context(v, query_len)
        context = self._update_context(context, v, scores_top, top_index)
        return context.transpose(1, 2).contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")

        self.inner_attention = attention
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        b, query_len, _ = queries.shape
        _, key_len, _ = keys.shape

        queries = self.query_projection(queries).view(b, query_len, self.n_heads, self.head_dim)
        keys = self.key_projection(keys).view(b, key_len, self.n_heads, self.head_dim)
        values = self.value_projection(values).view(b, key_len, self.n_heads, self.head_dim)

        out, attn = self.inner_attention(queries, keys, values, attn_mask=attn_mask)
        out = out.reshape(b, query_len, self.n_heads * self.head_dim)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y), attn


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.down_conv = nn.Conv1d(
            c_in, c_in, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.down_conv(x.transpose(1, 2))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x.transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers[:-1], self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        new_x, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        new_x, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x + self.dropout(new_x)
        y = self.norm2(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class LTSF_Informer(nn.Module):
    """
    Informer-style long-sequence forecaster adapted to tsai's single-input API.

    Input shape:  [batch, channels, seq_len]
    Output shape: [batch, target_channels, pred_len]
    """

    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        pred_dim=None,
        label_len=None,
        factor=5,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        dropout=0.1,
        attn_dropout=0.1,
        attn="prob",
        activation="gelu",
        distil=True,
        **kwargs,
    ):
        super().__init__()

        self.c_in = int(c_in)
        self.seq_len = int(seq_len)
        self.c_out, self.pred_len = _resolve_output_shape(c_out, pred_dim)
        self.label_len = int(label_len) if label_len is not None else min(48, max(1, self.seq_len // 2))
        self.label_len = min(self.label_len, self.seq_len)

        attention_cls = ProbAttention if attn == "prob" else FullAttention

        self.enc_embedding = DataEmbedding(self.c_in, d_model, dropout=dropout)
        self.dec_embedding = DataEmbedding(self.c_in, d_model, dropout=dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        attention_cls(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=attn_dropout,
                        )
                        if attention_cls is ProbAttention
                        else attention_cls(mask_flag=False, attention_dropout=attn_dropout),
                        d_model=d_model,
                        n_heads=n_heads,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            [
                ConvLayer(d_model)
                for _ in range(e_layers - 1)
            ]
            if distil and e_layers > 1
            else None,
            norm_layer=nn.LayerNorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        attention_cls(
                            mask_flag=True,
                            factor=factor,
                            attention_dropout=attn_dropout,
                        )
                        if attention_cls is ProbAttention
                        else attention_cls(mask_flag=True, attention_dropout=attn_dropout),
                        d_model=d_model,
                        n_heads=n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=attn_dropout),
                        d_model=d_model,
                        n_heads=n_heads,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, self.c_out)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Informer 输入期望 3 维 [B, C, L]，实际为 {x.ndim} 维")

        x_enc = x.transpose(1, 2)
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)

        known_tokens = x_enc[:, -self.label_len :, :]
        future_tokens = x_enc.new_zeros(x_enc.size(0), self.pred_len, self.c_in)
        x_dec = torch.cat([known_tokens, future_tokens], dim=1)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out)
        dec_out = self.projection(dec_out[:, -self.pred_len :, :])
        return dec_out.transpose(1, 2)


Informer = LTSF_Informer


def train_informer(X, y, splits, preproc_pipe, exp_pipe):
    label_len = min(48, max(1, X.shape[-1] // 2))
    arch_config = {
        "label_len": label_len,
        "factor": 5,
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 128,
        "dropout": 0.1,
        "attn_dropout": 0.1,
        "attn": "prob",
        "activation": "gelu",
        "distil": True,
    }

    print("\n========== Informer 模型配置 ==========")
    for key, value in arch_config.items():
        print(f"  {key}: {value}")

    learn = TSForecaster(
        X,
        y,
        splits=splits,
        batch_size=16,
        path="tsai/models",
        pipelines=[preproc_pipe, exp_pipe],
        arch=Informer,
        arch_config=arch_config,
        metrics=[rmse, mse, mae],
        cbs=[],
    )
    print("[OK] TSForecaster实例化成功")

    lr_max = learn.lr_find().valley
    lr = min(lr_max, 1e-4)
    print(f"[OK] 最优学习率: {lr_max}", f"（实际使用学习率: {lr}）")

    n_epochs = 50
    learn.fit_one_cycle(n_epochs, lr_max=lr)

    learn.export("informer.pt")
    print("[OK] 模型训练完成并已导出")
    return learn


def evaluate_informer(learn, X, y, splits):
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


def _resolve_output_shape(c_out, pred_dim):
    if isinstance(pred_dim, (list, tuple)):
        if len(pred_dim) == 0:
            raise ValueError("pred_dim 不能为空")
        if len(pred_dim) == 1:
            return int(c_out), int(pred_dim[0])
        return int(pred_dim[0]), int(pred_dim[-1])
    if pred_dim is None:
        return int(c_out), 1
    return int(c_out), int(pred_dim)


def _log_sample_size(length, factor):
    return max(1, int(factor * math.ceil(math.log(max(length, 2)))))


def _causal_mask(query_len, key_len, device):
    mask = torch.triu(
        torch.ones(query_len, key_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)


def _prob_mask(batch_size, n_heads, query_len, value_len, index, device):
    base_mask = _causal_mask(query_len, value_len, device).expand(
        batch_size, n_heads, query_len, value_len
    )
    gather_index = index.unsqueeze(-1).expand(-1, -1, -1, value_len)
    return base_mask.gather(2, gather_index)


if __name__ == "__main__":
    print("Informer模型定义完成")
    print("\n使用说明:")
    print("1. 先运行 data_preparation_single.py 或 data_preparation.py 准备数据")
    print("2. 然后运行 train_informer(X, y, splits, preproc_pipe, exp_pipe) 进行训练")
    print("3. 最后运行 evaluate_informer(learn, X, y, splits) 进行评估")
