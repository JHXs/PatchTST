# ST_PatchTST 修正版使用说明

## 问题总结

原始笔记本的数据准备阶段存在以下问题:

### 1. 数据合并错误
- **问题**: 使用 `pd.concat` 简单合并所有站点数据,导致时间戳混乱
- **后果**: 每个站点的数据无法正确对齐,无法构建正确的时空维度

### 2. 特征列命名冲突
- **问题**: 没有为不同站点的特征添加后缀
- **后果**: 多个站点的特征列名相同,数据互相覆盖

### 3. 维度不匹配
- **问题**: 模型期望 `[Batch, num_stations * feat_size, seq_len]`,但数据准备无法正确构建
- **后果**: 模型无法正确处理输入数据

## 修正方案

### 文件说明

1. **data_preparation.py** - 多站点数据准备脚本
2. **ST_PatchTST_model.py** - ST_PatchTST 模型定义和训练脚本

### 使用步骤

#### 第一步: 数据准备

```python
# 运行数据准备脚本
%run /home/hansel/Documents/ITProject/Python/PatchTST/data_preparation.py
```

**输出结果:**
- `X`: 输入数据,形状 `[Samples, num_stations * feat_size, seq_len]`
- `y`: 标签数据,形状 `[Samples, feat_size, pred_len]`
- `splits`: 数据集划分索引
- `preproc_pipe`: 预处理管道
- `exp_pipe`: 标准化管道
- `model_params.npz`: 模型参数文件

#### 第二步: 训练模型

```python
# 加载X, y
X = np.load('tsai/data/X.npz')['arr_0']
y = np.load('tsai/data/y.npz')['arr_0']
splits = load_object('tsai/data/splits.pkl')
preproc_pipe = load_object('tsai/data/preproc_pipe.pkl')
exp_pipe = load_object('tsai/data/exp_pipe.pkl')

# 训练模型
learn = train_st_patchtst(X, y, splits, preproc_pipe, exp_pipe)
```

**输出结果:**
- 学习率曲线
- 训练过程可视化
- 训练好的模型文件: `tsai/models/ST-PatchTST.pt`

#### 第三步: 评估模型

```python
# 评估模型
results_df, y_test_preds = evaluate_st_patchtst(learn, X, y, splits)
```

**输出结果:**
- 验证集和测试集的MSE、MAE指标
- 预测结果

## 关键修正点

### 1. 数据合并逻辑

**修正前:**
```python
# 错误: 使用concat简单合并
df_raw = pd.concat(df_raw, ignore_index=True)
```

**修正后:**
```python
# 正确: 按时间对齐合并
for station_id in station_id_related_list:
    df_one_station = pd.read_csv(f'tsai/data/stations_data_Guangzhou/df_station_{station_id}.csv')
    df_one_station["time"] = pd.to_datetime(df_one_station["time"])

    # 为特征列添加站点ID后缀
    new_cols = {col: f"{col}_{station_id}" for col in original_feature_cols}
    df_one_station_processed = df_one_station[['time'] + original_feature_cols].rename(columns=new_cols)
    dfs_to_merge.append(df_one_station_processed)

# 按时间合并
df_raw = dfs_to_merge[0]
for i in range(1, len(dfs_to_merge)):
    df_raw = pd.merge(df_raw, dfs_to_merge[i], on='time', how='outer')
```

### 2. 特征列命名

**修正前:**
```python
# 错误: 所有站点使用相同的列名
# PM25_Concentration, PM25_Concentration, PM25_Concentration, ...
```

**修正后:**
```python
# 正确: 每个站点的特征添加站点ID后缀
# PM25_Concentration_9017, PM25_Concentration_9018, PM25_Concentration_9022, ...
new_cols = {col: f"{col}_{station_id}" for col in original_feature_cols}
```

### 3. 维度验证

**修正后添加了维度验证:**
```python
# 验证输入维度
assert X.shape[1] == n_vars_total, f"X的通道数错误! 期望{n_vars_total}, 实际{X.shape[1]}"
assert X.shape[2] == fcst_history, f"X的时间步数错误! 期望{fcst_history}, 实际{X.shape[2]}"

# 验证输出维度
assert y.shape[1] == n_vars_total, f"y的通道数错误! 期望{n_vars_total}, 实际{y.shape[1]}"
assert y.shape[2] == fcst_horizon, f"y的时间步数错误! 期望{fcst_horizon}, 实际{y.shape[2]}"
```

### 4. 模型初始化参数

**修正后:**
```python
# arch_config只包含自定义参数,不包含TSForecaster自动传递的参数
arch_config = {
    'num_stations': num_stations,  # 自定义参数
    'feat_size': feat_size,        # 自定义参数
    'arch_config': {               # 内部PatchTST的配置
        'n_layers': 3,
        'n_heads': 4,
        'd_model': 128,
        'd_ff': 512,
        'dropout': 0.2
    }
}
# c_in, c_out, seq_len, pred_dim 由TSForecaster自动传递
```

## 模型架构

### ST_PatchTST架构

```
输入: [Batch, num_stations * feat_size, seq_len]
  ↓
Reshape: [Batch, num_stations, feat_size, seq_len]
  ↓
Permute: [Batch, num_stations, seq_len, feat_size]
  ↓
空间融合层 (Conv2d)
  ├─ Conv2d(num_stations → 64, kernel=(1,1))
  ├─ BatchNorm2d(64)
  ├─ ReLU
  ├─ Conv2d(64 → 1, kernel=(1,1))
  ├─ BatchNorm2d(1)
  └─ ReLU
  ↓
输出: [Batch, 1, seq_len, feat_size]
  ↓
Squeeze: [Batch, seq_len, feat_size]
  ↓
Permute: [Batch, feat_size, seq_len]
  ↓
时间预测层 (PatchTST)
  └─ PatchTST(c_in=feat_size, c_out=feat_size, seq_len, pred_len)
  ↓
输出: [Batch, feat_size, pred_len]
```

## 参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `num_stations` | 站点数量 | 12 |
| `feat_size` | 每个站点的特征数 | 12 |
| `n_vars_total` | 总通道数 (num_stations * feat_size) | 144 |
| `seq_len` | 输入序列长度 | 168 |
| `pred_len` | 预测序列长度 | 24 |

## 数据维度

| 阶段 | 形状 | 说明 |
|------|------|------|
| 原始数据 | `(time_steps, 1 + num_stations * feat_size)` | DataFrame |
| X (输入) | `[Samples, num_stations * feat_size, seq_len]` | 3维张量 |
| y (标签) | `[Samples, num_stations * feat_size, pred_len]` | 3维张量 |
| 模型输入 | `[Batch, num_stations * feat_size, seq_len]` | 3维张量 |
| 模型输出 | `[Batch, feat_size, pred_len]` | 3维张量 |

## 常见问题

### Q1: 为什么模型输出只有 feat_size 个通道,而不是 num_stations * feat_size?

**A:** 因为空间融合层将所有站点融合为一个"虚拟超级站点",所以输出只需要预测融合后的特征。这是ST_PatchTST的设计理念:先在空间上融合,再在时间上预测。

### Q2: 如果想预测每个站点的未来值,应该怎么修改?

**A:** 有两种方案:
1. 修改模型,移除空间融合层,直接对每个站点独立预测
2. 在模型输出后添加一个空间解耦层,将融合后的预测值分配到各个站点

### Q3: 数据准备后如何验证维度是否正确?

**A:** 运行数据准备脚本后,会自动进行维度验证,并打印相关信息:
```python
print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")
print(f"预期X形状应为: [Samples, {n_vars_total}, {fcst_history}]")
print(f"预期y形状应为: [Samples, {n_vars_total}, {fcst_horizon}]")
```

