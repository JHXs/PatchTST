"""
修正后的数据准备代码 - ST_PatchTST模型
主要修正:
1. 正确合并多站点数据,按时间对齐
2. 为每个站点的特征添加站点ID后缀
3. 确保数据维度正确: [Samples, num_stations * feat_size, seq_len]
"""

import sklearn
import pandas as pd
import numpy as np
from scipy import stats
from tsai.all import *

# ========== 1. 筛选相关站点 ==========
primary_var = "PM25_Concentration"
station_id_list = np.arange(9017, 9047).tolist() + [9058]
station_id_list = np.array(station_id_list)

r_thred = 0.85
center_station_id = 9022
station_id_related_list = []

df_one_station = pd.read_csv('tsai/data/stations_data_Guangzhou/df_station_{}.csv'.format(center_station_id))
v_list_1 = list(df_one_station[primary_var])

for station_id_other in station_id_list:
    df_one_station_other = pd.read_csv('tsai/data/stations_data_Guangzhou/df_station_{}.csv'.format(station_id_other))
    v_list_2 = list(df_one_station_other[primary_var])
    r, p = stats.pearsonr(v_list_1, v_list_2)
    if r > r_thred:
        station_id_related_list.append(station_id_other)

station_id_related_list = np.array(station_id_related_list)
print(f"相关站点数量: {len(station_id_related_list)}")
print(f"相关站点ID: {station_id_related_list}")

assert center_station_id in station_id_related_list, (
    f"中心站点 {center_station_id} 不在相关站点列表中，无法构造中心站输出。"
)
center_station_idx = int(np.where(station_id_related_list == center_station_id)[0][0])
print(f"中心站点ID: {center_station_id}")
print(f"中心站点索引 (center_station_idx): {center_station_idx}")

# ========== 2. 正确合并多站点数据 ==========
dfs_to_merge = []

# 读取第一个站点获取原始特征列名
df_first_station = pd.read_csv(
    'tsai/data/stations_data_Guangzhou/df_station_{}.csv'.format(station_id_related_list[0])
)
original_feature_cols = [col for col in df_first_station.columns
                        if col not in ['station_id', 'time']]

print(f"每个站点的特征数: {len(original_feature_cols)}")
print(f"特征列: {original_feature_cols}")

# 为每个站点加载数据并重命名特征列
for station_id in station_id_related_list:
    df_one_station = pd.read_csv(
        f'tsai/data/stations_data_Guangzhou/df_station_{station_id}.csv'
    )

    # 转换时间列
    df_one_station["time"] = pd.to_datetime(df_one_station["time"])

    # 为特征列添加站点ID后缀,例如: PM25_Concentration -> PM25_Concentration_9022
    new_cols = {col: f"{col}_{station_id}" for col in original_feature_cols}

    # 选择并重命名列
    df_one_station_processed = df_one_station[['time'] + original_feature_cols].rename(columns=new_cols)

    dfs_to_merge.append(df_one_station_processed)

# 按时间合并所有站点数据 (使用outer join确保所有时间戳都保留)
df_raw = dfs_to_merge[0]
for i in range(1, len(dfs_to_merge)):
    df_raw = pd.merge(df_raw, dfs_to_merge[i], on='time', how='outer')

# 按时间排序并重置索引
df_raw = df_raw.sort_values("time").reset_index(drop=True)

print(f"\n合并后的数据形状: {df_raw.shape}")
print(f"合并后的列数: {len(df_raw.columns)}")
print(f"\n前5行数据:")
print(df_raw.head())

# ========== 3. 定义模型参数 ==========
num_stations = len(station_id_related_list)
feat_size = len(original_feature_cols)
n_vars_total = num_stations * feat_size

print(f"\n模型参数:")
print(f"  站点数量 (num_stations): {num_stations}")
print(f"  每站点特征数 (feat_size): {feat_size}")
print(f"  总通道数 (n_vars_total): {n_vars_total}")
print(f"  中心站点索引 (center_station_idx): {center_station_idx}")

# ========== 4. 数据预处理 ==========
datetime_col = "time"
freq = '1h'
method = 'ffill'
value = 0

columns_for_preproc = [col for col in df_raw.columns if col != datetime_col]

preproc_pipe = sklearn.pipeline.Pipeline([
    ('shrinker', TSShrinkDataFrame()),
    ('drop_duplicates', TSDropDuplicates(datetime_col=datetime_col)),
    ('add_mts', TSAddMissingTimestamps(datetime_col=datetime_col, freq=freq)),
    ('fill_missing', TSFillMissing(columns=columns_for_preproc, method=method, value=value)),
], verbose=True)

df = preproc_pipe.fit_transform(df_raw)

# 保存预处理管道
mkdir('tsai/data', exist_ok=True, parents=True)
save_object(preproc_pipe, 'tsai/data/preproc_pipe.pkl')

print(f"\n预处理后数据形状: {df.shape}")

# ========== 5. 数据标准化 ==========
fcst_history = 168
fcst_horizon = 24
valid_size = 0.1
test_size = 0.2

# 先生成splits以获取训练集索引
splits = get_forecasting_splits(
    df,
    fcst_history=fcst_history,
    fcst_horizon=fcst_horizon,
    datetime_col=datetime_col,
    valid_size=valid_size,
    test_size=test_size
)

save_object(splits, 'tsai/data/splits.pkl')

train_split = splits[0]
print(f"\n数据集划分:")
print(f"  训练集样本数: {len(train_split)}")
print(f"  验证集样本数: {len(splits[1])}")
print(f"  测试集样本数: {len(splits[2])}")

# 标准化
columns_for_exp_pipe = df.columns[1:]
exp_pipe = sklearn.pipeline.Pipeline([
    ('scaler', TSStandardScaler(columns=columns_for_exp_pipe)),
], verbose=True)

df_scaled = exp_pipe.fit_transform(df, scaler__idxs=train_split)

# 保存标准化管道
save_object(exp_pipe, 'tsai/data/exp_pipe.pkl')

# ========== 6. 准备时间序列数据 ==========
x_vars = [col for col in df.columns if col != datetime_col]

# 为y_vars只保留center_station_id对应的列
center_vars = [col for col in df.columns if col.endswith(f"_{center_station_id}")]

print(f"\n输入变量数量 (多个站点): {len(x_vars)}")
print(f"输出变量数量 (中心站点{center_station_id}): {len(center_vars)}")
print(f"输出变量列: {center_vars}")

# 准备X: 使用所有站点的所有变量
X, _ = prepare_forecasting_data(
    df,
    fcst_history=fcst_history,
    fcst_horizon=fcst_horizon,
    x_vars=x_vars,
    y_vars=x_vars
)

# 准备y: 只使用center_station_id的变量
_, y = prepare_forecasting_data(
    df,
    fcst_history=fcst_history,
    fcst_horizon=fcst_horizon,
    x_vars=x_vars,
    y_vars=center_vars
)

print(f"\nX.shape: {X.shape}")
print(f"y.shape: {y.shape}")
print(f"预期X形状: [Samples, {n_vars_total}, {fcst_history}]")
print(f"预期y形状: [Samples, {feat_size}, {fcst_horizon}]")

# 验证维度
assert X.shape[1] == n_vars_total, f"X的通道数错误! 期望{n_vars_total}, 实际{X.shape[1]}"
assert X.shape[2] == fcst_history, f"X的时间步数错误! 期望{fcst_history}, 实际{X.shape[2]}"
assert y.shape[1] == feat_size, f"y的通道数错误! 期望{feat_size}, 实际{y.shape[1]}"
assert y.shape[2] == fcst_horizon, f"y的时间步数错误! 期望{fcst_horizon}, 实际{y.shape[2]}"

print("\n✓ 数据维度验证通过!")

np.savez('tsai/data/X.npz', X)
np.savez('tsai/data/y.npz', y)

# ========== 8. 模型配置 ==========
arch_config = {
    'num_stations': num_stations,
    'feat_size': feat_size,
    'arch_config': {
        'n_layers': 3,
        'n_heads': 4,
        'd_model': 16,
        'd_ff': 128,
        'dropout': 0.2
    }
}

print(f"\n模型配置:")
print(f"  arch_config: {arch_config}")

# 保存关键参数供后续使用
np.savez('tsai/data/model_params.npz',
         num_stations=num_stations,
         feat_size=feat_size,
         n_vars_total=n_vars_total,
         fcst_history=fcst_history,
         fcst_horizon=fcst_horizon,
         center_station_id=center_station_id,
         center_station_idx=center_station_idx)

print("\n✓ 数据准备完成!")
