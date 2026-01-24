"""
单站点数据准备代码 - PatchTST模型
"""

import sklearn
import pandas as pd
import numpy as np
from scipy import stats
from tsai.all import *


def load_data(file_path: str) -> pd.DataFrame:
    """读取并排序数据"""
    df_raw = pd.read_csv(file_path)
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw = df_raw.sort_values("time").reset_index(drop=True)
    print(df_raw.head())
    return df_raw


def preprocess_data(df_raw: pd.DataFrame, datetime_col: str = "time", 
                    freq: str = '1h', method: str = 'ffill', value: int = 0) -> tuple:
    """数据预处理：去重、填充缺失时间戳、处理缺失值"""
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
    return df, preproc_pipe


def get_data_splits(df: pd.DataFrame, fcst_history: int, fcst_horizon: int,
                    datetime_col: str = "time", valid_size: float = 0.1, 
                    test_size: float = 0.2) -> tuple:
    """划分训练集、验证集、测试集"""
    splits = get_forecasting_splits(
        df, fcst_history=fcst_history, fcst_horizon=fcst_horizon,
        datetime_col=datetime_col, valid_size=valid_size, test_size=test_size,
        show_plot=False
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集样本数: {len(splits[0])}")
    print(f"  验证集样本数: {len(splits[1])}")
    print(f"  测试集样本数: {len(splits[2])}")
    
    return splits


def standardize_data(df: pd.DataFrame, train_split) -> tuple:
    """数据标准化"""
    columns_for_exp_pipe = df.columns[1:]
    exp_pipe = sklearn.pipeline.Pipeline([
        ('scaler', TSStandardScaler(columns=columns_for_exp_pipe)),
    ], verbose=True)
    
    df_scaled = exp_pipe.fit_transform(df, scaler__idxs=train_split)
    
    # 保存管道以备后续使用
    mkdir('tsai/data', exist_ok=True, parents=True)
    save_object(exp_pipe, 'tsai/data/exp_pipe.pkl')
    
    return df_scaled, exp_pipe


def prepare_time_series(df: pd.DataFrame, fcst_history: int, fcst_horizon: int) -> tuple:
    """准备时间序列数据"""
    print(df.columns)
    x_vars = df.columns[2:]  # 从第3列开始取值
    y_vars = df.columns[2:]
    
    X, y = prepare_forecasting_data(
        df, fcst_history=fcst_history, fcst_horizon=fcst_horizon,
        x_vars=x_vars, y_vars=y_vars
    )
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")
    return X, y


def save_data(X: np.ndarray, y: np.ndarray, fcst_history: int, fcst_horizon: int, splits, preproc_pipe, exp_pipe):
    """保存数据和参数"""
    np.savez('tsai/data/X.npz', X)
    np.savez('tsai/data/y.npz', y)
    
    # 使用 pickle 保存 splits（因为它们长度不同，无法用numpy保存）
    mkdir('tsai/data', exist_ok=True, parents=True)
    save_object(splits, 'tsai/data/splits.pkl')
    
    # 保存关键参数供后续使用
    np.savez('tsai/data/model_params.npz',
             fcst_history=fcst_history,
             fcst_horizon=fcst_horizon)

    # 保存数据预处理管道
    mkdir('tsai/data', exist_ok=True, parents=True)
    save_object(preproc_pipe, 'tsai/data/preproc_pipe.pkl')

    # 保存数据标准化管道
    mkdir('tsai/data', exist_ok=True, parents=True)
    save_object(exp_pipe, 'tsai/data/exp_pipe.pkl')


def main():
    """主函数：执行完整的数据准备流程"""
    # 配置参数
    file_path = "tsai/data/stations_data_Guangzhou/df_station_9022.csv"
    datetime_col = "time"
    fcst_history = 168  # 历史步数
    fcst_horizon = 24   # 预测步数
    valid_size = 0.1    # 验证集比例
    test_size = 0.2     # 测试集比例
    
    # 1. 读取数据
    df_raw = load_data(file_path)
    
    # 2. 数据预处理
    df, preproc_pipe = preprocess_data(df_raw, datetime_col=datetime_col)
    
    # 3. 数据集划分
    splits = get_data_splits(
        df, fcst_history=fcst_history, fcst_horizon=fcst_horizon,
        datetime_col=datetime_col, valid_size=valid_size, test_size=test_size
    )
    train_split = splits[0]
    
    # 4. 数据标准化
    df_scaled, exp_pipe = standardize_data(df, train_split)
    
    # 5. 准备时间序列数据
    X, y = prepare_time_series(df, fcst_history, fcst_horizon)
    
    # 6. 保存数据
    save_data(X, y, fcst_history, fcst_horizon, splits, preproc_pipe, exp_pipe)
    
    print("\n✓ 数据准备完成!")
    
    return X, y, splits, preproc_pipe, exp_pipe


if __name__ == "__main__":
    X, y, splits, preproc_pipe, exp_pipe = main()