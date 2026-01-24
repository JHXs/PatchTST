import pandas as pd
from tsai.all import *
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_patchtst(X, y, splits, preproc_pipe, exp_pipe):

    # ========== 模型配置 ==========
    arch_config = dict(
        n_layers=1,  # number of encoder layers
        n_heads=1,  # number of heads
        d_model=6,  # dimension of model
        d_ff=2,  # dimension of fully connected network
        attn_dropout=0.0, # dropout applied to the attention weights
        dropout=0.8,  # dropout applied to all linear layers in the encoder except q,k&v projections
        patch_len=2,  # length of the patch applied to the time series to create patches
        stride=2,  # stride used when creating patches
        padding_patch=True,  # padding_patch
    )

    print(f"\n模型配置:")
    print(f"  arch_config: {arch_config}")

    learn = TSForecaster(X, y, splits=splits, batch_size=16, path="tsai/models", pipelines=[preproc_pipe, exp_pipe],
                     arch="PatchTST", arch_config=arch_config, metrics=[mse, mae], cbs=[])
    print("✓ TSForecaster实例化成功")

    # 查找学习率
    lr_max = learn.lr_find().valley
    print(f"✓ 最优学习率: {lr_max}")

    # 训练模型
    n_epochs = 50
    lr = lr_max
    learn.fit_one_cycle(n_epochs, lr_max=lr)

    # 导出模型
    learn.export('patchTST.pt')
    print("✓ 模型训练完成并已导出")

    return learn

def evaluate_patchtst(learn, X, y, splits):
    scaled_preds, *_ = learn.get_X_preds(X[splits[1]])
    scaled_preds = to_np(scaled_preds)
    print(f"scaled_preds.shape: {scaled_preds.shape}")

    scaled_y_true = y[splits[1]]
    results_df = pd.DataFrame(columns=["mse", "mae"]) # 创建空的 dataframe
    results_df.loc["valid", "mse"] = mean_squared_error(scaled_y_true.flatten(), scaled_preds.flatten())
    results_df.loc["valid", "mae"] = mean_absolute_error(scaled_y_true.flatten(), scaled_preds.flatten())

    y_test_preds, *_ = learn.get_X_preds(X[splits[2]])
    y_test_preds = to_np(y_test_preds)
    print(f"y_test_preds.shape: {y_test_preds.shape}")

    y_test = y[splits[2]]
    # results_df = pd.DataFrame(columns=["mse", "mae"])
    results_df.loc["test", "mse"] = mean_squared_error(y_test.flatten(), y_test_preds.flatten())
    results_df.loc["test", "mae"] = mean_absolute_error(y_test.flatten(), y_test_preds.flatten())
    # print(f"results_df: {results_df}")
    return results_df

if __name__ == "__main__":
    print("先运行 data_preparation_single.py 准备单站点数据\n")
    print("运行 train_patchtst 方法开始训练\n")
    print("运行 evaluate_patchtst 方法开始评估\n")