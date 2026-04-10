import numpy as np
from tsai.all import *
from evaluation_utils import (
    build_results_dataframes,
    build_split_evaluation,
    infer_num_metrics,
    load_target_names,
    print_evaluation_results,
)

def train_patchtst(X, y, splits, preproc_pipe, exp_pipe):

    # ========== 模型配置 ==========
    arch_config = dict(
        n_layers=3,  # number of encoder layers
        n_heads=4,  # number of heads
        d_model=16,  # dimension of model
        d_ff=128,  # dimension of fully connected network
        attn_dropout=0.0, # dropout applied to the attention weights
        dropout=0.2,  # dropout applied to all linear layers in the encoder except q,k&v projections
        patch_len=4,  # length of the patch applied to the time series to create patches
        stride=2,  # stride used when creating patches
        padding_patch=True,  # padding_patch
    )

    print(f"\n模型配置:")
    print(f"  arch_config: {arch_config}")

    learn = TSForecaster(X, y, splits=splits, batch_size=16, path="tsai/models", pipelines=[preproc_pipe, exp_pipe],
                     arch="PatchTST", arch_config=arch_config, metrics=[rmse, mse, mae], cbs=[])
    print("✓ TSForecaster实例化成功")

    # 查找学习率
    lr_max = learn.lr_find().valley
    lr = min(lr_max, 1e-4)  # 设置学习率上限，避免过大
    print(f"✓ 最优学习率: {lr_max}", f"（实际使用学习率: {lr}）")

    # 训练模型
    n_epochs = 50
    learn.fit_one_cycle(n_epochs, lr_max=lr)

    # 导出模型
    learn.export('patchTST.pt')
    print("✓ 模型训练完成并已导出")

    return learn

def evaluate_patchtst(learn, X, y, splits):
    target_names = load_target_names(expected_count=infer_num_metrics(y))
    summary_rows = []
    per_metric_rows = []

    for split_name, split_idxs in (("valid", splits[1]), ("test", splits[2])):
        preds, *_ = learn.get_X_preds(X[split_idxs])
        preds = to_np(preds)
        print(f"{split_name} 预测形状: {preds.shape}")

        y_true = y[split_idxs]
        summary_row, split_metric_rows = build_split_evaluation(y_true, preds, split_name, target_names)
        summary_rows.append(summary_row)
        per_metric_rows.extend(split_metric_rows)

    summary_df, per_metric_df = build_results_dataframes(summary_rows, per_metric_rows)
    print_evaluation_results(summary_df, per_metric_df)
    return {"summary": summary_df, "per_metric": per_metric_df}

if __name__ == "__main__":
    print("先运行 data_preparation_single.py 准备单站点数据\n")
    print("运行 train_patchtst 方法开始训练\n")
    print("运行 evaluate_patchtst 方法开始评估\n")
