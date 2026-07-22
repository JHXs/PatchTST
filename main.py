import pandas as pd
import numpy as np
from tsai.all import *
# from data_preparation import X, y, splits, preproc_pipe, exp_pipe
from CT_PatchTST_model import CT_PatchTST, train_ct_patchtst, evaluate_ct_patchtst
from Informer_model import Informer, train_informer, evaluate_informer
from ST_PatchTST_model import ST_PatchTST, train_st_patchtst, evaluate_st_patchtst
from PatchTST import PatchTST, train_patchtst, evaluate_patchtst

# 加载X, y
X = np.load('tsai/data/X.npz')['arr_0']
y = np.load('tsai/data/y.npz')['arr_0']
splits = load_object('tsai/data/splits.pkl')
preproc_pipe = load_object('tsai/data/preproc_pipe.pkl')
exp_pipe = load_object('tsai/data/exp_pipe.pkl')

print(f"加载X形状: {X.shape}", f"加载y形状: {y.shape}")
# print(f"加载splits: {splits}")
# print(f"加载preproc_pipe: {preproc_pipe}")
# print(f"加载exp_pipe: {exp_pipe}")

train_model = "pa"

if train_model == "st":
    print("训练ST_PatchTST模型")
    learn = train_st_patchtst(X, y, splits, preproc_pipe, exp_pipe)
elif train_model == "ct":
    print("训练CT_PatchTST模型")
    learn = train_ct_patchtst(X, y, splits, preproc_pipe, exp_pipe)
elif train_model == "informer":
    print("训练Informer模型")
    learn = train_informer(X, y, splits, preproc_pipe, exp_pipe)
elif train_model in ("pa", "patchtst"):
    print("训练PatchTST模型")
    learn = train_patchtst(X, y, splits, preproc_pipe, exp_pipe)
else:
    raise ValueError(f"未知模型类型: {train_model}")

if train_model == "st":
    print("评估ST_PatchTST模型")
    evaluation_results = evaluate_st_patchtst(learn, X, y, splits)
elif train_model == "ct":
    print("评估CT_PatchTST模型")
    summary_df, test_preds = evaluate_ct_patchtst(learn, X, y, splits)
    evaluation_results = {"summary": summary_df, "test_preds": test_preds}
elif train_model == "informer":
    print("评估Informer模型")
    evaluation_results = evaluate_informer(learn, X, y, splits)
else:
    print("评估PatchTST模型")
    evaluation_results = evaluate_patchtst(learn, X, y, splits)
print(f"\n返回结果包含: {list(evaluation_results.keys())}")
