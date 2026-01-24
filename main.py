import pandas as pd
import numpy as np
from tsai.all import *
# from data_preparation import X, y, splits, preproc_pipe, exp_pipe
from CT_PatchTST_model import CT_PatchTST, train_ct_patchtst
from ST_PatchTST_model import ST_PatchTST, train_st_patchtst
from PatchTST import PatchTST, train_patchtst, evaluate_patchtst

# 加载X, y
X = np.load('tsai/data/X.npz')['arr_0']
y = np.load('tsai/data/y.npz')['arr_0']
splits = load_object('tsai/data/splits.pkl')
preproc_pipe = load_object('tsai/data/preproc_pipe.pkl')
exp_pipe = load_object('tsai/data/exp_pipe.pkl')

print(f"加载X形状: {X.shape}", f"加载y形状: {y.shape}")
print(f"加载splits: {splits}")
print(f"加载preproc_pipe: {preproc_pipe}")
print(f"加载exp_pipe: {exp_pipe}")

learn = train_st_patchtst(X, y, splits, preproc_pipe, exp_pipe) 
# learn = train_patchtst(X, y, splits, preproc_pipe, exp_pipe)

results_df = evaluate_patchtst(learn, X, y, splits)
print(f"results_df: {results_df}")