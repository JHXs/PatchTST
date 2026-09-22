# B3 Best-of-baselines 与 ST 配对

负的 `ST−best` 与相对变化表示 ST 的 RMSE 更低。缺失 ST 产物显式标为 `missing`。

| city | history | horizon | winner_arm | winner_arm_mean_rmse_ugm3 | best_of_baselines_mean_rmse_ugm3 | st_mean_rmse_ugm3 | mean_st_minus_best_rmse_ugm3 | mean_st_relative_change_vs_best_percent | st_better_direction_count | st_pairing_status |
|---|---|---|---|---|---|---|---|---|---|---|
| beijing | 24 | 1 | multi_gru_matched | 19.677170 | 19.651992 | 20.879179 | 1.227187 | 6.256514 | 0/5 | available |
| beijing | 24 | 3 | multi_gru_default | 29.438311 | 29.438311 | 30.610879 | 1.172568 | 3.992808 | 0/3 | available |
| beijing | 24 | 6 | multi_gru_default | 39.179899 | 39.179899 | 39.913925 | 0.734027 | 1.878894 | 0/3 | available |
| beijing | 24 | 12 | multi_gru_default | 53.191242 | 51.818420 | 51.622506 | -0.195914 | -0.355603 | 1/3 | available |
| beijing | 24 | 24 | plain_mix_patchtst_all_default | 64.046549 | 64.009141 | 64.831502 | 0.822361 | 1.297319 | 0/3 | available |
| beijing | 48 | 1 | multi_gru_default | 19.989205 | 19.989205 | 20.820853 | 0.831648 | 4.176736 | 0/3 | available |
| beijing | 48 | 3 | multi_gru_default | 29.158254 | 29.158254 | 31.138627 | 1.980373 | 6.795378 | 0/3 | available |
| beijing | 48 | 6 | multi_gru_default | 38.987895 | 38.987895 | 41.838913 | 2.851018 | 7.327555 | 0/3 | available |
| beijing | 48 | 12 | multi_gru_default | 51.918429 | 51.918429 | 55.104867 | 3.186438 | 6.132119 | 0/3 | available |
| beijing | 48 | 24 | multi_gru_default | 66.614878 | 65.431978 | 69.501508 | 4.069530 | 6.269225 | 0/3 | available |
| beijing | 72 | 1 | multi_gru_default | 19.887283 | 19.887283 | 21.105325 | 1.218042 | 6.131186 | 0/3 | available |
| beijing | 72 | 3 | multi_gru_default | 29.194021 | 29.194021 | 31.272501 | 2.078480 | 7.134328 | 0/3 | available |
| beijing | 72 | 6 | multi_gru_default | 39.192560 | 39.192560 | 41.393878 | 2.201318 | 5.648193 | 0/3 | available |
| beijing | 72 | 12 | multi_gru_default | 51.430056 | 51.430056 | 55.277826 | 3.847769 | 7.492767 | 0/3 | available |
| beijing | 72 | 24 | multi_gru_default | 63.846198 | 63.846198 | 66.301552 | 2.455354 | 3.971436 | 0/3 | available |
| beijing | 168 | 1 | multi_gru_default | 19.938643 | 19.938643 | 22.083808 | 2.145165 | 10.760938 | 0/3 | available |
| beijing | 168 | 3 | trad_spatial_linear | 38.583023 | 33.792023 | 31.923554 | -1.868468 | -3.657696 | 1/2 | available |
| beijing | 168 | 6 | multi_gru_default | 39.141388 | 39.141388 | 42.100684 | 2.959296 | 7.576796 | 0/5 | available |
| beijing | 168 | 12 | multi_gru_default | 50.887287 | 54.149971 | 55.826781 | 1.676810 | 3.517773 | 1/3 | available |
| beijing | 168 | 24 | multi_gru_default | 61.288498 | 64.028744 | 69.033765 | 5.005021 | 8.158528 | 1/3 | available |
