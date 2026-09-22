# S3 主表：多站点模型 vs 最强单站点基线

> **解释边界：信息集不同：单站点基线仅使用中心站 PM2.5；ST/ST+频域额外使用邻站。相对提升只能归因于新增跨站信息的价值，不代表机制优于同信息集模型。**

`mean_reduction_percent > 0` 表示多站点模型 RMSE 更低；比较按种子配对，广州先在 8 个站内按种子池化。

| city | history | horizon | model_arm | best_single_station_arm | best_single_station_mean_rmse_ugm3 | model_mean_rmse_ugm3 | mean_reduction_percent | better_direction_count | pairing_status |
|---|---|---|---|---|---|---|---|---|---|
| beijing | 24 | 1 | st | center_lstm_matched | 20.830436 | 20.879179 | -0.236954 | 1/5 | complete |
| beijing | 24 | 1 | st_plus_frequency | center_lstm_matched | 20.830436 | 20.867889 | -0.182780 | 1/5 | complete |
| beijing | 24 | 3 | st | center_lstm_default | 31.219953 | 30.610879 | 1.949527 | 3/3 | complete |
| beijing | 24 | 3 | st_plus_frequency | center_lstm_default | 31.219953 | 30.553990 | 2.131722 | 3/3 | complete |
| beijing | 24 | 6 | st | center_gru_default | 40.126275 | 39.913925 | 0.528531 | 3/3 | complete |
| beijing | 24 | 6 | st_plus_frequency | center_gru_default | 40.126275 | 39.668437 | 1.140481 | 3/3 | complete |
| beijing | 24 | 12 | st | center_gru_default | 50.229370 | 51.622506 | -2.773569 | 0/3 | complete |
| beijing | 24 | 12 | st_plus_frequency | center_gru_default | 50.229370 | 51.201346 | -1.934854 | 0/3 | complete |
| beijing | 24 | 24 | st | center_gru_default | 60.197141 | 64.831502 | -7.703758 | 0/3 | complete |
| beijing | 24 | 24 | st_plus_frequency | center_gru_default | 60.197141 | 63.665273 | -5.765404 | 0/3 | complete |
| beijing | 48 | 1 | st | center_lstm_default | 21.110797 | 20.820853 | 1.374867 | 3/3 | complete |
| beijing | 48 | 1 | st_plus_frequency | center_lstm_default | 21.110797 | 20.781076 | 1.562962 | 3/3 | complete |
| beijing | 48 | 3 | st | center_lstm_default | 31.292734 | 31.138627 | 0.491426 | 2/3 | complete |
| beijing | 48 | 3 | st_plus_frequency | center_lstm_default | 31.292734 | 31.041485 | 0.801930 | 3/3 | complete |
| beijing | 48 | 6 | st | center_gru_default | 40.343953 | 41.838913 | -3.706807 | 0/3 | complete |
| beijing | 48 | 6 | st_plus_frequency | center_gru_default | 40.343953 | 41.475622 | -2.806419 | 0/3 | complete |
| beijing | 48 | 12 | st | center_gru_default | 50.181844 | 55.104867 | -9.800946 | 0/3 | complete |
| beijing | 48 | 12 | st_plus_frequency | center_gru_default | 50.181844 | 54.425363 | -8.446837 | 0/3 | complete |
| beijing | 48 | 24 | st | trad_ridge | 59.971481 | 69.501508 | -15.890931 | 0/3 | complete |
| beijing | 48 | 24 | st_plus_frequency | trad_ridge | 59.971481 | 68.403910 | -14.060732 | 0/3 | complete |
| beijing | 72 | 1 | st | center_lstm_default | 21.178962 | 21.105325 | 0.346646 | 2/3 | complete |
| beijing | 72 | 1 | st_plus_frequency | center_lstm_default | 21.178962 | 21.056562 | 0.576447 | 2/3 | complete |
| beijing | 72 | 3 | st | center_gru_default | 31.304386 | 31.272501 | 0.102981 | 2/3 | complete |
| beijing | 72 | 3 | st_plus_frequency | center_gru_default | 31.304386 | 31.214760 | 0.287554 | 2/3 | complete |
| beijing | 72 | 6 | st | center_gru_default | 40.007339 | 41.393878 | -3.466188 | 0/3 | complete |
| beijing | 72 | 6 | st_plus_frequency | center_gru_default | 40.007339 | 41.097383 | -2.725066 | 0/3 | complete |
| beijing | 72 | 12 | st | center_gru_default | 50.114943 | 55.277826 | -10.300056 | 0/3 | complete |
| beijing | 72 | 12 | st_plus_frequency | center_gru_default | 50.114943 | 54.865368 | -9.477363 | 0/3 | complete |
| beijing | 72 | 24 | st | center_gru_default | 59.614346 | 66.301552 | -11.215295 | 0/3 | complete |
| beijing | 72 | 24 | st_plus_frequency | center_gru_default | 59.614346 | 65.880727 | -10.509525 | 0/3 | complete |
| beijing | 168 | 1 | st | center_lstm_default | 21.258548 | 22.083808 | -3.882661 | 0/3 | complete |
| beijing | 168 | 1 | st_plus_frequency | center_lstm_default | 21.258548 | 21.797305 | -2.535019 | 0/3 | complete |
| beijing | 168 | 3 | st | center_patchtst_frozen | 32.167862 | 31.860429 | 0.955756 | 3/3 | complete |
| beijing | 168 | 3 | st_plus_frequency | center_patchtst_frozen | 32.167862 | 31.727272 | 1.369703 | 3/3 | complete |
| beijing | 168 | 6 | st | center_gru_matched | 40.161269 | 42.100684 | -4.828645 | 0/5 | complete |
| beijing | 168 | 6 | st_plus_frequency | center_gru_matched | 40.161269 | 41.929550 | -4.402430 | 0/5 | complete |
| beijing | 168 | 12 | st | trad_ridge | 51.320641 | 55.826781 | -8.780367 | 0/3 | complete |
| beijing | 168 | 12 | st_plus_frequency | trad_ridge | 51.320641 | 55.742360 | -8.615870 | 0/3 | complete |
| beijing | 168 | 24 | st | trad_ridge | 60.610630 | 69.033765 | -13.897125 | 0/3 | complete |
| beijing | 168 | 24 | st_plus_frequency | trad_ridge | 60.610630 | 68.741023 | -13.414137 | 0/3 | complete |
| guangzhou | 24 | 1 | st | center_gru_matched | 8.653739 | 8.514909 | 1.600536 | 3/3 | complete |
| guangzhou | 24 | 1 | st_plus_frequency | center_gru_matched | 8.653739 | 8.467184 | 2.151890 | 3/3 | complete |
| guangzhou | 168 | 6 | st | center_gru_matched | 14.125269 | 14.288717 | -1.157512 | 0/3 | complete |
| guangzhou | 168 | 6 | st_plus_frequency | center_gru_matched | 14.125269 | 14.212184 | -0.615932 | 0/3 | complete |
