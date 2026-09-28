# S3 主表：多站点模型 vs 最强单站点基线

> **解释边界：信息集不同：单站点基线仅使用中心站 PM2.5；ST/ST+频域额外使用邻站。相对提升只能归因于新增跨站信息的价值，不代表机制优于同信息集模型。**

`mean_reduction_percent > 0` 表示多站点模型 RMSE 更低；比较按种子配对，广州先在 8 个站内按种子池化。

| city | history | horizon | model_arm | best_single_station_arm | best_single_station_mean_rmse_ugm3 | model_mean_rmse_ugm3 | mean_reduction_percent | better_direction_count | pairing_status |
|---|---|---|---|---|---|---|---|---|---|
| beijing | 24 | 1 | st | center_lstm_matched | 20.830436 | 20.879179 | -0.236954 | 1/5 | complete |
| beijing | 24 | 1 | st_plus_frequency | center_lstm_matched | 20.830436 | 20.867889 | -0.182780 | 1/5 | complete |
| beijing | 24 | 3 | st | center_lstm_default | 31.219953 | 30.610879 | 1.949527 | 3/3 | complete |
| beijing | 24 | 3 | st_plus_frequency | center_lstm_default | 31.219953 | 30.553990 | 2.131722 | 3/3 | complete |
| beijing | 24 | 6 | st | center_lstm_default | 40.803276 | 39.913925 | 2.178857 | 3/3 | complete |
| beijing | 24 | 6 | st_plus_frequency | center_lstm_default | 40.803276 | 39.668437 | 2.780719 | 3/3 | complete |
| beijing | 24 | 12 | st | center_lstm_default | 50.935741 | 51.622506 | -1.359700 | 0/3 | complete |
| beijing | 24 | 12 | st_plus_frequency | center_lstm_default | 50.935741 | 51.201346 | -0.531957 | 1/3 | complete |
| beijing | 24 | 24 | st | center_lstm_default | 60.850169 | 64.831502 | -6.543008 | 0/3 | complete |
| beijing | 24 | 24 | st_plus_frequency | center_lstm_default | 60.850169 | 63.665273 | -4.626250 | 0/3 | complete |
| beijing | 48 | 1 | st | center_lstm_default | 21.110797 | 20.820853 | 1.374867 | 3/3 | complete |
| beijing | 48 | 1 | st_plus_frequency | center_lstm_default | 21.110797 | 20.781076 | 1.562962 | 3/3 | complete |
| beijing | 48 | 3 | st | center_lstm_default | 31.292734 | 31.138627 | 0.491426 | 2/3 | complete |
| beijing | 48 | 3 | st_plus_frequency | center_lstm_default | 31.292734 | 31.041485 | 0.801930 | 3/3 | complete |
| beijing | 48 | 6 | st | center_lstm_default | 41.607222 | 41.838913 | -0.564691 | 1/3 | complete |
| beijing | 48 | 6 | st_plus_frequency | center_lstm_default | 41.607222 | 41.475622 | 0.308549 | 1/3 | complete |
| beijing | 48 | 12 | st | center_lstm_default | 53.805130 | 55.104867 | -2.412790 | 1/3 | complete |
| beijing | 48 | 12 | st_plus_frequency | center_lstm_default | 53.805130 | 54.425363 | -1.150690 | 1/3 | complete |
| beijing | 48 | 24 | st | center_lstm_default | 64.348930 | 69.501508 | -8.062971 | 0/3 | complete |
| beijing | 48 | 24 | st_plus_frequency | center_lstm_default | 64.348930 | 68.403910 | -6.360750 | 0/3 | complete |
| beijing | 72 | 1 | st | center_gru_default | 21.356263 | 21.105325 | 1.175266 | 3/3 | complete |
| beijing | 72 | 1 | st_plus_frequency | center_gru_default | 21.356263 | 21.056562 | 1.403784 | 3/3 | complete |
| beijing | 72 | 3 | st | center_gru_default | 31.304386 | 31.272501 | 0.102981 | 2/3 | complete |
| beijing | 72 | 3 | st_plus_frequency | center_gru_default | 31.304386 | 31.214760 | 0.287554 | 2/3 | complete |
| beijing | 72 | 6 | st | center_gru_default | 40.007339 | 41.393878 | -3.466188 | 0/3 | complete |
| beijing | 72 | 6 | st_plus_frequency | center_gru_default | 40.007339 | 41.097383 | -2.725066 | 0/3 | complete |
| beijing | 72 | 12 | st | center_lstm_default | 52.885333 | 55.277826 | -4.542250 | 0/3 | complete |
| beijing | 72 | 12 | st_plus_frequency | center_lstm_default | 52.885333 | 54.865368 | -3.764441 | 0/3 | complete |
| beijing | 72 | 24 | st | center_lstm_default | 61.628635 | 66.301552 | -7.644068 | 0/3 | complete |
| beijing | 72 | 24 | st_plus_frequency | center_lstm_default | 61.628635 | 65.880727 | -6.959666 | 0/3 | complete |
| beijing | 168 | 1 | st | center_gru_default | 21.469984 | 22.083808 | -2.855891 | 0/3 | complete |
| beijing | 168 | 1 | st_plus_frequency | center_gru_default | 21.469984 | 21.797305 | -1.523292 | 0/3 | complete |
| beijing | 168 | 3 | st | center_tcn_default | 58.020618 | 31.860429 | 43.669636 | 3/3 | complete |
| beijing | 168 | 3 | st_plus_frequency | center_tcn_default | 58.020618 | 31.727272 | 43.916658 | 3/3 | complete |
| beijing | 168 | 6 | st | center_gru_default | 40.195742 | 42.100684 | -4.738787 | 0/5 | complete |
| beijing | 168 | 6 | st_plus_frequency | center_gru_default | 40.195742 | 41.929550 | -4.313091 | 0/5 | complete |
| beijing | 168 | 12 | st | center_tcn_default | 64.798435 | 55.826781 | 13.750807 | 3/3 | complete |
| beijing | 168 | 12 | st_plus_frequency | center_tcn_default | 64.798435 | 55.742360 | 13.876987 | 3/3 | complete |
| beijing | 168 | 24 | st | trad_ridge | 60.610630 | 69.033765 | -13.897125 | 0/3 | complete |
| beijing | 168 | 24 | st_plus_frequency | trad_ridge | 60.610630 | 68.741023 | -13.414137 | 0/3 | complete |
| guangzhou | 24 | 1 | st | center_gru_matched | 8.653739 | 8.514909 | 1.600536 | 3/3 | complete |
| guangzhou | 24 | 1 | st_plus_frequency | center_gru_matched | 8.653739 | 8.467184 | 2.151890 | 3/3 | complete |
| guangzhou | 168 | 6 | st | center_gru_matched | 14.125269 | 14.288717 | -1.157512 | 0/3 | complete |
| guangzhou | 168 | 6 | st_plus_frequency | center_gru_matched | 14.125269 | 14.212184 | -0.615932 | 0/3 | complete |
