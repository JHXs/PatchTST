| split | block | task | variant | k | seeds | rmse_mean | rmse_std | rmse_reduction_mean | rmse_reduction_std | improved_seeds | mae_reduction_mean | smape_delta_mean | disable_rmse_increase_mean | shuffle_rmse_increase_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | end_to_end_dev | 24$\rightarrow$1 | centre-only gate |  | 3 | 21.4451 | 0.1693 | -0.3280 | 0.7431 | 1 | -2.7443 | 1.1476 | nan | nan |
| test | end_to_end_dev | 24$\rightarrow$1 | pairwise gate |  | 3 | 21.4484 | 0.0667 | -0.3434 | 0.0740 | 0 | -4.7801 | 2.5336 | nan | nan |
| test | end_to_end_dev | 24$\rightarrow$1 | pairwise + delta (input) |  | 3 | 21.3348 | 0.0863 | 0.1884 | 0.0936 | 3 | -0.0032 | 0.1221 | nan | nan |
| test | end_to_end_dev | 24$\rightarrow$1 | pairwise + delta (forecast) | 7 | 3 | 20.9357 | 0.4455 | 2.6954 | 2.0577 | 3 | -0.0111 | 1.2984 | 2.8307 | 6.4451 |
| test | end_to_end_dev | 24$\rightarrow$1 | sparse + delta (forecast) | 7 | 3 | 21.0170 | 0.5220 | 2.3199 | 2.2700 | 2 | -1.4932 | 1.4927 | 2.5223 | 5.2985 |
| test | end_to_end_dev | 24$\rightarrow$1 | station-bias + delta (forecast) |  | 3 | 20.9272 | 0.4624 | 2.7353 | 2.1173 | 3 | -0.0272 | 1.3187 | 2.9557 | 6.8664 |
| test | end_to_end_dev | 168$\rightarrow$6 | pairwise + delta (input) |  | 3 | 41.9969 | 0.7217 | 0.4558 | 1.1475 | 2 | 0.5710 | -0.1715 | nan | nan |
| test | end_to_end_dev | 168$\rightarrow$6 | pairwise + delta (forecast) | 7 | 3 | 42.1668 | 0.9005 | 1.0788 | 1.2347 | 2 | 0.7484 | -0.4061 | 0.4997 | 1.1116 |
| test | end_to_end_dev | 168$\rightarrow$6 | sparse + delta (forecast) | 7 | 3 | 42.1110 | 0.8415 | 1.2089 | 1.1125 | 2 | 0.9912 | -0.4533 | 0.5867 | 0.7872 |
| test | end_to_end_dev | 168$\rightarrow$6 | station-bias + delta (forecast) |  | 3 | 42.1740 | 0.8998 | 1.0618 | 1.2529 | 2 | 0.7406 | -0.3995 | 0.4989 | 1.1068 |
| test | frozen_production | 24$\rightarrow$1 | sparse + station-bias + delta | 5 | 5 | 20.6250 | 0.1985 | 3.1188 | 0.2409 | 5 | 1.1838 | 0.5249 | 3.2371 | 9.4129 |
| test | frozen_production | 168$\rightarrow$6 | sparse + station-bias + delta | 5 | 5 | 41.9195 | 0.2220 | 0.8220 | 0.1991 | 5 | 1.1744 | -0.3210 | 0.8434 | 2.3034 |
