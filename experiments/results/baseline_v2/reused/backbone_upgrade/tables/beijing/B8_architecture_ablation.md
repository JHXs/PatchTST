| protocol | task | variant | seeds | rmse_mean | rmse_std | rmse_reduction_mean | rmse_reduction_std | improved_seeds | mae_reduction_mean | smape_delta_mean | disable_rmse_increase_mean | shuffle_rmse_increase_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| end-to-end | 24$\rightarrow$1 | centre-only gate | 3 | 21.6398 | 0.1731 | -0.5777 | 0.8980 | 1 | -2.9870 | 1.3355 | 0.3461 | 0.9821 |
| end-to-end | 24$\rightarrow$1 | confidence-gated + delta (forecast) | 3 | 20.9338 | 0.4112 | 2.7043 | 1.8950 | 3 | 0.0271 | 1.2686 | 2.7776 | 5.3427 |
| end-to-end | 24$\rightarrow$1 | pairwise gate | 3 | 21.4737 | 0.2546 | 0.1980 | 0.6409 | 2 | -1.2806 | 1.3166 | 0.5487 | 1.8843 |
| end-to-end | 24$\rightarrow$1 | pairwise + delta (input) | 3 | 21.4101 | 0.0759 | 0.4896 | 0.5869 | 3 | 0.3174 | 0.0531 | 0.1006 | 0.0149 |
| end-to-end | 24$\rightarrow$1 | pairwise + delta (forecast) | 3 | 20.9357 | 0.4455 | 2.6954 | 2.0577 | 3 | -0.0111 | 1.2984 | 2.8307 | 6.4451 |
| end-to-end | 24$\rightarrow$1 | sparse + delta (forecast) | 3 | 20.9368 | 0.4309 | 2.6926 | 1.8218 | 3 | -0.3011 | 1.2154 | 2.9459 | 5.4929 |
| end-to-end | 24$\rightarrow$1 | sparse + station-bias + delta | 3 | 20.9550 | 0.4901 | 2.6084 | 2.1025 | 3 | -0.3252 | 1.1386 | 3.0676 | 5.8679 |
| end-to-end | 24$\rightarrow$1 | station-bias + delta (forecast) | 3 | 20.9272 | 0.4624 | 2.7353 | 2.1173 | 3 | -0.0272 | 1.3187 | 2.9557 | 6.8664 |
| end-to-end | 168$\rightarrow$6 | centre-only gate | 3 | 43.5992 | 0.1940 | -2.3050 | 2.0607 | 0 | -5.5083 | 3.5480 | 3.6596 | 6.3293 |
| end-to-end | 168$\rightarrow$6 | confidence-gated + delta (forecast) | 3 | 42.3277 | 0.9373 | 0.7014 | 1.3765 | 2 | 0.4061 | -0.3207 | 0.3601 | 0.6870 |
| end-to-end | 168$\rightarrow$6 | pairwise gate | 3 | 44.1476 | 0.7963 | -3.5919 | 2.7540 | 0 | -6.7199 | 4.2244 | 4.5361 | 5.9783 |
| end-to-end | 168$\rightarrow$6 | pairwise + delta (input) | 3 | 42.5818 | 0.9513 | 0.1117 | 0.2223 | 2 | 0.0378 | 0.2290 | 1.7579 | 0.8283 |
| end-to-end | 168$\rightarrow$6 | pairwise + delta (forecast) | 3 | 42.1668 | 0.9005 | 1.0788 | 1.2347 | 2 | 0.7484 | -0.4061 | 0.4997 | 1.1116 |
| end-to-end | 168$\rightarrow$6 | sparse + delta (forecast) | 3 | 41.9626 | 0.7392 | 1.5567 | 0.6354 | 3 | 1.4007 | -0.5180 | 0.6718 | 0.8341 |
| end-to-end | 168$\rightarrow$6 | sparse + station-bias + delta | 3 | 42.0027 | 0.7590 | 1.4627 | 0.7653 | 3 | 1.2827 | -0.4917 | 0.6205 | 0.7823 |
| end-to-end | 168$\rightarrow$6 | station-bias + delta (forecast) | 3 | 42.1740 | 0.8998 | 1.0618 | 1.2529 | 2 | 0.7406 | -0.3995 | 0.4989 | 1.1068 |
| frozen + degraded init | 24$\rightarrow$1 | centre-only gate | 3 | 21.5107 | 0.1190 | 0.0235 | 0.0403 | 2 | -0.2958 | 0.2148 | 0.0413 | 0.0026 |
| frozen + degraded init | 24$\rightarrow$1 | confidence-gated + delta (forecast) | 3 | 20.9320 | 0.1000 | 2.7132 | 0.1237 | 3 | 0.6236 | 0.5739 | 2.8144 | 7.6570 |
| frozen + degraded init | 24$\rightarrow$1 | pairwise gate | 3 | 21.5244 | 0.1088 | -0.0401 | 0.1211 | 1 | -0.5224 | 0.7563 | -0.0344 | 0.3964 |
| frozen + degraded init | 24$\rightarrow$1 | pairwise + delta (input) | 3 | 20.9904 | 0.4712 | 2.4401 | 2.2410 | 2 | -1.3770 | 0.4694 | 2.7807 | 22.0727 |
| frozen + degraded init | 24$\rightarrow$1 | pairwise + delta (forecast) | 3 | 20.9149 | 0.0954 | 2.7926 | 0.1262 | 3 | 0.7023 | 0.5709 | 2.8984 | 8.7835 |
| frozen + degraded init | 24$\rightarrow$1 | sparse + delta (forecast) | 3 | 20.9361 | 0.1016 | 2.6931 | 0.4566 | 3 | 0.4812 | 0.6940 | 2.7843 | 7.2840 |
| frozen + degraded init | 24$\rightarrow$1 | sparse + station-bias + delta | 3 | 20.8109 | 0.1136 | 3.2755 | 0.3706 | 3 | 0.5659 | 0.8210 | 3.4360 | 9.6033 |
| frozen + degraded init | 24$\rightarrow$1 | station-bias + delta (forecast) | 3 | 20.8404 | 0.0945 | 3.1383 | 0.3019 | 3 | 0.4524 | 0.8686 | 3.2419 | 10.5389 |
| frozen + degraded init | 168$\rightarrow$6 | centre-only gate | 3 | 42.7384 | 0.8505 | -0.2592 | 0.2182 | 0 | 0.0873 | 1.0049 | -0.1128 | -0.0171 |
| frozen + degraded init | 168$\rightarrow$6 | confidence-gated + delta (forecast) | 3 | 42.4483 | 0.8129 | 0.4204 | 0.1802 | 3 | 0.3960 | -0.3784 | 0.5128 | 0.9721 |
| frozen + degraded init | 168$\rightarrow$6 | pairwise gate | 3 | 42.7946 | 0.9105 | -0.3893 | 0.3010 | 0 | -0.0304 | 1.7127 | -0.1836 | 0.0853 |
| frozen + degraded init | 168$\rightarrow$6 | pairwise + delta (input) | 3 | 41.6630 | 0.4637 | 2.2518 | 0.9470 | 3 | 1.9622 | 0.2057 | 3.1020 | 7.0893 |
| frozen + degraded init | 168$\rightarrow$6 | pairwise + delta (forecast) | 3 | 42.4327 | 0.7818 | 0.4561 | 0.2601 | 3 | 0.4080 | -0.3910 | 0.5491 | 1.3337 |
| frozen + degraded init | 168$\rightarrow$6 | sparse + delta (forecast) | 3 | 42.4275 | 0.8123 | 0.4692 | 0.1800 | 3 | 0.3003 | -0.2795 | 0.5585 | 1.2430 |
| frozen + degraded init | 168$\rightarrow$6 | sparse + station-bias + delta | 3 | 42.3438 | 0.8264 | 0.6661 | 0.1935 | 3 | 0.5965 | -0.4463 | 0.7607 | 2.1231 |
| frozen + degraded init | 168$\rightarrow$6 | station-bias + delta (forecast) | 3 | 42.4231 | 0.7827 | 0.4785 | 0.2592 | 3 | 0.4393 | -0.4034 | 0.5717 | 1.3602 |
