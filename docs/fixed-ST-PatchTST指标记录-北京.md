# 训练指标记录——北京数据集

# ST-PatchTST

## 24*1h pre 1h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.168085  0.409982  0.207124
test   0.169758  0.412017  0.194113

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.122499 0.349998 0.210463
valid PM10_Concentration 0.004461 0.066790 0.030134
valid  NO2_Concentration 0.113197 0.336447 0.226816
valid   CO_Concentration 0.207278 0.455278 0.279369
valid   O3_Concentration 0.023338 0.152769 0.092913
valid  SO2_Concentration 0.167741 0.409562 0.272296
valid            weather 0.347016 0.589081 0.209736
valid        temperature 0.014918 0.122140 0.081907
valid           pressure 0.000198 0.014059 0.007958
valid           humidity 0.095119 0.308414 0.166814
valid         wind_speed 0.563985 0.750989 0.533640
valid     wind_direction 0.357274 0.597724 0.373438

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.092380 0.303941 0.160054
 test PM10_Concentration 0.013515 0.116256 0.055321
 test  NO2_Concentration 0.086573 0.294233 0.193224
 test   CO_Concentration 0.063979 0.252940 0.150630
 test   O3_Concentration 0.036297 0.190518 0.128912
 test  SO2_Concentration 0.083897 0.289650 0.162328
 test            weather 0.364591 0.603814 0.221562
 test        temperature 0.018715 0.136804 0.091028
 test           pressure 0.000220 0.014825 0.008646
 test           humidity 0.128804 0.358893 0.198975
 test         wind_speed 0.748541 0.865182 0.586702
 test     wind_direction 0.399580 0.632124 0.371971
```

## 120h pre 4h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.520671  0.721575  0.451515
test   0.454898  0.674461  0.427490

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.794602 0.891405 0.619066
valid PM10_Concentration 0.018505 0.136034 0.086165
valid  NO2_Concentration 0.594979 0.771349 0.573613
valid   CO_Concentration 0.956708 0.978114 0.696834
valid   O3_Concentration 0.095660 0.309289 0.231036
valid  SO2_Concentration 0.930020 0.964376 0.708476
valid            weather 0.951960 0.975684 0.599972
valid        temperature 0.047936 0.218943 0.171791
valid           pressure 0.000739 0.027182 0.019715
valid           humidity 0.333429 0.577433 0.410020
valid         wind_speed 1.117138 1.056947 0.764934
valid     wind_direction 0.406372 0.637473 0.536553

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.520543 0.721487 0.488097
 test PM10_Concentration 0.077183 0.277817 0.170178
 test  NO2_Concentration 0.443916 0.666270 0.499920
 test   CO_Concentration 0.329124 0.573693 0.402951
 test   O3_Concentration 0.211438 0.459823 0.345718
 test  SO2_Concentration 0.486383 0.697412 0.462619
 test            weather 1.007102 1.003545 0.605378
 test        temperature 0.069684 0.263977 0.197143
 test           pressure 0.000982 0.031341 0.022746
 test           humidity 0.419159 0.647425 0.472866
 test         wind_speed 1.401042 1.183656 0.886795
 test     wind_direction 0.492219 0.701583 0.575472
```

## 168h pre 8h patch_len=4
```
总体评估指标:
            mse      rmse       mae
split
valid  0.429775  0.655573  0.397641
test   0.385067  0.620537  0.379264

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.599872 0.774514 0.516761
valid PM10_Concentration 0.014346 0.119776 0.072449
valid  NO2_Concentration 0.477245 0.690829 0.498291
valid   CO_Concentration 0.770439 0.877746 0.602484
valid   O3_Concentration 0.079913 0.282688 0.205574
valid  SO2_Concentration 0.745592 0.863477 0.617153
valid            weather 0.781671 0.884122 0.509885
valid        temperature 0.041034 0.202569 0.155490
valid           pressure 0.000522 0.022841 0.016189
valid           humidity 0.265928 0.515682 0.352756
valid         wind_speed 0.986135 0.993043 0.712481
valid     wind_direction 0.394609 0.628179 0.512176

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.395654 0.629011 0.409291
 test PM10_Concentration 0.061109 0.247202 0.143947
 test  NO2_Concentration 0.365230 0.604342 0.439078
 test   CO_Concentration 0.260869 0.510754 0.348369
 test   O3_Concentration 0.170719 0.413182 0.304593
 test  SO2_Concentration 0.387127 0.622196 0.399745
 test            weather 0.849455 0.921659 0.525118
 test        temperature 0.054568 0.233597 0.172793
 test           pressure 0.000671 0.025912 0.018431
 test           humidity 0.341315 0.584222 0.409258
 test         wind_speed 1.258436 1.121801 0.830025
 test     wind_direction 0.475643 0.689669 0.550521
```


## 24*7h pre 24h patch_len=4

- alpha_max = 0.65

``` 
总体评估指标:
            mse      rmse       mae
split
valid  0.718140  0.847431  0.553452
test   0.577318  0.759814  0.503408

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.751006 0.866606 0.622441
 test PM10_Concentration 0.096553 0.310729 0.204042
 test  NO2_Concentration 0.561781 0.749520 0.579737
 test   CO_Concentration 0.449743 0.670628 0.489838
 test   O3_Concentration 0.253846 0.503831 0.385200
 test  SO2_Concentration 0.705858 0.840154 0.591289 
 test            weather 1.357729 1.165216 0.778445
 test        temperature 0.095670 0.309306 0.237096
 test           pressure 0.001927 0.043897 0.032758
 test           humidity 0.540413 0.735128 0.559101
 test         wind_speed 1.597111 1.263768 0.957896
 test     wind_direction 0.516177 0.718455 0.603051
```

- alpha_max = 0.5
```
总体评估指标:
            mse      rmse       mae
split
valid  0.711954  0.843774  0.549043
test   0.576418  0.759222  0.501759

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 1.213946 1.101792 0.809702
valid PM10_Concentration 0.026919 0.164069 0.111043
valid  NO2_Concentration 0.827018 0.909405 0.706205
valid   CO_Concentration 1.368105 1.169660 0.872496
valid   O3_Concentration 0.121577 0.348679 0.270752
valid  SO2_Concentration 1.363553 1.167713 0.892885
valid            weather 1.368799 1.169957 0.800588
valid        temperature 0.059917 0.244780 0.195029
valid           pressure 0.001330 0.036463 0.027156
valid           humidity 0.457911 0.676691 0.501096
valid         wind_speed 1.300270 1.140294 0.839278
valid     wind_direction 0.434109 0.658870 0.562280

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.748795 0.865330 0.620158
 test PM10_Concentration 0.097923 0.312927 0.204974
 test  NO2_Concentration 0.562773 0.750182 0.579575
 test   CO_Concentration 0.450078 0.670878 0.487569
 test   O3_Concentration 0.258306 0.508238 0.386894
 test  SO2_Concentration 0.707341 0.841036 0.591958
 test            weather 1.333236 1.154659 0.760710
 test        temperature 0.097415 0.312114 0.238697
 test           pressure 0.001941 0.044054 0.032739
 test           humidity 0.539569 0.734554 0.556316
 test         wind_speed 1.596628 1.263577 0.956413
 test     wind_direction 0.523010 0.723194 0.605101
```

## 24*14h pre 48h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.941412  0.970264  0.653305
test   0.707456  0.841104  0.576341

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 1.816080 1.347620 1.037730
valid PM10_Concentration 0.037731 0.194246 0.144110
valid  NO2_Concentration 1.076141 1.037372 0.844766
valid   CO_Concentration 1.906194 1.380650 1.087807
valid   O3_Concentration 0.121562 0.348657 0.278178
valid  SO2_Concentration 1.844506 1.358126 1.085882
valid            weather 2.041705 1.428882 1.058847
valid        temperature 0.073846 0.271747 0.216167
valid           pressure 0.002400 0.048994 0.037412
valid           humidity 0.547758 0.740106 0.575694
valid         wind_speed 1.368257 1.169725 0.875599
valid     wind_direction 0.460767 0.678798 0.597467

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.836274 0.914480 0.676462
 test PM10_Concentration 0.115799 0.340292 0.230487
 test  NO2_Concentration 0.672230 0.819896 0.650117
 test   CO_Concentration 0.580458 0.761878 0.573770
 test   O3_Concentration 0.276357 0.525696 0.403455
 test  SO2_Concentration 0.967986 0.983863 0.754199
 test            weather 1.836099 1.355027 0.971737
 test        temperature 0.161637 0.402041 0.317030
 test           pressure 0.002788 0.052800 0.042187
 test           humidity 0.701322 0.837450 0.651449
 test         wind_speed 1.824658 1.350799 1.030879
 test     wind_direction 0.513869 0.716847 0.614320
```

# PatchTST

## 24*1h pre 1h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.169118  0.411239  0.209130
test   0.169559  0.411776  0.195286

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.124035 0.352185 0.212327
valid PM10_Concentration 0.004513 0.067179 0.030360
valid  NO2_Concentration 0.113675 0.337157 0.227085
valid   CO_Concentration 0.206787 0.454739 0.279411
valid   O3_Concentration 0.023446 0.153122 0.093199
valid  SO2_Concentration 0.168579 0.410583 0.273754
valid            weather 0.360508 0.600423 0.222071
valid        temperature 0.015184 0.123222 0.082806
valid           pressure 0.000201 0.014188 0.008063
valid           humidity 0.095045 0.308294 0.165660
valid         wind_speed 0.567255 0.753163 0.535996
valid     wind_direction 0.350182 0.591762 0.378832

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.091580 0.302622 0.159466
 test PM10_Concentration 0.013755 0.117281 0.055968
 test  NO2_Concentration 0.087168 0.295242 0.193654
 test   CO_Concentration 0.063515 0.252022 0.150527
 test   O3_Concentration 0.037047 0.192475 0.130563
 test  SO2_Concentration 0.083981 0.289795 0.162569
 test            weather 0.371348 0.609384 0.230252
 test        temperature 0.018630 0.136491 0.090874
 test           pressure 0.000224 0.014961 0.008722
 test           humidity 0.129010 0.359180 0.199748
 test         wind_speed 0.745159 0.863226 0.583489
 test     wind_direction 0.393294 0.627132 0.377598
```

## 120h pre 4h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.303888  0.551260  0.316634
test   0.282006  0.531043  0.301436

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.363453 0.602871 0.377318
valid PM10_Concentration 0.009688 0.098429 0.053707
valid  NO2_Concentration 0.306587 0.553703 0.382116
valid   CO_Concentration 0.529276 0.727514 0.471313
valid   O3_Concentration 0.055649 0.235900 0.159696
valid  SO2_Concentration 0.452536 0.672708 0.452354
valid            weather 0.566982 0.752982 0.385660
valid        temperature 0.028375 0.168448 0.124510
valid           pressure 0.000306 0.017484 0.011752
valid           humidity 0.169059 0.411167 0.262974
valid         wind_speed 0.800114 0.894491 0.638838
valid     wind_direction 0.364629 0.603845 0.479370

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.230057 0.479642 0.289514
 test PM10_Concentration 0.036613 0.191346 0.101980
 test  NO2_Concentration 0.241314 0.491237 0.337304
 test   CO_Concentration 0.163315 0.404122 0.261200
 test   O3_Concentration 0.102698 0.320465 0.225992
 test  SO2_Concentration 0.238202 0.488059 0.296379
 test            weather 0.611273 0.781840 0.396138
 test        temperature 0.034223 0.184994 0.133997
 test           pressure 0.000382 0.019547 0.013057
 test           humidity 0.233647 0.483371 0.314991
 test         wind_speed 1.051959 1.025651 0.740599
 test     wind_direction 0.440393 0.663621 0.506077
```

## 24*7h pre 8h patch_len=4
```
总体评估指标:
            mse      rmse       mae
split
valid  0.428489  0.654591  0.396912
test   0.383935  0.619625  0.376714

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.596409 0.772275 0.515353
valid PM10_Concentration 0.014338 0.119741 0.072311
valid  NO2_Concentration 0.476914 0.690590 0.497877
valid   CO_Concentration 0.777188 0.881583 0.607355
valid   O3_Concentration 0.080318 0.283405 0.204831
valid  SO2_Concentration 0.739619 0.860011 0.615230
valid            weather 0.770937 0.878030 0.501764
valid        temperature 0.040561 0.201399 0.153875
valid           pressure 0.000492 0.022174 0.015668
valid           humidity 0.263573 0.513393 0.349789
valid         wind_speed 0.986204 0.993078 0.714688
valid     wind_direction 0.395320 0.628745 0.514207

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.395365 0.628780 0.409168
 test PM10_Concentration 0.060625 0.246222 0.142589
 test  NO2_Concentration 0.365402 0.604485 0.438349
 test   CO_Concentration 0.260731 0.510618 0.347989
 test   O3_Concentration 0.170511 0.412929 0.303054
 test  SO2_Concentration 0.386680 0.621836 0.398459
 test            weather 0.840529 0.916804 0.520230
 test        temperature 0.053433 0.231156 0.170131
 test           pressure 0.000636 0.025222 0.017885
 test           humidity 0.338347 0.581676 0.404956
 test         wind_speed 1.270455 1.127145 0.830514
 test     wind_direction 0.464502 0.681544 0.537249
```

## 24*7h pre 12h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.520330  0.721339  0.451874
test   0.456223  0.675443  0.428471

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.799722 0.894272 0.622731
valid PM10_Concentration 0.018540 0.136161 0.086833
valid  NO2_Concentration 0.602628 0.776291 0.576833
valid   CO_Concentration 0.958225 0.978890 0.697929
valid   O3_Concentration 0.096034 0.309894 0.230524
valid  SO2_Concentration 0.929318 0.964012 0.708771
valid            weather 0.944620 0.971916 0.604274
valid        temperature 0.048358 0.219905 0.172516
valid           pressure 0.000759 0.027541 0.020062
valid           humidity 0.333577 0.577562 0.410596
valid         wind_speed 1.105525 1.051440 0.759546
valid     wind_direction 0.406649 0.637690 0.531874

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.515357 0.717884 0.487662
 test PM10_Concentration 0.075989 0.275661 0.168372
 test  NO2_Concentration 0.445603 0.667535 0.499010
 test   CO_Concentration 0.330115 0.574556 0.404520
 test   O3_Concentration 0.214377 0.463009 0.347500
 test  SO2_Concentration 0.489443 0.699602 0.464801
 test            weather 1.007915 1.003950 0.610095
 test        temperature 0.070329 0.265197 0.198052
 test           pressure 0.000989 0.031451 0.022904
 test           humidity 0.419693 0.647837 0.472549
 test         wind_speed 1.408117 1.186641 0.887412
 test     wind_direction 0.496747 0.704803 0.578773
```

## 24*7h pre 24h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.713195  0.844509  0.551073
test   0.577591  0.759994  0.504562

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 1.245648 1.116086 0.821307
valid PM10_Concentration 0.027559 0.166010 0.113063
valid  NO2_Concentration 0.825982 0.908836 0.706826
valid   CO_Concentration 1.363342 1.167622 0.867750
valid   O3_Concentration 0.122454 0.349935 0.272690
valid  SO2_Concentration 1.353357 1.163339 0.884453
valid            weather 1.378478 1.174086 0.817623
valid        temperature 0.060868 0.246715 0.196413
valid           pressure 0.001397 0.037382 0.027825
valid           humidity 0.468065 0.684153 0.508759
valid         wind_speed 1.278613 1.130758 0.827550
valid     wind_direction 0.432579 0.657707 0.568621

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.751732 0.867025 0.623338
 test PM10_Concentration 0.096389 0.310467 0.204511
 test  NO2_Concentration 0.561451 0.749301 0.579645
 test   CO_Concentration 0.447590 0.669021 0.487944
 test   O3_Concentration 0.257326 0.507273 0.387900
 test  SO2_Concentration 0.707153 0.840924 0.591154
 test            weather 1.343058 1.158904 0.766901
 test        temperature 0.098675 0.314126 0.241621
 test           pressure 0.002003 0.044756 0.033517
 test           humidity 0.543274 0.737071 0.560948
 test         wind_speed 1.593368 1.262287 0.956007
 test     wind_direction 0.529071 0.727373 0.621265
```

## 24*14h pre 48h patch_len=4
```
总体评估指标:
            mse      rmse       mae
split
valid  0.928024  0.963340  0.650000
test   0.714203  0.845105  0.580293

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 1.795113 1.339818 1.025174
valid PM10_Concentration 0.036686 0.191536 0.141594
valid  NO2_Concentration 1.056926 1.028069 0.837920
valid   CO_Concentration 1.900524 1.378595 1.081933
valid   O3_Concentration 0.121721 0.348886 0.279473
valid  SO2_Concentration 1.835950 1.354972 1.082025
valid            weather 1.953036 1.397511 1.052055
valid        temperature 0.073847 0.271747 0.215498
valid           pressure 0.002308 0.048045 0.037017
valid           humidity 0.537411 0.733083 0.572367
valid         wind_speed 1.367321 1.169325 0.877844
valid     wind_direction 0.455447 0.674868 0.597098

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.848402 0.921088 0.683986
 test PM10_Concentration 0.111702 0.334219 0.228127
 test  NO2_Concentration 0.686919 0.828806 0.658272
 test   CO_Concentration 0.602215 0.776025 0.581094
 test   O3_Concentration 0.274177 0.523619 0.403583
 test  SO2_Concentration 0.979300 0.989596 0.754910
 test            weather 1.838033 1.355741 0.975884
 test        temperature 0.164282 0.405317 0.320121
 test           pressure 0.002792 0.052842 0.042105
 test           humidity 0.712329 0.843996 0.657808
 test         wind_speed 1.827610 1.351891 1.032733
 test     wind_direction 0.522677 0.722964 0.624893
```
