# 指标记录-广州数据集

# ST-PatchTST

``` ST-PatchTST
验证集评估...
验证集预测形状: (852, 12, 24)                                                                                                             

测试集评估...
测试集预测形状: (1705, 12, 24)                                                                                                            

评估结果:
            mse       mae
valid  0.932143  0.641901
test   0.498816  0.483552
results_df:             mse       mae
valid  0.932143  0.641901
test   0.498816  0.483552
```

- alpha_max=0.05
```alpha_max=0.05
评估结果:
            mse       mae
valid  0.920702  0.639035
test   0.514104   0.49062
results_df:             mse       mae
valid  0.920702  0.639035
test   0.514104   0.49062
```

- alpha_max=0.1
```alpha_max=0.1
评估结果:
            mse       mae
valid  0.915271  0.639124
test   0.498283   0.48382
results_df:             mse       mae
valid  0.915271  0.639124
test   0.498283   0.48382
```

- alpha_max=0.15
```alpha_max=0.15
评估结果:
            mse       mae
valid  0.920619   0.64063
test   0.495451  0.480426
results_df:             mse       mae
valid  0.920619   0.64063
test   0.495451  0.480426
```

- alpha_max=0.2
```0.2
评估结果:
            mse       mae
valid  0.903114  0.638271
test   0.495508  0.483788
results_df:             mse       mae
valid  0.903114  0.638271
test   0.495508  0.483788
```

- alpha_max=0.25
```0.25
评估结果:
            mse       mae
valid  0.909522  0.636546
test   0.494325  0.481512
results_df:             mse       mae
valid  0.909522  0.636546
test   0.494325  0.481512
```

- alpha_max=0.3
```0.3
评估结果:
            mse       mae
valid  0.924146  0.640857
test   0.499157  0.483304
results_df:             mse       mae
valid  0.924146  0.640857
test   0.499157  0.483304
```

- alpha_max=0.35
```0.35
评估结果:
            mse       mae
valid  0.916332  0.637274
test     0.4951  0.480465
results_df:             mse       mae
valid  0.916332  0.637274
test     0.4951  0.480465
```

- alpha_max=0.4
```0.4
评估结果:
            mse       mae
valid  0.913749   0.63867
test   0.493991  0.481313
results_df:             mse       mae
valid  0.913749   0.63867
test   0.493991  0.481313
```

- alpha_max=0.45
```0.45
评估结果:
            mse       mae
valid  0.926632  0.643501
test   0.497149   0.48268
results_df:             mse       mae
valid  0.926632  0.643501
test   0.497149   0.48268
```

- alpha_max=0.5
```0.5
评估结果:
            mse       mae
valid  0.912644  0.636734
test   0.492882  0.479971
results_df:             mse       mae
valid  0.912644  0.636734
test   0.492882  0.479971
```

- alpha_max=0.55
```0.55
评估结果:
            mse       mae
valid  0.908302  0.632966
test   0.493156  0.480401
results_df:             mse       mae
valid  0.908302  0.632966
test   0.493156  0.480401
```

- alpha_max=0.6
```0.6
评估结果:
            mse       mae
valid  0.935142  0.644051
test    0.50123   0.48473
results_df:             mse       mae
valid  0.935142  0.644051
test    0.50123   0.48473
```

- alpha_max=0.65
```0.65
评估结果:
            mse       mae
valid  0.902753   0.63362
test   0.491026  0.478893
results_df:             mse       mae
valid  0.902753   0.63362
test   0.491026  0.478893
```

- alpha_max=0.7
```
评估结果:
            mse       mae
valid  0.937002  0.647065
test   0.498935  0.485294
results_df:             mse       mae
valid  0.937002  0.647065
test   0.498935  0.485294
```

- alpha_max=0.75
```
评估结果:
            mse       mae
valid  0.918926  0.639287
test   0.495311  0.482106
results_df:             mse       mae
valid  0.918926  0.639287
test   0.495311  0.482106
```

- alpha_max=0.8
```
评估结果:
            mse       mae
valid  0.907311  0.636054
test   0.494767  0.480595
results_df:             mse       mae
valid  0.907311  0.636054
test   0.494767  0.480595
```

- alpha_max=0.85
```
评估结果:
            mse       mae
valid  0.914542  0.637368
test   0.491165  0.479045
results_df:             mse       mae
valid  0.914542  0.637368
test   0.491165  0.479045
```

- alpha_max=0.9
```
评估结果:
            mse       mae
valid  0.913314  0.639371
test   0.493793  0.482693
results_df:             mse       mae
valid  0.913314  0.639371
test   0.493793  0.482693
```

- alpha_max=0.95
```
评估结果:
            mse       mae
valid  0.922468  0.640413
test   0.500305  0.485335
results_df:             mse       mae
valid  0.922468  0.640413
test   0.500305  0.485335
```

- alpha_max=1
```
评估结果:
            mse       mae
valid  0.922602  0.638745
test   0.495423   0.48142
results_df:             mse       mae
valid  0.922602  0.638745
test   0.495423   0.48142
```

> 下面都以 alpha_max=0.65 进行测试

## 24*14h pre 48h patch_len=4
```
总体评估指标:
            mse      rmse       mae
split
valid  1.203171  1.096891  0.765699
test   0.643296  0.802057  0.570819

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 1.933321 1.390439 1.053745
valid PM10_Concentration 1.850369 1.360283 0.996041
valid  NO2_Concentration 1.793651 1.339273 1.029486
valid   CO_Concentration 1.068053 1.033466 0.753249
valid   O3_Concentration 0.297886 0.545790 0.412195
valid  SO2_Concentration 1.110226 1.053673 0.790433
valid            weather 3.262631 1.806275 1.201007
valid        temperature 0.253002 0.502993 0.409838
valid           pressure 0.126943 0.356291 0.277853
valid           humidity 0.623655 0.789718 0.625824
valid         wind_speed 1.023181 1.011524 0.754016
valid     wind_direction 1.095131 1.046485 0.884697

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 1.049454 1.024429 0.743346
 test PM10_Concentration 0.976330 0.988094 0.712988
 test  NO2_Concentration 0.628425 0.792733 0.578644
 test   CO_Concentration 0.795909 0.892137 0.604157
 test   O3_Concentration 0.243816 0.493777 0.321333
 test  SO2_Concentration 0.484903 0.696350 0.498293
 test            weather 0.534462 0.731069 0.500311
 test        temperature 0.314655 0.560941 0.439511
 test           pressure 0.181562 0.426101 0.335940
 test           humidity 0.347307 0.589328 0.433086
 test         wind_speed 1.306837 1.143170 0.888011
 test     wind_direction 0.855888 0.925142 0.794210
```

## 24*7h pre 24h patch_len=4
24*30h？？？

```
总体评估指标:
            mse      rmse       mae
split                              
valid  0.943550  0.971365  0.671632
test   0.582291  0.763080  0.533270

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 1.570306 1.253119 0.946215
valid PM10_Concentration 1.524055 1.234526 0.911529
valid  NO2_Concentration 1.284870 1.133521 0.856747
valid   CO_Concentration 0.760759 0.872215 0.645797
valid   O3_Concentration 0.265072 0.514852 0.392904
valid  SO2_Concentration 0.985842 0.992896 0.728178
valid            weather 2.114744 1.454216 0.894419
valid        temperature 0.177597 0.421423 0.338510
valid           pressure 0.082321 0.286916 0.216505
valid           humidity 0.473285 0.687957 0.520973
valid         wind_speed 1.006012 1.003002 0.750640
valid     wind_direction 1.077733 1.038139 0.857162

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.938926 0.968982 0.710018
 test PM10_Concentration 0.846173 0.919877 0.662873
 test  NO2_Concentration 0.581267 0.762408 0.556937
 test   CO_Concentration 0.762500 0.873213 0.593058
 test   O3_Concentration 0.215861 0.464609 0.297689
 test  SO2_Concentration 0.462721 0.680236 0.469498
 test            weather 0.498362 0.705948 0.476974
 test        temperature 0.216241 0.465017 0.358323
 test           pressure 0.106535 0.326396 0.254135
 test           humidity 0.287048 0.535769 0.393264
 test         wind_speed 1.197702 1.094396 0.842510
 test     wind_direction 0.874159 0.934965 0.783963

返回结果包含: ['summary', 'per_metric']
```

## 168h pre 12h patch_len=4
```
总体评估指标:
            mse      rmse       mae
split                              
valid  0.702973  0.838435  0.552420
test   0.419097  0.647377  0.431265

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.896767 0.946978 0.690456
valid PM10_Concentration 0.898461 0.947872 0.671539
valid  NO2_Concentration 1.111367 1.054214 0.769781
valid   CO_Concentration 0.531464 0.729016 0.491212
valid   O3_Concentration 0.234550 0.484304 0.352481
valid  SO2_Concentration 0.832255 0.912280 0.639402
valid            weather 1.386116 1.177334 0.547569
valid        temperature 0.127888 0.357615 0.281074
valid           pressure 0.040918 0.202282 0.148443
valid           humidity 0.407131 0.638068 0.487133
valid         wind_speed 0.936945 0.967959 0.726159
valid     wind_direction 1.031806 1.015778 0.823790

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.514002 0.716939 0.503789
 test PM10_Concentration 0.499693 0.706890 0.478056
 test  NO2_Concentration 0.475449 0.689528 0.478225
 test   CO_Concentration 0.370652 0.608812 0.383855
 test   O3_Concentration 0.190444 0.436399 0.271158
 test  SO2_Concentration 0.341919 0.584738 0.391489
 test            weather 0.250706 0.500706 0.302617
 test        temperature 0.131252 0.362287 0.263127
 test           pressure 0.061634 0.248261 0.190662
 test           humidity 0.239020 0.488896 0.347486
 test         wind_speed 1.093147 1.045537 0.797974
 test     wind_direction 0.861250 0.928036 0.766737
```

## 24*7h pre 8h patch_len=4
```
总体评估指标:
            mse      rmse       mae
split
valid  0.593444  0.770353  0.496454
test   0.364571  0.603798  0.392971

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.664567 0.815210 0.591596
valid PM10_Concentration 0.702726 0.838288 0.587783
valid  NO2_Concentration 0.896433 0.946802 0.667818
valid   CO_Concentration 0.441168 0.664205 0.435096
valid   O3_Concentration 0.197294 0.444178 0.318324
valid  SO2_Concentration 0.692063 0.831903 0.556848
valid            weather 1.204554 1.097522 0.485905
valid        temperature 0.100691 0.317318 0.242925
valid           pressure 0.029079 0.170525 0.123632
valid           humidity 0.317388 0.563372 0.428937
valid         wind_speed 0.869240 0.932331 0.696383
valid     wind_direction 1.006125 1.003058 0.822200

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.434625 0.659261 0.462489
 test PM10_Concentration 0.418680 0.647055 0.429702
 test  NO2_Concentration 0.395624 0.628987 0.423469
 test   CO_Concentration 0.287405 0.536101 0.329038
 test   O3_Concentration 0.165250 0.406509 0.245229
 test  SO2_Concentration 0.290972 0.539418 0.353228
 test            weather 0.215553 0.464277 0.272356
 test        temperature 0.094416 0.307272 0.220238
 test           pressure 0.040103 0.200257 0.151780
 test           humidity 0.188274 0.433905 0.302853
 test         wind_speed 1.013652 1.006803 0.762601
 test     wind_direction 0.830305 0.911211 0.762668
```

## 168h pre 12h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.518437  0.720026  0.451930
test   0.455183  0.674672  0.428212

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.791957 0.889920 0.615692
valid PM10_Concentration 0.018462 0.135876 0.086211
valid  NO2_Concentration 0.593420 0.770337 0.572044
valid   CO_Concentration 0.961368 0.980494 0.697578
valid   O3_Concentration 0.095441 0.308935 0.230553
valid  SO2_Concentration 0.923172 0.960818 0.705458
valid            weather 0.927114 0.962867 0.603292
valid        temperature 0.047661 0.218314 0.170801
valid           pressure 0.000729 0.027009 0.019554
valid           humidity 0.334788 0.578609 0.411098
valid         wind_speed 1.112635 1.054815 0.764273
valid     wind_direction 0.414496 0.643814 0.546612

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.520172 0.721229 0.486481
 test PM10_Concentration 0.076856 0.277229 0.169514
 test  NO2_Concentration 0.443584 0.666021 0.497930
 test   CO_Concentration 0.329450 0.573977 0.404382
 test   O3_Concentration 0.212567 0.461050 0.345181
 test  SO2_Concentration 0.488612 0.699008 0.463372
 test            weather 0.997775 0.998887 0.609449
 test        temperature 0.068690 0.262088 0.195104
 test           pressure 0.000961 0.030992 0.022454
 test           humidity 0.418026 0.646550 0.471631
 test         wind_speed 1.407625 1.186434 0.890387
 test     wind_direction 0.497874 0.705602 0.582659
```

## 24*5h pre 4h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split                              
valid  0.424245  0.651341  0.402955
test   0.281841  0.530887  0.328884

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.371538 0.609539 0.435087
valid PM10_Concentration 0.442816 0.665444 0.457100
valid  NO2_Concentration 0.603867 0.777089 0.520346
valid   CO_Concentration 0.289900 0.538424 0.336895
valid   O3_Concentration 0.126188 0.355229 0.245076
valid  SO2_Concentration 0.487343 0.698100 0.444656
valid            weather 0.749669 0.865834 0.341882
valid        temperature 0.058231 0.241312 0.184216
valid           pressure 0.015957 0.126322 0.094869
valid           humidity 0.205886 0.453747 0.333760
valid         wind_speed 0.749314 0.865629 0.647965
valid     wind_direction 0.990232 0.995104 0.793610

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.301987 0.549534 0.386122
 test PM10_Concentration 0.285229 0.534069 0.346679
 test  NO2_Concentration 0.271981 0.521518 0.343122
 test   CO_Concentration 0.172805 0.415699 0.251252
 test   O3_Concentration 0.104085 0.322622 0.193640
 test  SO2_Concentration 0.210623 0.458937 0.287504
 test            weather 0.143330 0.378589 0.207600
 test        temperature 0.050776 0.225335 0.158669
 test           pressure 0.019564 0.139871 0.106647
 test           humidity 0.117976 0.343476 0.232430
 test         wind_speed 0.865324 0.930228 0.687986
 test     wind_direction 0.838405 0.915645 0.744952

返回结果包含: ['summary', 'per_metric']
```

## 24*1h pre 1h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split                              
valid  0.266419  0.516158  0.290215
test   0.200723  0.448021  0.243441

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.160580 0.400724 0.292161
valid PM10_Concentration 0.219935 0.468972 0.314111
valid  NO2_Concentration 0.282514 0.531520 0.344757
valid   CO_Concentration 0.127524 0.357105 0.214250
valid   O3_Concentration 0.060817 0.246612 0.164816
valid  SO2_Concentration 0.222182 0.471362 0.280880
valid            weather 0.332275 0.576433 0.185704
valid        temperature 0.027363 0.165417 0.123653
valid           pressure 0.007840 0.088543 0.067043
valid           humidity 0.102468 0.320106 0.227022
valid         wind_speed 0.624321 0.790140 0.570275
valid     wind_direction 1.029213 1.014501 0.697911

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.174497 0.417728 0.304487
 test PM10_Concentration 0.131793 0.363034 0.235603
 test  NO2_Concentration 0.116502 0.341324 0.222723
 test   CO_Concentration 0.060816 0.246609 0.152535
 test   O3_Concentration 0.038367 0.195874 0.117884
 test  SO2_Concentration 0.104381 0.323080 0.192104
 test            weather 0.066719 0.258300 0.116094
 test        temperature 0.017109 0.130801 0.093542
 test           pressure 0.007867 0.088697 0.068905
 test           humidity 0.045776 0.213953 0.141456
 test         wind_speed 0.676744 0.822645 0.573938
 test     wind_direction 0.968101 0.983921 0.702017
 ```

# PatchTST

## 24*14h pre 48h patch_len = 4
```
总体评估指标:
            mse      rmse       mae
split
valid  1.213977  1.101806  0.764994
test   0.640086  0.800054  0.569354

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 1.966615 1.402360 1.055416
valid PM10_Concentration 1.870583 1.367693 0.996625
valid  NO2_Concentration 1.804939 1.343480 1.027833
valid   CO_Concentration 1.020708 1.010301 0.734192
valid   O3_Concentration 0.301660 0.549236 0.412228
valid  SO2_Concentration 1.105900 1.051618 0.780963
valid            weather 3.363626 1.834019 1.221655
valid        temperature 0.259819 0.509724 0.417245
valid           pressure 0.124293 0.352552 0.272588
valid           humidity 0.631330 0.794563 0.629240
valid         wind_speed 1.022435 1.011155 0.746166
valid     wind_direction 1.095817 1.046813 0.885777

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 1.020202 1.010050 0.729309
 test PM10_Concentration 0.952743 0.976085 0.701487
 test  NO2_Concentration 0.624166 0.790042 0.571889
 test   CO_Concentration 0.794088 0.891116 0.606814
 test   O3_Concentration 0.244706 0.494677 0.322507
 test  SO2_Concentration 0.479506 0.692464 0.495981
 test            weather 0.558410 0.747268 0.507341
 test        temperature 0.325020 0.570106 0.451970
 test           pressure 0.189221 0.434995 0.339203
 test           humidity 0.344828 0.587221 0.431702
 test         wind_speed 1.294030 1.137554 0.879174
 test     wind_direction 0.854113 0.924182 0.794874
```

## 24*7h pre 24h patch_len = 4
```PatchTST
scaled_preds.shape: (852, 12, 24)                                                                                                         
y_test_preds.shape: (1705, 12, 24)                                                                                                        
results_df:             mse       mae
valid  0.972787  0.655101
test   0.513733  0.489772

# 1e-4学习率
results_df:             mse       mae
valid   0.92363  0.640434
test   0.496582  0.481753
```

```
总体评估指标:
            mse      rmse       mae
split                              
valid  0.943620  0.971401  0.673503
test   0.594569  0.771083  0.541719

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 1.548519 1.244395 0.931943
valid PM10_Concentration 1.477940 1.215706 0.896226
valid  NO2_Concentration 1.341682 1.158310 0.890096
valid   CO_Concentration 0.869977 0.932726 0.677231
valid   O3_Concentration 0.275466 0.524849 0.393022
valid  SO2_Concentration 0.927975 0.963314 0.716961
valid            weather 2.078682 1.441763 0.866796
valid        temperature 0.183523 0.428396 0.342820
valid           pressure 0.092198 0.303642 0.227720
valid           humidity 0.487008 0.697860 0.533596
valid         wind_speed 0.970447 0.985112 0.732332
valid     wind_direction 1.070024 1.034420 0.873292

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.992407 0.996196 0.735838
 test PM10_Concentration 0.859355 0.927014 0.680253
 test  NO2_Concentration 0.604719 0.777637 0.566035
 test   CO_Concentration 0.834349 0.913427 0.641515
 test   O3_Concentration 0.233200 0.482908 0.310280
 test  SO2_Concentration 0.528165 0.726750 0.512058
 test            weather 0.416071 0.645035 0.419234
 test        temperature 0.221267 0.470390 0.357932
 test           pressure 0.106096 0.325724 0.255001
 test           humidity 0.281972 0.531010 0.387569
 test         wind_speed 1.194599 1.092977 0.845857
 test     wind_direction 0.862630 0.928779 0.789050

返回结果包含: ['summary', 'per_metric']
```

## 168h pre 12h

```
总体评估指标:
            mse      rmse       mae
split                              
valid  0.702704  0.838274  0.551254
test   0.413619  0.643132  0.428850

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.887625 0.942139 0.690484
valid PM10_Concentration 0.889330 0.943043 0.667340
valid  NO2_Concentration 1.095721 1.046767 0.761749
valid   CO_Concentration 0.515917 0.718274 0.483766
valid   O3_Concentration 0.233260 0.482970 0.353403
valid  SO2_Concentration 0.807291 0.898494 0.626722
valid            weather 1.490394 1.220817 0.571670
valid        temperature 0.127845 0.357554 0.274969
valid           pressure 0.040991 0.202464 0.144994
valid           humidity 0.391778 0.625921 0.479648
valid         wind_speed 0.937901 0.968453 0.726042
valid     wind_direction 1.014389 1.007169 0.834256

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.518531 0.720091 0.506961
 test PM10_Concentration 0.498144 0.705793 0.478158
 test  NO2_Concentration 0.463206 0.680593 0.468310
 test   CO_Concentration 0.354159 0.595112 0.373610
 test   O3_Concentration 0.195306 0.441935 0.271348
 test  SO2_Concentration 0.330840 0.575187 0.383696
 test            weather 0.249699 0.499699 0.305631
 test        temperature 0.127761 0.357437 0.258242
 test           pressure 0.061696 0.248388 0.188762
 test           humidity 0.230527 0.480132 0.340517
 test         wind_speed 1.092432 1.045195 0.800703
 test     wind_direction 0.841122 0.917127 0.770261

返回结果包含: ['summary', 'per_metric']
```

## 24*7h pre 8h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.592150  0.769513  0.497052
test   0.366632  0.605502  0.394049

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.663919 0.814812 0.591924
valid PM10_Concentration 0.703564 0.838787 0.587739
valid  NO2_Concentration 0.899089 0.948203 0.673163
valid   CO_Concentration 0.440759 0.663897 0.433951
valid   O3_Concentration 0.198506 0.445540 0.320584
valid  SO2_Concentration 0.695557 0.834001 0.560015
valid            weather 1.181193 1.086827 0.481950
valid        temperature 0.099880 0.316038 0.241765
valid           pressure 0.029290 0.171142 0.124561
valid           humidity 0.317440 0.563418 0.429040
valid         wind_speed 0.864349 0.929703 0.696694
valid     wind_direction 1.012250 1.006106 0.823237

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.438245 0.662001 0.464388
 test PM10_Concentration 0.422586 0.650066 0.431961
 test  NO2_Concentration 0.400948 0.633204 0.426484
 test   CO_Concentration 0.290679 0.539147 0.331574
 test   O3_Concentration 0.165689 0.407049 0.247107
 test  SO2_Concentration 0.291578 0.539980 0.355141
 test            weather 0.216151 0.464921 0.273159
 test        temperature 0.094363 0.307185 0.219917
 test           pressure 0.040907 0.202254 0.153340
 test           humidity 0.184408 0.429428 0.300783
 test         wind_speed 1.015014 1.007479 0.764003
 test     wind_direction 0.839022 0.915982 0.760728
```


## 24*5h pre 4h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split                              
valid  0.420393  0.648377  0.401455
test   0.281385  0.530457  0.329506

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.366244 0.605181 0.432658
valid PM10_Concentration 0.438444 0.662151 0.453982
valid  NO2_Concentration 0.590970 0.768746 0.516248
valid   CO_Concentration 0.278567 0.527795 0.330232
valid   O3_Concentration 0.127512 0.357088 0.247021
valid  SO2_Concentration 0.480813 0.693407 0.439058
valid            weather 0.762890 0.873436 0.349240
valid        temperature 0.058472 0.241810 0.182871
valid           pressure 0.016537 0.128597 0.095966
valid           humidity 0.194589 0.441123 0.324937
valid         wind_speed 0.750548 0.866342 0.649097
valid     wind_direction 0.979123 0.989506 0.796148

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.314940 0.561195 0.395477
 test PM10_Concentration 0.286832 0.535567 0.348290
 test  NO2_Concentration 0.270930 0.520510 0.342821
 test   CO_Concentration 0.173452 0.416476 0.251501
 test   O3_Concentration 0.106920 0.326986 0.193641
 test  SO2_Concentration 0.210729 0.459052 0.288405
 test            weather 0.144125 0.379638 0.208136
 test        temperature 0.049613 0.222739 0.157186
 test           pressure 0.019679 0.140282 0.107198
 test           humidity 0.112799 0.335855 0.225423
 test         wind_speed 0.864625 0.929852 0.690823
 test     wind_direction 0.821974 0.906628 0.745170

返回结果包含: ['summary', 'per_metric']
```

## 24*1h pre 1h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split                              
valid  0.245023  0.494998  0.277246
test   0.188637  0.434323  0.238235

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.142718 0.377780 0.275099
valid PM10_Concentration 0.216913 0.465739 0.307298
valid  NO2_Concentration 0.237697 0.487541 0.315125
valid   CO_Concentration 0.118375 0.344057 0.203244
valid   O3_Concentration 0.051778 0.227548 0.146156
valid  SO2_Concentration 0.197297 0.444182 0.263413
valid            weather 0.306221 0.553372 0.159808
valid        temperature 0.022501 0.150005 0.109154
valid           pressure 0.007060 0.084027 0.063536
valid           humidity 0.084033 0.289885 0.193993
valid         wind_speed 0.613708 0.783395 0.576429
valid     wind_direction 0.941976 0.970554 0.713693

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.175872 0.419371 0.306937
 test PM10_Concentration 0.126095 0.355099 0.229961
 test  NO2_Concentration 0.104947 0.323955 0.208362
 test   CO_Concentration 0.056040 0.236728 0.145161
 test   O3_Concentration 0.031838 0.178432 0.106573
 test  SO2_Concentration 0.098211 0.313387 0.184156
 test            weather 0.065021 0.254991 0.110452
 test        temperature 0.014510 0.120458 0.084896
 test           pressure 0.006739 0.082094 0.063684
 test           humidity 0.038196 0.195437 0.122696
 test         wind_speed 0.686816 0.828744 0.587815
 test     wind_direction 0.859353 0.927013 0.708126

返回结果包含: ['summary', 'per_metric']
```

# Informer

## 24*14h pre 48h 

```
评估Informer模型

valid 预测形状: (828, 12, 48)

test 预测形状: (1657, 12, 48)

总体评估指标:
            mse      rmse       mae
split
valid  1.590833  1.261282  0.915796
test   1.153990  1.074239  0.858816

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 3.066031 1.751009 1.357035
valid PM10_Concentration 2.712382 1.646931 1.254399
valid  NO2_Concentration 2.800277 1.673403 1.282640
valid   CO_Concentration 1.151825 1.073231 0.838662
valid   O3_Concentration 0.630311 0.793921 0.540600
valid  SO2_Concentration 1.593496 1.262338 0.946388
valid            weather 2.310271 1.519957 0.911704
valid        temperature 0.678025 0.823423 0.671878
valid           pressure 0.299477 0.547245 0.455130
valid           humidity 1.405771 1.185652 0.936475
valid         wind_speed 1.381733 1.175471 0.912666
valid     wind_direction 1.060396 1.029755 0.881969

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 1.528593 1.236363 1.004990
 test PM10_Concentration 1.638412 1.280005 1.066788
 test  NO2_Concentration 1.038785 1.019208 0.781354
 test   CO_Concentration 0.981646 0.990780 0.714601
 test   O3_Concentration 0.819164 0.905077 0.772705
 test  SO2_Concentration 1.265251 1.124834 0.973801
 test            weather 0.900969 0.949194 0.751204
 test        temperature 0.606316 0.778663 0.633596
 test           pressure 0.533389 0.730335 0.572685
 test           humidity 1.612058 1.269669 1.094205
 test         wind_speed 2.066463 1.437520 1.140010
 test     wind_direction 0.856833 0.925652 0.799849
```

## 24*7h pre 24h 

```
评估Informer模型

valid 预测形状: (852, 12, 24)

test 预测形状: (1705, 12, 24)

总体评估指标:
            mse      rmse       mae
split
valid  1.561316  1.249526  0.902105
test   0.989494  0.994733  0.774932

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 2.223313 1.491078 1.248485
valid PM10_Concentration 1.985329 1.409017 1.113104
valid  NO2_Concentration 2.822337 1.679981 1.290845
valid   CO_Concentration 1.220705 1.104855 0.821634
valid   O3_Concentration 0.590965 0.768742 0.583044
valid  SO2_Concentration 1.707368 1.306663 0.906655
valid            weather 2.047666 1.430967 0.652461
valid        temperature 0.531312 0.728911 0.588063
valid           pressure 0.301238 0.548851 0.452683
valid           humidity 1.246629 1.116525 0.912570
valid         wind_speed 2.704654 1.644583 1.352093
valid     wind_direction 1.354279 1.163735 0.903620

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 1.552116 1.245839 1.026237
 test PM10_Concentration 1.607746 1.267969 1.056862
 test  NO2_Concentration 1.203177 1.096894 0.862656
 test   CO_Concentration 0.825509 0.908575 0.716479
 test   O3_Concentration 0.743106 0.862036 0.709292
 test  SO2_Concentration 0.705091 0.839697 0.666069
 test            weather 0.560908 0.748938 0.545631
 test        temperature 0.447442 0.668911 0.529024
 test           pressure 0.370788 0.608923 0.488220
 test           humidity 0.762149 0.873012 0.723318
 test         wind_speed 2.159163 1.469409 1.168881
 test     wind_direction 0.936738 0.967852 0.806509
```

## 168h pre 12h

```

```
