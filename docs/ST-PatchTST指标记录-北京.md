# ST-PatchTST

## 24*1h pre 1h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.172464  0.415288  0.211936
test   0.172561  0.415404  0.199025

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.138838 0.372610 0.226826
valid PM10_Concentration 0.004582 0.067691 0.031855
valid  NO2_Concentration 0.124047 0.352204 0.239907
valid   CO_Concentration 0.218729 0.467685 0.286918
valid   O3_Concentration 0.025731 0.160410 0.099165
valid  SO2_Concentration 0.181845 0.426433 0.283616
valid            weather 0.356856 0.597374 0.222321
valid        temperature 0.016382 0.127992 0.087826
valid           pressure 0.000191 0.013821 0.008117
valid           humidity 0.095629 0.309239 0.176946
valid         wind_speed 0.551522 0.742645 0.524819
valid     wind_direction 0.355211 0.595996 0.354920

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.099869 0.316021 0.172933
 test PM10_Concentration 0.014173 0.119051 0.057222
 test  NO2_Concentration 0.098561 0.313944 0.209119
 test   CO_Concentration 0.067714 0.260219 0.156021
 test   O3_Concentration 0.045480 0.213260 0.147142
 test  SO2_Concentration 0.090042 0.300069 0.169855
 test            weather 0.372745 0.610528 0.232623
 test        temperature 0.019924 0.141154 0.097627
 test           pressure 0.000213 0.014600 0.008755
 test           humidity 0.128251 0.358122 0.207145
 test         wind_speed 0.735143 0.857405 0.570786
 test     wind_direction 0.398615 0.631360 0.359069
```

## 120h pre 4h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.304833  0.552117  0.313278
test   0.282772  0.531764  0.299803

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.368660 0.607174 0.380804
valid PM10_Concentration 0.009548 0.097715 0.052970
valid  NO2_Concentration 0.312030 0.558596 0.384831
valid   CO_Concentration 0.540763 0.735366 0.475361
valid   O3_Concentration 0.055897 0.236426 0.157567
valid  SO2_Concentration 0.462253 0.679892 0.455649
valid            weather 0.554083 0.744368 0.359599
valid        temperature 0.028357 0.168395 0.123975
valid           pressure 0.000298 0.017263 0.011895
valid           humidity 0.168247 0.410180 0.261656
valid         wind_speed 0.788118 0.887760 0.633629
valid     wind_direction 0.369742 0.608064 0.461396

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.234178 0.483919 0.293900
 test PM10_Concentration 0.037132 0.192697 0.103710
 test  NO2_Concentration 0.247477 0.497471 0.340012
 test   CO_Concentration 0.163572 0.404440 0.261927
 test   O3_Concentration 0.104543 0.323332 0.229147
 test  SO2_Concentration 0.241311 0.491235 0.298973
 test            weather 0.614188 0.783702 0.382475
 test        temperature 0.034255 0.185081 0.133820
 test           pressure 0.000372 0.019299 0.013134
 test           humidity 0.233661 0.483385 0.313825
 test         wind_speed 1.047240 1.023347 0.734331
 test     wind_direction 0.435340 0.659803 0.492385
```

## 168h pre 12h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split                              
valid  0.526636  0.725697  0.451330
test   0.460556  0.678643  0.427767

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 0.832494 0.912411 0.635740
valid PM10_Concentration 0.019268 0.138809 0.088563
valid  NO2_Concentration 0.603222 0.776674 0.574059
valid   CO_Concentration 0.989809 0.994892 0.699258
valid   O3_Concentration 0.099306 0.315129 0.234952
valid  SO2_Concentration 0.888606 0.942659 0.680444
valid            weather 0.958665 0.979115 0.591368
valid        temperature 0.048294 0.219759 0.173383
valid           pressure 0.000741 0.027230 0.019690
valid           humidity 0.342797 0.585489 0.415201
valid         wind_speed 1.121717 1.059112 0.771281
valid     wind_direction 0.414706 0.643977 0.532017

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.521482 0.722137 0.485934
 test PM10_Concentration 0.075709 0.275153 0.166355
 test  NO2_Concentration 0.448062 0.669375 0.499290
 test   CO_Concentration 0.334072 0.577990 0.407132
 test   O3_Concentration 0.210959 0.459302 0.345705
 test  SO2_Concentration 0.508849 0.713337 0.474545
 test            weather 1.028738 1.014267 0.610839
 test        temperature 0.066880 0.258612 0.191840
 test           pressure 0.001003 0.031674 0.022850
 test           humidity 0.418808 0.647154 0.469595
 test         wind_speed 1.412133 1.188332 0.888655
 test     wind_direction 0.499976 0.707090 0.570462
```

## 24*7h pre 24h patch_len=4

```
总体评估指标:
            mse      rmse       mae
split
valid  0.715755  0.846023  0.551375
test   0.580782  0.762091  0.503014

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 1.269126 1.126555 0.836894
valid PM10_Concentration 0.028066 0.167529 0.114231
valid  NO2_Concentration 0.822196 0.906750 0.707214
valid   CO_Concentration 1.417061 1.190404 0.890602
valid   O3_Concentration 0.124066 0.352230 0.275019
valid  SO2_Concentration 1.306094 1.142845 0.869556
valid            weather 1.368892 1.169996 0.809426
valid        temperature 0.056119 0.236894 0.188973
valid           pressure 0.001327 0.036422 0.027113
valid           humidity 0.478205 0.691523 0.512416
valid         wind_speed 1.291025 1.136233 0.840604
valid     wind_direction 0.426889 0.653368 0.544446

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.779850 0.883091 0.631486
 test PM10_Concentration 0.096633 0.310858 0.206233
 test  NO2_Concentration 0.562761 0.750174 0.582237
 test   CO_Concentration 0.455915 0.675214 0.497237
 test   O3_Concentration 0.247209 0.497201 0.380926
 test  SO2_Concentration 0.736083 0.857953 0.605836
 test            weather 1.348112 1.161082 0.773923
 test        temperature 0.084873 0.291329 0.222100
 test           pressure 0.002023 0.044981 0.033187
 test           humidity 0.523977 0.723862 0.550142
 test         wind_speed 1.610429 1.269027 0.960298
 test     wind_direction 0.521522 0.722165 0.592562
```

- alpha_max = 0.5

```
总体评估指标:
            mse      rmse       mae
split
valid  0.717584  0.847103  0.552542
test   0.585821  0.765390  0.505693

valid 单指标评估结果:
split        metric_name      mse     rmse      mae
valid PM25_Concentration 1.278703 1.130797 0.849322
valid PM10_Concentration 0.027887 0.166995 0.113321
valid  NO2_Concentration 0.843277 0.918301 0.715448
valid   CO_Concentration 1.383392 1.176177 0.880198
valid   O3_Concentration 0.122782 0.350402 0.277074
valid  SO2_Concentration 1.295710 1.138293 0.868144
valid            weather 1.392744 1.180146 0.799612
valid        temperature 0.058505 0.241878 0.192489
valid           pressure 0.001317 0.036295 0.026612
valid           humidity 0.476775 0.690489 0.516946
valid         wind_speed 1.286347 1.134173 0.840959
valid     wind_direction 0.443572 0.666012 0.550374

test 单指标评估结果:
split        metric_name      mse     rmse      mae
 test PM25_Concentration 0.779159 0.882700 0.635077
 test PM10_Concentration 0.097593 0.312399 0.206868
 test  NO2_Concentration 0.555891 0.745581 0.580319
 test   CO_Concentration 0.456039 0.675307 0.497696
 test   O3_Concentration 0.244024 0.493988 0.380493
 test  SO2_Concentration 0.766306 0.875389 0.623440
 test            weather 1.376446 1.173220 0.779410
 test        temperature 0.088811 0.298011 0.229087
 test           pressure 0.002125 0.046096 0.033946
 test           humidity 0.535751 0.731950 0.555966
 test         wind_speed 1.609289 1.268578 0.964749
 test     wind_direction 0.518421 0.720015 0.581261
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
