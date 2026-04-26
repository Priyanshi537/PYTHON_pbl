# Air Quality Category Prediction Report

- Dataset: `AirQualityUCI.csv`
- Total modeled rows: 9325
- Training rows: 7460
- Test rows: 1865
- Selected features: PT08.S2(NMHC), C6H6(GT), PT08.S1(CO), PT08.S5(O3), CO(GT), PT08.S3(NOx), NO2(GT), PT08.S4(NO2), lag1_C6H6(GT), NOx(GT), lag1_CO(GT), lag1_NOx(GT), T, RH, hour

## Target Distribution

- Good: 1555
- Satisfactory: 1554
- Moderate: 1554
- Poor: 1554
- Very Poor: 1553
- Severe: 1555

## logistic_regression

- Accuracy: 0.5298
- Macro Precision: 0.5101
- Macro Recall: 0.5014
- Macro F1: 0.4833
- Macro ROC AUC: 0.7110

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Good | 0.6283 | 0.8847 | 0.7348 | 451 |
| Satisfactory | 0.4746 | 0.2601 | 0.3360 | 323 |
| Moderate | 0.4395 | 0.2371 | 0.3080 | 291 |
| Poor | 0.3497 | 0.5535 | 0.4286 | 271 |
| Very Poor | 0.5000 | 0.3000 | 0.3750 | 260 |
| Severe | 0.6688 | 0.7732 | 0.7172 | 269 |

Confusion Matrix:

```text
399 24 12 15 1 0
161 84 44 32 1 1
49 50 69 116 3 4
16 12 25 150 46 22
9 5 3 89 78 76
1 2 4 27 27 208
```

## knn

- Accuracy: 0.5169
- Macro Precision: 0.4918
- Macro Recall: 0.4995
- Macro F1: 0.4923
- Macro ROC AUC: 0.6836

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Good | 0.7716 | 0.7339 | 0.7523 | 451 |
| Satisfactory | 0.4256 | 0.3189 | 0.3646 | 323 |
| Moderate | 0.3696 | 0.4433 | 0.4031 | 291 |
| Poor | 0.3238 | 0.3358 | 0.3297 | 271 |
| Very Poor | 0.4145 | 0.3731 | 0.3927 | 260 |
| Severe | 0.6455 | 0.7918 | 0.7112 | 269 |

Confusion Matrix:

```text
331 74 37 8 1 0
74 103 108 34 4 0
14 44 129 74 24 6
8 19 54 91 71 28
2 2 17 59 97 83
0 0 4 15 37 213
```

## svm

- Accuracy: 0.5458
- Macro Precision: 0.5173
- Macro Recall: 0.5245
- Macro F1: 0.5183
- Macro ROC AUC: 0.7163

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Good | 0.7788 | 0.7805 | 0.7796 | 451 |
| Satisfactory | 0.4606 | 0.4706 | 0.4655 | 323 |
| Moderate | 0.4147 | 0.3093 | 0.3543 | 291 |
| Poor | 0.3529 | 0.3985 | 0.3744 | 271 |
| Very Poor | 0.4393 | 0.4038 | 0.4208 | 260 |
| Severe | 0.6573 | 0.7844 | 0.7153 | 269 |

Confusion Matrix:

```text
352 78 14 6 1 0
77 152 67 22 4 1
15 74 90 91 17 4
6 16 40 108 82 19
2 10 2 55 105 86
0 0 4 24 30 211
```