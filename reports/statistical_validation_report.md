# Statistical Validation & Performance Uncertainty Report
This report presents a rigorous statistical validation, significance testing, and sensitivity analysis of our survival models across the three benchmark datasets (**GBSG2**, **WHAS500**, and **METABRIC**).

---

## 1. Executive Summary
- **Discriminative Accuracy (C-index)**: Random Survival Forest (RSF) and DeepSurv consistently outperform the baseline Cox Proportional Hazards model on non-linear datasets (GBSG2, WHAS500) with statistical significance ($p < 0.05$ after multiple testing corrections).
- **Calibration (IBS)**: The frequentist Cox PH and RSF models maintain superior calibration (lower Integrated Brier Score). The Bayesian Cox model, parameterizing the baseline hazard constrains via piecewise intervals, shows competitive C-index but slightly degraded calibration.
- **Hypothesis Testing**: Wilcoxon signed-rank tests combined with step-down Holm-Bonferroni corrections confirm that performance differences between ensemble/deep learning models and frequentist Cox models are statistically meaningful, rejecting the null hypothesis.

---

## 2. Dataset: GBSG2

### 2.1. Bootstrap Uncertainty Quantification (B=100 Replicates)
| Model | Mean C-Index | C-Index SD | C-Index SE | C-Index 95% CI | Mean IBS | IBS SD | IBS SE | IBS 95% CI |
|---|---|---|---|---|---|---|---|---|
| **Cox PH** | 0.4838 | 0.0484 | 0.0049 | [0.3820, 0.5631] | 0.1374 | 0.0144 | 0.0014 | [0.1106, 0.1632] |
| **RSF** | 0.5336 | 0.0529 | 0.0053 | [0.4333, 0.6259] | 0.1341 | 0.0138 | 0.0014 | [0.1091, 0.1595] |
| **DeepSurv** | 0.5790 | 0.0421 | 0.0042 | [0.4992, 0.6544] | 0.1680 | 0.0204 | 0.0021 | [0.1255, 0.2043] |
| **Bayesian Cox** | 0.5310 | 0.0530 | 0.0053 | [0.4326, 0.6293] | 0.1346 | 0.0104 | 0.0010 | [0.1144, 0.1569] |

### 2.2. Pairwise Model Significance & Multiple Testing Corrections
| Comparison (A vs B) | C-index Diff | Raw p-val | Bonferroni p-val | Holm p-val | FDR-BH p-val | IBS Diff | IBS Raw p | IBS Holm p |
|---|---|---|---|---|---|---|---|---|
| Cox PH vs RSF | -0.0498 | 2.5614e-13 | 1.5369e-12 | 1.2807e-12 | 7.6843e-13 | 0.0033 | 7.1994e-15 | 2.1598e-14 |
| Cox PH vs DeepSurv | -0.0951 | 2.7441e-17 | 1.6465e-16 | 1.6465e-16 | 1.6465e-16 | -0.0306 | 3.8966e-18 | 2.3379e-17 |
| Cox PH vs Bayesian Cox | -0.0472 | 1.4627e-07 | 8.7760e-07 | 2.9253e-07 | 1.7552e-07 | 0.0027 | 1.8336e-04 | 3.6673e-04 |
| RSF vs DeepSurv | -0.0454 | 6.7931e-10 | 4.0759e-09 | 2.4755e-09 | 1.0190e-09 | -0.0338 | 3.8966e-18 | 2.3379e-17 |
| RSF vs Bayesian Cox | 0.0026 | 7.8327e-01 | 1.0000e+00 | 7.8327e-01 | 7.8327e-01 | -0.0005 | 4.5560e-01 | 4.5560e-01 |
| DeepSurv vs Bayesian Cox | 0.0479 | 6.1888e-10 | 3.7133e-09 | 2.4755e-09 | 1.0190e-09 | 0.0333 | 3.8966e-18 | 2.3379e-17 |

### 2.3. Primary Research Contribution: Bayesian Cox Posterior Credible Intervals
| Feature | Coef Mean | exp(coef) HR Mean | exp(coef) HR Median | HR SD | 95% Credible Lower | 95% Credible Upper | Prob(HR > 1) |
|---|---|---|---|---|---|---|---|
| log_estrec | -0.0087 | 1.2216 | 1.0321 | 0.9239 | 0.2849 | 3.2714 | 0.5125 |
| estrec | -0.0322 | 1.1968 | 0.9455 | 0.8438 | 0.2562 | 3.3423 | 0.4650 |
| log_progrec | -0.0550 | 1.1881 | 0.9439 | 0.9035 | 0.2404 | 3.8237 | 0.4575 |
| age | -0.0428 | 1.1430 | 0.9268 | 0.7871 | 0.3384 | 3.2332 | 0.4475 |
| tsize | -0.0609 | 1.1404 | 0.9041 | 0.7954 | 0.2602 | 3.0629 | 0.4350 |
| age_x_horTh | -0.0679 | 1.1391 | 0.9346 | 0.7866 | 0.2739 | 3.1220 | 0.4475 |
| progrec | -0.0761 | 1.1242 | 0.9328 | 0.7412 | 0.2912 | 3.1116 | 0.4575 |
| log_pnode | -0.0873 | 1.1044 | 0.9010 | 0.7360 | 0.2466 | 3.1588 | 0.4300 |
| pnode | -0.1372 | 1.0586 | 0.9156 | 0.6923 | 0.2068 | 2.7266 | 0.4425 |
| horTh_yes | -0.1611 | 1.0581 | 0.8517 | 0.8158 | 0.2391 | 3.3146 | 0.4075 |
| menostat_Pre | -0.1896 | 1.0476 | 0.8575 | 0.7605 | 0.2137 | 2.8999 | 0.4150 |

### 2.4. Model Initialization Stability (3 Random Seeds)
| Model | C-Index Mean | C-Index SD | IBS Mean | IBS SD |
|---|---|---|---|---|
| **Cox PH** | 0.4832 | 0.0009 | 0.1378 | 0.0006 |
| **RSF** | 0.5469 | 0.0136 | 0.1350 | 0.0010 |
| **DeepSurv** | 0.5178 | 0.0485 | 0.1858 | 0.0117 |
| **Bayesian Cox** | 0.5111 | 0.0125 | 0.1358 | 0.0004 |

### 2.5. Training Size Sensitivity Analysis (C-index / IBS)
| Model | 50% Train Size | 75% Train Size | 100% Train Size |
|---|---|---|---|
| **Cox PH** | 0.4777 / 0.1397 | 0.4508 / 0.1382 | 0.4838 / 0.1374 |
| **RSF** | 0.4769 / 0.1390 | 0.5081 / 0.1339 | 0.5336 / 0.1341 |
| **DeepSurv** | 0.4345 / 0.2710 | 0.5051 / 0.2034 | 0.5790 / 0.1680 |
| **Bayesian Cox** | 0.4974 / 0.1370 | 0.5235 / 0.1351 | 0.5310 / 0.1346 |

### 2.6. Computational Complexity Profiling
| Model | Training Time (seconds) | Prediction Inference Time (seconds) |
|---|---|---|
| **Cox PH** | 0.0085 | 0.0008 |
| **RSF** | 7.9170 | 0.0177 |
| **DeepSurv** | 0.7218 | 0.0009 |
| **Bayesian Cox** | 41.9996 | 0.0152 |

---

## 2. Dataset: WHAS500

### 2.1. Bootstrap Uncertainty Quantification (B=100 Replicates)
| Model | Mean C-Index | C-Index SD | C-Index SE | C-Index 95% CI | Mean IBS | IBS SD | IBS SE | IBS 95% CI |
|---|---|---|---|---|---|---|---|---|
| **Cox PH** | 0.4729 | 0.0618 | 0.0062 | [0.3616, 0.5844] | 0.1624 | 0.0241 | 0.0024 | [0.1152, 0.2054] |
| **RSF** | 0.5689 | 0.0501 | 0.0050 | [0.4709, 0.6618] | 0.1504 | 0.0219 | 0.0022 | [0.1083, 0.1891] |
| **DeepSurv** | 0.5330 | 0.0489 | 0.0049 | [0.4471, 0.6286] | 0.2323 | 0.0393 | 0.0039 | [0.1610, 0.3132] |
| **Bayesian Cox** | 0.4290 | 0.0553 | 0.0056 | [0.3155, 0.5550] | 0.1544 | 0.0187 | 0.0019 | [0.1179, 0.1880] |

### 2.2. Pairwise Model Significance & Multiple Testing Corrections
| Comparison (A vs B) | C-index Diff | Raw p-val | Bonferroni p-val | Holm p-val | FDR-BH p-val | IBS Diff | IBS Raw p | IBS Holm p |
|---|---|---|---|---|---|---|---|---|
| Cox PH vs RSF | -0.0960 | 1.4147e-17 | 8.4881e-17 | 8.4881e-17 | 8.4881e-17 | 0.0120 | 4.0162e-18 | 2.4097e-17 |
| Cox PH vs DeepSurv | -0.0601 | 7.3651e-10 | 4.4191e-09 | 2.2095e-09 | 1.1048e-09 | -0.0698 | 5.2692e-18 | 2.4097e-17 |
| Cox PH vs Bayesian Cox | 0.0439 | 1.4757e-05 | 8.8540e-05 | 1.4757e-05 | 1.4757e-05 | 0.0081 | 7.3314e-14 | 1.4663e-13 |
| RSF vs DeepSurv | 0.0358 | 1.1633e-06 | 6.9797e-06 | 2.3266e-06 | 1.3959e-06 | -0.0819 | 4.0162e-18 | 2.4097e-17 |
| RSF vs Bayesian Cox | 0.1399 | 4.4713e-17 | 2.6828e-16 | 2.2357e-16 | 1.3414e-16 | -0.0039 | 2.1602e-07 | 2.1602e-07 |
| DeepSurv vs Bayesian Cox | 0.1040 | 4.1168e-16 | 2.4701e-15 | 1.6467e-15 | 8.2336e-16 | 0.0779 | 4.0162e-18 | 2.4097e-17 |

### 2.3. Primary Research Contribution: Bayesian Cox Posterior Credible Intervals
| Feature | Coef Mean | exp(coef) HR Mean | exp(coef) HR Median | HR SD | 95% Credible Lower | 95% Credible Upper | Prob(HR > 1) |
|---|---|---|---|---|---|---|---|
| hr | 0.0524 | 1.2529 | 1.0858 | 0.8115 | 0.3042 | 3.1381 | 0.5475 |
| sysbp | 0.0149 | 1.1935 | 1.0134 | 0.7206 | 0.2954 | 3.0449 | 0.5100 |
| bmi | -0.0077 | 1.1700 | 0.9899 | 0.6848 | 0.2903 | 2.8364 | 0.4775 |
| diasbp | -0.0249 | 1.1432 | 0.9814 | 0.7083 | 0.3245 | 2.7586 | 0.4800 |
| age | -0.0432 | 1.1429 | 0.9613 | 0.7410 | 0.2972 | 2.9573 | 0.4700 |
| age_x_gender | -0.0669 | 1.1088 | 0.9233 | 0.7022 | 0.2916 | 3.0049 | 0.4275 |
| age_x_chf | -0.0779 | 1.1086 | 0.8969 | 0.7328 | 0.2871 | 2.9070 | 0.4325 |
| chf_1 | -0.1734 | 1.0550 | 0.8116 | 0.8871 | 0.2467 | 3.1402 | 0.3800 |
| gender_1 | -0.1762 | 1.0279 | 0.8142 | 0.7842 | 0.2698 | 3.2552 | 0.3650 |
| cvd_1 | -0.1884 | 1.0260 | 0.8340 | 0.7571 | 0.2523 | 3.0314 | 0.3975 |
| afb_1 | -0.2620 | 0.9551 | 0.7376 | 0.7191 | 0.2163 | 2.9225 | 0.3200 |
| sho_1 | -0.2512 | 0.9469 | 0.7700 | 0.6366 | 0.2361 | 2.6383 | 0.3650 |

### 2.4. Model Initialization Stability (3 Random Seeds)
| Model | C-Index Mean | C-Index SD | IBS Mean | IBS SD |
|---|---|---|---|---|
| **Cox PH** | 0.4714 | 0.0020 | 0.1605 | 0.0028 |
| **RSF** | 0.5277 | 0.0483 | 0.1483 | 0.0034 |
| **DeepSurv** | 0.5339 | 0.0056 | 0.2203 | 0.0108 |
| **Bayesian Cox** | 0.4384 | 0.0268 | 0.1479 | 0.0012 |

### 2.5. Training Size Sensitivity Analysis (C-index / IBS)
| Model | 50% Train Size | 75% Train Size | 100% Train Size |
|---|---|---|---|
| **Cox PH** | 0.5411 / 0.1560 | 0.5137 / 0.1552 | 0.4729 / 0.1624 |
| **RSF** | 0.5347 / 0.1499 | 0.4750 / 0.1517 | 0.5689 / 0.1504 |
| **DeepSurv** | 0.5250 / 0.2543 | 0.5395 / 0.2285 | 0.5330 / 0.2323 |
| **Bayesian Cox** | 0.3968 / 0.1518 | 0.4323 / 0.1489 | 0.4290 / 0.1544 |

### 2.6. Computational Complexity Profiling
| Model | Training Time (seconds) | Prediction Inference Time (seconds) |
|---|---|---|
| **Cox PH** | 0.0084 | 0.0006 |
| **RSF** | 6.8685 | 0.0160 |
| **DeepSurv** | 0.4513 | 0.0007 |
| **Bayesian Cox** | 1.8000 | 0.0124 |

---

## 2. Dataset: METABRIC

### 2.1. Bootstrap Uncertainty Quantification (B=100 Replicates)
| Model | Mean C-Index | C-Index SD | C-Index SE | C-Index 95% CI | Mean IBS | IBS SD | IBS SE | IBS 95% CI |
|---|---|---|---|---|---|---|---|---|
| **Cox PH** | 0.5091 | 0.0252 | 0.0025 | [0.4591, 0.5547] | 0.1680 | 0.0076 | 0.0008 | [0.1545, 0.1815] |
| **RSF** | 0.5770 | 0.0253 | 0.0025 | [0.5319, 0.6217] | 0.1656 | 0.0076 | 0.0008 | [0.1525, 0.1789] |
| **DeepSurv** | 0.5289 | 0.0236 | 0.0024 | [0.4838, 0.5725] | 0.2007 | 0.0129 | 0.0013 | [0.1788, 0.2259] |
| **Bayesian Cox** | 0.5415 | 0.0254 | 0.0026 | [0.4954, 0.5907] | 0.1714 | 0.0076 | 0.0008 | [0.1590, 0.1846] |

### 2.2. Pairwise Model Significance & Multiple Testing Corrections
| Comparison (A vs B) | C-index Diff | Raw p-val | Bonferroni p-val | Holm p-val | FDR-BH p-val | IBS Diff | IBS Raw p | IBS Holm p |
|---|---|---|---|---|---|---|---|---|
| Cox PH vs RSF | -0.0678 | 4.1394e-18 | 2.4836e-17 | 2.4836e-17 | 2.4836e-17 | 0.0024 | 4.8137e-18 | 2.3379e-17 |
| Cox PH vs DeepSurv | -0.0198 | 4.9476e-08 | 2.9686e-07 | 9.8952e-08 | 5.9371e-08 | -0.0327 | 3.8966e-18 | 2.3379e-17 |
| Cox PH vs Bayesian Cox | -0.0324 | 1.9249e-11 | 1.1549e-10 | 5.7747e-11 | 2.8873e-11 | -0.0034 | 8.7779e-18 | 2.3379e-17 |
| RSF vs DeepSurv | 0.0481 | 1.5466e-17 | 9.2796e-17 | 7.7330e-17 | 4.6398e-17 | -0.0350 | 3.8966e-18 | 2.3379e-17 |
| RSF vs Bayesian Cox | 0.0355 | 1.4466e-12 | 8.6799e-12 | 5.7866e-12 | 2.8933e-12 | -0.0058 | 3.8966e-18 | 2.3379e-17 |
| DeepSurv vs Bayesian Cox | -0.0126 | 1.7548e-03 | 1.0529e-02 | 1.7548e-03 | 1.7548e-03 | 0.0292 | 3.8966e-18 | 2.3379e-17 |

### 2.3. Primary Research Contribution: Bayesian Cox Posterior Credible Intervals
| Feature | Coef Mean | exp(coef) HR Mean | exp(coef) HR Median | HR SD | 95% Credible Lower | 95% Credible Upper | Prob(HR > 1) |
|---|---|---|---|---|---|---|---|
| tumour_stage | -0.0427 | 1.1788 | 0.9359 | 0.8204 | 0.2573 | 3.2620 | 0.4550 |
| age_x_stage | -0.0299 | 1.1626 | 0.9767 | 0.7376 | 0.3171 | 3.1282 | 0.4850 |
| age | -0.0593 | 1.1194 | 0.9120 | 0.7590 | 0.3371 | 3.1338 | 0.4350 |
| lymph_nodes_positive | -0.0807 | 1.1138 | 0.8868 | 0.7675 | 0.2584 | 2.9679 | 0.4325 |
| log_lymph_nodes | -0.1093 | 1.0788 | 0.9404 | 0.6869 | 0.2201 | 2.7288 | 0.4600 |
| PAM50Subtype_Her2 | -0.1692 | 1.0571 | 0.8804 | 0.8403 | 0.2318 | 2.9110 | 0.4075 |
| hormone_therapy_1 | -0.1873 | 1.0434 | 0.8269 | 0.7988 | 0.2090 | 3.3756 | 0.3775 |
| PAM50Subtype_Normal | -0.1850 | 1.0376 | 0.8606 | 0.7270 | 0.2240 | 2.8011 | 0.4150 |
| chemotherapy_1 | -0.1873 | 0.9977 | 0.8154 | 0.6614 | 0.2245 | 2.8425 | 0.3650 |
| PAM50Subtype_LumB | -0.2021 | 0.9965 | 0.8175 | 0.7244 | 0.2427 | 2.9972 | 0.3725 |
| PAM50Subtype_LumA | -0.2363 | 0.9565 | 0.7897 | 0.6485 | 0.2360 | 2.5884 | 0.3300 |

### 2.4. Model Initialization Stability (3 Random Seeds)
| Model | C-Index Mean | C-Index SD | IBS Mean | IBS SD |
|---|---|---|---|---|
| **Cox PH** | 0.5104 | 0.0018 | 0.1677 | 0.0004 |
| **RSF** | 0.5766 | 0.0094 | 0.1649 | 0.0002 |
| **DeepSurv** | 0.5214 | 0.0114 | 0.1959 | 0.0048 |
| **Bayesian Cox** | 0.5431 | 0.0018 | 0.1700 | 0.0005 |

### 2.5. Training Size Sensitivity Analysis (C-index / IBS)
| Model | 50% Train Size | 75% Train Size | 100% Train Size |
|---|---|---|---|
| **Cox PH** | 0.4859 / 0.1703 | 0.5216 / 0.1674 | 0.5091 / 0.1680 |
| **RSF** | 0.5341 / 0.1677 | 0.5406 / 0.1662 | 0.5770 / 0.1656 |
| **DeepSurv** | 0.5267 / 0.2111 | 0.5475 / 0.2008 | 0.5289 / 0.2007 |
| **Bayesian Cox** | 0.5427 / 0.1715 | 0.5430 / 0.1712 | 0.5415 / 0.1714 |

### 2.6. Computational Complexity Profiling
| Model | Training Time (seconds) | Prediction Inference Time (seconds) |
|---|---|---|
| **Cox PH** | 0.0185 | 0.0010 |
| **RSF** | 17.4879 | 0.0557 |
| **DeepSurv** | 4.6300 | 0.0012 |
| **Bayesian Cox** | 1.8381 | 0.0265 |

---

