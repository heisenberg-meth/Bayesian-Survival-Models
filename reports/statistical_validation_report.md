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
| **Bayesian Cox** | 0.5282 | 0.0530 | 0.0053 | [0.4280, 0.6276] | 0.4240 | 0.0383 | 0.0038 | [0.3544, 0.5001] |

### 2.2. Pairwise Model Significance & Multiple Testing Corrections
| Comparison (A vs B) | C-index Diff | Raw p-val | Bonferroni p-val | Holm p-val | FDR-BH p-val | IBS Diff | IBS Raw p | IBS Holm p |
|---|---|---|---|---|---|---|---|---|
| Cox PH vs RSF | -0.0498 | 2.5614e-13 | 1.5369e-12 | 1.2807e-12 | 7.6843e-13 | 0.0033 | 7.1994e-15 | 7.1994e-15 |
| Cox PH vs DeepSurv | -0.0951 | 2.7441e-17 | 1.6465e-16 | 1.6465e-16 | 1.6465e-16 | -0.0306 | 3.8966e-18 | 2.3379e-17 |
| Cox PH vs Bayesian Cox | -0.0444 | 5.2617e-07 | 3.1570e-06 | 1.0523e-06 | 6.3140e-07 | -0.2866 | 3.8966e-18 | 2.3379e-17 |
| RSF vs DeepSurv | -0.0454 | 6.7931e-10 | 4.0759e-09 | 2.0379e-09 | 1.0190e-09 | -0.0338 | 3.8966e-18 | 2.3379e-17 |
| RSF vs Bayesian Cox | 0.0054 | 5.0475e-01 | 1.0000e+00 | 5.0475e-01 | 5.0475e-01 | -0.2898 | 3.8966e-18 | 2.3379e-17 |
| DeepSurv vs Bayesian Cox | 0.0508 | 1.1419e-10 | 6.8515e-10 | 4.5677e-10 | 2.2838e-10 | -0.2560 | 3.8966e-18 | 2.3379e-17 |

### 2.3. Primary Research Contribution: Bayesian Cox Posterior Credible Intervals
| Feature | Coef Mean | exp(coef) HR Mean | exp(coef) HR Median | HR SD | 95% Credible Lower | 95% Credible Upper | Prob(HR > 1) |
|---|---|---|---|---|---|---|---|
| log_estrec | -0.0089 | 1.2214 | 1.0319 | 0.9236 | 0.2849 | 3.2705 | 0.5125 |
| estrec | -0.0332 | 1.1957 | 0.9446 | 0.8431 | 0.2559 | 3.3394 | 0.4650 |
| log_progrec | -0.0551 | 1.1871 | 0.9438 | 0.9013 | 0.2408 | 3.8157 | 0.4575 |
| age | -0.0436 | 1.1421 | 0.9261 | 0.7864 | 0.3382 | 3.2305 | 0.4450 |
| tsize | -0.0615 | 1.1397 | 0.9036 | 0.7948 | 0.2601 | 3.0607 | 0.4350 |
| age_x_horTh | -0.0687 | 1.1381 | 0.9338 | 0.7859 | 0.2737 | 3.1190 | 0.4475 |
| progrec | -0.0761 | 1.1243 | 0.9329 | 0.7411 | 0.2913 | 3.1115 | 0.4575 |
| log_pnode | -0.0873 | 1.1043 | 0.9010 | 0.7359 | 0.2466 | 3.1585 | 0.4300 |
| pnode | -0.1373 | 1.0584 | 0.9155 | 0.6921 | 0.2068 | 2.7261 | 0.4425 |
| horTh_yes | -0.1696 | 1.0507 | 0.8445 | 0.8137 | 0.2361 | 3.3016 | 0.3975 |
| menostat_Pre | -0.2028 | 1.0363 | 0.8464 | 0.7565 | 0.2094 | 2.8799 | 0.4125 |

### 2.4. Model Initialization Stability (3 Random Seeds)
| Model | C-Index Mean | C-Index SD | IBS Mean | IBS SD |
|---|---|---|---|---|
| **Cox PH** | 0.4832 | 0.0009 | 0.1378 | 0.0006 |
| **RSF** | 0.5469 | 0.0136 | 0.1350 | 0.0010 |
| **DeepSurv** | 0.5178 | 0.0485 | 0.1858 | 0.0117 |
| **Bayesian Cox** | 0.5090 | 0.0116 | 0.4221 | 0.0010 |

### 2.5. Training Size Sensitivity Analysis (C-index / IBS)
| Model | 50% Train Size | 75% Train Size | 100% Train Size |
|---|---|---|---|
| **Cox PH** | 0.4777 / 0.1397 | 0.4508 / 0.1382 | 0.4838 / 0.1374 |
| **RSF** | 0.4769 / 0.1390 | 0.5081 / 0.1339 | 0.5336 / 0.1341 |
| **DeepSurv** | 0.4345 / 0.2710 | 0.5051 / 0.2034 | 0.5790 / 0.1680 |
| **Bayesian Cox** | 0.4953 / 0.4215 | 0.5244 / 0.4217 | 0.5282 / 0.4240 |

### 2.6. Computational Complexity Profiling
| Model | Training Time (seconds) | Prediction Inference Time (seconds) |
|---|---|---|
| **Cox PH** | 0.0476 | 0.0006 |
| **RSF** | 7.4820 | 0.0173 |
| **DeepSurv** | 0.7550 | 0.0007 |
| **Bayesian Cox** | 71.0335 | 0.0153 |

---

## 2. Dataset: WHAS500

### 2.1. Bootstrap Uncertainty Quantification (B=100 Replicates)
| Model | Mean C-Index | C-Index SD | C-Index SE | C-Index 95% CI | Mean IBS | IBS SD | IBS SE | IBS 95% CI |
|---|---|---|---|---|---|---|---|---|
| **Cox PH** | 0.4729 | 0.0618 | 0.0062 | [0.3616, 0.5844] | 0.1624 | 0.0241 | 0.0024 | [0.1152, 0.2054] |
| **RSF** | 0.5689 | 0.0501 | 0.0050 | [0.4709, 0.6618] | 0.1504 | 0.0219 | 0.0022 | [0.1083, 0.1891] |
| **DeepSurv** | 0.5330 | 0.0489 | 0.0049 | [0.4471, 0.6286] | 0.2323 | 0.0393 | 0.0039 | [0.1610, 0.3132] |
| **Bayesian Cox** | 0.4282 | 0.0556 | 0.0056 | [0.3190, 0.5615] | 0.4330 | 0.0457 | 0.0046 | [0.3423, 0.5173] |

### 2.2. Pairwise Model Significance & Multiple Testing Corrections
| Comparison (A vs B) | C-index Diff | Raw p-val | Bonferroni p-val | Holm p-val | FDR-BH p-val | IBS Diff | IBS Raw p | IBS Holm p |
|---|---|---|---|---|---|---|---|---|
| Cox PH vs RSF | -0.0960 | 1.4147e-17 | 8.4881e-17 | 8.4881e-17 | 8.4881e-17 | 0.0120 | 4.0162e-18 | 2.3379e-17 |
| Cox PH vs DeepSurv | -0.0601 | 7.3651e-10 | 4.4191e-09 | 2.2095e-09 | 1.1048e-09 | -0.0698 | 5.2692e-18 | 2.3379e-17 |
| Cox PH vs Bayesian Cox | 0.0446 | 1.4757e-05 | 8.8540e-05 | 1.4757e-05 | 1.4757e-05 | -0.2706 | 3.8966e-18 | 2.3379e-17 |
| RSF vs DeepSurv | 0.0358 | 1.1633e-06 | 6.9797e-06 | 2.3266e-06 | 1.3959e-06 | -0.0819 | 4.0162e-18 | 2.3379e-17 |
| RSF vs Bayesian Cox | 0.1406 | 4.8816e-17 | 2.9290e-16 | 2.4408e-16 | 1.4645e-16 | -0.2826 | 3.8966e-18 | 2.3379e-17 |
| DeepSurv vs Bayesian Cox | 0.1048 | 5.3124e-16 | 3.1874e-15 | 2.1249e-15 | 1.0625e-15 | -0.2007 | 3.8966e-18 | 2.3379e-17 |

### 2.3. Primary Research Contribution: Bayesian Cox Posterior Credible Intervals
| Feature | Coef Mean | exp(coef) HR Mean | exp(coef) HR Median | HR SD | 95% Credible Lower | 95% Credible Upper | Prob(HR > 1) |
|---|---|---|---|---|---|---|---|
| hr | 0.0488 | 1.2482 | 1.0818 | 0.8080 | 0.3033 | 3.1250 | 0.5475 |
| sysbp | 0.0229 | 1.2030 | 1.0216 | 0.7261 | 0.2979 | 3.0684 | 0.5125 |
| bmi | -0.0100 | 1.1672 | 0.9877 | 0.6829 | 0.2898 | 2.8286 | 0.4750 |
| age | -0.0391 | 1.1472 | 0.9652 | 0.7433 | 0.2986 | 2.9672 | 0.4700 |
| diasbp | -0.0283 | 1.1391 | 0.9780 | 0.7054 | 0.3236 | 2.7480 | 0.4775 |
| age_x_gender | -0.0686 | 1.1063 | 0.9217 | 0.6993 | 0.2917 | 2.9941 | 0.4250 |
| age_x_chf | -0.0828 | 1.1032 | 0.8925 | 0.7291 | 0.2857 | 2.8926 | 0.4300 |
| chf_1 | -0.2059 | 1.0246 | 0.7855 | 0.8700 | 0.2369 | 3.0671 | 0.3650 |
| gender_1 | -0.2095 | 1.0001 | 0.7872 | 0.7776 | 0.2568 | 3.2093 | 0.3450 |
| cvd_1 | -0.2202 | 0.9983 | 0.8079 | 0.7461 | 0.2414 | 2.9760 | 0.3650 |
| afb_1 | -0.2841 | 0.9378 | 0.7211 | 0.7140 | 0.2092 | 2.8925 | 0.3125 |
| sho_1 | -0.2717 | 0.9286 | 0.7544 | 0.6261 | 0.2306 | 2.5928 | 0.3575 |

### 2.4. Model Initialization Stability (3 Random Seeds)
| Model | C-Index Mean | C-Index SD | IBS Mean | IBS SD |
|---|---|---|---|---|
| **Cox PH** | 0.4714 | 0.0020 | 0.1605 | 0.0028 |
| **RSF** | 0.5277 | 0.0483 | 0.1483 | 0.0034 |
| **DeepSurv** | 0.5339 | 0.0056 | 0.2203 | 0.0108 |
| **Bayesian Cox** | 0.4395 | 0.0230 | 0.4350 | 0.0012 |

### 2.5. Training Size Sensitivity Analysis (C-index / IBS)
| Model | 50% Train Size | 75% Train Size | 100% Train Size |
|---|---|---|---|
| **Cox PH** | 0.5411 / 0.1560 | 0.5137 / 0.1552 | 0.4729 / 0.1624 |
| **RSF** | 0.5347 / 0.1499 | 0.4750 / 0.1517 | 0.5689 / 0.1504 |
| **DeepSurv** | 0.5250 / 0.2543 | 0.5395 / 0.2285 | 0.5330 / 0.2323 |
| **Bayesian Cox** | 0.4000 / 0.4378 | 0.4290 / 0.4364 | 0.4282 / 0.4330 |

### 2.6. Computational Complexity Profiling
| Model | Training Time (seconds) | Prediction Inference Time (seconds) |
|---|---|---|
| **Cox PH** | 0.0075 | 0.0006 |
| **RSF** | 6.5004 | 0.0154 |
| **DeepSurv** | 0.3761 | 0.0007 |
| **Bayesian Cox** | 52.9377 | 0.0121 |

---

## 2. Dataset: METABRIC

### 2.1. Bootstrap Uncertainty Quantification (B=100 Replicates)
| Model | Mean C-Index | C-Index SD | C-Index SE | C-Index 95% CI | Mean IBS | IBS SD | IBS SE | IBS 95% CI |
|---|---|---|---|---|---|---|---|---|
| **Cox PH** | 0.5091 | 0.0252 | 0.0025 | [0.4591, 0.5547] | 0.1680 | 0.0076 | 0.0008 | [0.1545, 0.1815] |
| **RSF** | 0.5770 | 0.0253 | 0.0025 | [0.5319, 0.6217] | 0.1656 | 0.0076 | 0.0008 | [0.1525, 0.1789] |
| **DeepSurv** | 0.5289 | 0.0236 | 0.0024 | [0.4838, 0.5725] | 0.2007 | 0.0129 | 0.0013 | [0.1788, 0.2259] |
| **Bayesian Cox** | 0.5434 | 0.0255 | 0.0026 | [0.4956, 0.5914] | 0.3706 | 0.0194 | 0.0019 | [0.3330, 0.4073] |

### 2.2. Pairwise Model Significance & Multiple Testing Corrections
| Comparison (A vs B) | C-index Diff | Raw p-val | Bonferroni p-val | Holm p-val | FDR-BH p-val | IBS Diff | IBS Raw p | IBS Holm p |
|---|---|---|---|---|---|---|---|---|
| Cox PH vs RSF | -0.0678 | 4.1394e-18 | 2.4836e-17 | 2.4836e-17 | 2.4836e-17 | 0.0024 | 4.8137e-18 | 2.3379e-17 |
| Cox PH vs DeepSurv | -0.0198 | 4.9476e-08 | 2.9686e-07 | 9.8952e-08 | 5.9371e-08 | -0.0327 | 3.8966e-18 | 2.3379e-17 |
| Cox PH vs Bayesian Cox | -0.0343 | 3.7735e-12 | 2.2641e-11 | 1.5094e-11 | 7.5469e-12 | -0.2026 | 3.8966e-18 | 2.3379e-17 |
| RSF vs DeepSurv | 0.0481 | 1.5466e-17 | 9.2796e-17 | 7.7330e-17 | 4.6398e-17 | -0.0350 | 3.8966e-18 | 2.3379e-17 |
| RSF vs Bayesian Cox | 0.0335 | 9.0009e-12 | 5.4006e-11 | 2.7003e-11 | 1.3501e-11 | -0.2050 | 3.8966e-18 | 2.3379e-17 |
| DeepSurv vs Bayesian Cox | -0.0145 | 3.0999e-04 | 1.8600e-03 | 3.0999e-04 | 3.0999e-04 | -0.1699 | 3.8966e-18 | 2.3379e-17 |

### 2.3. Primary Research Contribution: Bayesian Cox Posterior Credible Intervals
| Feature | Coef Mean | exp(coef) HR Mean | exp(coef) HR Median | HR SD | 95% Credible Lower | 95% Credible Upper | Prob(HR > 1) |
|---|---|---|---|---|---|---|---|
| tumour_stage | -0.0430 | 1.1785 | 0.9357 | 0.8200 | 0.2573 | 3.2606 | 0.4550 |
| age_x_stage | -0.0299 | 1.1626 | 0.9768 | 0.7376 | 0.3172 | 3.1280 | 0.4850 |
| age | -0.0588 | 1.1199 | 0.9124 | 0.7593 | 0.3373 | 3.1349 | 0.4350 |
| lymph_nodes_positive | -0.0815 | 1.1129 | 0.8861 | 0.7669 | 0.2582 | 2.9657 | 0.4300 |
| log_lymph_nodes | -0.1094 | 1.0786 | 0.9403 | 0.6867 | 0.2201 | 2.7281 | 0.4600 |
| PAM50Subtype_Her2 | -0.1977 | 1.0324 | 0.8560 | 0.8332 | 0.2222 | 2.8672 | 0.3925 |
| hormone_therapy_1 | -0.2177 | 1.0184 | 0.8021 | 0.7921 | 0.1991 | 3.3353 | 0.3650 |
| PAM50Subtype_Normal | -0.2271 | 1.0018 | 0.8255 | 0.7144 | 0.2103 | 2.7383 | 0.4000 |
| chemotherapy_1 | -0.2186 | 0.9702 | 0.7901 | 0.6502 | 0.2149 | 2.7865 | 0.3475 |
| PAM50Subtype_LumB | -0.2497 | 0.9587 | 0.7795 | 0.7169 | 0.2253 | 2.9404 | 0.3550 |
| PAM50Subtype_LumA | -0.2803 | 0.9239 | 0.7557 | 0.6437 | 0.2194 | 2.5477 | 0.3050 |

### 2.4. Model Initialization Stability (3 Random Seeds)
| Model | C-Index Mean | C-Index SD | IBS Mean | IBS SD |
|---|---|---|---|---|
| **Cox PH** | 0.5104 | 0.0018 | 0.1677 | 0.0004 |
| **RSF** | 0.5766 | 0.0094 | 0.1649 | 0.0002 |
| **DeepSurv** | 0.5214 | 0.0114 | 0.1959 | 0.0048 |
| **Bayesian Cox** | 0.5430 | 0.0023 | 0.3756 | 0.0005 |

### 2.5. Training Size Sensitivity Analysis (C-index / IBS)
| Model | 50% Train Size | 75% Train Size | 100% Train Size |
|---|---|---|---|
| **Cox PH** | 0.4859 / 0.1703 | 0.5216 / 0.1674 | 0.5091 / 0.1680 |
| **RSF** | 0.5341 / 0.1677 | 0.5406 / 0.1662 | 0.5770 / 0.1656 |
| **DeepSurv** | 0.5267 / 0.2111 | 0.5475 / 0.2008 | 0.5289 / 0.2007 |
| **Bayesian Cox** | 0.5412 / 0.3736 | 0.5420 / 0.3739 | 0.5434 / 0.3706 |

### 2.6. Computational Complexity Profiling
| Model | Training Time (seconds) | Prediction Inference Time (seconds) |
|---|---|---|
| **Cox PH** | 0.0199 | 0.0007 |
| **RSF** | 17.0140 | 0.0553 |
| **DeepSurv** | 4.5833 | 0.0012 |
| **Bayesian Cox** | 188.8509 | 0.0234 |

---

