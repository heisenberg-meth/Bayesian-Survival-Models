
(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/download_datasets.py
Exporting GBSG2 dataset to /home/loverboy/Downloads/Personal/Bayesian COX/data/raw/gbsg2.csv...
sksurv or pandas load failed (No module named 'sksurv'). Generating synthetic GBSG2 dataset...
Saved benchmark GBSG2 dataset (686 records) to /home/loverboy/Downloads/Personal/Bayesian COX/data/raw/gbsg2.csv.
Exporting WHAS500 dataset to /home/loverboy/Downloads/Personal/Bayesian COX/data/raw/whas500.csv...
sksurv or pandas load failed (No module named 'sksurv'). Generating synthetic WHAS500 dataset...
Saved benchmark WHAS500 dataset (500 records) to /home/loverboy/Downloads/Personal/Bayesian COX/data/raw/whas500.csv.
Exporting METABRIC dataset to /home/loverboy/Downloads/Personal/Bayesian COX/data/raw/metabric.csv...
METABRIC dataset already exists at /home/loverboy/Downloads/Personal/Bayesian COX/data/raw/metabric.csv.

Dataset preparation completed successfully!
(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/prepare_dataset.py
==============================================================================================

STARTING PHASE 4 — DATA PREPROCESSING & DATASET PREPARATION
============================================================

[+] Processing dataset: 'GBSG2'...
    Raw Shape: 686 rows, 9 columns
    ✓ Processed Train Split: 479 samples (Censoring: 55.11%)
    ✓ Processed Val Split:   102 samples (Censoring: 54.9%)
    ✓ Processed Test Split:  105 samples (Censoring: 55.24%)
    ✓ Processed Features:    11 columns: ['age', 'tsize', 'pnode', 'progrec', 'estrec', 'log_pnode', 'log_progrec', 'log_estrec', 'age_x_horTh', 'horTh_yes', 'menostat_Pre']

[+] Processing dataset: 'WHAS500'...
    Raw Shape: 500 rows, 12 columns
    ✓ Processed Train Split: 349 samples (Censoring: 57.31%)
    ✓ Processed Val Split:   74 samples (Censoring: 58.11%)
    ✓ Processed Test Split:  77 samples (Censoring: 57.14%)
    ✓ Processed Features:    12 columns: ['age', 'hr', 'sysbp', 'diasbp', 'bmi', 'age_x_chf', 'age_x_gender', 'gender_1', 'cvd_1', 'afb_1', 'sho_1', 'chf_1']

[+] Processing dataset: 'METABRIC'...
    Raw Shape: 1904 rows, 8 columns
    ✓ Processed Train Split: 1332 samples (Censoring: 42.04%)
    ✓ Processed Val Split:   285 samples (Censoring: 42.11%)
    ✓ Processed Test Split:  287 samples (Censoring: 41.81%)
    ✓ Processed Features:    11 columns: ['age', 'lymph_nodes_positive', 'log_lymph_nodes', 'age_x_stage', 'tumour_stage', 'chemotherapy_1', 'hormone_therapy_1', 'PAM50Subtype_Her2', 'PAM50Subtype_LumA', 'PAM50Subtype_LumB', 'PAM50Subtype_Normal']

============================================================
ALL DATASETS SUCCESSFULLY PREPROCESSED AND STORED IN data/processed/
====================================================================

(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/prepare_dataset.py
==============================================================================================

STARTING PHASE 4 — DATA PREPROCESSING & DATASET PREPARATION
============================================================

[+] Processing dataset: 'GBSG2'...
    Raw Shape: 686 rows, 9 columns
    ✓ Processed Train Split: 479 samples (Censoring: 55.11%)
    ✓ Processed Val Split:   102 samples (Censoring: 54.9%)
    ✓ Processed Test Split:  105 samples (Censoring: 55.24%)
    ✓ Processed Features:    11 columns: ['age', 'tsize', 'pnode', 'progrec', 'estrec', 'log_pnode', 'log_progrec', 'log_estrec', 'age_x_horTh', 'horTh_yes', 'menostat_Pre']

[+] Processing dataset: 'WHAS500'...
    Raw Shape: 500 rows, 12 columns
    ✓ Processed Train Split: 349 samples (Censoring: 57.31%)
    ✓ Processed Val Split:   74 samples (Censoring: 58.11%)
    ✓ Processed Test Split:  77 samples (Censoring: 57.14%)
    ✓ Processed Features:    12 columns: ['age', 'hr', 'sysbp', 'diasbp', 'bmi', 'age_x_chf', 'age_x_gender', 'gender_1', 'cvd_1', 'afb_1', 'sho_1', 'chf_1']

[+] Processing dataset: 'METABRIC'...
    Raw Shape: 1904 rows, 8 columns
    ✓ Processed Train Split: 1332 samples (Censoring: 42.04%)
    ✓ Processed Val Split:   285 samples (Censoring: 42.11%)
    ✓ Processed Test Split:  287 samples (Censoring: 41.81%)
    ✓ Processed Features:    11 columns: ['age', 'lymph_nodes_positive', 'log_lymph_nodes', 'age_x_stage', 'tumour_stage', 'chemotherapy_1', 'hormone_therapy_1', 'PAM50Subtype_Her2', 'PAM50Subtype_LumA', 'PAM50Subtype_LumB', 'PAM50Subtype_Normal']

============================================================
ALL DATASETS SUCCESSFULLY PREPROCESSED AND STORED IN data/processed/
====================================================================

(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/run_eda.py
======================================================================================

STARTING PHASE 3 — DATASET UNDERSTANDING & EDA
===============================================

Processing Dataset: GBSG2 (gbsg2.csv)...
✓ Completed analysis & visualizations for GBSG2.

Processing Dataset: WHAS500 (whas500.csv)...
✓ Completed analysis & visualizations for WHAS500.

Processing Dataset: METABRIC (metabric.csv)...
✓ Completed analysis & visualizations for METABRIC.

Saved structured EDA statistics to /home/loverboy/Downloads/Personal/Bayesian COX/reports/tables/eda_results.json
Saved complete EDA markdown report to /home/loverboy/Downloads/Personal/Bayesian COX/reports/eda_report.md

Phase 3 EDA successfully executed!
(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/benchmark_log_rank.py
Original score:  0.05133079730873379
Optimized score: 0.05133079730873379
Correctness verified!
Original execution time (200 runs):  0.9296 seconds
Optimized execution time (200 runs): 0.4230 seconds
Speedup factor: 2.20x
(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$



(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/run_cox_ph.py
/home/loverboy/Downloads/Personal/Bayesian COX/.venv/lib64/python3.10/site-packages/arviz/__init__.py:50: FutureWarning:
ArviZ is undergoing a major refactor to improve flexibility and extensibility while maintaining a user-friendly interface.
Some upcoming changes may be backward incompatible.
For details and migration guidance, visit: https://python.arviz.org/en/latest/user_guide/migration_guide.html
  warn(
=======

STARTING PHASE 6 — COX PROPORTIONAL HAZARDS MODEL (BASELINE)
=============================================================

Resolved Configuration (source: config/model.yaml):
    L2 Regularization:   0.0001
===============================

[+] Executing Cox PH on Dataset: 'GBSG2'

    Model Coefficients & Hazard Ratios:
     feature  coef (beta)  exp(coef) HR  se(coef)  95% CI Lower  95% CI Upper         z        p
   horTh_yes     0.737299      2.090281  0.433657      0.893446      4.890366  1.700187 0.089096
 age_x_horTh    -0.340573      0.711363  0.221323      0.460996      1.097703 -1.538805 0.123852
menostat_Pre     0.168850      1.183942  0.140888      0.898264      1.560476  1.198463 0.230737
         age     0.111800      1.118289  0.095035      0.928236      1.347254  1.176406 0.239433
      estrec     0.062384      1.064371  0.112263      0.854147      1.326334  0.555692 0.578421
       tsize     0.032380      1.032910  0.072062      0.896855      1.189604  0.449336 0.653189
 log_progrec     0.047982      1.049152  0.118831      0.831166      1.324309  0.403786 0.686370
     progrec    -0.035602      0.965024  0.125907      0.753987      1.235130 -0.282765 0.777357
  log_estrec    -0.029711      0.970726  0.122040      0.764213      1.233044 -0.243458 0.807651
   log_pnode    -0.041100      0.959733  0.171538      0.685700      1.343281 -0.239597 0.810643
       pnode    -0.003680      0.996327  0.178852      0.701714      1.414632 -0.020574 0.983586

    Proportional Hazards Assumption Test (Schoenfeld Residuals):
     feature     rho  p_value  ph_satisfied
         age -0.0624   0.3624          True
       tsize  0.0522   0.4463          True
       pnode  0.0067   0.9227          True
     progrec  0.0574   0.4026          True
      estrec  0.1330   0.0515          True
   log_pnode -0.0290   0.6725          True
 log_progrec  0.0206   0.7639          True
  log_estrec  0.0828   0.2269          True
 age_x_horTh  0.0788   0.2496          True
   horTh_yes  0.1113   0.1036          True
menostat_Pre  0.1260   0.0652          True

    Test C-Index:              0.482 ± 0.0103
    Test Integrated Brier:     0.1386
    5-Fold CV Mean C-Index:    0.4485 ± 0.0592
    5-Fold CV Mean IBS:        0.1503 ± 0.0268

[+] Executing Cox PH on Dataset: 'WHAS500'

    Model Coefficients & Hazard Ratios:
     feature  coef (beta)  exp(coef) HR  se(coef)  95% CI Lower  95% CI Upper         z        p
          hr     0.177975      1.194795  0.090176      1.001231      1.425781  1.973641 0.048423
       chf_1     0.876629      2.402786  0.638437      0.687490      8.397766  1.373085 0.169726
       afb_1    -0.370918      0.690101  0.270177      0.406381      1.171904 -1.372869 0.169793
age_x_gender     0.306559      1.358742  0.283829      0.778999      2.369938  1.080086 0.280104
       sysbp    -0.091541      0.912524  0.086260      0.770580      1.080614 -1.061220 0.288590
   age_x_chf    -0.256808      0.773517  0.252292      0.471752      1.268310 -1.017898 0.308727
       sho_1    -0.406033      0.666288  0.402102      0.302961      1.465338 -1.009778 0.312602
         age    -0.109056      0.896680  0.117843      0.711750      1.129660 -0.925434 0.354740
    gender_1    -0.509039      0.601073  0.565072      0.198577      1.819389 -0.900840 0.367673
      diasbp    -0.018500      0.981670  0.086890      0.827948      1.163932 -0.212914 0.831394
       cvd_1     0.037073      1.037769  0.199737      0.701586      1.535043  0.185611 0.852750
         bmi    -0.015665      0.984457  0.084575      0.834075      1.161954 -0.185215 0.853061

    Proportional Hazards Assumption Test (Schoenfeld Residuals):
     feature     rho  p_value  ph_satisfied
         age  0.0263   0.7501          True
          hr  0.1143   0.1651          True
       sysbp  0.1039   0.2074          True
      diasbp -0.1465   0.0746          True
         bmi -0.0465   0.5737          True
   age_x_chf  0.1455   0.0767          True
age_x_gender -0.0818   0.3210          True
    gender_1 -0.0740   0.3699          True
       cvd_1 -0.1587   0.0532          True
       afb_1 -0.1099   0.1823          True
       sho_1  0.1411   0.0861          True
       chf_1  0.1641   0.0456         False

    Test C-Index:              0.4685 ± 0.0142
    Test Integrated Brier:     0.1565
    5-Fold CV Mean C-Index:    0.5129 ± 0.0501
    5-Fold CV Mean IBS:        0.151 ± 0.0166

[+] Executing Cox PH on Dataset: 'METABRIC'

    Model Coefficients & Hazard Ratios:
             feature  coef (beta)  exp(coef) HR  se(coef)  95% CI Lower  95% CI Upper         z        p
   PAM50Subtype_Her2     0.176862      1.193466  0.112883      0.956583      1.489010  1.566778 0.117167
     log_lymph_nodes    -0.129869      0.878210  0.092628      0.732405      1.053041 -1.402047 0.160901
lymph_nodes_positive     0.121986      1.129739  0.092512      0.942389      1.354333  1.318603 0.187302
 PAM50Subtype_Normal     0.141711      1.152244  0.115091      0.919553      1.443817  1.231296 0.218212
      chemotherapy_1     0.074856      1.077729  0.078082      0.924794      1.255954  0.958688 0.337716
   PAM50Subtype_LumB     0.069424      1.071891  0.112685      0.859471      1.336810  0.616091 0.537835
   hormone_therapy_1    -0.036363      0.964290  0.074155      0.833846      1.115140 -0.490372 0.623871
        tumour_stage    -0.049648      0.951564  0.109750      0.767391      1.179938 -0.452375 0.650999
   PAM50Subtype_LumA     0.049437      1.050679  0.116813      0.835674      1.321002  0.423214 0.672139
         age_x_stage     0.045955      1.047027  0.157188      0.769409      1.424817  0.292358 0.770013
                 age    -0.014453      0.985651  0.107458      0.798460      1.216727 -0.134499 0.893008

    Proportional Hazards Assumption Test (Schoenfeld Residuals):
             feature     rho  p_value  ph_satisfied
                 age -0.0239   0.5076          True
lymph_nodes_positive  0.0083   0.8189          True
     log_lymph_nodes  0.0039   0.9146          True
         age_x_stage -0.0146   0.6862          True
        tumour_stage  0.0033   0.9272          True
      chemotherapy_1  0.0294   0.4139          True
   hormone_therapy_1 -0.0300   0.4059          True
   PAM50Subtype_Her2 -0.0446   0.2154          True
   PAM50Subtype_LumA  0.0063   0.8602          True
   PAM50Subtype_LumB -0.0357   0.3217          True
 PAM50Subtype_Normal  0.0586   0.1036          True

    Test C-Index:              0.5129 ± 0.0032
    Test Integrated Brier:     0.1671
    5-Fold CV Mean C-Index:    0.489 ± 0.0267
    5-Fold CV Mean IBS:        0.1696 ± 0.0079

============================================================
COX PH BASELINE EXECUTION COMPLETE! Results saved to /home/loverboy/Downloads/Personal/Bayesian COX/reports/tables/cox_ph_results.json
======================================================================================================================================

(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/run_rsf.py
======================================================================================

STARTING PHASE 6 — RANDOM SURVIVAL FORESTS (RSF)
=================================================

Resolved Configuration (source: config/model.yaml):
    n_estimators:        100
    max_depth:           8
    min_samples_split:   10
    min_samples_leaf:    5
    max_features:        sqrt
    bootstrap:           True
    Grid search combos:  8
==========================

[+] Processing Dataset: 'GBSG2'
    Running grid search hyperparameter tuning on validation set...
      Combo 1: {'n_estimators': 50, 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5189
      Combo 2: {'n_estimators': 50, 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5079
      Combo 3: {'n_estimators': 50, 'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5268
      Combo 4: {'n_estimators': 50, 'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5202
      Combo 5: {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5379
      Combo 6: {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5273
      Combo 7: {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5128
      Combo 8: {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5185
    [--->] Optimal Parameters: {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} with Val C-Index = 0.5379

    Permutation Feature Importances (VIMP):
     feature  importance (VIMP)
         age           0.052899
   log_pnode           0.048095
       tsize           0.046124
 log_progrec           0.039463
  log_estrec           0.034315
      estrec           0.031310
     progrec           0.019272
 age_x_horTh           0.019234
menostat_Pre           0.007694
       pnode           0.007100
   horTh_yes           0.001129

    Test C-Index:              0.5441 ± 0.0103
    Test Integrated Brier:     0.1356
    Test Time-dependent AUC:   {'auc_time_255.0': 0.6632, 'auc_time_617.0': 0.5495, 'auc_time_1418.0': 0.5828}
    5-Fold CV Mean C-Index:    0.4411 ± 0.0725
    5-Fold CV Mean IBS:        0.1481 ± 0.0273

    Baseline Comparison vs Cox PH:
    - Cox PH Test C-Index:     0.482
    - RSF Test C-Index:        0.5441
    - Cox PH Test IBS:         0.1386
    - RSF Test IBS:            0.1356

[+] Processing Dataset: 'WHAS500'
    Running grid search hyperparameter tuning on validation set...
      Combo 1: {'n_estimators': 50, 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.3875
      Combo 2: {'n_estimators': 50, 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.3855
      Combo 3: {'n_estimators': 50, 'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.4084
      Combo 4: {'n_estimators': 50, 'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.4074
      Combo 5: {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.3924
      Combo 6: {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.4074
      Combo 7: {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.3964
      Combo 8: {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.4133
    [--->] Optimal Parameters: {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} with Val C-Index = 0.4133

    Permutation Feature Importances (VIMP):
     feature  importance (VIMP)
          hr           0.109218
         age           0.078846
       sysbp           0.066500
         bmi           0.059532
      diasbp           0.054306
age_x_gender           0.040180
       cvd_1           0.022192
    gender_1           0.012535
       afb_1           0.012384
   age_x_chf           0.009733
       chf_1           0.002992
       sho_1           0.001780

    Test C-Index:              0.5581 ± 0.0141
    Test Integrated Brier:     0.1457
    Test Time-dependent AUC:   {'auc_time_265.0': 0.5482, 'auc_time_522.0': 0.5775, 'auc_time_1010.0': 0.5812}
    5-Fold CV Mean C-Index:    0.5415 ± 0.0334
    5-Fold CV Mean IBS:        0.1505 ± 0.0188

    Baseline Comparison vs Cox PH:
    - Cox PH Test C-Index:     0.4685
    - RSF Test C-Index:        0.5581
    - Cox PH Test IBS:         0.1565
    - RSF Test IBS:            0.1457

[+] Processing Dataset: 'METABRIC'
    Running grid search hyperparameter tuning on validation set...
      Combo 1: {'n_estimators': 50, 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5191
      Combo 2: {'n_estimators': 50, 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5249
      Combo 3: {'n_estimators': 50, 'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5068
      Combo 4: {'n_estimators': 50, 'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.4931
      Combo 5: {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5407
      Combo 6: {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5446
      Combo 7: {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5108
      Combo 8: {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} -> Val C-Index = 0.5117
    [--->] Optimal Parameters: {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'bootstrap': True} with Val C-Index = 0.5446

    Permutation Feature Importances (VIMP):
             feature  importance (VIMP)
                 age           0.032318
         age_x_stage           0.022725
     log_lymph_nodes           0.021316
lymph_nodes_positive           0.019775
   PAM50Subtype_Her2           0.017601
        tumour_stage           0.013871
   hormone_therapy_1           0.012324
      chemotherapy_1           0.010721
   PAM50Subtype_LumA           0.006769
   PAM50Subtype_LumB           0.003377
 PAM50Subtype_Normal           0.001918

    Test C-Index:              0.5746 ± 0.0032
    Test Integrated Brier:     0.165
    Test Time-dependent AUC:   {'auc_time_41.8': 0.5566, 'auc_time_95.1': 0.5721, 'auc_time_172.6': 0.6324}
    5-Fold CV Mean C-Index:    0.5098 ± 0.0281
    5-Fold CV Mean IBS:        0.1689 ± 0.0076

    Baseline Comparison vs Cox PH:
    - Cox PH Test C-Index:     0.5129
    - RSF Test C-Index:        0.5746
    - Cox PH Test IBS:         0.1671
    - RSF Test IBS:            0.165

============================================================
RANDOM SURVIVAL FOREST EXECUTION COMPLETE! Results saved to /home/loverboy/Downloads/Personal/Bayesian COX/reports/tables/rsf_results.json
==========================================================================================================================================

(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$



(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/run_deepsurv.py
===========================================================================================

STARTING PHASE 7 — DEEPSURV (DEEP LEARNING SURVIVAL)
=====================================================

Resolved Configuration (source: config/model.yaml):
    hidden_dims:         [64, 32]
    l2_reg:              0.001
    max_iter:            300
    Grid search combos:  6
==========================

[+] Executing DeepSurv Pipeline on Dataset: 'GBSG2'
    Running validation set grid search hyperparameter tuning...
      Combo 1: {'hidden_dims': [16, 8], 'l2_reg': 0.0001} -> Val C-Index = 0.5075
      Combo 2: {'hidden_dims': [16, 8], 'l2_reg': 0.001} -> Val C-Index = 0.5392
      Combo 3: {'hidden_dims': [32, 16], 'l2_reg': 0.0001} -> Val C-Index = 0.5132
      Combo 4: {'hidden_dims': [32, 16], 'l2_reg': 0.001} -> Val C-Index = 0.5264
      Combo 5: {'hidden_dims': [32, 16], 'l2_reg': 0.01} -> Val C-Index = 0.4652
      Combo 6: {'hidden_dims': [64, 32], 'l2_reg': 0.001} -> Val C-Index = 0.5018
    [--->] Optimal Parameters: {'hidden_dims': [16, 8], 'l2_reg': 0.001} with Val C-Index = 0.5392

    Neural Network Architecture Summary:
                  layer    shape
            Input Layer (11, 16)
  Hidden Layer 1 (SELU)  (16, 8)
Output Layer (Log-Risk)   (8, 1)
  Total Loss Iterations      320

    Permutation Feature Importances (VIMP):
     feature  importance (VIMP)
       tsize           0.040478
      estrec           0.039176
   log_pnode           0.026105
 age_x_horTh           0.025224
  log_estrec           0.024229
     progrec           0.022545
       pnode           0.014698
   horTh_yes           0.010584
         age           0.008995
menostat_Pre           0.008038
 log_progrec           0.000000

    Test C-Index:              0.5411 ± 0.0103
    Test Integrated Brier:     0.1939
    5-Fold CV Mean C-Index:    0.4983 ± 0.0742
    5-Fold CV Mean IBS:        0.2469 ± 0.0472

    Multi-Model Comparative Test C-Index:
    - Cox PH Baseline:         0.482
    - Random Survival Forest:  0.5441
    - DeepSurv Neural Net:     0.5411

[+] Executing DeepSurv Pipeline on Dataset: 'WHAS500'
    Running validation set grid search hyperparameter tuning...
      Combo 1: {'hidden_dims': [16, 8], 'l2_reg': 0.0001} -> Val C-Index = 0.4452
      Combo 2: {'hidden_dims': [16, 8], 'l2_reg': 0.001} -> Val C-Index = 0.4532
      Combo 3: {'hidden_dims': [32, 16], 'l2_reg': 0.0001} -> Val C-Index = 0.3406
      Combo 4: {'hidden_dims': [32, 16], 'l2_reg': 0.001} -> Val C-Index = 0.4422
      Combo 5: {'hidden_dims': [32, 16], 'l2_reg': 0.01} -> Val C-Index = 0.3884
      Combo 6: {'hidden_dims': [64, 32], 'l2_reg': 0.001} -> Val C-Index = 0.3974
    [--->] Optimal Parameters: {'hidden_dims': [16, 8], 'l2_reg': 0.001} with Val C-Index = 0.4532

    Neural Network Architecture Summary:
                  layer    shape
            Input Layer (12, 16)
  Hidden Layer 1 (SELU)  (16, 8)
Output Layer (Log-Risk)   (8, 1)
  Total Loss Iterations      315

    Permutation Feature Importances (VIMP):
     feature  importance (VIMP)
       afb_1           0.028365
         bmi           0.018329
         age           0.007726
          hr           0.003636
       chf_1           0.000076
       sysbp           0.000000
      diasbp           0.000000
   age_x_chf           0.000000
    gender_1           0.000000
age_x_gender           0.000000
       cvd_1           0.000000
       sho_1           0.000000

    Test C-Index:              0.5492 ± 0.0141
    Test Integrated Brier:     0.24
    5-Fold CV Mean C-Index:    0.524 ± 0.0535
    5-Fold CV Mean IBS:        0.2712 ± 0.0395

    Multi-Model Comparative Test C-Index:
    - Cox PH Baseline:         0.4685
    - Random Survival Forest:  0.5581
    - DeepSurv Neural Net:     0.5492

[+] Executing DeepSurv Pipeline on Dataset: 'METABRIC'
    Running validation set grid search hyperparameter tuning...
      Combo 1: {'hidden_dims': [16, 8], 'l2_reg': 0.0001} -> Val C-Index = 0.4792
      Combo 2: {'hidden_dims': [16, 8], 'l2_reg': 0.001} -> Val C-Index = 0.5023
      Combo 3: {'hidden_dims': [32, 16], 'l2_reg': 0.0001} -> Val C-Index = 0.4937
      Combo 4: {'hidden_dims': [32, 16], 'l2_reg': 0.001} -> Val C-Index = 0.5029
      Combo 5: {'hidden_dims': [32, 16], 'l2_reg': 0.01} -> Val C-Index = 0.4980
      Combo 6: {'hidden_dims': [64, 32], 'l2_reg': 0.001} -> Val C-Index = 0.4969
    [--->] Optimal Parameters: {'hidden_dims': [32, 16], 'l2_reg': 0.001} with Val C-Index = 0.5029

    Neural Network Architecture Summary:
                  layer    shape
            Input Layer (11, 32)
  Hidden Layer 1 (SELU) (32, 16)
Output Layer (Log-Risk)  (16, 1)
  Total Loss Iterations      327

    Permutation Feature Importances (VIMP):
             feature  importance (VIMP)
     log_lymph_nodes           0.032152
         age_x_stage           0.018844
                 age           0.012736
   PAM50Subtype_LumB           0.009651
   hormone_therapy_1           0.008233
 PAM50Subtype_Normal           0.006892
   PAM50Subtype_LumA           0.005095
        tumour_stage           0.004887
      chemotherapy_1           0.001508
lymph_nodes_positive           0.000000
   PAM50Subtype_Her2           0.000000

    Test C-Index:              0.5493 ± 0.0032
    Test Integrated Brier:     0.2086
    5-Fold CV Mean C-Index:    0.4987 ± 0.0133
    5-Fold CV Mean IBS:        0.2188 ± 0.0099

    Multi-Model Comparative Test C-Index:
    - Cox PH Baseline:         0.5129
    - Random Survival Forest:  0.5746
    - DeepSurv Neural Net:     0.5493

============================================================
DEEPSURV EXECUTION COMPLETE! Results saved to /home/loverboy/Downloads/Personal/Bayesian COX/reports/tables/deepsurv_results.json
=================================================================================================================================

(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$



(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/run_bayesian_cox.py
STARTING PHASE 9 — BAYESIAN COX MODEL (ADVI)
  Resolved Configuration (source: config/model.yaml + CLI overrides):
    Inference Method:    advi
    Coefficient Prior:   normal (params:)
    Piecewise Intervals: 6
    Draws:               2000
    Tune:                1000
    Chains:              4
    ADVI Iterations:     10000
    Target Accept:       0.95
=============================

[+] Fitting Bayesian Cox Model on Dataset: 'GBSG2'
Finished [100%]: Average Loss = 3,017.8

    Posterior Hazard Ratio Summary (Top 5 Features):
    feature  exp(coef) HR Mean  95% Credible Lower  95% Credible Upper  Prob(HR > 1)
age_x_horTh           2.390709            1.522805            3.567327        1.0000
log_progrec           1.187156            0.753812            1.802906        0.7510
 log_estrec           1.085044            0.684090            1.608382        0.6110
  log_pnode           1.071498            0.660550            1.594813        0.5795
      tsize           0.988003            0.667613            1.381142        0.4430

    Test C-Index:              0.5017 ± 0.0103
    Test Integrated Brier:     0.2478
Finished [100%]: Average Loss = 1.5212e+05
Finished [100%]: Average Loss = 1.4142e+05
Finished [100%]: Average Loss = 33,034
Finished [100%]: Average Loss = 1.3435e+05
Finished [100%]: Average Loss = 1.3997e+05
    5-Fold CV Mean C-Index:    0.5035 ± 0.0582
    5-Fold CV Mean IBS:        0.4224 ± 0.0349

    Four-Model Comparative Test C-Index:
    1. Cox PH Baseline:        0.482
    2. Random Survival Forest: 0.5441
    3. DeepSurv Neural Net:    0.5411
    4. Bayesian Cox (PyMC):    0.5017

[+] Fitting Bayesian Cox Model on Dataset: 'WHAS500'
Finished [100%]: Average Loss = 1,145.6

    Posterior Hazard Ratio Summary (Top 5 Features):
     feature  exp(coef) HR Mean  95% Credible Lower  95% Credible Upper  Prob(HR > 1)
age_x_gender           2.788568            1.928269            3.903461        1.0000
   age_x_chf           1.818004            1.286777            2.502457        1.0000
          hr           1.125154            0.846559            1.458111        0.7780
       sysbp           1.065280            0.794379            1.377340        0.6600
         bmi           0.936565            0.697406            1.218811        0.3025

    Test C-Index:              0.4403 ± 0.0141
    Test Integrated Brier:     0.2332
Finished [100%]: Average Loss = 9,723.4
Finished [100%]: Average Loss = 10,047
Finished [100%]: Average Loss = 9,594.3
Finished [100%]: Average Loss = 10,548
Finished [100%]: Average Loss = 10,878
    5-Fold CV Mean C-Index:    0.4989 ± 0.0376
    5-Fold CV Mean IBS:        0.4419 ± 0.0454

    Four-Model Comparative Test C-Index:
    1. Cox PH Baseline:        0.4685
    2. Random Survival Forest: 0.5581
    3. DeepSurv Neural Net:    0.5492
    4. Bayesian Cox (PyMC):    0.4403

[+] Fitting Bayesian Cox Model on Dataset: 'METABRIC'
Finished [100%]: Average Loss = 3,146.4

    Posterior Hazard Ratio Summary (Top 5 Features):
             feature  exp(coef) HR Mean  95% Credible Lower  95% Credible Upper  Prob(HR > 1)
                 age           1.031975            0.736769            1.440679        0.5365
     log_lymph_nodes           1.021861            0.673186            1.470442        0.5115
         age_x_stage           0.998709            0.660721            1.442445        0.4390
        tumour_stage           0.994945            0.688214            1.377520        0.4460
lymph_nodes_positive           0.949405            0.615043            1.371192        0.3545

    Test C-Index:              0.5376 ± 0.0032
    Test Integrated Brier:     0.1785
Finished [100%]: Average Loss = 5,933.7
Finished [100%]: Average Loss = 5,842.2
Finished [100%]: Average Loss = 5,788.5
Finished [100%]: Average Loss = 4,895.7
Finished [100%]: Average Loss = 5,557.2
    5-Fold CV Mean C-Index:    0.4945 ± 0.0145
    5-Fold CV Mean IBS:        0.2125 ± 0.0055

    Four-Model Comparative Test C-Index:
    1. Cox PH Baseline:        0.5129
    2. Random Survival Forest: 0.5746
    3. DeepSurv Neural Net:    0.5493
    4. Bayesian Cox (PyMC):    0.5376

============================================================
BAYESIAN COX EXECUTION COMPLETE! Results saved to /home/loverboy/Downloads/Personal/Bayesian COX/reports/tables/bayesian_cox_results.json
=========================================================================================================================================

(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$



(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/evaluate.py
=======================================================================================

RUNNING UNIFIED MODEL COMPARISON & BENCHMARK SUITE
==================================================

[+] Successfully loaded results. Generating plots...
    - Comparison plot saved to /home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/model_performance_comparison.png
[+] Performing Ranking Analysis...

Overall Model Rankings (Average Rank across 3 Datasets):

* Cox PH Baseline           | C-Index Rank: 3.67 | IBS Rank: 2.00 | Overall Rank: 2.83
* Random Survival Forest    | C-Index Rank: 1.00 | IBS Rank: 1.00 | Overall Rank: 1.00
* DeepSurv Neural Net       | C-Index Rank: 2.00 | IBS Rank: 3.67 | Overall Rank: 2.83
* Bayesian Cox (ADVI)       | C-Index Rank: 3.33 | IBS Rank: 3.33 | Overall Rank: 3.33

[+] Running Statistical Significance Tests...

* Friedman test for C-Index: Chi2 = 2.7600, p-value = 4.3013e-01
* Friedman test for IBS:     Chi2 = 37.4800, p-value = 3.6418e-08
* Wilcoxon signed-rank test (Bayesian Cox (ADVI) vs Cox PH Baseline):
  - C-Index: p-value = 0.7197 (stat=53.0)
  - IBS:     p-value = 0.0001 (stat=0.0)
* Wilcoxon signed-rank test (Bayesian Cox (ADVI) vs Random Survival Forest):
  - C-Index: p-value = 0.8904 (stat=57.0)
  - IBS:     p-value = 0.0001 (stat=0.0)
* Wilcoxon signed-rank test (Bayesian Cox (ADVI) vs DeepSurv Neural Net):
  - C-Index: p-value = 0.6788 (stat=52.0)
  - IBS:     p-value = 0.0054 (stat=13.0)

[+] Saved comparative json/csv tables.
[+] Wrote model_comparison_report.md successfully.

============================================================
UNIFIED MODEL COMPARISON SUITE COMPLETE!
========================================

(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$



(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/run_statistical_validation.py
[+] STATISTICAL VALIDATION ON DATASET: GBSG2
  --> Fitting model: Cox PH
  --> Fitting model: RSF
  --> Fitting model: DeepSurv
  --> Fitting model: Bayesian Cox
Finished [100%]: Average Loss = 1.5724e+08

  [+] Executing 100 Bootstrap Replicates...
  [+] Computing Pairwise Significance and corrections...
  [+] Extracting Bayesian Cox Posterior Credible Intervals...
  [+] Running Model Stability trials (3 seeds)...
Finished [100%]: Average Loss = 1.5724e+08
Finished [100%]: Average Loss = 4.4889e+08
Finished [100%]: Average Loss = 1.8824e+08
  [+] Running Model Sensitivity trials (training sizes)...
Finished [100%]: Average Loss = 1.5615e+08
Finished [100%]: Average Loss = 1.4865e+08
[+] STATISTICAL VALIDATION ON DATASET: WHAS500
  --> Fitting model: Cox PH
  --> Fitting model: RSF
  --> Fitting model: DeepSurv
  --> Fitting model: Bayesian Cox
Finished [100%]: Average Loss = 2.8984e+05

  [+] Executing 100 Bootstrap Replicates...
  [+] Computing Pairwise Significance and corrections...
  [+] Extracting Bayesian Cox Posterior Credible Intervals...
  [+] Running Model Stability trials (3 seeds)...
Finished [100%]: Average Loss = 2.8984e+05
Finished [100%]: Average Loss = 2.9069e+05
Finished [100%]: Average Loss = 3.7073e+05
  [+] Running Model Sensitivity trials (training sizes)...
Finished [100%]: Average Loss = 1.4263e+05
Finished [100%]: Average Loss = 2.2173e+05
[+] STATISTICAL VALIDATION ON DATASET: METABRIC
  --> Fitting model: Cox PH
  --> Fitting model: RSF
  --> Fitting model: DeepSurv
  --> Fitting model: Bayesian Cox
Finished [100%]: Average Loss = 5.6411e+08

  [+] Executing 100 Bootstrap Replicates...
  [+] Computing Pairwise Significance and corrections...
  [+] Extracting Bayesian Cox Posterior Credible Intervals...
  [+] Running Model Stability trials (3 seeds)...
Finished [100%]: Average Loss = 5.6411e+08
Finished [100%]: Average Loss = 6.8359e+07
Finished [100%]: Average Loss = 2.1804e+08
  [+] Running Model Sensitivity trials (training sizes)...
Finished [100%]: Average Loss = 2.1251e+05
Finished [100%]: Average Loss = 2.6793e+05

[+] Statistical validation results saved to /home/loverboy/Downloads/Personal/Bayesian COX/reports/tables/statistical_validation.json
[+] Statistical validation markdown report saved to /home/loverboy/Downloads/Personal/Bayesian COX/reports/statistical_validation_report.md
(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$


(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/explain.py
RUNNING EXPLAINABILITY & FEATURE IMPORTANCE SYNTHESIS
[+] Successfully generated reports/explainability_report.md
(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$



(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python scripts/run_extensions.py
=============================================================================================

STARTING PHASE 12 — ADVANCED RESEARCH EXTENSIONS
=================================================

============================================================
RUNNING STEP 12.1 — ABLATION STUDY (PRIOR DISTRIBUTIONS)
=========================================================

  Training Bayesian Cox Model with 'normal' prior...
Finished [100%]: Average Loss = 1.5724e+08
    -> C-index: 0.5248, IBS: 0.4217
  Training Bayesian Cox Model with 'student-t' prior...
Finished [100%]: Average Loss = 1.5724e+08
    -> C-index: 0.5248, IBS: 0.4217
  Training Bayesian Cox Model with 'laplace' prior...
Finished [100%]: Average Loss = 1.5724e+08
    -> C-index: 0.5248, IBS: 0.4217

============================================================
RUNNING STEP 12.4 — CENSORING ROBUSTNESS ANALYSIS
==================================================

  Simulating training set with target censoring rate = 60.0%...
    Training Cox PH...
    Training RSF...
    Training DeepSurv...
    Training Bayesian Cox...
Finished [100%]: Average Loss = 3.3424e+07
    -> Results (C-index) at 60.0% Censoring: Cox PH=0.4782, RSF=0.5034, DeepSurv=0.5351, Bayesian Cox=0.5244
  Simulating training set with target censoring rate = 75.0%...
    Training Cox PH...
    Training RSF...
    Training DeepSurv...
    Training Bayesian Cox...
Finished [100%]: Average Loss = 3.2898e+07
    -> Results (C-index) at 75.0% Censoring: Cox PH=0.4876, RSF=0.5103, DeepSurv=0.5398, Bayesian Cox=0.5244
  Simulating training set with target censoring rate = 90.0%...
    Training Cox PH...
    Training RSF...
    Training DeepSurv...
    Training Bayesian Cox...
Finished [100%]: Average Loss = 3.2825e+07
    -> Results (C-index) at 90.0% Censoring: Cox PH=0.5805, RSF=0.5565, DeepSurv=0.5938, Bayesian Cox=0.5235
  [+] Saved line plot to: /home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/extension_censoring_robustness.png

============================================================
RUNNING STEP 12.5 — SMALL DATASET ROBUSTNESS ANALYSIS
======================================================

  Training with 20.0% of GBSG2 training dataset...
Finished [100%]: Average Loss = 3.5224e+05
    -> Results (C-index) at 20.0% training size: Cox PH=0.4452, RSF=0.4486, DeepSurv=0.3926, Bayesian Cox=0.5368
  Training with 40.0% of GBSG2 training dataset...
Finished [100%]: Average Loss = 4.9339e+05
    -> Results (C-index) at 40.0% training size: Cox PH=0.4786, RSF=0.4876, DeepSurv=0.4050, Bayesian Cox=0.5492
  Training with 60.0% of GBSG2 training dataset...
Finished [100%]: Average Loss = 1.0176e+06
    -> Results (C-index) at 60.0% training size: Cox PH=0.4435, RSF=0.4756, DeepSurv=0.4409, Bayesian Cox=0.5394
  Training with 80.0% of GBSG2 training dataset...
Finished [100%]: Average Loss = 1.4207e+06
    -> Results (C-index) at 80.0% training size: Cox PH=0.4756, RSF=0.4722, DeepSurv=0.4824, Bayesian Cox=0.5415
  Training with 100.0% of GBSG2 training dataset...
Finished [100%]: Average Loss = 3.3341e+07
    -> Results (C-index) at 100.0% training size: Cox PH=0.4820, RSF=0.5188, DeepSurv=0.5634, Bayesian Cox=0.5244
  [+] Saved line plot to: /home/loverboy/Downloads/Personal/Bayesian COX/reports/figures/extension_small_dataset_analysis.png

============================================================
PHASE 12 EXTENSIONS COMPLETED SUCCESSFULLY
==========================================

(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$



(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$ python tests/run_tests.py
======================================================================================

RUNNING CUSTOM TEST DISCOVERY & EXECUTION
=========================================

[+] Scanning module: tests.test_data
  No test functions found.

[+] Scanning module: tests.test_metrics
  No test functions found.

[+] Scanning module: tests.test_models
  Running test_bayesian_cox_model... Finished [100%]: Average Loss = 462.46
Only 50 samples per chain. Reliable r-hat and ESS diagnostics require longer chains for accurate estimate.
Initializing NUTS using jitter+adapt_diag...
Sequential sampling (2 chains in 1 job)
NUTS: [beta, log_lambda]
Sampling 2 chains for 50 tune and 50 draw iterations (100 + 100 draws total) took 12 seconds.
The number of samples is too small to check convergence reliably.
PASSED
  Running test_deepsurv_model... PASSED
  Running test_rsf_model... PASSED
  Running test_survival_tree_basic... PASSED

[+] Scanning module: tests.test_pipeline
  No test functions found.

[+] Scanning module: tests.test_training
  No test functions found.
Test Summary: 4 run, 0 failed.
(.venv) loverboy@Loverboy:~/Downloads/Personal/Bayesian COX$
