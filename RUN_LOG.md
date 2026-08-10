# Project Run Log

This file contains the commands and their outputs for running the Bayesian COX project.

## Environment

- Python version: `python3.10`
- Virtual environment: `.venv`
- Date: $(date)

## Steps

### 1. Download Datasets

**Command**:
```bash
source .venv/bin/activate && python scripts/download_datasets.py
```

**Output**:
```
Exporting GBSG2 dataset to /home/loverboy/Downloads/Personal/Bayesian COX/data/raw/gbsg2.csv...
Successfully exported GBSG2 dataset (686 records) from sksurv.
Exporting WHAS500 dataset to /home/loverboy/Downloads/Personal/Bayesian COX/data/raw/whas500.csv...
Successfully exported WHAS500 dataset (500 records) from sksurv.
Exporting METABRIC dataset to /home/loverboy/Downloads/Personal/Bayesian COX/data/raw/metabric.csv...
METABRIC dataset already exists at /home/loverboy/Downloads/Personal/Bayesian COX/data/raw/metabric.csv.

Dataset preparation completed successfully!
```


### 2. Prepare Datasets

**Command**:
```bash
source .venv/bin/activate && python scripts/prepare_dataset.py
```

**Output**:
```
============================================================
STARTING PHASE 4 — DATA PREPROCESSING & DATASET PREPARATION
============================================================

[+] Processing dataset: 'GBSG2'...
    Raw Shape: 686 rows, 10 columns
Traceback (most recent call last):
  File "/home/loverboy/Downloads/Personal/Bayesian COX/scripts/prepare_dataset.py", line 91, in <module>
    main()
  File "/home/loverboy/Downloads/Personal/Bayesian COX/scripts/prepare_dataset.py", line 69, in main
    metadata = pipeline.run(raw_df, output_dir=DATA_DIR)
  File "/home/loverboy/Downloads/Personal/Bayesian COX/src/data/preprocessing.py", line 99, in run
    X_train_proc, X_val_proc, X_test_proc = self._fit_transform_features(
  File "/home/loverboy/Downloads/Personal/Bayesian COX/src/data/preprocessing.py", line 281, in _fit_transform_features
    mean_v = float(df_train_enc[c_col].mean())
  File "/home/loverboy/Downloads/Personal/Bayesian COX/.venv/lib64/python3.10/site-packages/pandas/core/series.py", line 6570, in mean
    return NDFrame.mean(self, axis, skipna, numeric_only, **kwargs)
  File "/home/loverboy/Downloads/Personal/Bayesian COX/.venv/lib64/python3.10/site-packages/pandas/core/generic.py", line 12485, in mean
    return self._stat_function(
  File "/home/loverboy/Downloads/Personal/Bayesian COX/.venv/lib64/python3.10/site-packages/pandas/core/generic.py", line 12442, in _stat_function
    return self._reduce(
  File "/home/loverboy/Downloads/Personal/Bayesian COX/.venv/lib64/python3.10/site-packages/pandas/core/series.py", line 6478, in _reduce
    return op(delegate, skipna=skipna, **kwds)
  File "/home/loverboy/Downloads/Personal/Bayesian COX/.venv/lib64/python3.10/site-packages/pandas/core/nanops.py", line 147, in f
    result = alt(values, axis=axis, skipna=skipna, **kwds)
  File "/home/loverboy/Downloads/Personal/Bayesian COX/.venv/lib64/python3.10/site-packages/pandas/core/nanops.py", line 404, in new_func
    result = func(values, axis=axis, skipna=skipna, **kwargs)
  File "/home/loverboy/Downloads/Personal/Bayesian COX/.venv/lib64/python3.10/site-packages/pandas/core/nanops.py", line 720, in nanmean
    the_sum = _ensure_numeric(the_sum)
  File "/home/loverboy/Downloads/Personal/Bayesian COX/.venv/lib64/python3.10/site-packages/pandas/core/nanops.py", line 1701, in _ensure_numeric
    raise TypeError(f"Could not convert string '{x}' to numeric")
TypeError: Could not convert string 'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
```

### 3. Run Cox PH Model

**Command**:
```bash
source .venv/bin/activate && python scripts/run_cox_ph.py
```

**Output**:
```
============================================================
STARTING PHASE 6 — COX PROPORTIONAL HAZARDS MODEL (BASELINE)
============================================================
  Resolved Configuration (source: config/model.yaml):
    L2 Regularization:   0.0001
============================================================

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
        tumour_stage    -0.049648      0.951564  0.109750  0.109750      1.179938 -0.452375 0.650999
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
============================================================
```


### 4. Run Random Survival Forest
