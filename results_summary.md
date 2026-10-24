
## Step 1 — source LLM fine-tuning (mean ± std over seeds)

| dataset | AUC | F1 | ACC | n |
|---|---|---|---|---|
| Cardiovascular-Disease-dataset | 0.800 ± 0.003 | 0.729 ± 0.007 | 0.731 ± 0.005 | 10 |
| car | 0.998 ± 0.003 | 0.975 ± 0.017 | 0.975 ± 0.017 | 10 |
| churn | 0.894 ± 0.015 | 0.892 ± 0.011 | 0.899 ± 0.009 | 10 |
| credit-g | 0.743 ± 0.053 | 0.711 ± 0.040 | 0.724 ± 0.044 | 9 |
| mfeat-fourier | 0.958 ± 0.005 | 0.687 ± 0.030 | 0.697 ± 0.023 | 10 |
| optdigits | 0.998 ± 0.001 | 0.941 ± 0.015 | 0.941 ± 0.015 | 10 |
| seismic_bumps | 0.764 ± 0.031 | 0.902 ± 0.001 | 0.932 ± 0.004 | 10 |
| sick | 0.976 ± 0.027 | 0.973 ± 0.016 | 0.975 ± 0.011 | 10 |
| steel-plate-fault | 1.000 ± 0.000 | 0.972 ± 0.017 | 0.972 ± 0.016 | 10 |

## Step 3 — LATTLE transfer (mean ± std over folds)

| pair | AUC | F1 | ACC | n |
|---|---|---|---|---|
| Cardiovascular-Disease-dataset_Diabetes | 0.802 ± 0.062 | 0.600 ± 0.093 | 0.736 ± 0.052 | 10 |
| car_pc1 | 0.842 ± 0.051 | 0.287 ± 0.105 | 0.928 ± 0.011 | 10 |
| churn_cmc | 0.710 ± 0.026 | 0.505 ± 0.033 | 0.527 ± 0.035 | 10 |
| credit-g_Diabetes | 0.790 ± 0.063 | 0.600 ± 0.072 | 0.732 ± 0.047 | 10 |
| mfeat-fourier_vehicle | 0.952 ± 0.009 | 0.808 ± 0.027 | 0.808 ± 0.026 | 10 |
| optdigits_vehicle | 0.946 ± 0.009 | 0.792 ± 0.027 | 0.794 ± 0.025 | 10 |
| seismic_bumps_cylinder-bands | 0.836 ± 0.029 | 0.807 ± 0.036 | 0.775 ± 0.040 | 10 |
| sick_cmc | 0.717 ± 0.028 | 0.512 ± 0.029 | 0.533 ± 0.035 | 10 |
| steel-plate-fault_cylinder-bands | 0.840 ± 0.032 | 0.805 ± 0.029 | 0.772 ± 0.033 | 10 |
| steel-plate-fault_pc1 | 0.822 ± 0.066 | 0.291 ± 0.115 | 0.928 ± 0.010 | 10 |

Overall mean transfer AUC: 0.826 over 10 pairs / 100 runs
