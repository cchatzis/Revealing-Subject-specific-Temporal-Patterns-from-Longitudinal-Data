# Revealing-Subject-specific-Temporal-Patterns-from-Longitudinal-Data

This repository contains the official code for the paper:
---------------
This repository contains data and code to reproduce the findings and figures presented in the manuscript.

### Data Availability
Individual-level clinical data from the COPSAC<sub>2010</sub> cohort are not publicly available to protect participant privacy, in accordance with the Danish Data Protection Act and European Regulation 2016/679 of the European Parliament and of the Council (GDPR) that prohibit distribution even in pseudo-anonymized form. Data can be made available under a joint research collaboration by contacting COPSAC’s data protection officer (administration@dbac.dk). 

### Important note

> ⚠️ We are utilizing the latest version of [MatCoupLy](https://github.com/MarieRoald/matcouply) with the initial imputation from factors disabled.

### Reproducing the reproducibility results

> ⚠️ This step can take several hours on CPU.

The reproducibility experiments in the paper can be re-run as follows:

1. Pre-process the data
2. Run `uniqueness_check.py <model> <r> <l_B> <dataset> <gender>`, where: `<model>` is chosen from `cmf, parafac2` `cp`, `<r>` denotes the number of componets, `<l_B>` can be used to optionally include ridge penalty on all factors, `<dataset>` is one of `-m` and `-s`, with `-m` indicating the metabolomics dataset while `-s` the sensitization, and lastly `<gender>` indicates the gender of the subjects to use in the analysis. The `<gender>` argument has no effect when runing with `-s`. Running this will produce the file `<dataset>/results/uniqueness/factors_<model>_<r>_components_l_B_<l_B>_<gender>.pkl`, which is a pickle object containt a list of tuples in the format `(factors, loss)`.
3. Run `uniquenes_analysis_<dataset>.ipynb` to obtain an overview of the reproducibility results for all models. This notebook produces `<dataset>/df_A_uniq_<gender>.pkl` and `<dataset>/df_CB_uniq_<gender>.pkl` which contain the FMS_A and FMS_C*B values to plot in the boxplots of the paper.
4. Run the respective cells in `paper_plots.ipynb` to reproduce the exact figure of the paper.

### Reproducing the replicability results

> ⚠️ This step can take several hours on CPU.

The reprlicability experiments in the paper can be re-run as follows:

1. Pre-process the data
2. Run `replicability_check.py <model> <r> <l_B> <dataset> <gender>`, where: `<model>` is chosen from `cmf, parafac2` `cp`, `<r>` denotes the number of componets, `<l_B>` can be used to optionally include ridge penalty on all factors, `<dataset>` is one of `-m` and `-s`, with `-m` indicating the metabolomics dataset while `-s` the sensitization, and lastly `<gender>` indicates the gender of the subjects to use in the analysis. The `<gender>` argument has no effect when runing with `-s`. Running this will produce the file `<dataset>/results/uniqueness/factors_<model>_<r>_components_l_B_<l_B>_<gender>.pkl`, which is a pickle object containt a list of tuples in the format `(factors, loss)`.
3. Run `replicability_analysis_<dataset>.ipynb` to obtain an overview of the reproducibility results for all models. This notebook produces `<dataset>/df_A_replic_<gender>.pkl` and `<dataset>/df_CB_replic_<gender>.pkl` which contain the FMS_A and FMS_C*B values to plot in the boxplots of the paper.
4. Run the respective cells in `paper_plots.ipynb` to reproduce the exact figure of the paper.


### Reproducing the paper plots

This can be done by running the appropriate cells from `paper_plots.ipynb`.


## Directory Structure

```
.
├── Metabolomics/
│   ├── results/
│   │   └── uniqueness/         # Contains intermediate uniqueness-related results 
│   │   └── replicability/        # Contains intermediate replicability-related results 
│   ├── df_A_uniq_males.pkl
│   ├── df_CB_uniq_males.pkl
│   ├── df_A_uniq_females.pkl
│   ├── df_CB_uniq_females.pkl
│   ├── df_A_replic_males.pkl
│   ├── df_CB_replic_males.pkl
│   ├── df_A_replic_females.pkl
│   ├── df_CB_replic_females.pkl
├── Sensitization/
│   └── results/
│   │   └── uniqueness/        # Contains intermediate uniqueness-related results 
│   │   └── replicability/        # Contains intermediate replicability-related results  
│   ├── df_A_uniq.pkl
│   ├── df_CB_uniq.pkl
│   ├── df_A_replic.pkl
│   ├── df_CB_replic.pkl
├── uniqueness_check.py
├── replicability_check.py
├── uniqueness_analysis_metabolomics.py
├── uniqueness_analysis_sensitization.py
├── replicability_analysis_metabolomics.py
├── replicability_analysis_sensitization.py
├── paper_plots.ipynb            # reproduces the plots in the paper
├── plot_funcs.py            # utility functions supporting paper_plots.py
├── requirements.txt            # requirements for python virtual env
```

## Citation

<!-- @inproceedings{your_key_2026,
  title        = {Revealing Subject-specific Temporal Patterns from Longitudinal Data},
  author       = {Christos Chatzis, David Horner, Rasmus Bro, Ann-Marie Malby Schoos, Morten A. Rasmussen and Evrim Acar},
  booktitle    = {Venue Name},
  year         = {2026},
  url          = {https://arxiv.org/abs/xxxx.xxxxx}
} -->