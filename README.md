# Revealing-Subject-specific-Temporal-Patterns-from-Longitudinal-Data

This repository contains the official code for the paper.

### Data Availability
Individual-level clinical data from the COPSAC<sub>2010</sub> cohort are not publicly available to protect participant privacy, in accordance with the Danish Data Protection Act and European Regulation 2016/679 of the European Parliament and of the Council (GDPR) that prohibit distribution even in pseudo-anonymized form. Data can be made available under a joint research collaboration by contacting COPSAC’s data protection officer (administration@dbac.dk). 

### Important note

> ⚠️ We are utilizing the latest version of [MatCoupLy](https://github.com/MarieRoald/matcouply) with the initial imputation from factors disabled.
> To have the exact same toolbox as us, download the latest version of MatCoupLy and comment the following lines in `decomposition.py`:
>
> ```
> 943    if mask is not None:
> 944        if init == "random" or init == "svd" or init == "threshold_svd":
> 945            matrices = _update_imputed(tensor_slices=list(matrices), mask=mask, decomposition=None, method="mode-3")
> 946         else:  # If factors are provided from a "warmer" start (e.g. parafac2_als) use the factor estimates as initial guesses
> 947             matrices = _update_imputed(tensor_slices=list(matrices), mask=mask, decomposition=cmf, method="factors")
> ```
> The reason behind this is that initially imputing entries in a normalized dataset will result in entries higher in magnitude, which will affect the results. The [issue is raised](https://github.com/MarieRoald/matcouply/issues/12) and a pull request with a fix is planned.

### Analysis outline

The notebooks `metabolomics.ipynb` and `sensitization.ipynb` outlines the analysis of this work in the metabolomics and sensitization datasets, respectively.

### Reproducing the paper figures

**Factor plots:** The way we compute and plot factors is described in `metabolomics.ipynb` and `sensitization.ipynb`. In our case, the best performing run from the reproduciblity check is plotted, which we have supplied in the results folder. To reproduce the plot using the our *pre-fit* models you may use `paper_plots.ipynb`.

**Reproducibility results:** The script `uniqueness_check.py <model> <r> 0 <dataset> <gender>` runs the reproducibility check for the given model, dataset, rank and gender (only applicable in metabolomics). The output of our runs is stored in `Metabolomics/results/uniqueness/` and `Sensitization/results/uniqueness/`. To reproduce the plots of the paper from our runs you can use `paper_plots.ipynb`.

**Replicability results:** The script `replicability_check.py <model> <r> 0 <dataset> <gender> <splits>` runs the replicability check for the given model, dataset, rank and gender (only applicable in metabolomics). The output of our runs is stored in `Metabolomics/results/replicability/` and `Sensitization/results/replicability/`. To reproduce the plots of the paper from our runs you can use `paper_plots.ipynb`.

### Directory Structure

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
│   └── df_CB_replic_females.pkl
├── Sensitization/
│   └── results/
│   │   └── uniqueness/        # Contains intermediate uniqueness-related results 
│   │   └── replicability/        # Contains intermediate replicability-related results  
│   ├── df_A_uniq.pkl
│   ├── df_CB_uniq.pkl
│   ├── df_A_replic.pkl
│   └── df_CB_replic.pkl
├── metabolomics.ipynb # outline of the metabolomics data analysis
├── sensitization.ipynb # outline of the sensitization data analysis
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
