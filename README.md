
Neural correlates of descriptive and responsive speech in post-stroke aphasia
=============================================================================

## Introduction

The goal of this project was to explore the neural constructs of descriptive and responsive speech 
in post-stroke aphasia, by performing univariate Voxel-wise Lesion Symptom Mapping (VLSM) - corrected
for multiple comparisons - on four discourse constructs. 

Overall, the pipeline consists of four steps:
1.  Plotting the lesion overlap between subjects
2.  Correcting univariate VLSM output for multiple comparisons (using permutation testing)
3.  Plotting the corrected VLSM output brain clusters
4.  Determining the location and distribution of the VLSM output brain clusters.

These steps can be applied to other independent data. Note that the pipeline builds on univariate VLSM output 
generated using NiiStat software implemented in MATLAB (http://www.nitrc.org/projects/niistat). 
This pipeline specifically corrects NiiStat VLSM output for multiple comparisons 
using a Permutation Testing technique based on the methodology of Stark et al. (2019), and allows
to plot the VLSM output results and to localize the output in the brain.

Stark, B. C., Basilakos, A., Hickok, G., Rorden, C., Bonilha, L., & Fridriksson, J. (2019). Neural organization of speech production: A lesion-based study of error patterns in connected speech. _Cortex_, 117, 228–246. https://doi.org/10.1016/j.cortex.2019.02.029


## Installation

To install the latest code:
```
pip install git+https://github.com/EllaEycken/Barberis_Eycken_VLSM_corrected_repo.git
```
Requirements can be installed from the requirements.txt file doing something like this:
```
conda create --name neural-correlates python=3.11
pip install -r requirements.txt
```

## Reproduction of Results
### 1.  Plotting the lesion overlap between subjects
The main script related to plotting the lesion overlap is the ```ella_phd_vlsm_project/ella_phd_vlsm_code/vlsm_analysis_dec24/plot_lesion_overlap.py``` script.
### 2.  Correcting univariate VLSM output for multiple comparisons (using permutation testing)
The main script related to VLSM correction is  ```ella_phd_vlsm_project/ella_phd_vlsm_code/vlsm_analysis_dec24/calculate_and_plot_corrected_VLSM.py``` script.
### 3.  Plotting the corrected VLSM output brain clusters
The main script related to VLSM plotting is  ```ella_phd_vlsm_project/ella_phd_vlsm_code/vlsm_analysis_dec24/plot_VLSM_output.py``` script.
### 4.  Determining the location and distribution of the VLSM output brain clusters.
The main script related to localization and distribution of VLSM output clusters is  ```ella_phd_vlsm_project/ella_phd_vlsm_code/vlsm_analysis_dec24/calculate_cluster_distribution.py``` script.




## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data               Note: This pipeline doesn't store data here, but rather uses absolute paths to organizational drives.
                       This is of course not obligatory.
│   ├── external       <- Data from third party sources. 
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ella_phd_vlsm_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ella_phd_vlsm_project   <- **Source code for use in this project.**
    │
    ├── __init__.py             <- Makes ella_phd_vlsm_project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── vlsm_analysis_dec24                
    │   ├── __init__.py 
    │   ├── calculate_and_plot_corrected_VLSM.py 
                                <- Code to calculate and plot VLSM output,
                                corrected for multiple comparisons (using permutation testing)       
    │   └── calculate_cluster_distribution.py
                                <- Code to calculate the location and distribution of a VLSM
                                output cluster.
    │   └── plot_lesion_overlap.py
                                <- Code to plot the lesion overlap between subjects.
    │   └── plot_VLSM_output.py <- Code to plot the output of VLSM clusters.
    └── plots.py                <- Code to create visualizations
```

--------

