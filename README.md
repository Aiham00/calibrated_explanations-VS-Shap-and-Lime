# calibrated_explanations-VS-Shap-and-Lime
## Stability and Robustness for Multi-Class Classification

This repository contains the code for evaluating the **stability** and **robustness** of the [Calibrated Explanations](https://github.com/msolling/calibrated-explanations) framework in multi‑class classification settings.  
The experiments compare Calibrated Explanations (factual and counterfactual) against LIME and SHAP, both with and without Venn‑Abers calibration, across several datasets and models.

## Overview

- **Robustness** – How explanations vary when the underlying model is re‑fitted on different training/calibration splits (different random seeds).
- **Stability** – How explanations vary when only the No random seed (model fixed).

The code is organised as follows:

| File | Description |
|------|-------------|
| `Multi_Class_Experiments_rob.py` | Runs robustness experiments (varies model random seed). |
| `Multi_Class_Experiments_stab.py` | Runs stability experiments (varies explainer random seed). |
| `gru.py` | Defines the GRU+Attention model used as one of the classifiers. |
| `Multi_Classification_Analysis.ipynb` | Jupyter notebook to load results, compute metrics, and generate tables. |

## Datasets

The experiments use four publicly available datasets loaded via a `data_handler` included in this repository:

- **CIA** – (details as loaded by `data_handler.load_CIA()`)
- **AI4I** – (details as loaded by `data_handler.load_ai4i()`)
- **Azure** – (details as loaded by `data_handler.load_azure()`)
- **Pump** – (details as loaded by `data_handler.load_pump()`)

These datasets represent a mix of multi‑class classification tasks with varying numbers of features and instances.

## Models

Four classifiers are evaluated:

- **XGBoost** (`xgb.XGBClassifier`)
- **Random Forest** (`sklearn.ensemble.RandomForestClassifier`)
- **Decision Tree** (`sklearn.tree.DecisionTreeClassifier`)
- **GRU+Attention** – a custom PyTorch model (defined in `gru.py`) that treats each instance as a sequence of length 1.

## Explanation Methods

- **Calibrated Explanations (CE)** – factual explanations (feature weights for the predicted class) and counterfactual (alternative) explanations.
- **LIME** – from the `lime` library.
- **SHAP** – from the `shap` library.

Each explanation method is applied to both the **uncalibrated** model and the **Venn‑Abers calibrated** model (using the `venn_abers` package).  
For Calibrated Explanations, the calibration is already incorporated into the `CalibratedExplainer` class.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aiham00/calibrated_explanations-VS-Shap-and-Lime.git
   cd calibrated_explanations-VS-Shap-and-Lime

2. Install the required packages:
----------------------------------------
- numpy                     : 2.1.3
- scipy                     : 1.15.1
- pandas                    : 2.2.3
- scikit-learn              : 1.6.1
- xgboost                   : 3.0.0
- torch                     : 2.6.0+cpu
- torchvision               : 0.21.0+cpu
- lime                      : 0.2.0.1
- shap                      : 0.46.0
- calibrated-explanations   : v0.11.0-dev
- venn-abers                : 1.4.6
- jupyter-notebook          : 7.5.5

3. Running the experiments by runing all cells in jupyter notebook Multi_Classification_Analysis.ipynb
