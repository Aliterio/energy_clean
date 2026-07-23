# Appliance Energy Prediction

R pipeline accompanying the manuscript *"Time-Series-Aware Validation and
Interpretable Machine Learning for Household Appliance Energy Prediction"*
(Bahmanyar, 2026).

## Data
Dataset: Candanedo, Feldheim & Deramaix (2017), *Energy and Buildings*, 140, 81–97.
Source: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

## Pipeline (`appliance_energy_pipeline.R`)
1. Data cleaning and feature engineering (lag/rolling features, cyclical time encoding)
2. Exploratory data analysis (5 figures)
3. Six regression models under a time-ordered 70/15/15% train/val/test split:
   OLS, ridge, lasso, random forest, gradient boosting, SVM
4. SHAP interpretability analysis on the winning model
5. Residual diagnostics (ACF, Durbin-Watson, Ljung-Box, Shapiro-Wilk)

## Requirements
See package list at the top of the script. `fastshap` must be installed from
GitHub: `remotes::install_github("bgreenwell/fastshap")`.

## Usage
Place `energydata_complete.csv` in a `data/` folder, then run the script from
the repository root.
