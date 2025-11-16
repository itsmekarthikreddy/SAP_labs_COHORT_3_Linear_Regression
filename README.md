# SAP LABS COHORT_III Karthik_Reddy
# Bike Sharing Demand — BoomBikes (Linear Regression Project)

## Project overview
This project builds a multiple linear regression model to predict daily bike demand (cnt) for the BoomBikes dataset. The goal is to understand which factors influence demand and to provide a simple, interpretable model the business can use to plan resources and marketing after lockdowns end.

This repository contains a Jupyter notebook that walk through data loading, cleaning, feature engineering, model building (OLS), diagnostics and evaluation.

## Table of contents
- [Dataset](#dataset)  
- [Objective](#objective)  
- [Approach](#approach)  
- [Data preparation steps](#data-preparation-steps)  
- [Modeling and diagnostics](#modeling-and-diagnostics)  
- [Results to report](#results-to-report)  
- [How to run](#how-to-run)  
- [Dependencies](#dependencies)  
- [Acknowledgements & notes](#acknowledgements--notes)  
- [Contact](#contact)

## Dataset
The project uses the public Bike Sharing dataset (day-level) provided for the assignment. Key columns used:
- cnt — total daily bike rentals (target)
- casual, registered — components of cnt (dropped to avoid leakage)
- temp, atemp, hum, windspeed — numeric weather features
- season, weathersit, mnth, weekday, yr, holiday, workingday — categorical features

Refer to the dataset data dictionary for the exact meaning and coding of categorical variables.

## Objective
- Train an interpretable multiple linear regression model to predict daily bike demand (cnt).
- Identify which variables are statistically significant predictors.
- Evaluate model quality using R² on the held-out test set and perform residual diagnostics.

## Approach
- Convert categorical integer codes to descriptive strings (season, weathersit, month, weekday).
- Remove leakage: drop `casual` and `registered` because they sum to the target `cnt`.
- One-hot encode categorical variables with a reference level (drop_first) to avoid perfect multicollinearity.
- Split the data into train/test before any scaling (prevent data leakage).
- Scale numeric features using MinMaxScaler fit only on the training set.
- Fit OLS (statsmodels) on training data, perform iterative feature elimination based on p-values, and check multicollinearity using VIF.
- Evaluate final model on the test set using sklearn.r2_score and perform residual analysis.

## Data preparation steps (high level)
1. Load day.csv and remove columns: `instant`, `dteday`, `casual`, `registered`.
2. Map integer codes to category labels for `season`, `weathersit`, `mnth`, `weekday`.
3. Group rare weather categories (if present) to avoid tiny groups.
4. One-hot encode categorical columns (drop_first=True).
5. Split into X / y and train/test (70/30).
6. Fit MinMaxScaler on X_train numeric columns and transform X_test.
7. Ensure all features are numeric before calling statsmodels OLS.

## Modeling and diagnostics
- Initial OLS is fitted with all features (with constant).
- Remove features with high p-values iteratively (safe-drop checks included).
- Monitor multicollinearity with Variance Inflation Factor (VIF); drop highly collinear features (example: temp dropped because of atemp).
- After final model selection, show:
  - lr7.summary() (coefficients, p-values, R², adj. R²)
  - VIF table for final features
  - Residual plots (histogram, residual vs predicted, Q–Q plot)

## Results to report
Numeric results from the final run (fill verified from the notebook output):

- Train R² (final model on training set): 0.838  
- Train Adjusted R²: 0.830  
- Test R² (r2_score on held-out test set): 0.8302

Statistically significant predictors (p < 0.05) and coefficient direction (final model):
- Positive effect:
  - yr (positive; ~ +2019) — strong year-on-year increase in demand
  - atemp (positive; large coef) — higher apparent temperature increases rentals
  - workingday (positive) — modest increase on working days after controls
- Negative effect:
  - hum (negative) — higher humidity reduces demand
  - windspeed (negative) — higher wind reduces demand
  - season_spring (negative) — lower baseline in spring vs reference
  - mnth_Month_7, mnth_Month_8 (negative) — mid-summer months show reductions relative to baseline in this model specification
  - weathersit_Light_Rain_Snow (negative; ~ -2087) — large drop in demand under rain/snow
  - weathersit_Mist_Cloudy (negative; ~ -480) — reduced demand in misty/cloudy conditions

Top predictors by coefficient magnitude (for management attention):
1. atemp (positive) — largest meteorological driver of demand  
2. yr (positive) — strong annual growth effect captured by the model  
3. weathersit_Light_Rain_Snow (negative) — weather penalty with the largest single negative impact

Short interpretation / business implications
- The model explains a large share of variance (Test R² ≈ 0.83), so predictions are reasonably reliable for planning.  
- Year (market growth) and apparent temperature are the most important positive drivers. Management should expect baseline demand to be higher in later years and on warmer, comfortable days.  
- Poor weather (rain, snow, mist) causes large demand drops. Consider temporary pricing, sheltered docking, or targeted promotions on nearby clear-day forecasts to offset weather-driven dips. Capacity and staffing plans should prioritize warm, clear days and factor in strong seasonal patterns.

## How to run
1. Create and activate a virtual environment:
   - python3 -m venv .venv
   - source .venv/bin/activate
2. Install dependencies:
   - pip install -r requirements.txt
   Or manually:
   - pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
3. Open and run the notebook:
   - jupyter lab   (or `jupyter notebook`)
   - Run cells top → bottom to reproduce the analysis.


## Dependencies
- Python 3.10+  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- statsmodels

## Acknowledgements & notes
- Dataset and data dictionary were provided with the assignment.
- This notebook focuses on interpretable linear modeling rather than black-box prediction.

## Contact
Code related queries reach out to me at karthik.reddy01@sap.com
