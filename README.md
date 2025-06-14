# Implied Volatility (IV) Prediction 

NKSR Hackathon approaches
https://www.kaggle.com/competitions/nk-iv-prediction/ 

This document outlines the methodology for predicting missing Implied Volatility (IV) values in options data using feature engineering and XGBoost regression.

# EDA

1. **PCA**
- To find reduce multi-collinearity in the given features. For an IV column, the columns with IVs of nearby strikes (both call and put options +/- 200 strike) have high collinearity.
- Additionally, around 28 of 42 base features corresponded to 95% variance in distribution.
2. **Spline Interpolation**
- I used this to extrapolate IVs of columns in testing data which are absent in the training data. And, also to initialize the NaNs in test dataset. 
- I ended up not using this approach because the model trained on extrapolated values did not yield a good accuracy on testing data. 
3. **KMeans clustering**
- Used KMeans to cluster dataset based on base features (X0 ... X41). I found that the distribution of realtive size of clusters is very different in training and testing data. 
- This further indicates that the testing data does not follow a distribution similar to training data.


## 1. Feature Engineering

### ATM IV Calculation
The At-The-Money (ATM) Implied Volatility is an engineered feature calculated as:

```
ATM_IV = (call_IV_ATM + put_IV_ATM) / 2
```

Where:
- `call_IV_ATM` is the IV of the call option at the ATM strike
- `put_IV_ATM` is the IV of the put option at the ATM strike

The ATM strike is determined as the strike price closest to the current underlying stock value. If the exact ATM strike isn't available, I have rounded off to the nearest available strikes.

## 2. XGBoost Regressor Model

### Feature Set
The model uses:
1. Original features: X0 ... X41 (anonymized features from the dataset)
2. Other IV columns: For predicting any given IV column (e.g., call_IV_24000), all other available IV columns are used as features
3. Engineered feature: ATM_IV

### Handling Missing Values
The test dataset contains NaN values in some IV columns, representing the values needed to predict. XGBoost handles this elegantly through its tree-based approach:

- At each node split during training and prediction, the algorithm evaluates whether to send missing values left or right based on which direction improves the loss function
- No explicit imputation is needed for the missing target values we're trying to predict
- I tried using spline interpolation to initialize the NaN values but it yielded poor results compared to no initialization.

## 3. Iterative Prediction Process

I prepared the test data containing missing implied volatility (IV) values represented as NaNs. The initial XGBoost model is trained using the available data in the test set. This first model establishes our baseline predictions by leveraging all available information in the dataset. As mentioned above, XGBoost has native support for missing values (NaNs) in features and handles them automatically during both training and prediction without requiring imputation.

In each subsequent iteration, we train a fresh XGBoost model that incorporates the predictions from the previous iteration as part of its training data. This iterative approach progressively enhances the robustness of our predictions, as each new model benefits from the improved data completeness achieved in prior iterations. It essentially creates a self-improving system that converges toward increasingly accurate results.


## Advantages of This Approach

1. **Robust to Missing Data**: XGBoost's built-in handling of missing values eliminates need for complex imputation
2. **Feature Importance**: The model can reveal which IV strikes or features are most predictive

# Other things I tried

## 1. Pivoting the table
Convert a dataset containing individual option IVs (by strike and type) into a structured format suitable for model input, with one row per timestamp.

Each row in pivoted table represented one IV value:
- `timestamp`
- `underlying`
- `option_type` (call or put)
- `strike`
- `iv`

This is useful here as we are training a model on each individual IV as a separate prediction target.

## 2. Feature Engineering for pivoted table
- I created additional columns of moneyness (strike/underlying), log_moneyness, and moneyness_squared. 
- These features were important when trained on training data, especially with the non linear interactions captured by log and square functions. But they performed poorly on testing data.
