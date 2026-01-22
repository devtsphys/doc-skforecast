# skforecast Complete Reference Guide

## Overview

skforecast is a Python library that simplifies time series forecasting by using scikit-learn regressors as the forecasting model. It transforms time series forecasting into a regression problem using autoregressive features.

**Installation:**

```bash
pip install skforecast
```

-----

## Core Forecaster Classes

### ForecasterAutoreg

Autoregressive forecaster with a single time series.

**Key Parameters:**

- `regressor`: Any scikit-learn compatible regressor
- `lags`: Number of lagged values to use (int or array-like)
- `transformer_y`: Transformer for target variable (optional)
- `transformer_exog`: Transformer for exogenous variables (optional)
- `weight_func`: Custom function to create weights (optional)
- `differentiation`: Order of differentiation (optional)

### ForecasterAutoregCustom

Custom autoregressive forecaster with user-defined lag creation function.

**Key Parameters:**

- `regressor`: Any scikit-learn compatible regressor
- `fun_predictors`: Custom function to create predictors
- `window_size`: Size of window needed for custom function
- `transformer_y`: Transformer for target variable (optional)
- `transformer_exog`: Transformer for exogenous variables (optional)

### ForecasterAutoregDirect

Direct multi-step forecasting (one model per horizon).

**Key Parameters:**

- `regressor`: Any scikit-learn compatible regressor
- `lags`: Number of lagged values to use
- `steps`: Number of steps to forecast
- `transformer_y`: Transformer for target variable (optional)
- `transformer_exog`: Transformer for exogenous variables (optional)

### ForecasterAutoregMultiSeries

Autoregressive forecaster for multiple time series.

**Key Parameters:**

- `regressor`: Any scikit-learn compatible regressor
- `lags`: Number of lagged values to use
- `transformer_series`: Transformer for all series (optional)
- `transformer_exog`: Transformer for exogenous variables (optional)
- `weight_func`: Custom function to create weights (optional)
- `series_weights`: Dict with weights for each series (optional)

### ForecasterAutoregMultiSeriesCustom

Custom autoregressive forecaster for multiple series.

### ForecasterAutoregMultiVariate

Multivariate time series forecasting.

**Key Parameters:**

- `regressor`: Any scikit-learn compatible regressor
- `level`: Target variable to predict
- `lags`: Number of lagged values (can be dict for different lags per variable)
- `steps`: Number of steps to forecast
- `transformer_series`: Transformer for all series (optional)

### ForecasterSarimax

SARIMAX model wrapper.

**Key Parameters:**

- `regressor`: SARIMAX model from statsmodels
- `transformer_y`: Transformer for target variable (optional)
- `transformer_exog`: Transformer for exogenous variables (optional)

-----

## Method Reference Table

|Method                      |ForecasterAutoreg|ForecasterAutoregCustom|ForecasterAutoregDirect|ForecasterAutoregMultiSeries|ForecasterAutoregMultiVariate|ForecasterSarimax|Description                     |
|----------------------------|-----------------|-----------------------|-----------------------|----------------------------|-----------------------------|-----------------|--------------------------------|
|`fit()`                     |✓                |✓                      |✓                      |✓                           |✓                            |✓                |Train the forecaster            |
|`predict()`                 |✓                |✓                      |✓                      |✓                           |✓                            |✓                |Generate predictions            |
|`predict_interval()`        |✓                |✓                      |✓                      |✓                           |✓                            |✓                |Predict with intervals          |
|`predict_bootstrapping()`   |✓                |✓                      |✓                      |✓                           |✓                            |✗                |Predict using bootstrapping     |
|`predict_quantiles()`       |✓                |✓                      |✓                      |✓                           |✓                            |✗                |Predict quantiles               |
|`predict_dist()`            |✓                |✓                      |✓                      |✓                           |✓                            |✗                |Predict probability distribution|
|`set_params()`              |✓                |✓                      |✓                      |✓                           |✓                            |✓                |Set forecaster parameters       |
|`get_params()`              |✓                |✓                      |✓                      |✓                           |✓                            |✓                |Get forecaster parameters       |
|`set_lags()`                |✓                |✗                      |✓                      |✓                           |✓                            |✗                |Change lag configuration        |
|`set_out_sample_residuals()`|✓                |✓                      |✓                      |✓                           |✓                            |✗                |Set residuals for intervals     |
|`get_feature_importances()` |✓                |✓                      |✓                      |✓                           |✓                            |✗                |Get feature importance          |
|`summary()`                 |✓                |✓                      |✓                      |✓                           |✓                            |✓                |Print forecaster summary        |

-----

## Model Selection & Validation Functions

### backtesting_forecaster()

Backtesting with multiple validation folds.

**Parameters:**

- `forecaster`: Trained forecaster object
- `y`: Time series data
- `steps`: Forecast horizon
- `metric`: Evaluation metric (string or callable)
- `initial_train_size`: Size of initial training set
- `fixed_train_size`: Whether to use fixed or expanding window
- `gap`: Gap between train and test
- `allow_incomplete_fold`: Allow last fold to be incomplete
- `exog`: Exogenous variables (optional)
- `refit`: Whether to refit (bool or int for refit frequency)
- `interval`: Prediction intervals (optional)
- `n_boot`: Number of bootstrapping iterations
- `random_state`: Random seed
- `in_sample_residuals`: Use in-sample residuals for intervals
- `verbose`: Print progress

### backtesting_forecaster_multiseries()

Backtesting for multiple series.

### backtesting_sarimax()

Backtesting for SARIMAX models.

### cv_forecaster()

Cross-validation returning all predictions.

### grid_search_forecaster()

Hyperparameter grid search with backtesting.

**Parameters:**

- `forecaster`: Forecaster object
- `y`: Time series data
- `param_grid`: Dictionary with parameter combinations
- `steps`: Forecast horizon
- `metric`: Evaluation metric
- `initial_train_size`: Size of initial training
- `fixed_train_size`: Fixed or expanding window
- `gap`: Gap between train and test
- `allow_incomplete_fold`: Allow incomplete folds
- `exog`: Exogenous variables
- `lags_grid`: List of lag configurations to try
- `refit`: Refit strategy
- `return_best`: Return best model
- `verbose`: Print progress

### random_search_forecaster()

Random hyperparameter search.

### bayesian_search_forecaster()

Bayesian optimization for hyperparameter tuning.

-----

## Model Evaluation Metrics

### Available Metrics

- `mean_squared_error` (MSE)
- `mean_absolute_error` (MAE)
- `mean_absolute_percentage_error` (MAPE)
- `mean_squared_log_error` (MSLE)
- `root_mean_squared_error` (RMSE) - custom

-----

## Practical Examples

### Example 1: Basic Forecasting with ForecasterAutoreg

```python
import pandas as pd
import numpy as np
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Create sample data
dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
y = pd.Series(
    np.sin(np.arange(200) * 2 * np.pi / 50) + np.random.normal(0, 0.1, 200),
    index=dates
)

# Split data
train = y[:-30]
test = y[-30:]

# Create and train forecaster
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=15
)
forecaster.fit(y=train)

# Make predictions
predictions = forecaster.predict(steps=30)

# Evaluate
mse = mean_squared_error(test, predictions)
print(f"MSE: {mse:.4f}")
```

### Example 2: Forecasting with Exogenous Variables

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np

# Create time series with exogenous variable
dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
y = pd.Series(np.cumsum(np.random.randn(200)), index=dates)
exog = pd.DataFrame({
    'temperature': np.random.normal(20, 5, 200),
    'holiday': np.random.binomial(1, 0.1, 200)
}, index=dates)

# Split
train_y = y[:-30]
train_exog = exog[:-30]
test_exog = exog[-30:]

# Create forecaster
forecaster = ForecasterAutoreg(
    regressor=GradientBoostingRegressor(random_state=123),
    lags=10
)

# Fit with exogenous variables
forecaster.fit(y=train_y, exog=train_exog)

# Predict with future exogenous values
predictions = forecaster.predict(steps=30, exog=test_exog)
```

### Example 3: Prediction Intervals

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

# Create forecaster
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=15
)
forecaster.fit(y=train)

# Predict with intervals using bootstrapping
predictions = forecaster.predict_interval(
    steps=30,
    interval=[5, 95],  # 90% prediction interval
    n_boot=500,
    in_sample_residuals=True
)

print(predictions)
# Returns DataFrame with columns: pred, lower_bound, upper_bound
```

### Example 4: Backtesting

```python
from skforecast.model_selection import backtesting_forecaster
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

# Create forecaster
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=15
)

# Perform backtesting
metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=y,
    steps=10,
    metric='mean_squared_error',
    initial_train_size=100,
    fixed_train_size=False,  # Expanding window
    gap=0,
    allow_incomplete_fold=True,
    refit=True,
    verbose=True
)

print(f"Backtesting MSE: {metric:.4f}")
```

### Example 5: Grid Search for Hyperparameter Tuning

```python
from skforecast.model_selection import grid_search_forecaster
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

# Create forecaster
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=12
)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Define lags to try
lags_grid = [5, 10, 15, 20]

# Grid search
results = grid_search_forecaster(
    forecaster=forecaster,
    y=y,
    param_grid=param_grid,
    lags_grid=lags_grid,
    steps=10,
    metric='mean_squared_error',
    initial_train_size=100,
    fixed_train_size=False,
    refit=True,
    return_best=True,
    verbose=False
)

print(results)
```

### Example 6: Multiple Time Series Forecasting

```python
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Create multiple time series
dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
series = pd.DataFrame({
    'series_1': np.cumsum(np.random.randn(200)),
    'series_2': np.cumsum(np.random.randn(200)),
    'series_3': np.cumsum(np.random.randn(200))
}, index=dates)

# Split
train = series[:-30]
test = series[-30:]

# Create forecaster
forecaster = ForecasterAutoregMultiSeries(
    regressor=RandomForestRegressor(random_state=123),
    lags=15
)

# Fit
forecaster.fit(series=train)

# Predict all series
predictions = forecaster.predict(steps=30)

# Predict specific series
predictions_s1 = forecaster.predict(steps=30, levels='series_1')
```

### Example 7: Custom Lag Function

```python
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Define custom function to create predictors
def create_predictors(y):
    """
    Create custom predictors from a time series.
    
    Parameters
    ----------
    y : pandas Series
        Time series.
        
    Returns
    -------
    predictors : pandas DataFrame
        Predictors.
    """
    lags = y[-5:]  # Last 5 lags
    lag_mean = np.mean(lags)
    lag_std = np.std(lags)
    lag_min = np.min(lags)
    lag_max = np.max(lags)
    
    predictors = pd.DataFrame({
        'lag_1': lags.iloc[-1],
        'lag_2': lags.iloc[-2],
        'lag_mean': lag_mean,
        'lag_std': lag_std,
        'lag_range': lag_max - lag_min
    }, index=[0])
    
    return predictors

# Create forecaster
forecaster = ForecasterAutoregCustom(
    regressor=RandomForestRegressor(random_state=123),
    fun_predictors=create_predictors,
    window_size=5
)

forecaster.fit(y=train)
predictions = forecaster.predict(steps=30)
```

### Example 8: Direct Multi-Step Forecasting

```python
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.ensemble import GradientBoostingRegressor

# Create forecaster that trains one model per step
forecaster = ForecasterAutoregDirect(
    regressor=GradientBoostingRegressor(random_state=123),
    lags=15,
    steps=10  # Train 10 separate models
)

forecaster.fit(y=train)

# Predict 10 steps ahead
predictions = forecaster.predict(steps=10)
```

### Example 9: Data Preprocessing with Transformers

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Create forecaster with transformation
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=15,
    transformer_y=StandardScaler()  # Transform target variable
)

forecaster.fit(y=train)
predictions = forecaster.predict(steps=30)
# Predictions are automatically inverse-transformed
```

### Example 10: Multivariate Forecasting

```python
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Create multivariate time series
dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
series = pd.DataFrame({
    'temperature': np.cumsum(np.random.randn(200)),
    'humidity': np.cumsum(np.random.randn(200)),
    'pressure': np.cumsum(np.random.randn(200))
}, index=dates)

# Split
train = series[:-30]

# Create forecaster to predict 'temperature'
forecaster = ForecasterAutoregMultiVariate(
    regressor=RandomForestRegressor(random_state=123),
    level='temperature',  # Target variable
    lags=15,
    steps=10
)

forecaster.fit(series=train)

# Predict temperature using lags from all variables
predictions = forecaster.predict(steps=10)
```

### Example 11: Feature Importance Analysis

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=20
)
forecaster.fit(y=train)

# Get feature importance
importance = forecaster.get_feature_importances()
print(importance.sort_values(by='importance', ascending=False))
```

### Example 12: Weighted Time Series

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Define weight function (more weight to recent observations)
def custom_weights(index):
    """
    Exponential weights, more weight to recent observations.
    """
    weights = np.exp(np.arange(len(index)) / 20)
    return weights

forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=15,
    weight_func=custom_weights
)

forecaster.fit(y=train)
predictions = forecaster.predict(steps=30)
```

### Example 13: Differentiation for Non-Stationary Series

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge

# Create forecaster with first-order differentiation
forecaster = ForecasterAutoreg(
    regressor=Ridge(),
    lags=15,
    differentiation=1  # First-order differencing
)

forecaster.fit(y=train)
predictions = forecaster.predict(steps=30)
# Predictions are automatically integrated back
```

### Example 14: Quantile Predictions

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import GradientBoostingRegressor

forecaster = ForecasterAutoreg(
    regressor=GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=123),
    lags=15
)

forecaster.fit(y=train)

# Predict multiple quantiles
quantiles = forecaster.predict_quantiles(
    steps=30,
    quantiles=[0.1, 0.5, 0.9],
    n_boot=500
)

print(quantiles)
```

### Example 15: Probabilistic Forecasting

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=15
)

forecaster.fit(y=train)

# Get full probability distribution
distribution = forecaster.predict_dist(
    steps=30,
    n_boot=1000
)

# distribution is a 2D array: (steps, n_boot)
# Each row contains bootstrapped predictions for that step
```

-----

## Best Practices & Tips

### Choosing Lags

- Start with seasonal patterns (e.g., 7 for weekly, 12 for monthly)
- Use ACF/PACF plots to identify significant lags
- Try multiple lag configurations with grid search
- More lags ≠ better performance (risk of overfitting)

### Train/Test Splitting

- Always respect temporal order
- Use backtesting instead of random cross-validation
- Consider multiple folds to assess stability

### Handling Exogenous Variables

- Ensure exogenous variables are available for prediction period
- Scale exogenous variables consistently with training
- Be cautious of data leakage

### Model Selection

- Start simple (linear models, lags selection)
- Try ensemble methods (RandomForest, GradientBoosting)
- Use grid search for hyperparameter tuning
- Consider computational cost vs. accuracy trade-off

### Prediction Intervals

- Use bootstrapping for robust intervals
- Increase `n_boot` for smoother intervals (500-1000)
- Consider using out-of-sample residuals for better coverage

### Performance Optimization

- Use `fixed_train_size=True` for faster backtesting
- Reduce `n_boot` during experimentation
- Set `refit=False` or integer for less frequent refitting
- Use simpler models for very long series

-----

## Common Workflows

### Workflow 1: Complete Forecasting Pipeline

```python
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from sklearn.ensemble import RandomForestRegressor

# 1. Load and prepare data
y = pd.read_csv('data.csv', index_col=0, parse_dates=True)['value']

# 2. Initial train-test split
train = y[:-30]
test = y[-30:]

# 3. Hyperparameter tuning
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=12
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}

lags_grid = [7, 14, 21]

results = grid_search_forecaster(
    forecaster=forecaster,
    y=train,
    param_grid=param_grid,
    lags_grid=lags_grid,
    steps=10,
    metric='mean_squared_error',
    initial_train_size=int(len(train) * 0.7),
    return_best=True
)

# 4. Evaluate with backtesting
metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=train,
    steps=10,
    metric='mean_squared_error',
    initial_train_size=int(len(train) * 0.7),
    refit=True
)

print(f"CV MSE: {metric}")

# 5. Final predictions on test set
forecaster.fit(y=train)
final_predictions = forecaster.predict_interval(
    steps=30,
    interval=[5, 95],
    n_boot=1000
)
```

### Workflow 2: Multiple Series with Shared Model

```python
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection import backtesting_forecaster_multiseries
from sklearn.ensemble import GradientBoostingRegressor

# Load multiple series
series = pd.read_csv('multiseries.csv', index_col=0, parse_dates=True)

# Create forecaster
forecaster = ForecasterAutoregMultiSeries(
    regressor=GradientBoostingRegressor(random_state=123),
    lags=15
)

# Backtesting
metric, predictions = backtesting_forecaster_multiseries(
    forecaster=forecaster,
    series=series,
    steps=10,
    metric='mean_squared_error',
    initial_train_size=100,
    refit=True
)

# Fit and predict
forecaster.fit(series=series)
predictions = forecaster.predict(steps=30)
```

-----

## Troubleshooting

**Issue:** Poor forecast accuracy

- Try different lag configurations
- Add relevant exogenous variables
- Use more sophisticated models (ensemble methods)
- Check for non-stationarity (use differentiation)
- Ensure sufficient training data

**Issue:** Slow performance

- Reduce number of lags
- Use simpler models
- Reduce backtesting folds
- Use `fixed_train_size=True`
- Reduce `n_boot` for intervals

**Issue:** Wide prediction intervals

- Increase training data
- Reduce forecast horizon
- Use more stable models
- Consider if uncertainty is inherent

**Issue:** Memory errors

- Reduce `n_boot`
- Use fewer lags
- Process series in batches
- Use simpler models

-----

## Additional Resources

- **Documentation:** https://skforecast.org/
- **GitHub:** https://github.com/JoaquinAmatRodrigo/skforecast
- **Examples:** https://skforecast.org/latest/examples/examples.html
- **API Reference:** https://skforecast.org/latest/api/ForecasterAutoreg.html