import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from arch import arch_model


def calculate_historical_volatility(returns, window=30):
    """
    Calculate historical volatility on a rolling window of 30 days.
    """
    hvol = returns.rolling(window=window).std() * np.sqrt(252)
    return hvol

def test_stationarity(series):
    """
    Test stationarity of a time series using the Augmented Dickey-Fuller test.
    """
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

def fit_model(returns, model_type='Garch', distribution='StudentsT'):
    """ 
    Function to fit a GARCH(1,1) model.
    """
    # Rescale returns
    scaled_returns = returns * 100
    
    # Configure the model based on type
    if model_type.upper() == 'GJR-GARCH':
        model = arch_model(scaled_returns,
                         vol='GARCH',
                         p=1,
                         q=1,
                         o=1,  
                         power=2.0,  
                         dist=distribution)
    else:
        model = arch_model(scaled_returns,
                         vol=model_type,
                         p=1,
                         q=1,
                         o=1 if model_type=='EGARCH' else 0,
                         dist=distribution)
    
    # Fit the model
    results = model.fit(disp='off')
    
    # Get volatility forecasts
    volatility = pd.Series(
        np.sqrt(results.conditional_volatility) * np.sqrt(252) / 100,
        index=returns.index
    )
    
    return results, volatility

def find_best_order(returns, max_p=3, max_q=3, model_type='Garch', distribution='StudentsT'):
    """ 
    Find the best GARCH model order based on AIC and BIC.
    """
    best_aic = np.inf
    best_bic = np.inf
    best_order_aic = (0, 0)
    best_order_bic = (0, 0)
    
    results = {}
    
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                scaled_returns = returns * 100
                
                model = arch_model(scaled_returns,
                                 vol=model_type,
                                 p=p,
                                 q=q,
                                 o=1 if model_type=='EGARCH' else 0,
                                 dist=distribution)
                
                fitted = model.fit(disp='off')
                
                results[(p, q)] = {
                    'AIC': fitted.aic,
                    'BIC': fitted.bic,
                    'Log-Likelihood': fitted.loglikelihood
                }
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order_aic = (p, q)
                
                if fitted.bic < best_bic:
                    best_bic = fitted.bic
                    best_order_bic = (p, q)
                    
            except Exception as e:
                print(f"Error for p={p}, q={q}: {e}")
                results[(p, q)] = {'AIC': np.nan, 'BIC': np.nan, 'Log-Likelihood': np.nan}
    
    
    results_df = pd.DataFrame(results).T
    results_df.index.names = ['p', 'q']
    
    print(f"Best order by AIC: GARCH({best_order_aic[0]},{best_order_aic[1]})")
    print(f"Best order by BIC: GARCH({best_order_bic[0]},{best_order_bic[1]})")
    
    return results_df, best_order_aic, best_order_bic

def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual Values (Test Set)", dates=None):
    """
    Plot predictions against actual values.

    Args:
        y_true (array-like): Actual values.
        y_pred (array-like): Predicted values.
        title (str): Plot title.
        dates (array-like, optional): Dates to use as x-axis. If None, use default index.
    """
    plt.figure(figsize=(12, 6))
    
    x_axis = dates if dates is not None else range(len(y_true))
    plt.plot(x_axis, y_true, label='Actual Volatility', color='blue', alpha=0.7)
    plt.plot(x_axis, y_pred, label='Predictions', color='red', alpha=0.7)
    
    plt.title(title, fontsize=12, pad=15)
    plt.xlabel('Date' if dates is not None else 'Observations')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
