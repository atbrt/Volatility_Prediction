import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import check_array
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

    
from scipy.stats import loguniform

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, LayerNormalization,
    LSTM, Bidirectional, MultiHeadAttention, GlobalAveragePooling1D, Input,
    Attention, Reshape, Concatenate, GRU, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Historical volatility (HV) calculation over a 30-day window
def calculate_historical_volatility(returns, window=30):
    """
    Calculates historical volatility over a 30-day window.
    """
    # Calculate volatility as the rolling standard deviation over a 30-day window
    # and annualize (sqrt(252) is a common annualization factor for daily data)
    hvol = returns.rolling(window=window).std() * np.sqrt(252)
    return hvol

def calculate_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics for volatility forecasts.
    """
    # Calculate MAE and MSE
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    # Display results
    print("\nEvaluation metrics:")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")

    return mae, mse

def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual Values (Test Set)"):
    """
    Plots predictions against actual values.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot both series
    plt.plot(range(len(y_true)), y_true, label='Actual Volatility', color='blue', alpha=0.7)
    plt.plot(range(len(y_pred)), y_pred, label='Predictions', color='red', alpha=0.7)
    
    plt.title(title, fontsize=12, pad=15)
    plt.xlabel('Observations')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def create_sliding_window_dataset(data, n_past, n_future=1, stride=1, target_col_idx=None):
    """
    Creates a sliding window dataset from time series data.
    
    Args:
        data (numpy.ndarray): Input time series data. 
                             Shape should be (samples, features) for multivariate data
                             or (samples,) for univariate data.
        n_past (int): Number of past time steps to use for each prediction.
        n_future (int): Number of future time steps to predict. Default is 1.
        stride (int): Step size between consecutive windows. Default is 1.
        target_col_idx (int, optional): Index of the target column to predict in multivariate data.
                                      If None and data is multivariate, predicts all features.
    
    Returns:
        tuple: (X, y) where:
            - X is input sequences with shape (samples, n_past, features)
            - y is target values with shape (samples, n_future) or (samples, n_future, features)
    """
    # Check if data is univariate or multivariate and reshape accordingly
    if len(data.shape) == 1:
        # Univariate data: reshape to (samples, 1)
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    X, y = [], []
    
    # Create sliding windows with stride
    for i in range(0, n_samples - n_past - n_future + 1, stride):
        # Input sequence (past)
        X.append(data[i:i+n_past])
        
        # Target sequence (future)
        if target_col_idx is not None:
            # For univariate prediction in multivariate data
            y.append(data[i+n_past:i+n_past+n_future, target_col_idx])
        else:
            # For multivariate prediction or univariate data
            y.append(data[i+n_past:i+n_past+n_future])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape y for univariate prediction if n_future=1
    if n_future == 1 and target_col_idx is not None:
        y = y.reshape(-1, 1)
    
    return X, y

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Splits data into train, validation, and test sets.
    
    Args:
        X (numpy.ndarray): Input sequences.
        y (numpy.ndarray): Target values.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Split the data
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_time_series_data(df, feature_cols, target_col, n_past, n_future=1, 
                                              stride=1, 
                                              train_ratio=0.7, val_ratio=0.15, scale_data=True):
    """
    Improved version that allows including past values of the target variable as features.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        feature_cols (list): List of column names to use as features
        target_col (str): Name of the target column to predict
        n_past (int): Number of past time steps to include for each sequence
        n_future (int): Number of future time steps to predict. Default is 1.
        stride (int): Step between each consecutive sequence. Default is 1.
        target_lags (list): List of target variable lags to include as features
        train_ratio (float): Ratio of data for training
        val_ratio (float): Ratio of data for validation
        scale_data (bool): If True, normalize the data
        handle_nans (str): Method to handle NaN values ('fill' or None)
        include_target_as_feature (bool): If True, includes target lags as features
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y)
    """

    
    df_copy = df.copy()

    features_df = df_copy[feature_cols].copy()
    target_series = df_copy[target_col].copy()
    

    
    # Convert to numpy arrays
    features = features_df.values
    targets = target_series.values.reshape(-1, 1)
    
    # Create sequences
    feature_sequences = []
    target_sequences = []
    

    # Create sequences to predict t+1 with past values
    for i in range(0, len(features) - n_past - n_future + 1, stride):
        feature_sequences.append(features[i:i+n_past])
        target_sequences.append(targets[i+n_past:i+n_past+n_future])
    
    X = np.array(feature_sequences)
    y = np.array(target_sequences)
    
    # Flatten y if necessary for n_future=1
    if n_future == 1:
        y = y.reshape(-1, 1)
    
    # Split data before scaling 
    n = len(X)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Scale data after splitting
    scaler_X, scaler_y = None, None
    
    if scale_data:
 
        def safe_transform(scaler, data):
            data_safe = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
            return scaler.transform(data_safe)
        
        # Scaling for X
        n_samples_train, n_steps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(n_samples_train * n_steps, n_features)
        
        X_train_reshaped = np.nan_to_num(X_train_reshaped, nan=0.0, posinf=1e10, neginf=-1e10)
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
        
        X_train = X_train_scaled.reshape(n_samples_train, n_steps, n_features)
        
        n_samples_val, _, _ = X_val.shape
        X_val_reshaped = X_val.reshape(n_samples_val * n_steps, n_features)
        X_val_reshaped = np.nan_to_num(X_val_reshaped, nan=0.0, posinf=1e10, neginf=-1e10)
        X_val = scaler_X.transform(X_val_reshaped).reshape(n_samples_val, n_steps, n_features)
        
        n_samples_test, _, _ = X_test.shape
        X_test_reshaped = X_test.reshape(n_samples_test * n_steps, n_features)
        X_test_reshaped = np.nan_to_num(X_test_reshaped, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test = scaler_X.transform(X_test_reshaped).reshape(n_samples_test, n_steps, n_features)
        
        # Scaling for y
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        y_train_safe = np.nan_to_num(y_train, nan=0.0, posinf=1e10, neginf=-1e10)
        y_train = scaler_y.fit_transform(y_train_safe)
        
        y_val = safe_transform(scaler_y, y_val)
        y_test = safe_transform(scaler_y, y_test)
    
    # Display data shapes
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    

        
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

def evaluate_and_visualize_model(y_true, y_pred, model_name="Model", forecast_horizon=1, dates=None):
    """
    Evaluates model performance and visualizes predictions vs actual values.

    Args:
        y_true (numpy.ndarray): Actual target values.
        y_pred (numpy.ndarray): Predicted values from the model.
        model_name (str): Name of the model for display purposes.
        forecast_horizon (int): Forecast horizon used in the model.
        dates (array-like, optional): Dates to use as x-axis for plots.

    Returns:
        tuple: (mae, mse) calculated metrics
    """
    # Ensure inputs are flattened for metric calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate metrics
    mae, mse = calculate_metrics(y_true_flat, y_pred_flat)

    # Visualize predictions vs actual
    plot_title = f"{model_name} Predictions vs Actual Values (Forecast Horizon: {forecast_horizon})"
    x_axis = dates if dates is not None else range(len(y_true_flat))
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, y_true_flat, label='Actual Volatility', color='blue', alpha=0.7)
    plt.plot(x_axis, y_pred_flat, label='Predictions', color='red', alpha=0.7)
    plt.title(plot_title, fontsize=12, pad=15)
    plt.xlabel('Date' if dates is not None else 'Observations')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot error distribution
    plt.figure(figsize=(10, 6))
    errors = y_true_flat - y_pred_flat
    plt.hist(errors, bins=30, alpha=0.7)
    plt.title(f"Error Distribution - {model_name}")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot errors over time
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, errors)
    plt.title(f"Prediction Errors Over Time - {model_name}")
    plt.xlabel('Date' if dates is not None else 'Observation')
    plt.ylabel("Error (Actual - Predicted)")
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.show()

    # Return metrics for comparison with other models
    return mae, mse

def add_technical_indicators(df, column='historical_volatility', T=20, n=14, k=9, 
                             ema_short=12, ema_long=26, D=2):
    """
    Calculates several technical indicators on a specified column (default 'historical_volatility')
    and adds them to the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        column (str): Name of the column on which to calculate the indicators
        T (int): Period for Bollinger Bands
        n (int): Period for RSI
        k (int): Period for EMA smoothing
        ema_short (int): Short period for MACD
        ema_long (int): Long period for MACD
        D (int): Deviation factor for Bollinger Bands
    
    Returns:
        pandas.DataFrame: Original DataFrame with the new indicators added
    """
    result_df = df.copy()
    
    if column not in result_df.columns:
        raise ValueError(f"The column '{column}' does not exist in the DataFrame")
    
    series = result_df[column]
    
    # Bollinger Bands
    # Middle Band 
    result_df[f'{column}_middle_band'] = series.rolling(window=T).mean()
    
    rolling_std = series.rolling(window=T).std()
    
    # Upper Band and Lower Band
    result_df[f'{column}_upper_band'] = result_df[f'{column}_middle_band'] + (D * rolling_std)
    result_df[f'{column}_lower_band'] = result_df[f'{column}_middle_band'] - (D * rolling_std)
    
    # Momentum 
    result_df[f'{column}_momentum'] = series - series.shift(12)
    
    # Acceleration
    result_df[f'{column}_acceleration'] = result_df[f'{column}_momentum'] - result_df[f'{column}_momentum'].shift(12)
    
    # Exponential Moving Average (EMA)
    alpha = 2 / (1 + k)
    
    # Initialize EMA values
    ema_values = series.copy()
    ema_values[:k] = series[:k].mean()  # The first k points are initialized to the simple mean
    
    # Calculate EMA
    for i in range(k, len(series)):
        ema_values.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * ema_values.iloc[i-1]
    
    result_df[f'{column}_ema'] = ema_values
    
    # RSI
    delta = series.diff()
    
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = -loss  
    
    avg_gain = gain.rolling(window=n).mean()
    avg_loss = loss.rolling(window=n).mean()
    
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 1e-10)
    
    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    result_df[f'{column}_rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    # Calculate short and long term EMAs
    ema_short_series = series.ewm(span=ema_short, adjust=False).mean()
    ema_long_series = series.ewm(span=ema_long, adjust=False).mean()
    
    # MACD Line
    macd_line = ema_short_series - ema_long_series
    result_df[f'{column}_macd_line'] = macd_line
    
    # Signal Line (EMA of MACD Line)
    result_df[f'{column}_macd_signal'] = macd_line.ewm(span=9, adjust=False).mean()
    
    # MACD Histogram
    result_df[f'{column}_macd_hist'] = macd_line - result_df[f'{column}_macd_signal']
    
    return result_df

def add_predictive_features(df, window_short=7, window_long=30, correlation_window=20):
    """
    Adds predictive features for volatility without introducing data leakage.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the base data
        window_short (int): Short window for calculations (days)
        window_long (int): Long window for calculations (days)
        correlation_window (int): Window for correlation calculations (days)
    
    Returns:
        pandas.DataFrame: DataFrame with the new features
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # List of return columns
    return_cols = [col for col in df.columns if col.startswith('return_')]
    

    for col in return_cols:
        # Realized volatility over a short window 
        result_df[f'{col}_vol_{window_short}d'] = df[col].rolling(window=window_short).std() * np.sqrt(252)
    

    # Correlation between EUR/USD and equities
    result_df['corr_eurusd_sp500'] = df['return_eurousd'].rolling(window=correlation_window).corr(df['return_sp500'])
    result_df['corr_eurusd_dax'] = df['return_eurousd'].rolling(window=correlation_window).corr(df['return_dax'])
    
    # Correlation between interest rates
    result_df['corr_sofr_ester'] = df['return_sofr'].rolling(window=correlation_window).corr(df['return_ester'])
    

    # Spread between equity market returns
    result_df['spread_sp500_dax'] = df['return_sp500'] - df['return_dax'] #REMOVE ACCORDING TO RF
    

    # Short moving average minus long moving average
    for col in return_cols: 
        short_ma = df[col].rolling(window=window_short).mean()
        long_ma = df[col].rolling(window=window_long).mean()
        result_df[f'{col}_trend'] = short_ma - long_ma
    
    # Distance from long-term mean (normalized)
    for col in return_cols:  # Limit to the first 3 returns
        long_mean = df[col].rolling(window=window_long).mean()
        long_std = df[col].rolling(window=window_long).std()
        # Avoid division by zero
        long_std = long_std.replace(0, 1e-10)
        # Z-score: how many standard deviations is the current value from the mean
        result_df[f'{col}_zscore'] = (df[col] - long_mean) / long_std 
    
    # Ratio of equity/forex volatility
    sp500_vol = df['return_sp500'].rolling(window=window_short).std() * np.sqrt(252)
    eurousd_vol = df['return_eurousd'].rolling(window=window_short).std() * np.sqrt(252)

    # Avoid division by zero
    eurousd_vol = eurousd_vol.replace(0, 1e-10)
    result_df['vol_ratio_sp500_eurusd'] = sp500_vol / eurousd_vol
    
    # Cumulative returns over a short window 
    result_df['eurusd_cum_return_short'] = df['return_eurousd'].rolling(window=window_short).sum()
    
    # Calculate skewness over a rolling window 
    result_df['eurusd_skew'] = df['return_eurousd'].rolling(window=window_long).skew()
    
    # Regime change detection
    result_df['volatility_regime_change'] = (abs(df['historical_volatility'] - 
                                 df['historical_volatility'].rolling(window=10).mean()) >
                                 2*df['historical_volatility'].rolling(window=10).std()).astype(int)
    
    result_df['vol_growth'] = np.log(df['historical_volatility'] / df['historical_volatility'].shift(1))

    # Second derivative of volatility (acceleration of vol changes)
    result_df['vol_acceleration'] = result_df['vol_growth'].diff()

    # Detect large jumps in returns (potential regime shift indicators)
    for col in return_cols:
        # Calculate rolling mean and std of absolute returns
        rolling_mean_abs = df[col].abs().rolling(window=20).mean()
        rolling_std_abs = df[col].abs().rolling(window=20).std()
        
        # Flag extreme moves 
        result_df[f'{col}_jump_indicator'] = (df[col].abs() > (rolling_mean_abs + 3*rolling_std_abs)).astype(int)
        
        # Count recent jumps
        result_df[f'{col}_recent_jumps'] = result_df[f'{col}_jump_indicator'].rolling(window=10).sum()

    # Volatility of volatility (second order volatility)
    result_df['vol_of_vol'] = df['historical_volatility'].rolling(window=20).std()

    # EWMA volatility ratio (more responsive to recent changes)
    short_ewm_vol = df['return_eurousd'].ewm(span=7).std() * np.sqrt(252)
    long_ewm_vol = df['return_eurousd'].ewm(span=30).std() * np.sqrt(252)
    result_df['ewm_vol_ratio'] = short_ewm_vol / long_ewm_vol

    # Detect when correlations break down (often signals regime shifts)
    typical_corr = result_df['corr_eurusd_sp500'].rolling(window=60).mean()
    result_df['correlation_regime_change'] = (abs(result_df['corr_eurusd_sp500'] - typical_corr) > 
                                    2*result_df['corr_eurusd_sp500'].rolling(window=60).std()).astype(int)


    return result_df


def add_volatility_features(df, return_col='return_eurousd', 
                           window_size=30, stride=1, 
                           forecast_horizon=1, ewma_lambda=0.94):
    """
    Adds features based on GARCH, EGARCH, and EWMA models
    using a strict sliding window approach to avoid any data leakage.
    
    For each time point t, the models are fitted on the previous window_size days only,
    and the estimated parameters are used as features to predict volatility at time t+forecast_horizon.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        return_col (str): Name of the returns column to use
        window_size (int): Size of the observation window (previous days)
        stride (int): Step between each successive estimation
        forecast_horizon (int): Forecast horizon (usually 1 day)
        ewma_lambda (float): Decay parameter for EWMA (default: 0.94, RiskMetrics value)
        
    Returns:
        pandas.DataFrame: Original DataFrame with new features added
    """
    # Create a copy of the DataFrame to avoid in-place modifications
    result_df = df.copy()
    
    # Initialize columns for GARCH parameters
    result_df['garch_alpha'] = np.nan  
    result_df['garch_beta'] = np.nan  
    result_df['garch_omega'] = np.nan  
    
    # Initialize columns for EGARCH parameters
    result_df['egarch_omega'] = np.nan 
    result_df['egarch_alpha'] = np.nan 
    result_df['egarch_beta'] = np.nan   
    
    # Initialize columns for EWMA
    result_df['ewma_lambda'] = np.nan  
    
    # Initialize columns for forecasts
    result_df['garch_forecast'] = np.nan
    result_df['egarch_forecast'] = np.nan
    result_df['ewma_forecast'] = np.nan
    
    # Ensure returns are usable
    returns = result_df[return_col].copy()
  
    # For each time point (with a stride)
    # Start after window_size to have enough historical data
    for t in range(window_size, len(df) - forecast_horizon + 1, stride):
        if t % 100 == 0:
            print(f"Processing t = {t}/{len(df)}")
            
        try:
            # Define the strict observation window (past data only)
            start_idx = t - window_size
            end_idx = t  # Exclusive, so up to t-1 included
            
            # Extract returns from the window
            returns_window = returns.iloc[start_idx:end_idx].values
            
            # Fit GARCH(1,1) model
            garch_model = arch_model(returns_window, vol='GARCH', p=1, q=1, rescale=False)
            garch_result = garch_model.fit(disp='off', show_warning=False)
            
            # Extract parameters
            garch_params = garch_result.params
            alpha = garch_params['alpha[1]']
            beta = garch_params['beta[1]']
            omega = garch_params['omega']
            
            # Fit EGARCH(1,1) model
            egarch_model = arch_model(returns_window, vol='EGARCH', p=1, q=1)
            egarch_result = egarch_model.fit(disp='off', show_warning=False)
            
            # Extract parameters
            egarch_params = egarch_result.params
            egarch_omega = egarch_params['omega']
            egarch_alpha = egarch_params['alpha[1]']
            egarch_beta = egarch_params['beta[1]']
            
            # Calculate EWMA volatility and forecast
            # Initialize variance with empirical variance of first samples
            ewma_var = np.var(returns_window[:5])
            
            # Recursively calculate EWMA variance
            for i in range(len(returns_window)):
                r = returns_window[i]
                ewma_var = ewma_lambda * ewma_var + (1 - ewma_lambda) * r**2
            
            # EWMA forecast for t+1 is simply the current variance
            ewma_forecast = np.sqrt(ewma_var * 252)  # Annualized
            
            # Forecasts for GARCH and EGARCH
            garch_forecast = np.sqrt(garch_result.forecast().variance.values[-1][0] * 252)  # Annualized
            egarch_forecast = np.sqrt(egarch_result.forecast().variance.values[-1][0] * 252)
            
            # Store parameters for the prediction point (t + forecast_horizon)
            target_idx = t + forecast_horizon - 1  # -1 because we want to predict t + 1
            
            if target_idx < len(result_df):
                # Get the actual DataFrame index at this position
                actual_index = result_df.index[target_idx]
                
                # GARCH parameters
                result_df.loc[actual_index, 'garch_alpha'] = alpha
                result_df.loc[actual_index, 'garch_beta'] = beta
                result_df.loc[actual_index, 'garch_omega'] = omega
                
                # EGARCH parameters
                result_df.loc[actual_index, 'egarch_omega'] = egarch_omega
                result_df.loc[actual_index, 'egarch_alpha'] = egarch_alpha
                result_df.loc[actual_index, 'egarch_beta'] = egarch_beta
                
                # EWMA parameter (fixed or optimized)
                result_df.loc[actual_index, 'ewma_lambda'] = ewma_lambda
                
                # Forecasts
                result_df.loc[actual_index, 'garch_forecast'] = garch_forecast
                result_df.loc[actual_index, 'egarch_forecast'] = egarch_forecast
                result_df.loc[actual_index, 'ewma_forecast'] = ewma_forecast
                
        except Exception as e:
            print(f"Error at t={t}: {str(e)}")
            continue
    
    # Calculate derived features
    result_df['garch_persistence'] = result_df['garch_alpha'] + result_df['garch_beta'] 
    
    return result_df


def select_features_with_rfecv(df, target_col, train_ratio=0.7, val_ratio=0.15, cv=5, estimator=None, date_col='date', min_features=1, step=5):
    """
    Selects relevant features using RFECV, avoiding data leakage,
    and taking into account the chronology of dates.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the features and the target
        target_col (str): Name of the target column
        train_ratio (float): Ratio for the training set
        val_ratio (float): Ratio for the validation set
        cv (int): Number of folds for cross-validation
        estimator: Estimator to use for RFECV (default: RandomForestRegressor)
        date_col (str): Name of the date column
        
    Returns:
        pandas.DataFrame: DataFrame with only the selected features and the target
        list: List of selected feature names
    """

    
    # Create a copy of the DataFrame
    result_df = df.copy()
    
    # Sort by date
    result_df = result_df.sort_values(by=date_col)
        
    # Define feature columns (all except target and date)
    feature_cols = [col for col in df.columns if col != target_col and col != date_col]
        
    # Define indices for train/val/test sets
    dates = sorted(result_df[date_col].unique())
    n = len(dates)
    train_size = int(n * train_ratio)
        
    # Dates for the training set
    train_dates = dates[:train_size]
        
    # Filter training data
    train_mask = result_df[date_col].isin(train_dates)
    X_train = result_df[train_mask][feature_cols]
    y_train = result_df[train_mask][target_col]

    

    
    # Set default estimator if not specified: a random forest
    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Initialize and fit RFECV
    rfecv = RFECV(
        estimator=estimator,
        cv=cv,
        scoring='neg_mean_squared_error',
        min_features_to_select=min_features,
        n_jobs=-1,
        verbose=1,
        step=step
    )
    
    print(f"Running RFECV to select the best features...")
    rfecv.fit(X_train, y_train)
    
    # Get selected features
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfecv.support_[i]]
    
    # Display results
    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Selected features: {selected_features}")
    
    # Create a new DataFrame with only the selected features and the target
    if date_col in df.columns:
        selected_cols = selected_features + [target_col, date_col]
    else:
        selected_cols = selected_features + [target_col]
    
    result_df = df[selected_cols].copy()
    
    # Visualize scores for each number of features
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
    plt.xlabel('Number of features')
    plt.ylabel('Cross-validation score (neg_mean_squared_error)')
    plt.title('RFECV scores by number of features')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return result_df, selected_features


def create_lstm_tuned_model(input_shape, lstm_units=256, dropout_rate=0.2):
    """
    Creates an LSTM model for time series forecasting.
    
    Args:
        input_shape (tuple): Shape of input data (n_past, features).
        lstm_units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate for regularization.
    
    Returns:
        tensorflow.keras.Model: Compiled LSTM model.
    """


    dropout_rate= 0.15
    learning_rate= 0.0005
    activation="tanh"
    hidden_layers= 1
    
    model = Sequential()
    
    # Ensure lstm_units is an integer
    lstm_units = int(lstm_units)
    hidden_layers = int(hidden_layers)
    
    # Input layer

    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape))

    # After the LSTM layer with return_sequences=True

    model.add(Bidirectional(LSTM(lstm_units // 2)))
    
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer (if requested)
    if hidden_layers > 1:
        if bidirectional:
            model.add(Bidirectional(LSTM(lstm_units // 2)))
        else:
            model.add(LSTM(lstm_units // 2))
        model.add(Dropout(dropout_rate))
    
    # Dense layers
    model.add(Dense(16, activation=activation, kernel_regularizer=l2(0.001)))
    model.add(Dense(1))  # Output layer
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_gru_model(input_shape, gru_units=256, dropout_rate=0.15):
    """
    Creates a GRU model for time series forecasting.
    
    Args:
        input_shape (tuple): Shape of input data (n_past, features).
        gru_units (int): Number of GRU units.
        dropout_rate (float): Dropout rate for regularization.
        
    Returns:
        tensorflow.keras.Model: Compiled GRU model.
    """
    learning_rate = 0.0005
    activation = "tanh"
    hidden_layers = 1
    
    model = Sequential()
    
    # Input layer - Bidirectional GRU
    model.add(Bidirectional(GRU(gru_units, return_sequences=True), input_shape=input_shape))
    
    # Second GRU layer
    model.add(Bidirectional(GRU(gru_units // 2)))
    model.add(Dropout(dropout_rate))
    
    # Additional hidden layer if requested
    if hidden_layers > 1:
        model.add(Dense(gru_units // 4, activation=activation, kernel_regularizer=l2(0.001)))
        model.add(Dropout(dropout_rate))
    
    # Dense layers
    model.add(Dense(16, activation=activation, kernel_regularizer=l2(0.001)))
    model.add(Dense(1))  # Output layer
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model
