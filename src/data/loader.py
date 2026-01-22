"""
Data loading, preprocessing, and feature engineering module.

This module handles:
- Loading raw sales data
- Feature engineering (lags, rolling statistics, time features)
- Temporal train/calibration/test splitting
- Sequence preparation for deep learning models
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES FOR TYPE SAFETY
# =============================================================================

@dataclass
class DataSplit:
    """Container for a single data split."""
    X: np.ndarray
    y: np.ndarray
    dates: Optional[np.ndarray] = None
    
    def __len__(self) -> int:
        return len(self.y)


@dataclass
class TemporalSplits:
    """Container for train/calibration/test splits."""
    train: DataSplit
    calibration: DataSplit
    test: DataSplit
    feature_names: List[str]

    def summary(self) -> str:
        return (
            f"Train: {len(self.train)} samples | "
            f"Calibration: {len(self.calibration)} samples | "
            f"Test: {len(self.test)} samples"
        )


@dataclass
class RollingWindowSplit:
    """Container for a single rolling window split."""
    train: DataSplit
    calibration: DataSplit
    test: DataSplit
    window_idx: int
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    feature_names: List[str]

    def summary(self) -> str:
        return (
            f"Window {self.window_idx}: "
            f"Train={len(self.train)}, Cal={len(self.calibration)}, Test={len(self.test)} | "
            f"Test period: {self.test_start_date.date()} to {self.test_end_date.date()}"
        )


@dataclass
class SequenceData:
    """Container for sequence data (for DL models)."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    sequence_length: int


# =============================================================================
# DATA LOADING
# =============================================================================

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw sales data from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
        
    Returns
    -------
    pd.DataFrame
        Raw dataframe with date parsed.
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Loaded {len(df)} records")
    return df


def filter_store_item(
    df: pd.DataFrame, 
    store_id: int, 
    item_id: int
) -> pd.DataFrame:
    """
    Filter dataframe for specific store and item.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe.
    store_id : int
        Store ID to filter.
    item_id : int
        Item ID to filter.
        
    Returns
    -------
    pd.DataFrame
        Filtered and sorted dataframe.
    """
    filtered = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()
    filtered = filtered.sort_values('date').reset_index(drop=True)
    logger.info(f"Store {store_id}, Item {item_id}: {len(filtered)} records")
    return filtered


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'date' column.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with time features added.
    """
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['quarter'] = df['date'].dt.quarter
    return df


def create_lag_features(
    df: pd.DataFrame, 
    target_col: str = 'sales',
    lags: List[int] = [1, 7, 14, 28]
) -> pd.DataFrame:
    """
    Create lag features for the target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with target column.
    target_col : str
        Name of the target column.
    lags : List[int]
        List of lag periods to create.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with lag features added.
    """
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'sales',
    windows: List[int] = [7, 14, 28]
) -> pd.DataFrame:
    """
    Create rolling statistics features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with target column.
    target_col : str
        Name of the target column.
    windows : List[int]
        List of rolling window sizes.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with rolling features added.
    """
    df = df.copy()
    for window in windows:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(
            window=window, min_periods=1
        ).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(
            window=window, min_periods=1
        ).std()
        df[f'rolling_min_{window}'] = df[target_col].rolling(
            window=window, min_periods=1
        ).min()
        df[f'rolling_max_{window}'] = df[target_col].rolling(
            window=window, min_periods=1
        ).max()
    return df


def create_all_features(
    df: pd.DataFrame,
    lag_periods: List[int] = [1, 7, 28],
    rolling_windows: List[int] = [7, 28],
    warmup_period: int = 28
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create all features for the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with 'date' and 'sales' columns.
    lag_periods : List[int]
        Lag periods for lag features.
    rolling_windows : List[int]
        Window sizes for rolling features.
    warmup_period : int
        Number of initial rows to drop (to avoid NaN from lags).
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Processed dataframe and list of feature column names.
    """
    logger.info("Creating features...")
    
    # Create features
    df = create_time_features(df)
    df = create_lag_features(df, lags=lag_periods)
    df = create_rolling_features(df, windows=rolling_windows)
    
    # Handle missing values
    df = df.bfill().ffill()
    
    # Drop warmup period
    df = df.iloc[warmup_period:].reset_index(drop=True)
    
    # Define feature columns
    feature_cols = ['month', 'day_of_week', 'day_of_year']
    feature_cols += [f'sales_lag_{lag}' for lag in lag_periods]
    feature_cols += [f'rolling_mean_{w}' for w in rolling_windows]
    feature_cols += [f'rolling_std_{w}' for w in rolling_windows]
    
    logger.info(f"Created {len(feature_cols)} features, {len(df)} samples")
    
    return df, feature_cols


# =============================================================================
# TEMPORAL SPLITTING
# =============================================================================

def create_temporal_splits(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_years: List[int] = [2013, 2014],
    cal_years: List[int] = [2015, 2016],
    test_years: List[int] = [2017]
) -> TemporalSplits:
    """
    Create temporal train/calibration/test splits.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe with features.
    feature_cols : List[str]
        List of feature column names.
    train_years : List[int]
        Years for training data.
    cal_years : List[int]
        Years for calibration data.
    test_years : List[int]
        Years for test data.
        
    Returns
    -------
    TemporalSplits
        Container with all data splits.
    """
    logger.info("Creating temporal splits...")
    
    # Ensure year column exists
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    
    # Split data
    train_df = df[df['year'].isin(train_years)].copy()
    cal_df = df[df['year'].isin(cal_years)].copy()
    test_df = df[df['year'].isin(test_years)].copy()
    
    # Create DataSplit objects
    train_split = DataSplit(
        X=train_df[feature_cols].values,
        y=train_df['sales'].values,
        dates=train_df['date'].values
    )
    
    cal_split = DataSplit(
        X=cal_df[feature_cols].values,
        y=cal_df['sales'].values,
        dates=cal_df['date'].values
    )
    
    test_split = DataSplit(
        X=test_df[feature_cols].values,
        y=test_df['sales'].values,
        dates=test_df['date'].values
    )
    
    splits = TemporalSplits(
        train=train_split,
        calibration=cal_split,
        test=test_split,
        feature_names=feature_cols
    )
    
    logger.info(splits.summary())

    return splits


def create_rolling_window_splits(
    df: pd.DataFrame,
    feature_cols: List[str],
    initial_train_days: int = 730,  # 2 years
    calibration_days: int = 365,     # 1 year
    test_window_days: int = 30,      # 1 month prediction
    step_days: int = 30              # Roll forward by 1 month
) -> List[RollingWindowSplit]:
    """
    Create rolling window splits for time series cross-validation.

    This function creates multiple train/calibration/test splits by sliding
    a window through time. Each window predicts the next month.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe with features and date column.
    feature_cols : List[str]
        List of feature column names.
    initial_train_days : int
        Number of days in initial training period (default 730 = 2 years).
    calibration_days : int
        Number of days for calibration period (default 365 = 1 year).
    test_window_days : int
        Number of days to predict ahead (default 30 = 1 month).
    step_days : int
        Number of days to roll forward for each window (default 30 = 1 month).

    Returns
    -------
    List[RollingWindowSplit]
        List of rolling window splits, one for each time window.

    Examples
    --------
    With initial_train=730, cal=365, test=30, step=30:
    - Window 0: Train days 0-729, Cal days 730-1094, Test days 1095-1124
    - Window 1: Train days 30-759, Cal days 760-1124, Test days 1125-1154
    - Window 2: Train days 60-789, Cal days 790-1154, Test days 1155-1184
    - ...
    """
    logger.info("Creating rolling window splits...")
    logger.info(f"  Initial train: {initial_train_days} days")
    logger.info(f"  Calibration: {calibration_days} days")
    logger.info(f"  Test window: {test_window_days} days")
    logger.info(f"  Step size: {step_days} days")

    # Ensure data is sorted by date
    df = df.sort_values('date').reset_index(drop=True)

    splits = []
    window_idx = 0

    # Start position for first window
    start_pos = 0

    while True:
        # Calculate split positions
        train_end = start_pos + initial_train_days
        cal_end = train_end + calibration_days
        test_end = cal_end + test_window_days

        # Check if we have enough data
        if test_end > len(df):
            break

        # Extract data for this window
        train_df = df.iloc[start_pos:train_end].copy()
        cal_df = df.iloc[train_end:cal_end].copy()
        test_df = df.iloc[cal_end:test_end].copy()

        # Create DataSplit objects
        train_split = DataSplit(
            X=train_df[feature_cols].values,
            y=train_df['sales'].values,
            dates=train_df['date'].values
        )

        cal_split = DataSplit(
            X=cal_df[feature_cols].values,
            y=cal_df['sales'].values,
            dates=cal_df['date'].values
        )

        test_split = DataSplit(
            X=test_df[feature_cols].values,
            y=test_df['sales'].values,
            dates=test_df['date'].values
        )

        # Create rolling window split
        rolling_split = RollingWindowSplit(
            train=train_split,
            calibration=cal_split,
            test=test_split,
            window_idx=window_idx,
            test_start_date=test_df['date'].iloc[0],
            test_end_date=test_df['date'].iloc[-1],
            feature_names=feature_cols
        )

        splits.append(rolling_split)
        logger.info(f"  {rolling_split.summary()}")

        # Move to next window
        start_pos += step_days
        window_idx += 1

    logger.info(f"Created {len(splits)} rolling window splits")

    return splits


# =============================================================================
# SEQUENCE DATA PREPARATION (FOR DEEP LEARNING)
# =============================================================================

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int,
    prediction_horizon: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM/Transformer models.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).
    seq_length : int
        Length of input sequences (lookback window).
    prediction_horizon : int
        Number of days ahead to predict (default 30).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X_seq: shape (n_samples - seq_length - prediction_horizon + 1, seq_length, n_features)
        y_seq: shape (n_samples - seq_length - prediction_horizon + 1,)
    """
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X) - prediction_horizon + 1):
        X_seq.append(X[i - seq_length:i])
        y_seq.append(y[i + prediction_horizon - 1])
    return np.array(X_seq), np.array(y_seq)


def prepare_sequence_data(
    splits: TemporalSplits,
    seq_length: int = 28,
    prediction_horizon: int = 30
) -> SequenceData:
    """
    Prepare sequence data for deep learning models.

    For calibration and test sets, we include historical data from previous
    splits to create complete sequences.

    Parameters
    ----------
    splits : TemporalSplits
        Train/calibration/test splits.
    seq_length : int
        Sequence length (lookback window).
    prediction_horizon : int
        Number of days ahead to predict (default 30).

    Returns
    -------
    SequenceData
        Container with sequence data for all splits.
    """
    logger.info(f"Preparing sequence data with seq_length={seq_length}, prediction_horizon={prediction_horizon}")

    # Combine all data for sequence creation
    X_all = np.vstack([splits.train.X, splits.calibration.X, splits.test.X])
    y_all = np.concatenate([splits.train.y, splits.calibration.y, splits.test.y])

    # Create sequences from all data
    X_seq_all, y_seq_all = create_sequences(X_all, y_all, seq_length, prediction_horizon)
    
    # Calculate split indices (accounting for sequence length and prediction horizon)
    n_train = len(splits.train.y) - seq_length - prediction_horizon + 1
    n_cal = len(splits.calibration.y)
    n_test = len(splits.test.y)
    
    # Split back into train/cal/test
    X_train_seq = X_seq_all[:n_train]
    y_train_seq = y_seq_all[:n_train]
    
    X_cal_seq = X_seq_all[n_train:n_train + n_cal]
    y_cal_seq = y_seq_all[n_train:n_train + n_cal]
    
    X_test_seq = X_seq_all[n_train + n_cal:n_train + n_cal + n_test]
    y_test_seq = y_seq_all[n_train + n_cal:n_train + n_cal + n_test]
    
    logger.info(f"Train sequences: {X_train_seq.shape}")
    logger.info(f"Calibration sequences: {X_cal_seq.shape}")
    logger.info(f"Test sequences: {X_test_seq.shape}")
    
    return SequenceData(
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_cal=X_cal_seq,
        y_cal=y_cal_seq,
        X_test=X_test_seq,
        y_test=y_test_seq,
        sequence_length=seq_length
    )


def prepare_rolling_sequence_data(
    rolling_split: RollingWindowSplit,
    seq_length: int = 28,
    prediction_horizon: int = 30
) -> SequenceData:
    """
    Prepare sequence data for a single rolling window split.

    Parameters
    ----------
    rolling_split : RollingWindowSplit
        A single rolling window split.
    seq_length : int
        Sequence length (lookback window).
    prediction_horizon : int
        Number of days ahead to predict.

    Returns
    -------
    SequenceData
        Container with sequence data for this window.
    """
    # Combine all data for sequence creation
    X_all = np.vstack([rolling_split.train.X, rolling_split.calibration.X, rolling_split.test.X])
    y_all = np.concatenate([rolling_split.train.y, rolling_split.calibration.y, rolling_split.test.y])

    # Create sequences from all data
    X_seq_all, y_seq_all = create_sequences(X_all, y_all, seq_length, prediction_horizon)

    # Calculate split indices
    n_train = len(rolling_split.train.y) - seq_length - prediction_horizon + 1
    n_cal = len(rolling_split.calibration.y)
    n_test = len(rolling_split.test.y)

    # Split back into train/cal/test
    X_train_seq = X_seq_all[:n_train]
    y_train_seq = y_seq_all[:n_train]

    X_cal_seq = X_seq_all[n_train:n_train + n_cal]
    y_cal_seq = y_seq_all[n_train:n_train + n_cal]

    X_test_seq = X_seq_all[n_train + n_cal:n_train + n_cal + n_test]
    y_test_seq = y_seq_all[n_train + n_cal:n_train + n_cal + n_test]

    return SequenceData(
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_cal=X_cal_seq,
        y_cal=y_cal_seq,
        X_test=X_test_seq,
        y_test=y_test_seq,
        sequence_length=seq_length
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_and_prepare_data(
    filepath: str,
    store_id: int = 1,
    item_id: int = 1,
    lag_periods: List[int] = [1, 7, 28],
    rolling_windows: List[int] = [7, 28],
    train_years: List[int] = [2013, 2014],
    cal_years: List[int] = [2015, 2016],
    test_years: List[int] = [2017]
) -> TemporalSplits:
    """
    Convenience function to load and prepare data in one call.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    store_id : int
        Store ID to filter.
    item_id : int
        Item ID to filter.
    lag_periods : List[int]
        Lag periods for features.
    rolling_windows : List[int]
        Rolling window sizes.
    train_years, cal_years, test_years : List[int]
        Years for each split.
        
    Returns
    -------
    TemporalSplits
        Prepared data splits.
    """
    # Load raw data
    df = load_raw_data(filepath)
    
    # Filter for specific store/item
    df = filter_store_item(df, store_id, item_id)
    
    # Create features
    df, feature_cols = create_all_features(
        df, 
        lag_periods=lag_periods,
        rolling_windows=rolling_windows
    )
    
    # Create temporal splits
    splits = create_temporal_splits(
        df, 
        feature_cols,
        train_years=train_years,
        cal_years=cal_years,
        test_years=test_years
    )
    
    return splits


def load_multi_store_data(
    filepath: str,
    store_ids: List[int],
    item_ids: List[int],
    **kwargs
) -> Dict[Tuple[int, int], TemporalSplits]:
    """
    Load data for multiple store-item combinations.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    store_ids : List[int]
        List of store IDs.
    item_ids : List[int]
        List of item IDs.
    **kwargs
        Additional arguments passed to load_and_prepare_data.
        
    Returns
    -------
    Dict[Tuple[int, int], TemporalSplits]
        Dictionary mapping (store_id, item_id) to data splits.
    """
    logger.info(f"Loading data for {len(store_ids)} stores × {len(item_ids)} items")
    
    # Load raw data once
    df_raw = load_raw_data(filepath)
    
    results = {}
    for store_id in store_ids:
        for item_id in item_ids:
            try:
                df = filter_store_item(df_raw, store_id, item_id)
                if len(df) < 365:  # Skip if insufficient data
                    logger.warning(f"Skipping store {store_id}, item {item_id}: insufficient data")
                    continue
                    
                df, feature_cols = create_all_features(df, **kwargs)
                splits = create_temporal_splits(df, feature_cols)
                results[(store_id, item_id)] = splits
                
            except Exception as e:
                logger.error(f"Error processing store {store_id}, item {item_id}: {e}")
                continue
    
    logger.info(f"Successfully loaded {len(results)} store-item combinations")
    return results


def load_and_prepare_rolling_data(
    filepath: str,
    store_id: int = 1,
    item_id: int = 1,
    lag_periods: List[int] = [1, 7, 28],
    rolling_windows: List[int] = [7, 28],
    initial_train_days: int = 730,
    calibration_days: int = 365,
    test_window_days: int = 30,
    step_days: int = 30
) -> List[RollingWindowSplit]:
    """
    Convenience function to load and prepare rolling window data.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    store_id : int
        Store ID to filter.
    item_id : int
        Item ID to filter.
    lag_periods : List[int]
        Lag periods for features.
    rolling_windows : List[int]
        Rolling window sizes.
    initial_train_days : int
        Initial training period in days.
    calibration_days : int
        Calibration period in days.
    test_window_days : int
        Test window size in days.
    step_days : int
        Step size for rolling windows.

    Returns
    -------
    List[RollingWindowSplit]
        List of rolling window splits.
    """
    # Load raw data
    df = load_raw_data(filepath)

    # Filter for specific store/item
    df = filter_store_item(df, store_id, item_id)

    # Create features
    df, feature_cols = create_all_features(
        df,
        lag_periods=lag_periods,
        rolling_windows=rolling_windows
    )

    # Create rolling window splits
    splits = create_rolling_window_splits(
        df,
        feature_cols,
        initial_train_days=initial_train_days,
        calibration_days=calibration_days,
        test_window_days=test_window_days,
        step_days=step_days
    )

    return splits
