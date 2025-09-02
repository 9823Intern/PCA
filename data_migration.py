"""
PCA Data Migration Tool - CSV to xarray with FactSet Integration
"""
import xarray as xr
import pandas as pd
import numpy as np
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PCAConfig:
    """Configuration for PCA data migration."""
    # Data paths
    georges_db_path: str
    adj_close_csv: str
    pct_change_csv: str
    adj_close_xarray: str
    pct_change_xarray: str
    legacy_figionly_csv: str
    benchmark_config: str
    backtester_path: str
    
    # CSV structure
    bloomberg_ticker_column: int
    figi_column: int
    first_date_column: int
    date_format: str
    
    # xarray settings
    primary_id_dim: str
    date_dim: str
    compression: str
    compression_level: int
    
    # Validation thresholds
    min_tickers: int
    min_dates: int
    max_daily_return_threshold: float
    min_price_threshold: float
    max_price_threshold: float
    
    # FactSet settings
    factset_enabled: bool
    update_frequency: str
    max_days_behind: int
    batch_size: int
    retry_attempts: int
    
    # PCA parameters
    z_window: int
    z_min_periods: int
    entry_threshold: float
    exit_threshold: float
    min_r2: float
    dynamic_zscore: bool
    dynamic_lookback: int
    volatility_adjustment: bool
    momentum_filter: bool
    momentum_threshold: float
    
    # Backtest parameters
    start_date: str
    end_date: str
    initial_capital: float
    transaction_cost: float
    max_position_size: float
    rebalance_frequency: str
    
    # Logging
    log_level: str
    log_format: str
    log_file: str

def load_config(config_path: str = "config.json") -> PCAConfig:
    """Load configuration from JSON file with defaults fallback."""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return PCAConfig(
            # Data paths
            georges_db_path=config_dict["data_paths"]["georges_db_path"],
            adj_close_csv=config_dict["data_paths"]["adj_close_csv"],
            pct_change_csv=config_dict["data_paths"]["pct_change_csv"],
            adj_close_xarray=config_dict["data_paths"]["adj_close_xarray"],
            pct_change_xarray=config_dict["data_paths"]["pct_change_xarray"],
            legacy_figionly_csv=config_dict["data_paths"]["legacy_figionly_csv"],
            benchmark_config=config_dict["data_paths"]["benchmark_config"],
            backtester_path=config_dict["data_paths"]["backtester_path"],
            
            # CSV structure
            bloomberg_ticker_column=config_dict["csv_structure"]["bloomberg_ticker_column"],
            figi_column=config_dict["csv_structure"]["figi_column"],
            first_date_column=config_dict["csv_structure"]["first_date_column"],
            date_format=config_dict["csv_structure"]["date_format"],
            
            # xarray settings
            primary_id_dim=config_dict["xarray_settings"]["primary_id_dim"],
            date_dim=config_dict["xarray_settings"]["date_dim"],
            compression=config_dict["xarray_settings"]["compression"],
            compression_level=config_dict["xarray_settings"]["compression_level"],
            
            # Validation
            min_tickers=config_dict["validation"]["min_tickers"],
            min_dates=config_dict["validation"]["min_dates"],
            max_daily_return_threshold=config_dict["validation"]["max_daily_return_threshold"],
            min_price_threshold=config_dict["validation"]["min_price_threshold"],
            max_price_threshold=config_dict["validation"]["max_price_threshold"],
            
            # FactSet
            factset_enabled=config_dict["factset_config"]["enabled"],
            update_frequency=config_dict["factset_config"]["update_frequency"],
            max_days_behind=config_dict["factset_config"]["max_days_behind"],
            batch_size=config_dict["factset_config"]["batch_size"],
            retry_attempts=config_dict["factset_config"]["retry_attempts"],
            
            # PCA parameters
            z_window=config_dict["pca_parameters"]["z_window"],
            z_min_periods=config_dict["pca_parameters"]["z_min_periods"],
            entry_threshold=config_dict["pca_parameters"]["entry_threshold"],
            exit_threshold=config_dict["pca_parameters"]["exit_threshold"],
            min_r2=config_dict["pca_parameters"]["min_r2"],
            dynamic_zscore=config_dict["pca_parameters"]["dynamic_zscore"],
            dynamic_lookback=config_dict["pca_parameters"]["dynamic_lookback"],
            volatility_adjustment=config_dict["pca_parameters"]["volatility_adjustment"],
            momentum_filter=config_dict["pca_parameters"]["momentum_filter"],
            momentum_threshold=config_dict["pca_parameters"]["momentum_threshold"],
            
            # Backtest parameters
            start_date=config_dict["backtest_parameters"]["start_date"],
            end_date=config_dict["backtest_parameters"]["end_date"],
            initial_capital=config_dict["backtest_parameters"]["initial_capital"],
            transaction_cost=config_dict["backtest_parameters"]["transaction_cost"],
            max_position_size=config_dict["backtest_parameters"]["max_position_size"],
            rebalance_frequency=config_dict["backtest_parameters"]["rebalance_frequency"],
            
            # Logging
            log_level=config_dict["logging"]["level"],
            log_format=config_dict["logging"]["format"],
            log_file=config_dict["logging"]["file"]
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"⚠️ Config error: {e}. Using defaults.")
        return get_default_config()

def detect_console_unicode_support() -> bool:
    """Detect if console supports Unicode characters."""
    try:
        # Test if we can write Unicode to stdout
        sys.stdout.write('\u2713')  # Check mark
        sys.stdout.flush()
        return True
    except UnicodeEncodeError:
        return False

def get_safe_message_formatter(use_unicode: bool = None) -> Dict[str, str]:
    """Get console-safe message symbols."""
    if use_unicode is None:
        use_unicode = detect_console_unicode_support()
    
    if use_unicode:
        return {
            'success': '✅',
            'processing': '🔄', 
            'data': '📊',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️',
            'update': '📡',
            'migrate': '🔄',
            'validate': '✅',
            'export': '📤',
            'debug': '🧪'
        }
    else:
        return {
            'success': '[OK]',
            'processing': '[PROC]',
            'data': '[DATA]',
            'warning': '[WARN]',
            'error': '[ERROR]',
            'info': '[INFO]',
            'update': '[UPDATE]',
            'migrate': '[MIGRATE]',
            'validate': '[VALIDATE]',
            'export': '[EXPORT]',
            'debug': '[DEBUG]'
        }

class SafeConsoleHandler(logging.StreamHandler):
    """Console handler that safely handles Unicode characters."""
    
    def __init__(self):
        super().__init__(sys.stdout)
        self.use_unicode = detect_console_unicode_support()
        self.symbols = get_safe_message_formatter(self.use_unicode)
    
    def emit(self, record):
        try:
            # Replace Unicode symbols in message if console doesn't support them
            if not self.use_unicode and hasattr(record, 'msg'):
                msg = str(record.msg)
                for unicode_char, safe_char in [
                    ('✅', '[OK]'), ('🔄', '[PROC]'), ('📊', '[DATA]'),
                    ('⚠️', '[WARN]'), ('❌', '[ERROR]'), ('ℹ️', '[INFO]'),
                    ('📡', '[UPDATE]'), ('📤', '[EXPORT]'), ('🧪', '[DEBUG]')
                ]:
                    msg = msg.replace(unicode_char, safe_char)
                record.msg = msg
            
            super().emit(record)
        except UnicodeEncodeError:
            # Fallback: strip all non-ASCII characters
            if hasattr(record, 'msg'):
                record.msg = str(record.msg).encode('ascii', 'ignore').decode('ascii')
            super().emit(record)

def get_default_config() -> PCAConfig:
    """Return default configuration."""
    return PCAConfig(
        georges_db_path="C:/Users/bearj/Precept Dropbox/Enn Turn/Georges_DataBase",
        adj_close_csv="Securities_Adj_Close_MASTER.csv",
        pct_change_csv="Securities_Daily_Pct_Change_MASTER.csv",
        adj_close_xarray="securities_adj_close_master.nc",
        pct_change_xarray="securities_pct_change_master.nc",
        legacy_figionly_csv="figionly.csv",
        benchmark_config="benchmark_config.json",
        backtester_path="../Backtester",
        bloomberg_ticker_column=0,
        figi_column=1,
        first_date_column=2,
        date_format="mixed",
        primary_id_dim="figi",
        date_dim="date",
        compression="gzip",
        compression_level=6,
        min_tickers=10,
        min_dates=100,
        max_daily_return_threshold=0.5,
        min_price_threshold=0.01,
        max_price_threshold=10000.0,
        factset_enabled=True,
        update_frequency="daily",
        max_days_behind=5,
        batch_size=100,
        retry_attempts=3,
        z_window=60,
        z_min_periods=30,
        entry_threshold=2.0,
        exit_threshold=0.5,
        min_r2=0.1,
        dynamic_zscore=False,
        dynamic_lookback=252,
        volatility_adjustment=True,
        momentum_filter=False,
        momentum_threshold=0.1,
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_capital=100000.0,
        transaction_cost=0.001,
        max_position_size=0.05,
        rebalance_frequency="daily",
        log_level="INFO",
        log_format="%(asctime)s - %(levelname)s - %(message)s",
        log_file="pca_migration.log"
    )

class PCADataManager:
    """Manages PCA data migration from CSV to xarray format."""

    def __init__(self, config: PCAConfig):
        self.config = config
        self.georges_db_path = Path(config.georges_db_path)
        self.adj_close_xr_path = self.georges_db_path / config.adj_close_xarray
        self.pct_change_xr_path = self.georges_db_path / config.pct_change_xarray
        
        # Setup console-safe logging
        self.symbols = get_safe_message_formatter()
        
        # Clear any existing handlers to avoid duplicates
        logging.getLogger().handlers.clear()
        
        # Setup logging with safe handlers
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format=config.log_format,
            handlers=[
                logging.FileHandler(config.log_file, encoding='utf-8'),  # File supports UTF-8
                SafeConsoleHandler()  # Console-safe handler
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def migrate_csv_to_xarray(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """Convert CSV files to xarray format using config settings."""
        
        self.logger.info(f"{self.symbols['migrate']} Starting CSV to xarray migration...")
        
        # Load CSV files
        adj_close_df = pd.read_csv(self.georges_db_path / self.config.adj_close_csv)
        pct_change_df = pd.read_csv(self.georges_db_path / self.config.pct_change_csv)
        
        # Process each dataframe
        adj_close_xr = self._csv_to_xarray(adj_close_df, "adj_close")
        pct_change_xr = self._csv_to_xarray(pct_change_df, "pct_change")
        
        # Save with compression
        encoding = {
            adj_close_xr.name: {
                'zlib': True,
                'complevel': self.config.compression_level
            }
        }
        
        adj_close_xr.to_netcdf(self.adj_close_xr_path, encoding=encoding)
        
        encoding[pct_change_xr.name] = encoding.pop(adj_close_xr.name)
        pct_change_xr.to_netcdf(self.pct_change_xr_path, encoding=encoding)
        
        self.logger.info(f"{self.symbols['success']} Migration complete:")
        self.logger.info(f"   Adj Close: {adj_close_xr.shape} -> {self.adj_close_xr_path}")
        self.logger.info(f"   Pct Change: {pct_change_xr.shape} -> {self.pct_change_xr_path}")
        
        return adj_close_xr, pct_change_xr
    
    def _csv_to_xarray(self, df: pd.DataFrame, data_type: str) -> xr.DataArray:
        """Convert single CSV to xarray using config settings."""
        
        if df.empty:
            raise ValueError(f"Empty DataFrame for {data_type}")
        
        # Extract ID columns using config
        bloomberg_ticker = df.iloc[:, self.config.bloomberg_ticker_column].astype(str).str.strip()
        figi = df.iloc[:, self.config.figi_column].astype(str).str.strip()
        
        # Validate minimum data requirements
        if len(figi) < self.config.min_tickers:
            raise ValueError(f"Insufficient tickers: {len(figi)} < {self.config.min_tickers}")
        
        # Extract and validate date columns
        date_cols = df.columns[self.config.first_date_column:].tolist()
        valid_date_info = []
        
        for i, col in enumerate(date_cols):
            try:
                parsed_date = pd.to_datetime(col, errors='raise', format=self.config.date_format)
                valid_date_info.append((i + self.config.first_date_column, col, parsed_date))
            except:
                self.logger.warning(f"{self.symbols['warning']} Skipping invalid date column: {col}")
                continue
        
        if len(valid_date_info) < self.config.min_dates:
            raise ValueError(f"Insufficient dates: {len(valid_date_info)} < {self.config.min_dates}")
        
        # Sort by parsed date
        valid_date_info.sort(key=lambda x: x[2])
        
        # Extract data matrix
        valid_positions = [info[0] for info in valid_date_info]
        valid_dates = pd.DatetimeIndex([info[2] for info in valid_date_info])
        
        data_matrix = df.iloc[:, valid_positions].values.astype(float)
        
        # Validation checks
        self._validate_data_matrix(data_matrix, data_type)
        
        # Create xarray
        xr_data = xr.DataArray(
            data_matrix,
            dims=[self.config.primary_id_dim, self.config.date_dim],
            coords={
                self.config.primary_id_dim: figi,
                'bloomberg_ticker': (self.config.primary_id_dim, bloomberg_ticker),
                self.config.date_dim: valid_dates
            },
            name=data_type,
            attrs={
                'description': f'Security {data_type} data',
                'source': 'Georges_Database_CSV_Migration',
                'primary_id': self.config.primary_id_dim,
                'secondary_id': 'bloomberg_ticker',
                'created_at': pd.Timestamp.now().isoformat(),
                'config_used': str(self.config)
            }
        )
        
        return xr_data
    
    def _validate_data_matrix(self, data_matrix: np.ndarray, data_type: str) -> None:
        """Validate data matrix against config thresholds."""
        
        if data_type == "adj_close":
            # Price validation
            valid_prices = data_matrix[~np.isnan(data_matrix)]
            if len(valid_prices) > 0:
                min_price, max_price = valid_prices.min(), valid_prices.max()
                if min_price < self.config.min_price_threshold:
                    self.logger.warning(f"{self.symbols['warning']} Low prices detected: min={min_price}")
                if max_price > self.config.max_price_threshold:
                    self.logger.warning(f"{self.symbols['warning']} High prices detected: max={max_price}")
        
        elif data_type == "pct_change":
            # Return validation
            valid_returns = data_matrix[~np.isnan(data_matrix)]
            if len(valid_returns) > 0:
                extreme_returns = np.abs(valid_returns) > self.config.max_daily_return_threshold
                if extreme_returns.any():
                    count = extreme_returns.sum()
                    self.logger.warning(f"{self.symbols['warning']} Extreme returns detected: {count} observations > {self.config.max_daily_return_threshold}")
    
    def validate_migration(self, adj_close_xr: xr.DataArray, pct_change_xr: xr.DataArray) -> bool:
        """Validate migrated data using config thresholds."""
        
        self.logger.info(f"{self.symbols['validate']} Validating migrated data...")
        
        # Shape validation
        if adj_close_xr.shape != pct_change_xr.shape:
            self.logger.error(f"Shape mismatch: adj_close {adj_close_xr.shape} vs pct_change {pct_change_xr.shape}")
            return False
        
        # Coordinate validation
        if not adj_close_xr.figi.equals(pct_change_xr.figi):
            self.logger.error("FIGI coordinates don't match")
            return False
        
        if not adj_close_xr.date.equals(pct_change_xr.date):
            self.logger.error("Date coordinates don't match")
            return False
        
        # Minimum data requirements
        n_tickers, n_dates = adj_close_xr.shape
        if n_tickers < self.config.min_tickers:
            self.logger.error(f"Insufficient tickers: {n_tickers} < {self.config.min_tickers}")
            return False
        
        if n_dates < self.config.min_dates:
            self.logger.error(f"Insufficient dates: {n_dates} < {self.config.min_dates}")
            return False
        
        # Data quality checks
        adj_stats = {
            'min': float(adj_close_xr.min()),
            'max': float(adj_close_xr.max()),
            'mean': float(adj_close_xr.mean()),
            'nan_pct': float(adj_close_xr.isnull().mean() * 100)
        }
        
        pct_stats = {
            'min': float(pct_change_xr.min()),
            'max': float(pct_change_xr.max()),
            'mean': float(pct_change_xr.mean()),
            'nan_pct': float(pct_change_xr.isnull().mean() * 100)
        }
        
        self.logger.info(f"Adj Close stats: {adj_stats}")
        self.logger.info(f"Pct Change stats: {pct_stats}")
        
        # Flag potential issues
        if abs(pct_stats['mean']) > 0.01:
            self.logger.warning(f"High mean daily return: {pct_stats['mean']:.4f}")
        
        if pct_stats['nan_pct'] > 50:
            self.logger.warning(f"High NaN percentage: {pct_stats['nan_pct']:.1f}%")
        
        self.logger.info(f"{self.symbols['success']} Validation passed")
        return True
    
    def load_xarray_data(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """Load existing xarray data from netCDF files."""
        if not self.adj_close_xr_path.exists():
            raise FileNotFoundError(f"xarray file not found: {self.adj_close_xr_path}")
        
        adj_close = xr.open_dataarray(self.adj_close_xr_path)
        pct_change = xr.open_dataarray(self.pct_change_xr_path)
        
        self.logger.info(f"{self.symbols['data']} Loaded xarray data: {adj_close.shape}")
        return adj_close, pct_change
    
    def get_pandas_returns_for_pca(self) -> pd.DataFrame:
        """
        Convert xarray pct_change to pandas format for legacy compatibility.
        
        Returns: DataFrame with FIGI as index, dates as columns (existing format)
        """
        _, pct_change_xr = self.load_xarray_data()
        
        # Convert to pandas with FIGI as index, dates as columns
        df = pct_change_xr.to_pandas()
        df.index.name = "FIGI"
        df.index = df.index.map(str).str.strip().str.upper()
        
        self.logger.info(f"{self.symbols['export']} Converted to pandas format: {df.shape}")
        return df
    
    def get_xarray_returns(self) -> xr.DataArray:
        """
        Get xarray pct_change data directly for optimized processing.
        
        Returns: xarray.DataArray with dims=['figi', 'date']
        """
        _, pct_change_xr = self.load_xarray_data()
        
        # Ensure proper normalization
        pct_change_xr = pct_change_xr.assign_coords(
            figi=pct_change_xr.figi.astype(str).str.strip().str.upper()
        )
        
        self.logger.info(f"{self.symbols['data']} Loaded xarray returns: {pct_change_xr.shape}")
        return pct_change_xr
    
    def get_benchmark_subset_xarray(self, benchmark_tickers: list, returns_xr: xr.DataArray = None) -> xr.DataArray:
        """
        Get xarray subset for specific benchmark tickers.
        
        Args:
            benchmark_tickers: List of tickers for the benchmark
            returns_xr: Optional pre-loaded returns xarray
            
        Returns: Filtered xarray with only benchmark tickers
        """
        if returns_xr is None:
            returns_xr = self.get_xarray_returns()
        
        # Normalize ticker list
        normalized_tickers = [str(t).strip().upper() for t in benchmark_tickers]
        
        # Filter to available tickers
        available_tickers = [t for t in normalized_tickers if t in returns_xr.figi.values]
        
        if len(available_tickers) < 2:
            self.logger.warning(f"{self.symbols['warning']} Insufficient tickers for benchmark: {len(available_tickers)} < 2")
            return xr.DataArray()
        
        # Select subset
        subset = returns_xr.sel(figi=available_tickers)
        
        # Drop dates with all NaN for this subset
        valid_dates = ~subset.isnull().all(dim='figi')
        subset = subset.sel(date=subset.date[valid_dates])
        
        self.logger.info(f"{self.symbols['data']} Benchmark subset: {subset.shape} ({len(available_tickers)} tickers)")
        return subset
    
    def get_bloomberg_ticker_mapping(self) -> Dict[str, str]:
        """Extract FIGI -> Bloomberg Ticker mapping from xarray."""
        adj_close_xr, _ = self.load_xarray_data()
        
        mapping = dict(zip(
            adj_close_xr.figi.values,
            adj_close_xr.bloomberg_ticker.values
        ))
        
        self.logger.info(f"{self.symbols['data']} Extracted ticker mapping: {len(mapping)} entries")
        return mapping
    
    def update_with_factset(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Update xarray data with latest FactSet prices.
        
        Process:
        1. Check for missing dates since last update
        2. Fetch new adj close prices via FactSet
        3. Calculate new pct changes
        4. Append to xarray and save
        """
        if not self.config.factset_enabled:
            self.logger.info(f"{self.symbols['info']} FactSet updates disabled in config")
            return self.load_xarray_data()
        
        try:
            # Import FactSet interface
            import sys
            sys.path.append(str(Path(__file__).parent.parent / "Backtester"))
            from backtester.market_data_apis import FactsetInterface
        except ImportError as e:
            self.logger.error(f"FactSet import failed: {e}")
            return self.load_xarray_data()
        
        # Load existing data
        adj_close_xr, pct_change_xr = self.load_xarray_data()
        
        # Check if update needed
        last_date = adj_close_xr.date.max().values
        last_date_pd = pd.Timestamp(last_date)
        yesterday = pd.Timestamp.now().normalize() - timedelta(days=1)
        
        days_behind = (yesterday - last_date_pd).days
        
        if days_behind <= 0:
            self.logger.info(f"{self.symbols['success']} Data is up to date")
            return adj_close_xr, pct_change_xr
        
        if days_behind > self.config.max_days_behind:
            self.logger.warning(f"{self.symbols['warning']} Data is {days_behind} days behind (max: {self.config.max_days_behind})")
        
        # Fetch new data
        start_fetch = (last_date_pd + timedelta(days=1)).strftime('%Y-%m-%d')
        end_fetch = yesterday.strftime('%Y-%m-%d')
        
        self.logger.info(f"{self.symbols['update']} Fetching FactSet data: {start_fetch} to {end_fetch}")
        
        figis = adj_close_xr.figi.values.tolist()
        factset = FactsetInterface()
        
        try:
            new_price_data = factset.get_share_price(figis, start_fetch, end_fetch)
            
            if new_price_data is None or new_price_data.sizes['date'] == 0:
                self.logger.info(f"{self.symbols['info']} No new data available from FactSet")
                return adj_close_xr, pct_change_xr
            
            # Merge new data with existing
            updated_adj_close = xr.concat([adj_close_xr, new_price_data], dim='date')
            
            # Calculate new pct changes
            updated_pct_change = self._calculate_incremental_pct_change(
                adj_close_xr, new_price_data, pct_change_xr
            )
            
            # Save updated data
            encoding = {
                updated_adj_close.name: {
                    'zlib': True,
                    'complevel': self.config.compression_level
                }
            }
            
            updated_adj_close.to_netcdf(self.adj_close_xr_path, encoding=encoding)
            
            encoding[updated_pct_change.name] = encoding.pop(updated_adj_close.name)
            updated_pct_change.to_netcdf(self.pct_change_xr_path, encoding=encoding)
            
            self.logger.info(f"{self.symbols['success']} Updated data: {new_price_data.sizes['date']} new dates added")
            
            return updated_adj_close, updated_pct_change
            
        except Exception as e:
            self.logger.error(f"{self.symbols['error']} FactSet update failed: {e}")
            return adj_close_xr, pct_change_xr
    
    def _calculate_incremental_pct_change(self, 
                                        old_adj_close: xr.DataArray,
                                        new_adj_close: xr.DataArray, 
                                        old_pct_change: xr.DataArray) -> xr.DataArray:
        """Calculate pct change for new dates and merge with existing."""
        
        # Get last price from old data
        last_prices = old_adj_close.isel(date=-1)
        
        # Calculate pct change for new dates
        new_dates = new_adj_close.date.values
        new_pct_changes = []
        
        for i, date in enumerate(new_dates):
            if i == 0:
                # First new date: compare to last old date
                current_prices = new_adj_close.isel(date=i)
                pct_change = (current_prices / last_prices) - 1
            else:
                # Subsequent dates: compare to previous new date
                current_prices = new_adj_close.isel(date=i)
                prev_prices = new_adj_close.isel(date=i-1)
                pct_change = (current_prices / prev_prices) - 1
            
            new_pct_changes.append(pct_change)
        
        # Stack new pct changes
        new_pct_xr = xr.concat(new_pct_changes, dim='date')
        new_pct_xr = new_pct_xr.assign_coords(date=new_dates)
        
        # Merge with existing
        updated_pct_change = xr.concat([old_pct_change, new_pct_xr], dim='date')
        
        return updated_pct_change

# ==============================================
# XARRAY-NATIVE PCA FUNCTIONS FOR FAST PROCESSING
# ==============================================

class XArrayPCAProcessor:
    """High-performance PCA processing using xarray operations."""
    
    def __init__(self, data_manager: PCADataManager):
        self.data_manager = data_manager
        self.logger = data_manager.logger
        self.symbols = data_manager.symbols
    
    def run_benchmark_pca_xarray(self, 
                                benchmark_name: str,
                                benchmark_tickers: list,
                                returns_xr: xr.DataArray = None,
                                n_components: int = 2) -> Dict[str, xr.DataArray]:
        """
        Run PCA on benchmark using pure xarray operations for speed.
        
        Returns:
            Dict with 'pc_scores', 'loadings', 'explained_variance'
        """
        if returns_xr is None:
            returns_xr = self.data_manager.get_xarray_returns()
        
        # Get benchmark subset
        subset = self.data_manager.get_benchmark_subset_xarray(benchmark_tickers, returns_xr)
        
        if subset.size == 0:
            self.logger.warning(f"{self.symbols['warning']} Empty subset for {benchmark_name}")
            return {}
        
        # Drop NaN values (complete cases only)
        subset_clean = subset.dropna(dim='date', how='any')
        
        if subset_clean.sizes['date'] < 30:
            self.logger.warning(f"{self.symbols['warning']} Insufficient clean data for {benchmark_name}: {subset_clean.sizes['date']} days")
            return {}
        
        self.logger.info(f"{self.symbols['processing']} Running PCA for {benchmark_name}: {subset_clean.shape}")
        
        # Convert to numpy for sklearn (dates x tickers)
        X = subset_clean.values.T  # Transpose to (dates, tickers)
        
        # Standardize
        X_mean = np.nanmean(X, axis=0, keepdims=True)
        X_std = np.nanstd(X, axis=0, keepdims=True)
        X_std = np.where(X_std == 0, 1, X_std)  # Avoid division by zero
        X_standardized = (X - X_mean) / X_std
        
        # Run PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
        pc_scores = pca.fit_transform(X_standardized)
        
        # Convert back to xarray
        pc_scores_xr = xr.DataArray(
            pc_scores,
            dims=['date', 'component'],
            coords={
                'date': subset_clean.date,
                'component': [f'PC{i+1}' for i in range(pc_scores.shape[1])]
            },
            name=f'{benchmark_name}_pc_scores'
        )
        
        loadings_xr = xr.DataArray(
            pca.components_.T,
            dims=['figi', 'component'],
            coords={
                'figi': subset_clean.figi,
                'component': [f'PC{i+1}' for i in range(pca.components_.shape[0])]
            },
            name=f'{benchmark_name}_loadings'
        )
        
        explained_var_xr = xr.DataArray(
            pca.explained_variance_ratio_,
            dims=['component'],
            coords={'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))]},
            name=f'{benchmark_name}_explained_variance'
        )
        
        return {
            'pc_scores': pc_scores_xr,
            'loadings': loadings_xr,
            'explained_variance': explained_var_xr,
            'subset_data': subset_clean
        }
    
    def regress_stock_on_pc1_xarray(self, 
                                   stock_returns: xr.DataArray,
                                   pc1_series: xr.DataArray) -> Dict[str, xr.DataArray]:
        """
        Regress individual stock on PC1 using xarray operations.
        
        Returns: Dict with 'alpha', 'beta', 'r2', 'residuals'
        """
        # Align data
        stock_aligned, pc1_aligned = xr.align(stock_returns, pc1_series, join='inner')
        
        # Drop NaN values
        valid_mask = ~(stock_aligned.isnull() | pc1_aligned.isnull())
        y = stock_aligned.where(valid_mask, drop=True)
        x = pc1_aligned.where(valid_mask, drop=True)
        
        if len(y) < 30:
            # Insufficient data for regression
            return {
                'alpha': xr.DataArray(np.nan),
                'beta': xr.DataArray(np.nan),
                'r2': xr.DataArray(np.nan),
                'residuals': stock_returns * np.nan
            }
        
        # Calculate regression coefficients using xarray
        x_mean = x.mean()
        y_mean = y.mean()
        
        # Beta = Cov(x,y) / Var(x)
        cov_xy = ((x - x_mean) * (y - y_mean)).mean()
        var_x = ((x - x_mean) ** 2).mean()
        
        beta = cov_xy / var_x
        alpha = y_mean - beta * x_mean
        
        # R-squared
        y_pred = alpha + beta * x
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)
        
        # Calculate residuals for full time series
        pc1_full = pc1_series.reindex_like(stock_returns, method='nearest')
        predicted_full = alpha + beta * pc1_full
        residuals_full = stock_returns - predicted_full
        
        return {
            'alpha': alpha,
            'beta': beta, 
            'r2': r2,
            'residuals': residuals_full
        }
    
    def rolling_zscore_xarray(self, 
                             data: xr.DataArray,
                             window: int = None,
                             min_periods: int = None,
                             dynamic: bool = None) -> xr.DataArray:
        """
        Calculate rolling z-scores using xarray operations with optional dynamic scaling.
        
        Args:
            data: Input time series
            window: Rolling window size
            min_periods: Minimum periods for calculation
            dynamic: Use dynamic z-scoring based on volatility regime
        """
        if window is None:
            window = self.data_manager.config.z_window
        if min_periods is None:
            min_periods = self.data_manager.config.z_min_periods
        if dynamic is None:
            dynamic = self.data_manager.config.dynamic_zscore
        
        if dynamic:
            return self._dynamic_zscore_xarray(data, window, min_periods)
        else:
            return self._static_zscore_xarray(data, window, min_periods)
    
    def _static_zscore_xarray(self, data: xr.DataArray, window: int, min_periods: int) -> xr.DataArray:
        """Standard rolling z-score calculation."""
        rolling_mean = data.rolling(date=window, min_periods=min_periods, center=False).mean()
        rolling_std = data.rolling(date=window, min_periods=min_periods, center=False).std()
        z_scores = (data - rolling_mean) / rolling_std
        return z_scores
    
    def _dynamic_zscore_xarray(self, data: xr.DataArray, window: int, min_periods: int) -> xr.DataArray:
        """
        Dynamic z-scoring that adjusts for volatility regime.
        
        Uses longer lookback during high volatility periods for more stable thresholds.
        """
        # Calculate base z-scores
        base_z = self._static_zscore_xarray(data, window, min_periods)
        
        # Calculate volatility regime (rolling volatility of the data itself)
        vol_window = self.data_manager.config.dynamic_lookback
        rolling_vol = data.rolling(date=vol_window, min_periods=vol_window//2).std()
        vol_zscore = self._static_zscore_xarray(rolling_vol, vol_window//4, vol_window//8)
        
        # Adjust z-scores based on volatility regime
        # High vol regime (vol_zscore > 1): Use longer window for more stable thresholds
        # Low vol regime (vol_zscore < -1): Use shorter window for more responsive signals
        
        high_vol_mask = vol_zscore > 1.0
        low_vol_mask = vol_zscore < -1.0
        
        # Calculate alternative z-scores for different regimes
        long_window_z = self._static_zscore_xarray(data, int(window * 1.5), min_periods)
        short_window_z = self._static_zscore_xarray(data, int(window * 0.7), min_periods)
        
        # Combine based on volatility regime
        dynamic_z = xr.where(high_vol_mask, long_window_z, base_z)
        dynamic_z = xr.where(low_vol_mask, short_window_z, dynamic_z)
        
        return dynamic_z
    
    def stat_arb_signals_xarray(self,
                               z_scores: xr.DataArray,
                               entry: float = None,
                               exit_: float = None,
                               returns_data: xr.DataArray = None) -> xr.DataArray:
        """
        Generate stat arb signals using xarray operations with enhanced logic.
        
        Args:
            z_scores: Z-score time series
            entry: Entry threshold
            exit_: Exit threshold  
            returns_data: Optional returns data for momentum filtering
        """
        if entry is None:
            entry = self.data_manager.config.entry_threshold
        if exit_ is None:
            exit_ = self.data_manager.config.exit_threshold
        
        # Apply volatility adjustment to thresholds
        if self.data_manager.config.volatility_adjustment and returns_data is not None:
            entry, exit_ = self._adjust_thresholds_for_volatility(entry, exit_, returns_data)
        
        # Initialize signals
        signals = xr.zeros_like(z_scores, dtype=int)
        
        # Vectorized signal generation per stock
        for figi in z_scores.figi.values:
            z_stock = z_scores.sel(figi=figi)
            
            # Apply momentum filter if enabled
            if self.data_manager.config.momentum_filter and returns_data is not None:
                momentum = self._calculate_momentum(returns_data.sel(figi=figi))
                if not self._momentum_filter_passed(momentum):
                    continue  # Skip this stock due to momentum filter
            
            signal_stock = xr.zeros_like(z_stock, dtype=int)
            
            # Convert to numpy for state-based logic
            z_vals = z_stock.values
            sig_vals = signal_stock.values
            
            pos = 0
            for i, z_val in enumerate(z_vals):
                if np.isnan(z_val):
                    sig_vals[i] = 0
                    continue
                
                # Exit if within band
                if abs(z_val) <= exit_:
                    pos = 0
                else:
                    # Enter positions when outside entry thresholds
                    if z_val >= entry:
                        pos = -1  # Short overvalued
                    elif z_val <= -entry:
                        pos = 1   # Long undervalued
                
                sig_vals[i] = pos
            
            # Update signals for this stock
            signals.loc[{'figi': figi}] = sig_vals
        
        return signals
    
    def _adjust_thresholds_for_volatility(self, entry: float, exit_: float, returns_data: xr.DataArray) -> tuple:
        """Adjust entry/exit thresholds based on current volatility regime."""
        # Calculate recent volatility (last 20 days)
        recent_vol = returns_data.rolling(date=20, min_periods=10).std().isel(date=-1)
        median_vol = recent_vol.median()
        
        # Adjust thresholds: higher vol = higher thresholds
        vol_multiplier = (recent_vol / median_vol).clip(0.5, 2.0)
        
        adjusted_entry = entry * vol_multiplier.mean().values
        adjusted_exit = exit_ * vol_multiplier.mean().values
        
        return float(adjusted_entry), float(adjusted_exit)
    
    def _calculate_momentum(self, returns_series: xr.DataArray, lookback: int = 20) -> xr.DataArray:
        """Calculate momentum score for momentum filtering."""
        # Simple momentum: cumulative return over lookback period
        momentum = (1 + returns_series).rolling(date=lookback, min_periods=lookback//2).apply(np.prod) - 1
        return momentum
    
    def _momentum_filter_passed(self, momentum: xr.DataArray) -> bool:
        """Check if momentum filter criteria are met."""
        recent_momentum = momentum.isel(date=-1)
        threshold = self.data_manager.config.momentum_threshold
        
        # Pass filter if momentum is within acceptable range (not too trending)
        return abs(float(recent_momentum)) <= threshold
    
    def process_benchmark_xarray(self,
                                benchmark_name: str,
                                benchmark_tickers: list,
                                returns_xr: xr.DataArray = None) -> Dict[str, xr.DataArray]:
        """
        Complete benchmark processing using pure xarray operations.
        
        Returns comprehensive results for the benchmark.
        """
        self.logger.info(f"{self.symbols['processing']} Processing benchmark {benchmark_name} with xarray...")
        
        if returns_xr is None:
            returns_xr = self.data_manager.get_xarray_returns()
        
        # Run PCA
        pca_results = self.run_benchmark_pca_xarray(benchmark_name, benchmark_tickers, returns_xr)
        
        if not pca_results:
            return {}
        
        subset_data = pca_results['subset_data']
        pc1_scores = pca_results['pc_scores'].sel(component='PC1')
        
        # Process each stock in the benchmark
        stock_results = {}
        
        for figi in subset_data.figi.values:
            stock_returns = subset_data.sel(figi=figi)
            
            # Regression on PC1
            reg_results = self.regress_stock_on_pc1_xarray(stock_returns, pc1_scores)
            
            # Rolling z-scores
            z_scores = self.rolling_zscore_xarray(reg_results['residuals'])
            
            # Generate signals
            signals = self.stat_arb_signals_xarray(z_scores)
            
            stock_results[figi] = {
                'alpha': reg_results['alpha'],
                'beta': reg_results['beta'],
                'r2': reg_results['r2'],
                'z_scores': z_scores,
                'signals': signals,
                'residuals': reg_results['residuals']
            }
        
        # Combine results
        all_z_scores = xr.concat([stock_results[figi]['z_scores'] for figi in stock_results.keys()], 
                                dim='figi')
        all_z_scores = all_z_scores.assign_coords(figi=list(stock_results.keys()))
        
        all_signals = xr.concat([stock_results[figi]['signals'] for figi in stock_results.keys()], 
                               dim='figi')
        all_signals = all_signals.assign_coords(figi=list(stock_results.keys()))
        
        # Summary statistics
        summary_stats = xr.Dataset({
            'alpha': xr.DataArray([stock_results[figi]['alpha'].values for figi in stock_results.keys()],
                                 dims=['figi'], coords={'figi': list(stock_results.keys())}),
            'beta': xr.DataArray([stock_results[figi]['beta'].values for figi in stock_results.keys()],
                                dims=['figi'], coords={'figi': list(stock_results.keys())}),
            'r2': xr.DataArray([stock_results[figi]['r2'].values for figi in stock_results.keys()],
                              dims=['figi'], coords={'figi': list(stock_results.keys())})
        })
        
        self.logger.info(f"{self.symbols['success']} Completed {benchmark_name}: {len(stock_results)} stocks processed")
        
        return {
            'benchmark_name': benchmark_name,
            'pca_results': pca_results,
            'z_scores': all_z_scores,
            'signals': all_signals,
            'summary_stats': summary_stats,
            'stock_results': stock_results
        }

# ==============================================
# BACKTEST FUNCTIONALITY
# ==============================================

class XArrayBacktester:
    """Fast backtesting using xarray operations."""
    
    def __init__(self, data_manager: PCADataManager):
        self.data_manager = data_manager
        self.logger = data_manager.logger
        self.symbols = data_manager.symbols
        self.processor = XArrayPCAProcessor(data_manager)
    
    def run_backtest(self, benchmarks: dict, **override_params) -> Dict[str, Dict]:
        """
        Run complete backtest across all benchmarks using xarray.
        
        Args:
            benchmarks: Dict of benchmark_name -> ticker_list
            **override_params: Override config parameters
        """
        self.logger.info(f"{self.symbols['processing']} Starting xarray backtest...")
        
        # Load data
        returns_xr = self.data_manager.get_xarray_returns()
        
        # Filter date range if specified
        start_date = override_params.get('start_date', self.data_manager.config.start_date)
        end_date = override_params.get('end_date', self.data_manager.config.end_date)
        
        if start_date and end_date:
            date_mask = (returns_xr.date >= pd.to_datetime(start_date)) & (returns_xr.date <= pd.to_datetime(end_date))
            returns_xr = returns_xr.sel(date=returns_xr.date[date_mask])
            self.logger.info(f"{self.symbols['data']} Filtered to date range: {start_date} to {end_date}")
        
        # Process each benchmark
        benchmark_results = {}
        
        for benchmark_name, tickers in benchmarks.items():
            self.logger.info(f"{self.symbols['processing']} Backtesting {benchmark_name}...")
            
            try:
                result = self.processor.process_benchmark_xarray(benchmark_name, tickers, returns_xr)
                if result:
                    # Calculate backtest metrics
                    performance = self._calculate_backtest_performance(result, returns_xr)
                    result['performance'] = performance
                    benchmark_results[benchmark_name] = result
                    
                    self.logger.info(f"{self.symbols['success']} {benchmark_name}: Sharpe={performance.get('sharpe', 'N/A'):.2f}")
                
            except Exception as e:
                self.logger.error(f"{self.symbols['error']} Failed to backtest {benchmark_name}: {e}")
                continue
        
        return benchmark_results
    
    def _calculate_backtest_performance(self, benchmark_result: dict, returns_xr: xr.DataArray) -> dict:
        """Calculate performance metrics for a benchmark result."""
        
        signals = benchmark_result['signals']
        
        # Get returns for the same stocks and dates
        stock_figis = signals.figi.values
        signal_dates = signals.date.values
        
        # Align returns data
        aligned_returns = returns_xr.sel(figi=stock_figis, date=signal_dates)
        
        # Lag signals by 1 day (signal at close, trade next day)
        lagged_signals = signals.shift(date=1).fillna(0)
        
        # Calculate strategy returns (element-wise multiplication then sum across stocks)
        strategy_returns_xr = (lagged_signals * aligned_returns).sum(dim='figi')
        
        # Convert to pandas for performance calculations
        strategy_returns = strategy_returns_xr.to_pandas().dropna()
        
        if len(strategy_returns) == 0:
            return {'error': 'No valid returns'}
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + strategy_returns.mean()) ** 252 - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Additional metrics
        win_rate = (strategy_returns > 0).mean()
        profit_factor = strategy_returns[strategy_returns > 0].sum() / abs(strategy_returns[strategy_returns < 0].sum()) if (strategy_returns < 0).any() else np.inf
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_trades': (lagged_signals != 0).sum().values,
            'active_days': (lagged_signals != 0).any(dim='figi').sum().values
        }

def setup_argparse() -> argparse.ArgumentParser:
    """Setup comprehensive argument parser with all terminal options."""
    
    parser = argparse.ArgumentParser(
        description='PCA Statistical Arbitrage Tool - Config-Driven with Terminal Control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
USAGE EXAMPLES:
  # Basic migration
  python data_migration.py --action migrate
  
  # Quick backtest with custom parameters
  python data_migration.py --action backtest --entry-threshold 2.5 --dynamic-zscore
  
  # Live analysis with FactSet updates
  python data_migration.py --action update --action analyze --factset-enabled
  
  # Custom date range backtest
  python data_migration.py --action backtest --start-date 2022-01-01 --end-date 2023-12-31
  
  # Dynamic z-scoring with momentum filter
  python data_migration.py --action analyze --dynamic-zscore --momentum-filter --momentum-threshold 0.05
        """
    )
    
    # Config file
    parser.add_argument('--config', 
                       default='config.json',
                       help='Path to configuration file')
    
    # Actions
    parser.add_argument('--action', 
                       choices=['migrate', 'validate', 'update', 'info', 'export-pandas', 'analyze', 'backtest', 'help-config'],
                       default='migrate',
                       help='Action to perform')
    
    # === DATA PATHS ===
    parser.add_argument('--georges-db-path',
                       help='Override Georges database path')
    
    parser.add_argument('--benchmark-config',
                       help='Override benchmark config file path')
    
    # === PCA PARAMETERS ===
    parser.add_argument('--z-window', type=int,
                       help='Z-score rolling window size')
    
    parser.add_argument('--z-min-periods', type=int,
                       help='Minimum periods for z-score calculation')
    
    parser.add_argument('--entry-threshold', type=float,
                       help='Signal entry threshold')
    
    parser.add_argument('--exit-threshold', type=float,
                       help='Signal exit threshold')
    
    parser.add_argument('--min-r2', type=float,
                       help='Minimum R-squared for stock inclusion')
    
    # === DYNAMIC Z-SCORING ===
    parser.add_argument('--dynamic-zscore', action='store_true',
                       help='Enable dynamic z-scoring based on volatility regime')
    
    parser.add_argument('--no-dynamic-zscore', action='store_true',
                       help='Disable dynamic z-scoring (use static)')
    
    parser.add_argument('--dynamic-lookback', type=int,
                       help='Lookback period for dynamic z-score volatility calculation')
    
    parser.add_argument('--volatility-adjustment', action='store_true',
                       help='Adjust thresholds based on current volatility')
    
    parser.add_argument('--no-volatility-adjustment', action='store_true',
                       help='Disable volatility adjustment')
    
    # === MOMENTUM FILTERING ===
    parser.add_argument('--momentum-filter', action='store_true',
                       help='Enable momentum filtering (avoid trending stocks)')
    
    parser.add_argument('--no-momentum-filter', action='store_true',
                       help='Disable momentum filtering')
    
    parser.add_argument('--momentum-threshold', type=float,
                       help='Momentum threshold for filtering (lower = more restrictive)')
    
    # === BACKTEST PARAMETERS ===
    parser.add_argument('--start-date',
                       help='Backtest start date (YYYY-MM-DD)')
    
    parser.add_argument('--end-date',
                       help='Backtest end date (YYYY-MM-DD)')
    
    parser.add_argument('--initial-capital', type=float,
                       help='Initial capital for backtesting')
    
    parser.add_argument('--transaction-cost', type=float,
                       help='Transaction cost per trade (as decimal)')
    
    parser.add_argument('--max-position-size', type=float,
                       help='Maximum position size as fraction of capital')
    
    # === FACTSET OPTIONS ===
    parser.add_argument('--factset-enabled', action='store_true',
                       help='Enable FactSet updates')
    
    parser.add_argument('--no-factset', action='store_true',
                       help='Disable FactSet updates')
    
    parser.add_argument('--batch-size', type=int,
                       help='FactSet batch size for API calls')
    
    parser.add_argument('--max-days-behind', type=int,
                       help='Maximum days behind before warning')
    
    # === OUTPUT OPTIONS ===
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Override logging level')
    
    parser.add_argument('--output-format',
                       choices=['table', 'json', 'csv'],
                       default='table',
                       help='Output format for results')
    
    parser.add_argument('--save-results',
                       help='Save results to specified file path')
    
    # === ANALYSIS OPTIONS ===
    parser.add_argument('--benchmark',
                       help='Run analysis on specific benchmark only')
    
    parser.add_argument('--top-signals', type=int,
                       help='Show only top N signals by absolute z-score')
    
    parser.add_argument('--min-signal-strength', type=float,
                       help='Minimum signal strength (absolute z-score) to display')
    
    # === UTILITY OPTIONS ===
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress non-essential output')
    
    return parser

def apply_argparse_overrides(config: PCAConfig, args: argparse.Namespace) -> PCAConfig:
    """Apply comprehensive command line argument overrides to config."""
    
    # Create a copy to avoid modifying original
    config_dict = config.__dict__.copy()
    
    # === DATA PATHS ===
    if args.georges_db_path:
        config_dict['georges_db_path'] = args.georges_db_path
    
    if args.benchmark_config:
        config_dict['benchmark_config'] = args.benchmark_config
    
    # === PCA PARAMETERS ===
    if args.z_window:
        config_dict['z_window'] = args.z_window
    
    if args.z_min_periods:
        config_dict['z_min_periods'] = args.z_min_periods
    
    if args.entry_threshold:
        config_dict['entry_threshold'] = args.entry_threshold
    
    if args.exit_threshold:
        config_dict['exit_threshold'] = args.exit_threshold
    
    if args.min_r2:
        config_dict['min_r2'] = args.min_r2
    
    # === DYNAMIC Z-SCORING ===
    if args.dynamic_zscore:
        config_dict['dynamic_zscore'] = True
    
    if args.no_dynamic_zscore:
        config_dict['dynamic_zscore'] = False
    
    if args.dynamic_lookback:
        config_dict['dynamic_lookback'] = args.dynamic_lookback
    
    if args.volatility_adjustment:
        config_dict['volatility_adjustment'] = True
    
    if args.no_volatility_adjustment:
        config_dict['volatility_adjustment'] = False
    
    # === MOMENTUM FILTERING ===
    if args.momentum_filter:
        config_dict['momentum_filter'] = True
    
    if args.no_momentum_filter:
        config_dict['momentum_filter'] = False
    
    if args.momentum_threshold:
        config_dict['momentum_threshold'] = args.momentum_threshold
    
    # === BACKTEST PARAMETERS ===
    if args.start_date:
        config_dict['start_date'] = args.start_date
    
    if args.end_date:
        config_dict['end_date'] = args.end_date
    
    if args.initial_capital:
        config_dict['initial_capital'] = args.initial_capital
    
    if args.transaction_cost:
        config_dict['transaction_cost'] = args.transaction_cost
    
    if args.max_position_size:
        config_dict['max_position_size'] = args.max_position_size
    
    # === FACTSET OPTIONS ===
    if args.factset_enabled:
        config_dict['factset_enabled'] = True
    
    if args.no_factset:
        config_dict['factset_enabled'] = False
    
    if args.batch_size:
        config_dict['batch_size'] = args.batch_size
    
    if args.max_days_behind:
        config_dict['max_days_behind'] = args.max_days_behind
    
    # === LOGGING ===
    if args.log_level:
        config_dict['log_level'] = args.log_level
    
    # Adjust log level based on verbose/quiet flags
    if args.verbose:
        config_dict['log_level'] = 'DEBUG'
    elif args.quiet:
        config_dict['log_level'] = 'WARNING'
    
    return PCAConfig(**config_dict)

def load_benchmarks_from_config(config: PCAConfig) -> dict:
    """Load benchmarks from config-specified file."""
    benchmark_path = Path(config.benchmark_config)
    if not benchmark_path.is_absolute():
        benchmark_path = Path.cwd() / benchmark_path
    
    with open(benchmark_path, "r") as f:
        benchmark_config = json.load(f)
    
    return {name: data["Tickers"] for name, data in benchmark_config["benchmarks"].items() if data["Tickers"]}

def print_config_help():
    """Print comprehensive guide to using the config file."""
    help_text = """
=== PCA STATISTICAL ARBITRAGE CONFIGURATION GUIDE ===

CONFIG FILE STRUCTURE (config.json):

📁 DATA PATHS:
  - georges_db_path: Main data directory path
  - adj_close_csv/pct_change_csv: CSV filenames  
  - adj_close_xarray/pct_change_xarray: xarray filenames
  - legacy_figionly_csv: Legacy CSV filename
  - benchmark_config: Benchmark definitions file
  - backtester_path: Path to Backtester package

🔧 CSV STRUCTURE:
  - bloomberg_ticker_column: Column index for Bloomberg tickers (0)
  - figi_column: Column index for FIGIs (1) 
  - first_date_column: First date column index (2)
  - date_format: Date parsing format ("mixed")

📊 PCA PARAMETERS:
  - z_window: Rolling window for z-score (60)
  - z_min_periods: Min periods for rolling calc (30)
  - entry_threshold: Signal entry threshold (2.0)
  - exit_threshold: Signal exit threshold (0.5)
  - min_r2: Minimum R-squared filter (0.1)

🎯 DYNAMIC Z-SCORING:
  - dynamic_zscore: Enable regime-aware z-scoring (false)
  - dynamic_lookback: Volatility regime lookback (252)
  - volatility_adjustment: Adjust thresholds by vol (true)
  - momentum_filter: Filter trending stocks (false)
  - momentum_threshold: Max momentum for entry (0.1)

📈 BACKTEST SETTINGS:
  - start_date/end_date: Backtest date range
  - initial_capital: Starting capital (100000)
  - transaction_cost: Cost per trade (0.001)
  - max_position_size: Max position as % of capital (0.05)

🔄 FACTSET INTEGRATION:
  - enabled: Use FactSet for updates (true)
  - batch_size: API batch size (100)
  - max_days_behind: Warning threshold (5)

📝 LOGGING:
  - level: Log level (INFO)
  - format: Log message format
  - file: Log file name

DYNAMIC Z-SCORING EXPLAINED:
- Static: Fixed rolling window for all periods
- Dynamic: Adapts window based on volatility regime
  * High volatility → Longer window (more stable)
  * Low volatility → Shorter window (more responsive)
  * Normal volatility → Standard window

MOMENTUM FILTERING:
- Filters out stocks with strong directional trends
- Mean reversion works better on range-bound stocks
- Lower threshold = more restrictive filtering

VOLATILITY ADJUSTMENT:
- Scales entry/exit thresholds by current volatility
- High vol periods → Higher thresholds
- Low vol periods → Lower thresholds
"""
    print(help_text)

def format_results(results: dict, format_type: str = 'table', top_n: int = None, min_strength: float = None) -> str:
    """Format results for display in specified format."""
    
    if format_type == 'json':
        import json
        return json.dumps(results, indent=2, default=str)
    
    elif format_type == 'csv':
        # Convert to DataFrame and return CSV string
        rows = []
        for benchmark_name, result in results.items():
            if 'summary_stats' in result:
                stats = result['summary_stats']
                for figi in stats.figi.values:
                    rows.append({
                        'benchmark': benchmark_name,
                        'figi': figi,
                        'alpha': float(stats.alpha.sel(figi=figi)),
                        'beta': float(stats.beta.sel(figi=figi)),
                        'r2': float(stats.r2.sel(figi=figi)),
                        'current_z': float(result['z_scores'].sel(figi=figi).isel(date=-1)),
                        'current_signal': int(result['signals'].sel(figi=figi).isel(date=-1))
                    })
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    
    else:  # table format
        output = []
        for benchmark_name, result in results.items():
            output.append(f"\n=== {benchmark_name} ===")
            
            if 'performance' in result:
                perf = result['performance']
                output.append(f"Performance: Sharpe={perf.get('sharpe', 0):.2f}, Return={perf.get('annual_return', 0):.2%}")
            
            if 'summary_stats' in result:
                stats = result['summary_stats']
                signals = result['signals']
                z_scores = result['z_scores']
                
                # Filter and sort signals
                current_signals = []
                for figi in stats.figi.values:
                    z_val = float(z_scores.sel(figi=figi).isel(date=-1))
                    sig_val = int(signals.sel(figi=figi).isel(date=-1))
                    
                    if min_strength and abs(z_val) < min_strength:
                        continue
                    
                    if sig_val != 0:
                        current_signals.append({
                            'figi': figi,
                            'z_score': z_val,
                            'signal': sig_val,
                            'r2': float(stats.r2.sel(figi=figi))
                        })
                
                # Sort by absolute z-score
                current_signals.sort(key=lambda x: abs(x['z_score']), reverse=True)
                
                # Apply top_n filter
                if top_n:
                    current_signals = current_signals[:top_n]
                
                if current_signals:
                    output.append("Active Signals:")
                    for sig in current_signals:
                        direction = "LONG" if sig['signal'] == 1 else "SHORT"
                        output.append(f"  {sig['figi']}: {direction} (z={sig['z_score']:.2f}, R²={sig['r2']:.3f})")
                else:
                    output.append("No active signals")
        
        return '\n'.join(output)

def main():
    """Enhanced main entry point with comprehensive terminal control."""
    
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Special action: help with config
    if args.action == 'help-config':
        print_config_help()
        return
    
    # Load config from file
    config = load_config(args.config)
    
    # Apply command line overrides
    config = apply_argparse_overrides(config, args)
    
    # Initialize manager
    manager = PCADataManager(config)
    
    # Get safe symbols for console output
    symbols = get_safe_message_formatter()
    
    # Execute action
    if args.dry_run:
        print(f"{symbols['debug']} DRY RUN MODE - No changes will be made")
        print(f"Config summary:")
        print(f"  Data path: {config.georges_db_path}")
        print(f"  Z-score: window={config.z_window}, entry={config.entry_threshold}, exit={config.exit_threshold}")
        print(f"  Dynamic: {config.dynamic_zscore}, Vol adj: {config.volatility_adjustment}")
        print(f"  Momentum filter: {config.momentum_filter}")
        return
    
    # === CORE ACTIONS ===
    if args.action == 'migrate':
        print(f"{symbols['migrate']} Starting CSV to xarray migration...")
        adj, pct = manager.migrate_csv_to_xarray()
        manager.validate_migration(adj, pct)
    
    elif args.action == 'validate':
        print(f"{symbols['validate']} Validating existing xarray data...")
        adj, pct = manager.load_xarray_data()
        manager.validate_migration(adj, pct)
    
    elif args.action == 'update':
        print(f"{symbols['update']} Updating with FactSet data...")
        manager.update_with_factset()
    
    elif args.action == 'info':
        try:
            adj, pct = manager.load_xarray_data()
            print(f"{symbols['data']} Data Info:")
            print(f"   Shape: {adj.shape}")
            print(f"   Date range: {adj.date.min().values} to {adj.date.max().values}")
            print(f"   Tickers: {adj.sizes['figi']}")
            print(f"   File sizes: {manager.adj_close_xr_path.stat().st_size / 1024 / 1024:.1f}MB")
            print(f"   Last update: {adj.attrs.get('created_at', 'Unknown')}")
        except FileNotFoundError:
            print(f"{symbols['error']} No xarray data found. Run --action migrate first.")
    
    # === ANALYSIS ACTIONS ===
    elif args.action == 'analyze':
        print(f"{symbols['processing']} Running PCA analysis with xarray...")
        
        # Load benchmarks
        benchmarks = load_benchmarks_from_config(config)
        
        # Filter to specific benchmark if requested
        if args.benchmark:
            if args.benchmark in benchmarks:
                benchmarks = {args.benchmark: benchmarks[args.benchmark]}
            else:
                print(f"{symbols['error']} Benchmark '{args.benchmark}' not found. Available: {list(benchmarks.keys())}")
                return
        
        # Run analysis
        processor = XArrayPCAProcessor(manager)
        results = {}
        
        for benchmark_name, tickers in benchmarks.items():
            result = processor.process_benchmark_xarray(benchmark_name, tickers)
            if result:
                results[benchmark_name] = result
        
        # Format and display results
        output = format_results(results, args.output_format, args.top_signals, args.min_signal_strength)
        print(output)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                f.write(output)
            print(f"{symbols['success']} Results saved to {args.save_results}")
    
    elif args.action == 'backtest':
        print(f"{symbols['processing']} Running backtest with xarray...")
        
        # Load benchmarks
        benchmarks = load_benchmarks_from_config(config)
        
        # Filter to specific benchmark if requested
        if args.benchmark:
            if args.benchmark in benchmarks:
                benchmarks = {args.benchmark: benchmarks[args.benchmark]}
            else:
                print(f"{symbols['error']} Benchmark '{args.benchmark}' not found")
                return
        
        # Run backtest
        backtester = XArrayBacktester(manager)
        results = backtester.run_backtest(benchmarks)
        
        # Display results
        print(f"\n{symbols['success']} BACKTEST RESULTS:")
        for benchmark_name, result in results.items():
            if 'performance' in result:
                perf = result['performance']
                print(f"\n{benchmark_name}:")
                print(f"  Annual Return: {perf.get('annual_return', 0):.2%}")
                print(f"  Volatility: {perf.get('volatility', 0):.2%}")
                print(f"  Sharpe Ratio: {perf.get('sharpe', 0):.2f}")
                print(f"  Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
                print(f"  Win Rate: {perf.get('win_rate', 0):.2%}")
                print(f"  Total Trades: {perf.get('n_trades', 0)}")
        
        # Save results if requested
        if args.save_results:
            output = format_results(results, args.output_format)
            with open(args.save_results, 'w') as f:
                f.write(output)
            print(f"{symbols['success']} Backtest results saved to {args.save_results}")
    
    elif args.action == 'export-pandas':
        print(f"{symbols['export']} Exporting pandas format for PCA...")
        returns_df = manager.get_pandas_returns_for_pca()
        ticker_mapping = manager.get_bloomberg_ticker_mapping()
        print(f"Returns shape: {returns_df.shape}")
        print(f"Ticker mapping: {len(ticker_mapping)} entries")
        print("Sample mapping:")
        for i, (figi, ticker) in enumerate(list(ticker_mapping.items())[:5]):
            print(f"  {figi} -> {ticker}")

if __name__ == "__main__":
    main()