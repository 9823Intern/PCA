"""
Example: Using Historical Signals for Backtesting (Config-Driven)
----------------------------------------------------------------

Enhanced script using config.json and xarray for faster backtesting.
Demonstrates both legacy pandas and new xarray approaches.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from data_migration import load_config, PCADataManager, XArrayBacktester
from pca_stat_arb_module import get_historical_signals

def example_xarray_backtest(config_path: str = "config.json"):
    """
    Example using new xarray-based backtesting for optimal performance.
    """
    print("=== XARRAY BACKTEST EXAMPLE ===")
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize data manager
    manager = PCADataManager(config)
    
    try:
        # Try to load xarray data
        returns_xr = manager.get_xarray_returns()
        print(f"[xarray] Loaded returns data: {returns_xr.shape}")
        
        # Load benchmarks
        benchmarks = load_benchmarks_from_config(config)
        print(f"[xarray] Loaded {len(benchmarks)} benchmarks")
        
        # Run fast xarray backtest
        backtester = XArrayBacktester(manager)
        results = backtester.run_backtest(benchmarks)
        
        # Display results
        print(f"\n=== XARRAY BACKTEST RESULTS ===")
        for benchmark_name, result in results.items():
            if 'performance' in result:
                perf = result['performance']
                print(f"\n{benchmark_name}:")
                print(f"  Annual Return: {perf.get('annual_return', 0):.2%}")
                print(f"  Sharpe Ratio: {perf.get('sharpe', 0):.2f}")
                print(f"  Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
                print(f"  Win Rate: {perf.get('win_rate', 0):.2%}")
                print(f"  Active Days: {perf.get('active_days', 0)}")
        
        return results
        
    except FileNotFoundError:
        print("[xarray] xarray data not found. Run migration first:")
        print("  python data_migration.py --action migrate")
        return None

def example_legacy_backtest(config_path: str = "config.json"):
    """
    Legacy example using pandas approach (for comparison/fallback).
    """
    print("=== LEGACY PANDAS BACKTEST EXAMPLE ===")
    
    # Load configuration
    config = load_config(config_path)
    
    # Load your returns data using config
    try:
        manager = PCADataManager(config)
        returns_df = manager.get_pandas_returns_for_pca()
        print(f"[legacy] Using xarray->pandas conversion: {returns_df.shape}")
        
    except FileNotFoundError:
        # Fallback to legacy CSV
        legacy_path = Path(config.georges_db_path) / config.legacy_figionly_csv
        try:
            returns_df = pd.read_csv(legacy_path, index_col=0, parse_dates=True)
            returns_df.index.name = "FIGI"
            
            # Clean and prepare data
            col_labels = returns_df.columns.map(str).str.strip()
            parsed_cols = pd.to_datetime(col_labels, errors="coerce", format="mixed")
            returns_df = returns_df.loc[:, ~parsed_cols.isna()]
            returns_df.columns = parsed_cols[~parsed_cols.isna()]
            returns_df.index = returns_df.index.map(str).str.strip().str.upper()
            print(f"[legacy] Using legacy CSV: {returns_df.shape}")
            
        except FileNotFoundError:
            print("No data found. Using mock data for demonstration...")
            # Create mock data for demonstration
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            np.random.seed(42)
            returns_data = np.random.normal(0.001, 0.02, (len(tickers), len(dates)))
            returns_df = pd.DataFrame(returns_data, index=tickers, columns=dates)
    
    # Load benchmarks using config
    try:
        benchmarks = load_benchmarks_from_config(config)
    except FileNotFoundError:
        print("Benchmark config not found. Using mock benchmarks...")
        benchmarks = {
            "Tech": ['AAPL', 'MSFT', 'GOOGL'],
            "Growth": ['AMZN', 'TSLA', 'GOOGL']
        }
    
    # Filter benchmarks to available tickers
    present = set(returns_df.index)
    filtered_benchmarks = {}
    for name, tks in benchmarks.items():
        cleaned = [str(t).strip().upper() for t in tks if str(t).strip().upper() in present]
        if len(cleaned) >= 2:
            filtered_benchmarks[name] = cleaned
    
    print(f"Using {len(filtered_benchmarks)} benchmarks with available data")
    
    # ===== GET HISTORICAL SIGNALS (LEGACY PANDAS) =====
    print("\n=== Retrieving Historical Signals (Legacy) ===")
    historical_data = get_historical_signals(
        returns_df=returns_df,
        benchmarks=filtered_benchmarks,
        z_window=config.z_window,
        z_min_periods=config.z_min_periods,
        entry=config.entry_threshold,
        exit_=config.exit_threshold,
    )
    
    # ===== ANALYZE RESULTS =====
    for benchmark_name, data in historical_data.items():
        signals = data['signals']
        z_scores = data['z_scores']
        pc1 = data['pc1']
        
        print(f"\n--- {benchmark_name} Results ---")
        print(f"Signal matrix shape: {signals.shape} (dates x stocks)")
        print(f"Date range: {signals.index.min()} to {signals.index.max()}")
        print(f"Stocks: {list(signals.columns)}")
        
        # Count signal activity
        total_signals = (signals != 0).sum().sum()
        long_signals = (signals == 1).sum().sum()
        short_signals = (signals == -1).sum().sum()
        
        print(f"Total signal days: {total_signals}")
        print(f"Long signals: {long_signals}")
        print(f"Short signals: {short_signals}")
        
        # Show recent signals
        if not signals.empty:
            recent_signals = signals.tail(5)
            print(f"\nRecent signals (last 5 days):")
            print(recent_signals)
            
    # ===== EXAMPLE BACKTEST CALCULATION =====
    print("\n=== Example Backtest Calculation ===")
    
    if historical_data:
        # Use first available benchmark for example
        benchmark_name = list(historical_data.keys())[0]
        signals = historical_data[benchmark_name]['signals']
        
        # Get returns for the same stocks and dates
        common_stocks = [s for s in signals.columns if s in returns_df.index]
        common_dates = signals.index.intersection(returns_df.columns)
        
        if common_stocks and len(common_dates) > 0:
            # Align data
            signal_matrix = signals.loc[common_dates, common_stocks]
            return_matrix = returns_df.loc[common_stocks, common_dates].T
            
            # Shift signals by 1 day for realistic entry (signal at close, enter next day)
            lagged_signals = signal_matrix.shift(1).fillna(0)
            
            # Calculate strategy returns
            strategy_returns = (lagged_signals * return_matrix).sum(axis=1)
            
            # Simple performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + strategy_returns.mean()) ** 252 - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            
            print(f"Example backtest for {benchmark_name}:")
            print(f"Total Return: {total_return:.2%}")
            print(f"Annualized Return: {annual_return:.2%}")
            print(f"Annualized Volatility: {volatility:.2%}")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            
            # Show signal utilization
            active_days = (lagged_signals != 0).any(axis=1).sum()
            print(f"Active trading days: {active_days}/{len(common_dates)} ({active_days/len(common_dates):.1%})")

def load_benchmarks_from_config(config):
    """Load benchmarks from config-specified file."""
    benchmark_path = Path(config.benchmark_config)
    if not benchmark_path.is_absolute():
        benchmark_path = Path.cwd() / benchmark_path
    
    with open(benchmark_path, "r") as f:
        benchmark_config = json.load(f)
    
    return {name: data["Tickers"] for name, data in benchmark_config["benchmarks"].items() if data["Tickers"]}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PCA Backtest Examples')
    parser.add_argument('--config', default='config.json', help='Config file path')
    parser.add_argument('--method', choices=['xarray', 'legacy', 'both'], default='both',
                       help='Which backtesting method to demonstrate')
    
    args = parser.parse_args()
    
    if args.method in ['xarray', 'both']:
        print("Running xarray backtest example...")
        xarray_results = example_xarray_backtest(args.config)
    
    if args.method in ['legacy', 'both']:
        print("\nRunning legacy pandas backtest example...")
        legacy_results = example_legacy_backtest(args.config)
    
    print("\n=== COMPARISON COMPLETE ===")
    print("The xarray method should be significantly faster for large datasets!") 
