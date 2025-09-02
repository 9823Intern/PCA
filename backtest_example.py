"""
Example: Using Historical Signals for Backtesting
------------------------------------------------

This script demonstrates how to retrieve full historical signals and z-scores
from the PCA stat arb module for backtesting purposes.
"""

import pandas as pd
import numpy as np
import json
from pca_stat_arb_module import get_historical_signals

def example_backtest_usage():
    """
    Example showing how to use historical signals for backtesting.
    """
    
    # Load your returns data (replace with your actual data)
    try:
        returns_df = pd.read_csv("C:/Users/EnnTurn/Precept Dropbox/Enn Turn/Georges_DataBase/figionly.csv", index_col=0, parse_dates=True)
        returns_df.index.name = "FIGI"
        
        # Clean and prepare data
        col_labels = returns_df.columns.map(str).str.strip()
        parsed_cols = pd.to_datetime(col_labels, errors="coerce", format="mixed")
        returns_df = returns_df.loc[:, ~parsed_cols.isna()]
        returns_df.columns = parsed_cols[~parsed_cols.isna()]
        returns_df.index = returns_df.index.map(str).str.strip().str.upper()
        
    except FileNotFoundError:
        print("Data file not found. Using mock data for demonstration...")
        # Create mock data for demonstration
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        np.random.seed(42)
        returns_data = np.random.normal(0.001, 0.02, (len(tickers), len(dates)))
        returns_df = pd.DataFrame(returns_data, index=tickers, columns=dates)
    
    # Load benchmarks
    try:
        with open("benchmark_config.json", "r") as f:
            config = json.load(f)
        benchmarks = {name: data["Tickers"] for name, data in config["benchmarks"].items() if data["Tickers"]}
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
    
    # ===== GET HISTORICAL SIGNALS =====
    print("\n=== Retrieving Historical Signals ===")
    historical_data = get_historical_signals(
        returns_df=returns_df,
        benchmarks=filtered_benchmarks,
        z_window=60,
        z_min_periods=30,
        entry=2.0,
        exit_=0.5,
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

if __name__ == "__main__":
    example_backtest_usage() 
