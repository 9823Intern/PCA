"""
PCA Stat Arb Runner (Real Data Ready)
------------------------------------

This script shows how to plug in your **actual returns DataFrame** and benchmark definitions.

- Replace the mock data generation with your real returns DataFrame.
- Expected format of `returns_df`:
    * index = tickers (str)
    * columns = dates (datetime-like)
    * values = daily returns in decimal (0.01 = 1%)
- Define benchmarks as dict[str, list[str]] mapping benchmark name -> tickers.

Once you have your real returns, drop them into `returns_df` and the pipeline will give
true PCA loadings, PC1/PC2 time series, residual z-scores, and signals.
"""

import pandas as pd
import json

from pca_stat_arb_module import pca_stat_arb_pipeline, extract_signals, best_benchmark_by_pc1_corr

# -------------------------------
# Replace this section with your real returns data
# -------------------------------
# Example:
# returns_df = pd.read_csv("my_returns.csv", index_col=0, parse_dates=True)
# returns_df.index.name = "ticker"
# returns_df.columns = pd.to_datetime(returns_df.columns)

# Load benchmarks from config file
with open("benchmark_config.json", "r") as f:
    config = json.load(f)

benchmarks = {name: data["Tickers"] for name, data in config["benchmarks"].items() if data["Tickers"]}

# -------------------------------
# Runner
# -------------------------------
if __name__ == "__main__":
    # >>> Replace with your real returns_df before running <<<
    try:
        returns_df
    except NameError:
        raise RuntimeError("Define your real returns_df before running this script!")

    # Run pipeline
    results = pca_stat_arb_pipeline(
        returns_df,
        benchmarks,
        z_window=60,
        z_min_periods=30,
        entry=2.0,
        exit_=0.5,
    )

    # Extract active signals
    signals = extract_signals(results.per_stock_table, signal_only=True)
    print("\n=== Active Signals ===")
    print(signals)

    # Show best benchmark mapping by PC1 correlation
    print("\n=== Best Benchmark Mapping ===")
    print(best_benchmark_by_pc1_corr(results.per_stock_table))

    # Access PC1/PC2 series for a benchmark (using first available benchmark)
    first_benchmark = list(benchmarks.keys())[0]
    pc1_series = results.benchmark_pcs[first_benchmark]["PC1"]
    print(f"\n{first_benchmark} PC1 head:")
    print(pc1_series.head())

    # Access stock loadings on PCs
    loadings = results.loadings[first_benchmark]
    print(f"\n{first_benchmark} Loadings:")
    print(loadings)
