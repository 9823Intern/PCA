"""
PCA Stat Arb Runner (Config-Driven)
-----------------------------------

Enhanced runner that uses config.json for all paths and parameters.
Supports both legacy CSV and new xarray data sources.
"""

import pandas as pd
import json
from pathlib import Path

from data_migration import load_config, PCADataManager
from pca_stat_arb_module import pca_stat_arb_pipeline, extract_signals, best_benchmark_by_pc1_corr

def load_data_source(config):
    """Load returns data from either xarray or legacy CSV based on availability."""
    
    # Try xarray first (preferred)
    try:
        manager = PCADataManager(config)
        returns_df = manager.get_pandas_returns_for_pca()
        print(f"[run] Using xarray data source: {returns_df.shape}")
        return returns_df
    except FileNotFoundError:
        print("[run] xarray data not found, falling back to legacy CSV...")
        
        # Fallback to legacy figionly.csv
        legacy_path = Path(config.georges_db_path) / config.legacy_figionly_csv
        try:
            returns_df = pd.read_csv(legacy_path, index_col=0, parse_dates=True)
            returns_df.index.name = "FIGI"
            col_labels = returns_df.columns.map(str).str.strip()
            parsed_cols = pd.to_datetime(col_labels, errors="coerce", format="mixed")
            returns_df = returns_df.loc[:, ~parsed_cols.isna()]
            returns_df.columns = parsed_cols[~parsed_cols.isna()]
            print(f"[run] Using legacy CSV: {returns_df.shape}")
            return returns_df
        except FileNotFoundError:
            raise RuntimeError(f"No data source found. Check paths in config:\n  legacy: {legacy_path}")

def load_benchmarks(config):
    """Load benchmarks from config file."""
    benchmark_path = Path(config.benchmark_config)
    if not benchmark_path.is_absolute():
        benchmark_path = Path.cwd() / benchmark_path
    
    with open(benchmark_path, "r") as f:
        benchmark_config = json.load(f)
    
    return {name: data["Tickers"] for name, data in benchmark_config["benchmarks"].items() if data["Tickers"]}

def filter_benchmarks(benchmarks, returns_df):
    """Filter benchmarks to available tickers."""
    # Normalize index to uppercase/stripped strings
    returns_df.index = returns_df.index.map(str).str.strip().str.upper()
    
    # Pre-filter benchmarks to tickers present in returns; require at least 2
    present = set(returns_df.index)
    filtered_benchmarks = {}
    for name, tks in benchmarks.items():
        cleaned = []
        for t in tks:
            tt = str(t).strip().upper()
            if tt in present:
                cleaned.append(tt)
        if len(cleaned) >= 2:
            filtered_benchmarks[name] = cleaned
    
    # Debug summary
    print("\n[run] filtered benchmarks:")
    for n, ts in filtered_benchmarks.items():
        print(f"  {n}: {len(ts)} tickers")
    print(f"[run] total benchmarks usable: {len(filtered_benchmarks)}\n")
    
    return filtered_benchmarks, returns_df

# Load configuration
config = load_config()

# Load data using config-driven approach
returns_df = load_data_source(config)

# Load benchmarks from config
benchmarks = load_benchmarks(config)

# Filter benchmarks to available data
filtered_benchmarks, returns_df = filter_benchmarks(benchmarks, returns_df)

# -------------------------------
# Runner
# -------------------------------
if __name__ == "__main__":
    # Run pipeline using config parameters
    results = pca_stat_arb_pipeline(
        returns_df,
        filtered_benchmarks,
        z_window=config.z_window,
        z_min_periods=config.z_min_periods,
        entry=config.entry_threshold,
        exit_=config.exit_threshold,
    )

    # Extract active signals
    signals = extract_signals(results.per_stock_table, signal_only=True)
    print("\n=== Active Signals ===")
    print(signals)

    # Show best benchmark mapping by PC1 correlation
    print("\n=== Best Benchmark Mapping ===")
    print(best_benchmark_by_pc1_corr(results.per_stock_table))

    # Access PC1/PC2 series for a benchmark (using first available benchmark)
    available = list(results.benchmark_pcs.keys())
    if not available:
        print("\nNo benchmark PCs were produced. Ensure your returns_df has data for benchmark tickers.")
    else:
        first_benchmark = available[0]
        pc1_series = results.benchmark_pcs[first_benchmark]["PC1"]
        print(f"\n{first_benchmark} PC1 head:")
        print(pc1_series.head())

        # Access stock loadings on PCs
        loadings = results.loadings[first_benchmark]
        print(f"\n{first_benchmark} Loadings:")
        print(loadings)
