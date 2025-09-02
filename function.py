from sklearn.decomposition import PCA
import json
import pandas as pd
import numpy as np
from pathlib import Path

from data_migration import load_config, PCADataManager

def load_data_source(config):
    """Load returns data from either xarray or legacy CSV based on availability."""
    
    # Try xarray first (preferred)
    try:
        manager = PCADataManager(config)
        returns_df = manager.get_pandas_returns_for_pca()
        print(f"[function] Using xarray data source: {returns_df.shape}")
        return returns_df
    except FileNotFoundError:
        print("[function] xarray data not found, falling back to legacy CSV...")
        
        # Fallback to legacy figionly.csv
        legacy_path = Path(config.georges_db_path) / config.legacy_figionly_csv
        try:
            adj_close_df = pd.read_csv(legacy_path, index_col=0, parse_dates=True).apply(pd.to_numeric, errors='coerce').sort_index()
            # Build returns like alpha-annihilation: pct_change and drop rows that are all NaN
            returns_df = adj_close_df.pct_change().dropna(how='all')
            print(f"[function] Using legacy CSV: {returns_df.shape}")
            return returns_df
        except FileNotFoundError:
            raise RuntimeError(f"No data source found. Check paths in config:\n  legacy: {legacy_path}")

def load_benchmark_config(config):
    """Load benchmark configuration from config-specified path."""
    benchmark_path = Path(config.benchmark_config)
    if not benchmark_path.is_absolute():
        benchmark_path = Path.cwd() / benchmark_path
    
    with open(benchmark_path, 'r') as f:
        benchmark_config = json.load(f)
    return {name: data['Tickers'] for name, data in benchmark_config['benchmarks'].items()}


def get_pca_for_benchmark(returns_df, tickers, n_components=2):
    """
    Get PCA for a given benchmark.

    Args:
        returns_df (pd.DataFrame): Returns matrix (pct_change), indexed by date.
        tickers (list): List of tickers to include in the PCA.
        n_components (int): Number of principal components to compute.
    """
    cols = [t for t in tickers if t in returns_df.columns]
    if len(cols) < 2:
        return {
            "scores": pd.DataFrame(),
            "loadings": pd.DataFrame(),
            "explained_var": []
        }

    df = returns_df[cols]
    df = df.dropna(axis=0, how='any')
    if df.shape[0] < 3:
        return {
            "scores": pd.DataFrame(),
            "loadings": pd.DataFrame(),
            "explained_var": []
        }

    std = df.std()
    use_cols = [c for c in df.columns if std.get(c, 0.0) > 1e-12]
    if len(use_cols) < 2:
        return {
            "scores": pd.DataFrame(),
            "loadings": pd.DataFrame(),
            "explained_var": []
        }

    X = df[use_cols].values
    k = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)

    return {
        "scores": pd.DataFrame(X_pca, index=df.index, columns=[f"PC{i+1}" for i in range(k)]),
        "loadings": pd.DataFrame(pca.components_.T, index=use_cols, columns=[f"PC{i+1}_loading" for i in range(k)]),
        "explained_var": pca.explained_variance_ratio_.tolist()
    }


def main():
    """Main function using config-driven approach."""
    # Load configuration
    config = load_config()
    
    # Load data using config-driven approach
    returns_df = load_data_source(config)
    
    # Load benchmark configuration
    benchmarks = load_benchmark_config(config)
    
    # Process each benchmark
    for benchmark_name, benchmark_tickers in benchmarks.items():
        if not benchmark_tickers:  # Skip empty benchmarks
            print(f"Skipping {benchmark_name}: no tickers defined")
            continue
        
        print(f"\n--- {benchmark_name} ---")
        result = get_pca_for_benchmark(returns_df, benchmark_tickers)
        if result["scores"].empty:
            print("Insufficient data after NaN/variance filtering")
            continue
        print(result["scores"].head())
        if not result["loadings"].empty and "PC1_loading" in result["loadings"].columns:
            print(result["loadings"].sort_values("PC1_loading", ascending=False).head())
        print(f"Explained variance: {result['explained_var']}")

if __name__ == "__main__":
    main()