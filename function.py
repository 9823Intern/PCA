from sklearn.decomposition import PCA
import json
import pandas as pd

adj_close_path = "C:/Users/EnnTurn/Precept Dropbox/Enn Turn/Georges_DataBase/figionly.csv"
adj_close_df = pd.read_csv(adj_close_path, index_col=0, parse_dates=True).apply(pd.to_numeric, errors='coerce').sort_index()

# Build returns like alpha-annihilation: pct_change and drop rows that are all NaN
returns_df = adj_close_df.pct_change().dropna(how='all')


def load_benchmark_config(config_path='benchmark_config.json'):
    """Load benchmark configuration and return mapping of benchmark to tickers."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return {name: data['Tickers'] for name, data in config['benchmarks'].items()}

import numpy as np


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


# Load benchmark configuration
benchmarks = load_benchmark_config()

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