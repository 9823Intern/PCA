from sklearn.decomposition import PCA
import json
import pandas as pd

adj_close_path = "C:/Users/EnnTurn/Precept Dropbox/Enn Turn/Georges_DataBase/Securities_Adj_Close_MASTER.csv"
adj_close_df = pd.read_csv(adj_close_path, index_col=1, parse_dates=True)

returns_path = "C:/Users/EnnTurn/Precept Dropbox/Enn Turn/Georges_DataBase/Securities_Daily_Pct_Change_MASTER.csv"
returns = pd.read_csv(returns_path, index_col=1, parse_dates=True)


def load_benchmark_config(config_path='benchmark_config.json'):
    """Load benchmark configuration and return mapping of benchmark to tickers."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return {name: data['Tickers'] for name, data in config['benchmarks'].items()}

import numpy as np

pca = PCA(n_components=2)
X_pca = pca.fit_transform(returns)

pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]

loadings = pd.DataFrame(
    pca.components_.T,
    index=returns.columns,
    columns=['PC1', 'PC2']
)

# variance explained

explained_var = pca.explained_variance_ratio_

def get_pca_for_benchmark(df, tickers, n_components=2):
    """
    Get PCA for a given benchmark.

    Args:
        df (pd.DataFrame): DataFrame containing returns data.
        tickers (list): List of tickers to include in the PCA.
        n_components (int): Number of principal components to compute.
    """
    df_benchmark = df[tickers].dropna(axis=1, how="any")
    returns = np.log(df_benchmark / df_benchmark.shift(1)).dropna()

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(returns)

    return {
        "scores": pd.DataFrame(X_pca, index=returns.index, columns=[f"PC{i+1}" for i in range(n_components)]),
        "loadings": pd.DataFrame(pca.components_.T, index=returns.columns, columns=[f"PC{i+1}_loading" for i in range(n_components)]),
        "explained_var": pca.explained_variance_ratio_
    }


# Load benchmark configuration
benchmarks = load_benchmark_config()

# Process each benchmark
for benchmark_name, benchmark_tickers in benchmarks.items():
    if not benchmark_tickers:  # Skip empty benchmarks
        print(f"Skipping {benchmark_name}: no tickers defined")
        continue
    
    print(f"\n--- {benchmark_name} ---")
    result = get_pca_for_benchmark(adj_close_df, benchmark_tickers)
    print(result["scores"].head())
    print(result["loadings"].sort_values("PC1_loading", ascending=False).head())
    print(f"Explained variance: {result['explained_var']}")