from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


# Helper functions
def ensure_datetime_columns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the columns are datetime (sorted ascending)."""
    out = returns_df.copy()
    out.columns = pd.to_datetime(out.columns)
    out = out.reindex(sorted(out.columns), axis=1)
    return out




def align_subset(returns_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Return a subset suitable for PCA despite IPO gaps.

    Steps:
    - Keep tickers present in index (normalized to upper/strip)
    - Trim to common start (max first non-NaN date across present tickers)
    - Require each ticker to have at least MIN_DAYS of data
    - Drop dates with any NaNs across the remaining tickers
    - Return empty if fewer than MIN_TICKERS or MIN_DAYS are available
    """
    MIN_DAYS = 30
    MIN_TICKERS = 2

    present = [str(t).strip().upper() for t in tickers if str(t).strip().upper() in returns_df.index]
    if len(present) < MIN_TICKERS:
        return pd.DataFrame()

    sub = returns_df.loc[present].copy()

    # Determine common start as the latest first-valid date among present tickers
    first_valid = sub.apply(lambda s: s.first_valid_index(), axis=1)
    valid_starts = [d for d in first_valid.tolist() if d is not None]
    if not valid_starts:
        return pd.DataFrame()
    common_start = max(valid_starts)

    # Trim to dates >= common_start
    sub = sub.loc[:, sub.columns >= common_start]

    # Filter tickers with at least MIN_DAYS available
    has_days = sub.notna().sum(axis=1)
    keep = has_days[has_days >= MIN_DAYS].index.tolist()
    if len(keep) < MIN_TICKERS:
        return pd.DataFrame()

    sub = sub.loc[keep]

    # Drop any dates with remaining NaNs across the kept tickers
    sub = sub.dropna(axis=1, how="any")

    # Final checks
    if sub.shape[0] < MIN_TICKERS or sub.shape[1] < MIN_DAYS:
        return pd.DataFrame()

    return sub

@dataclass
class BenchmarkPCA:
    name: str
    pc_time_series: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance_ratio: np.ndarray


def run_benchmark_pca(
    returns: pd.DataFrame,
    benchmark_name: str,
    benchmark_tickers: List[str],
    n_components: int = 2,
    standardize: bool = True,
    ) -> BenchmarkPCA:
    """Compute PCA for a benchmark and return PC time series + loadings.
    
    Parameters
    ----------
    returns_df : DataFrame (index=tickers, columns=dates)
    benchmark_name : str
    tickers : list[str]
    n_components : int, default=2
    standardize : bool, default=True


    Returns
    -------
    BenchmarkPCA
    """

    returns_prepped = ensure_datetime_columns(returns)
    subset = align_subset(returns_prepped, benchmark_tickers)
    if subset.shape[1] > 0:
        print(f"[pca] {benchmark_name}: tickers={subset.shape[0]} days={subset.shape[1]} span={subset.columns.min()}→{subset.columns.max()}")
    if subset.shape[0] < 2:
        print(f"[pca] {benchmark_name}: not enough tickers after alignment ({subset.shape[0]} < 2)")
        raise ValueError(f"Benchmark {benchmark_name} has less than 2 valid tickers")
    if subset.shape[1] < 2:
        print(f"[pca] {benchmark_name}: not enough days after alignment ({subset.shape[1]} < 2)")
        raise ValueError(f"Benchmark {benchmark_name} has less than 2 valid days")

    # sklearn expects shape (n_samples, n_features) -> (n_days,  n_tickers)
    X = subset.T
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_std = scaler.fit_transform(X.values)
    else:
        X_std = X.values

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_std) # (n_days, n_components)

    # Build outputs
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    pc_ts = pd.DataFrame(pcs, index=X.index, columns=pc_cols)
    loadings = pd.DataFrame(pca.components_.T, index = X.columns, columns=pc_cols)

    return BenchmarkPCA(
        name=benchmark_name,
        pc_time_series=pc_ts,
        loadings=loadings,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

# ----------------------
# Regression & Residuals
# ----------------------

@dataclass
class RegressionResult:
    alpha: float
    beta: float
    r2: float
    residuals: pd.Series # Indexed by date

def regress_stock_on_pc1(
    stock_series: pd.Series,
    pc1_series: pd.Series,
) -> RegressionResult:
    """OLS regression of stock returns on PC1 time series.

    
    Returns alpha, beta, R^2, and residuals aligned by date.
    """
    s=  pd.concat([stock_series.rename("y"), pc1_series.rename("pc1")], axis=1).dropna()
    if s.shape[0] < 30:
        # Need enough points to be meaningful
        return RegressionResult(alpha=np.nan, beta=np.nan, r2=np.nan, residuals=pd.Series(index=stock_series.index, dtype=np.float64))

    X = sm.add_constant(s["pc1"])
    y = s["y"]
    model = sm.OLS(y.values, X.values, missing="drop").fit()
    alpha = float(model.params[0])
    beta = float(model.params[1])
    r2 = float(model.rsquared)

    fitted = pd.Series(model.predict(X.values), index=s.index)
    resid = s["y"] - fitted

    # Reindex back to original stock_series index
    resid_full = resid.reindex(stock_series.index)
    return RegressionResult(alpha=alpha, beta=beta, r2=r2, residuals=resid_full)

def rolling_zscore(
    x: pd.Series,
    window: int = 60,
    min_periods: int = 30,
) -> pd.Series:

    m = x.rolling(window, min_periods=min_periods).mean()
    s = x.rolling(window, min_periods=min_periods).std(ddof=0)
    z_score = (x - m) / s
    return z_score

def stat_arb_signals(
    z: pd.Series,
    entry: float = 2.0,
    exit_: float = 0.5,
) -> pd.Series:
    signal = pd.Series(0, index=z.index, dtype=int)
    pos = 0
    for t, val in z.items():
        # Exit if within band
        if abs(val) <= exit_:
            pos = 0
        else:
            # Enter positions when outside entry thresholds
            if val >= entry:
                pos = -1
            elif val <= -entry:
                pos = 1
        signal.loc[t] = pos
    return signal

def pc_correlation(
    subset_returns: pd.DataFrame,
    pc_ts: pd.DataFrame,
) -> pd.DataFrame:

    aligned = subset_returns.T.join(pc_ts, how="inner")
    out = {}
    for tkr in subset_returns.index:
        if tkr not in aligned.columns:
            continue
    y = aligned[tkr]
    corr_pc1 = y.corr(aligned["PC1"]) if "PC1" in aligned else np.nan
    corr_pc2 = y.corr(aligned["PC2"]) if "PC2" in aligned else np.nan
    out[tkr] = {"pc1_corr": corr_pc1, "pc2_corr": corr_pc2} 
    return pd.DataFrame.from_dict(out, orient="index")

@dataclass
class BenchmarkResult:
    pca: BenchmarkPCA
    per_stock: pd.DataFrame # Multi-index (benchmark, stock) rows or index = stock
    historical_signals: Optional[pd.DataFrame] = None  # dates x stocks
    historical_z_scores: Optional[pd.DataFrame] = None  # dates x stocks


def process_benchmark(
    returns_df: pd.DataFrame,
    benchmark_name: str,
    tickers: List[str],
    z_window: int = 60,
    z_min_periods: int = 30,
    entry: float = 2.0,
    exit_: float = 0.5,
    ) -> BenchmarkResult:

    normalized_present = [str(t).strip().upper() for t in tickers if str(t).strip().upper() in returns_df.index]
    print(f"[process] {benchmark_name}: cfg={len(tickers)} present={len(normalized_present)}")

    pca_res = run_benchmark_pca(returns_df, benchmark_name, tickers, n_components=2, standardize=True)

    subset = align_subset(ensure_datetime_columns(returns_df), tickers)
    pc1 = pca_res.pc_time_series["PC1"]

    rows = []
    for tkr in subset.index:
        reg = regress_stock_on_pc1(subset.loc[tkr], pc1)
        z = rolling_zscore(reg.residuals, window=z_window, min_periods=z_min_periods)
        sig = stat_arb_signals(z, entry=entry, exit_=exit_)
        rows.append({
            "stock": tkr,
            "r2": reg.r2,
            "alpha": reg.alpha,
            "beta": reg.beta,
            "z": z.iloc[-1] if len(z.dropna()) else np.nan,
            "signal": int(sig.iloc[-1]) if len(sig.dropna()) else 0,
        })

    stock_df = pd.DataFrame(rows).set_index("stock").sort_index()

    # Add correlations to PCs
    corr_df = pc_correlation(subset, pca_res.pc_time_series)
    per_stock = stock_df.join(corr_df, how="left")
    per_stock.insert(0, "benchmark", benchmark_name)

    return BenchmarkResult(pca=pca_res, per_stock=per_stock)

@dataclass
class PipelineResult:
    benchmark_pcs: Dict[str, pd.DataFrame]
    loadings: Dict[str, pd.DataFrame]
    per_stock_table: pd.DataFrame


def pca_stat_arb_pipeline(
    returns_df: pd.DataFrame,
    benchmarks: Dict[str, List[str]],
    z_window: int = 60,
    z_min_periods: int = 30,
    entry: float = 2.0,
    exit_: float = 0.5,
    r2_filter: Optional[float] = None,
) -> PipelineResult:

    returns_df = ensure_datetime_columns(returns_df)
    
    pcs_dict: Dict[str, pd.DataFrame] = {}
    loadings_dict: Dict[str, pd.DataFrame] = {}
    per_stock_frames: List[pd.DataFrame] = []

    for name, tks in benchmarks.items():
        try:
            res = process_benchmark(
                returns_df,
                benchmark_name=name,
                tickers=tks,
                z_window=z_window,
                z_min_periods=z_min_periods,
                entry=entry,
                exit_=exit_,
            )
        except Exception as e:
            # Skip problematic benchmarks; append an empty per-stock frame with MultiIndex ['benchmark','stock']
            print(f"[pipeline] skipped {name}: {e}")
            empty = pd.DataFrame(columns=["r2", "alpha", "beta", "z", "signal", "pc1_corr", "pc2_corr"])
            empty.index = pd.MultiIndex.from_arrays([[], []], names=["benchmark", "stock"])
            per_stock_frames.append(empty)
            continue

        pcs_dict[name] = res.pca.pc_time_series
        loadings_dict[name] = res.pca.loadings

        df = res.per_stock.copy()
        df.index.name = "stock"
        df = df.reset_index().set_index(["benchmark", "stock"]).sort_index()
        per_stock_frames.append(df)

    per_stock_table = pd.concat(per_stock_frames, axis=0).sort_index()
    if r2_filter is not None:
        per_stock_table = per_stock_table[per_stock_table["r2"] >= r2_filter]

    return PipelineResult(
        benchmark_pcs=pcs_dict,
        loadings=loadings_dict,
        per_stock_table=per_stock_table,
    )

# ----------------------
# Convenience helpers
# ----------------------

def best_benchmark_by_pc1_corr(per_stock_table: pd.DataFrame) -> pd.Series:
    """Return the best benchmark for each stock based on PC1 correlation (NaN-safe)."""
    df = per_stock_table.reset_index()
    if df.empty or "pc1_corr" not in df.columns:
        return pd.Series(dtype=object)
    dfn = df.dropna(subset=["pc1_corr"]).copy()
    if dfn.empty:
        return pd.Series(dtype=object)
    idx = dfn.groupby("stock")["pc1_corr"].idxmax()
    best = dfn.loc[idx, ["stock","benchmark","pc1_corr"]]
    return best.set_index("stock")["benchmark"]

def extract_signals(per_stock_table: pd.DataFrame, signal_only: bool = True):
    if per_stock_table.empty:
        cols = ["benchmark","stock","z","signal","r2","pc1_corr","pc2_corr"]
        return pd.DataFrame(columns=cols)
    cols = [c for c in ["z","signal","r2","pc1_corr","pc2_corr"] if c in per_stock_table.columns]
    out = per_stock_table.reset_index()[["benchmark","stock"] + cols]
    if signal_only and "signal" in out.columns:
        out = out[out["signal"].abs() > 0]
    return out.sort_values(["benchmark","stock"]).reset_index(drop=True)


def get_historical_signals(
    returns_df: pd.DataFrame,
    benchmarks: Dict[str, List[str]],
    z_window: int = 60,
    z_min_periods: int = 30,
    entry: float = 2.0,
    exit_: float = 0.5,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Returns full historical signals and z-scores for backtesting.
    
    Parameters
    ----------
    returns_df : DataFrame (index=tickers, columns=dates)
    benchmarks : Dict[str, List[str]]
        Mapping of benchmark name to list of tickers
    z_window : int, default=60
        Rolling window for z-score calculation
    z_min_periods : int, default=30
        Minimum periods for rolling calculations
    entry : float, default=2.0
        Z-score threshold for signal entry
    exit_ : float, default=0.5
        Z-score threshold for signal exit
    
    Returns
    -------
    Dict with structure:
    {
        'benchmark_name': {
            'signals': DataFrame(dates x stocks),
            'z_scores': DataFrame(dates x stocks),
            'pc1': Series(dates),
            'pc2': Series(dates),
            'loadings': DataFrame(stocks x PCs),
            'explained_variance': array
        }
    }
    """
    returns_df = ensure_datetime_columns(returns_df)
    results = {}
    
    for benchmark_name, tickers in benchmarks.items():
        try:
            print(f"[historical] Processing {benchmark_name}...")
            
            # Run PCA for this benchmark
            pca_res = run_benchmark_pca(returns_df, benchmark_name, tickers, n_components=2, standardize=True)
            
            # Get aligned subset for this benchmark
            subset = align_subset(returns_df, tickers)
            if subset.empty:
                print(f"[historical] {benchmark_name}: No valid data after alignment")
                continue
                
            pc1 = pca_res.pc_time_series["PC1"]
            
            # Collect historical signals and z-scores for each stock
            signal_series = {}
            z_series = {}
            
            for tkr in subset.index:
                # Regress stock on PC1
                reg = regress_stock_on_pc1(subset.loc[tkr], pc1)
                
                # Calculate rolling z-scores
                z = rolling_zscore(reg.residuals, window=z_window, min_periods=z_min_periods)
                
                # Generate signals from z-scores
                sig = stat_arb_signals(z, entry=entry, exit_=exit_)
                
                signal_series[tkr] = sig
                z_series[tkr] = z
            
            # Convert to DataFrames with dates as index, stocks as columns
            signals_df = pd.DataFrame(signal_series)
            z_scores_df = pd.DataFrame(z_series)
            
            results[benchmark_name] = {
                'signals': signals_df,
                'z_scores': z_scores_df,
                'pc1': pca_res.pc_time_series["PC1"],
                'pc2': pca_res.pc_time_series["PC2"],
                'loadings': pca_res.loadings,
                'explained_variance': pca_res.explained_variance_ratio,
            }
            
            print(f"[historical] {benchmark_name}: Generated signals for {len(signal_series)} stocks over {len(signals_df)} dates")
            
        except Exception as e:
            print(f"[historical] Skipped {benchmark_name}: {e}")
            continue
    
    return results


def process_benchmark_with_history(
    returns_df: pd.DataFrame,
    benchmark_name: str,
    tickers: List[str],
    z_window: int = 60,
    z_min_periods: int = 30,
    entry: float = 2.0,
    exit_: float = 0.5,
    include_historical: bool = False,
    ) -> BenchmarkResult:
    """
    Extended version of process_benchmark that optionally includes historical signals.
    
    Parameters
    ----------
    include_historical : bool, default=False
        If True, includes full historical signals and z-scores in the result
    
    Returns
    -------
    BenchmarkResult with optional historical data
    """
    
    normalized_present = [str(t).strip().upper() for t in tickers if str(t).strip().upper() in returns_df.index]
    print(f"[process] {benchmark_name}: cfg={len(tickers)} present={len(normalized_present)}")

    pca_res = run_benchmark_pca(returns_df, benchmark_name, tickers, n_components=2, standardize=True)

    subset = align_subset(ensure_datetime_columns(returns_df), tickers)
    pc1 = pca_res.pc_time_series["PC1"]

    rows = []
    signal_series = {}
    z_series = {}
    
    for tkr in subset.index:
        reg = regress_stock_on_pc1(subset.loc[tkr], pc1)
        z = rolling_zscore(reg.residuals, window=z_window, min_periods=z_min_periods)
        sig = stat_arb_signals(z, entry=entry, exit_=exit_)
        
        # Summary stats (existing functionality)
        rows.append({
            "stock": tkr,
            "r2": reg.r2,
            "alpha": reg.alpha,
            "beta": reg.beta,
            "z": z.iloc[-1] if len(z.dropna()) else np.nan,
            "signal": int(sig.iloc[-1]) if len(sig.dropna()) else 0,
        })
        
        # Store historical data if requested
        if include_historical:
            signal_series[tkr] = sig
            z_series[tkr] = z

    stock_df = pd.DataFrame(rows).set_index("stock").sort_index()

    # Add correlations to PCs
    corr_df = pc_correlation(subset, pca_res.pc_time_series)
    per_stock = stock_df.join(corr_df, how="left")
    per_stock.insert(0, "benchmark", benchmark_name)

    # Prepare historical data if requested
    historical_signals = None
    historical_z_scores = None
    if include_historical and signal_series:
        historical_signals = pd.DataFrame(signal_series)
        historical_z_scores = pd.DataFrame(z_series)

    return BenchmarkResult(
        pca=pca_res, 
        per_stock=per_stock,
        historical_signals=historical_signals,
        historical_z_scores=historical_z_scores
    )

