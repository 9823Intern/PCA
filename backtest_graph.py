import argparse
import sys
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pca_stat_arb_module import get_historical_signals

def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Backtest graphs: cumulative return, drawdown, rolling metrics, distribution")
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--csv", type=str, help="Path to CSV containing daily returns with a date column")
    mode.add_argument("--use-example", action="store_true", help="Use the same flow as backtest_example.py to get data and plot strategy results")
    p.add_argument("--date-col", type=str, default=None, help="Name of the date column (auto-detect if omitted)")
    p.add_argument("--cols", type=str, nargs="*", default=None, help="Return columns to plot (default: all numeric)")
    p.add_argument("--window", type=int, default=63, help="Rolling window days for Sharpe/Volatility")
    p.add_argument("--title", type=str, default=None, help="Figure title")
    # Example-mode parameters
    p.add_argument("--example-csv", type=str, default="C:/Users/EnnTurn/Precept Dropbox/Enn Turn/Georges_DataBase/figionly.csv", help="CSV path used in example flow")
    p.add_argument("--benchmark-config", type=str, default="benchmark_config.json", help="Benchmark config JSON used in example flow")
    p.add_argument("--benchmark", type=str, default=None, help="Benchmark name to plot in example flow (default: first available)")
    p.add_argument("--z-window", type=int, default=60)
    p.add_argument("--z-min-periods", type=int, default=30)
    p.add_argument("--entry", type=float, default=2.0)
    p.add_argument("--exit", type=float, default=0.5)

    args = p.parse_args(list(argv) if argv is not None else None)

    if args.use_example:
        # Reproduce backtest_example.py flow without modifying it
        try:
            returns_df = pd.read_csv(args.example_csv, index_col=0, parse_dates=True)
            returns_df.index.name = "FIGI"
            col_labels = returns_df.columns.map(str).str.strip()
            parsed_cols = pd.to_datetime(col_labels, errors="coerce", format="mixed")
            returns_df = returns_df.loc[:, ~parsed_cols.isna()]
            returns_df.columns = parsed_cols[~parsed_cols.isna()]
            returns_df.index = returns_df.index.map(str).str.strip().str.upper()
        except FileNotFoundError:
            # Fallback mock, consistent with example
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            np.random.seed(42)
            returns_data = np.random.normal(0.001, 0.02, (len(tickers), len(dates)))
            returns_df = pd.DataFrame(returns_data, index=tickers, columns=dates)

        try:
            with open(args.benchmark_config, "r") as f:
                config = json.load(f)
            benchmarks = {name: data["Tickers"] for name, data in config["benchmarks"].items() if data["Tickers"]}
        except FileNotFoundError:
            benchmarks = {
                "Tech": ['AAPL', 'MSFT', 'GOOGL'],
                "Growth": ['AMZN', 'TSLA', 'GOOGL']
            }

        present = set(returns_df.index)
        filtered_benchmarks = {}
        for name, tks in benchmarks.items():
            cleaned = [str(t).strip().upper() for t in tks]  # type: ignore[attr-defined]
            cleaned = [t for t in cleaned if t in present]
            if len(cleaned) >= 2:
                filtered_benchmarks[name] = cleaned

        hist = get_historical_signals(
            returns_df=returns_df,
            benchmarks=filtered_benchmarks,
            z_window=args.z_window,
            z_min_periods=args.z_min_periods,
            entry=args.entry,
            exit_=args.exit,
        )
        if not hist:
            print("no historical signals produced.")
            return 1

        bmk = args.benchmark if args.benchmark in hist else (list(hist.keys())[0])
        signals = hist[bmk]['signals']
        common_stocks = [s for s in signals.columns if s in returns_df.index]
        common_dates = signals.index.intersection(returns_df.columns)
        if not common_stocks or len(common_dates) == 0:
            print("no overlapping dates/stocks to compute strategy.")
            return 1

        signal_matrix = signals.loc[common_dates, common_stocks]
        return_matrix = returns_df.loc[common_stocks, common_dates].T
        lagged_signals = signal_matrix.shift(1).fillna(0)
        strategy_returns = (lagged_signals * return_matrix).sum(axis=1)

        plot_df = pd.DataFrame({f"strategy_{bmk}": strategy_returns})
        ttl = args.title or f"Backtest (example) - {bmk}"
        plot_backtest(plot_df, labels=list(plot_df.columns), window=args.window, title=ttl)
        return 0

    # CSV mode
    if not args.csv:
        print("either --csv or --use-example is required.")
        return 2
    returns_df = load_returns_csv(args.csv, date_col=args.date_col, cols=args.cols)
    if returns_df.empty:
        print("no data to plot from csv.")
        return 1

    plot_backtest(returns_df, labels=args.cols, window=args.window, title=args.title)
    return 0
