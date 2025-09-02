import logging
import os
import sys
import xarray as xr

import backtester as bt

from ..configurations import configurations, Configuration, BacktestDescription
from .model_inputs import ModelInputs


def print_best_results_summary():
    """Print a summary of the best results found so far across all tests."""
    if not hasattr(run_backtest, 'best_results') or run_backtest.best_results is None:
        print("No tests have been run yet.")
        return
    
    print("\n" + "="*80)
    print(f"🏆 BEST RESULTS SUMMARY (out of {run_backtest.test_counter} tests)")
    print("="*80)
    
    config = run_backtest.best_config
    print("\n📥 BEST CONFIGURATION:")
    print_backtest_configuration(
        num_ranks=config['num_ranks'],
        num_long_ranks=config['num_long_ranks'],
        num_short_ranks=config['num_short_ranks'],
        long_weight_total=config['long_weight_total'],
        short_weight_total=config['short_weight_total'],
        z_score_tolerance=config['z_score_tolerance'],
        factor_weights=config['factor_weights'],
        mkt_cap_filter_threshold=config['mkt_cap_filter_threshold'],
        share_price_filter_threshold=config['share_price_filter_threshold'],
        dollar_volume_filter_threshold=config['dollar_volume_filter_threshold'],
        header="",
        prefix="   "
    )
    
    print("\n📈 BEST RESULTS:")
    print(f"   • Sharpe Ratio: {config['sharpe']:.4f}")
    print(f"   • Total Return: {config['return']:.2%}")
    print(f"   • Tests Run: {run_backtest.test_counter}")
    print("="*80)


def print_config_summary(
        num_ranks: int,
        long_ranks: int,
        short_ranks: int,
        factor_weights: dict[str, tuple[float, str | None]],
        compact: bool = True) -> str:
    """
    Create a compact string summary of configuration for inline printing.
    
    Returns:
        Formatted string with configuration summary
    """
    if compact:
        # Compact one-line format
        factors = ", ".join([f"{k}={v[0]:.2f}" for k, v in factor_weights.items() if v[0] > 0])
        return f"ranks={num_ranks}, L/S={long_ranks}/{short_ranks}, factors=[{factors}]"
    else:
        # Multi-line format
        lines = []
        lines.append(f"Ranks: {num_ranks} (L={long_ranks}, S={short_ranks})")
        lines.append("Factors: " + ", ".join([f"{k}={v[0]:.2f}" for k, v in factor_weights.items() if v[0] > 0]))
        return "\n".join(lines)


def print_backtest_configuration(
        num_ranks: int,
        num_long_ranks: int,
        num_short_ranks: int,
        long_weight_total: float,
        short_weight_total: float,
        z_score_tolerance: float,
        factor_weights: dict[str, tuple[float, str | None]],
        mkt_cap_filter_threshold: float,
        share_price_filter_threshold: float,
        dollar_volume_filter_threshold: float,
        header: str = "BACKTEST CONFIGURATION",
        show_reproducible_code: bool = False,
        signal_sets_path: str = None,
        prefix: str = "") -> None:
    """
    Print a formatted summary of backtest configuration parameters.
    
    Args:
        All backtest parameters...
        header: Optional header text for the configuration section
        show_reproducible_code: If True, also prints the exact code to reproduce
        signal_sets_path: Path for signal sets (only needed if show_reproducible_code=True)
        prefix: Optional prefix for all lines (for indentation)
    """
    if header:
        print(f"{prefix}{header}")
        print(f"{prefix}" + "-"*len(header))
    
    print(f"{prefix}Portfolio Parameters:")
    print(f"{prefix}   • Number of ranks: {num_ranks}")
    print(f"{prefix}   • Long ranks: {num_long_ranks}")
    print(f"{prefix}   • Short ranks: {num_short_ranks}")
    print(f"{prefix}   • Long weight total: {long_weight_total}")
    print(f"{prefix}   • Short weight total: {short_weight_total}")
    print(f"{prefix}   • Z-score tolerance: {z_score_tolerance}")
    
    print(f"{prefix}Factor Weights:")
    for factor_name, (weight, factor_type) in factor_weights.items():
        if weight > 0:
            print(f"{prefix}   • {factor_name}: {weight:.3f} ({factor_type or 'base'})")
        elif weight == 0 and factor_type:  # Show special factors even if weight is 0
            print(f"{prefix}   • {factor_name}: {weight:.3f} ({factor_type}) [disabled]")
    
    print(f"{prefix}Universe Filters:")
    print(f"{prefix}   • Market cap threshold: ${mkt_cap_filter_threshold}M")
    print(f"{prefix}   • Share price threshold: ${share_price_filter_threshold}")
    print(f"{prefix}   • Dollar volume threshold: ${dollar_volume_filter_threshold}M")
    
    if show_reproducible_code:
        print(f"\n{prefix}TO REPRODUCE:")
        print(f"{prefix}run_backtest(")
        print(f"{prefix}    model_inputs=mi,")
        print(f"{prefix}    num_ranks={num_ranks},")
        print(f"{prefix}    num_long_ranks={num_long_ranks},")
        print(f"{prefix}    num_short_ranks={num_short_ranks},")
        print(f"{prefix}    long_weight_total={long_weight_total},")
        print(f"{prefix}    short_weight_total={short_weight_total},")
        print(f"{prefix}    z_score_tolerance={z_score_tolerance},")
        print(f"{prefix}    factor_weights={factor_weights},")
        print(f"{prefix}    mkt_cap_filter_threshold={mkt_cap_filter_threshold},")
        print(f"{prefix}    share_price_filter_threshold={share_price_filter_threshold},")
        print(f"{prefix}    dollar_volume_filter_threshold={dollar_volume_filter_threshold},")
        if signal_sets_path:
            print(f"{prefix}    signal_sets_path='{signal_sets_path}',")
        print(f"{prefix}    results_manager=mi.rm")
        print(f"{prefix})")


"""
Do not create a context in this script. It is created according to the backtest
cfg settings in ModelInputs.
"""
cfg = configurations.fund

# Create ModelInputs with configuration integration
mi = ModelInputs(
    config=cfg, 
    verbose=True,
    usage_mode='backtest',
    save_to_temp_cache=False,
    use_temp_cache=False
)
# Example: Override configuration parameters for a specific backtest
def run_backtest_with_config_override():
    """Example of how to override config parameters for a specific backtest run."""
    import copy
    
    # Create a copy of the base configuration
    single_bt_config = copy.deepcopy(configurations.fund)
    single_bt_config.mkt_cap_filter_threshold = 200  # override the default
    
    # Create new ModelInputs with the modified configuration
    mi_override = ModelInputs(
        config=single_bt_config,
        verbose=True,
        usage_mode='backtest',
        save_to_temp_cache=False,
        use_temp_cache=False
    )
    
    # Run backtest with the modified ModelInputs and configuration
    # Note: Using equal weights (1/4 each)
    return run_backtest_using_model_inputs(mi_override,
                                         transcript_weight=1/4, 
                                         revision_weight=1/4, 
                                         e_to_p_weight=1/4,
                                         eps_ntm_weight=1/4)

# Simpler approach: Use ModelInputs parameter override feature
def run_backtest_with_simple_override():
    """Simpler approach using ModelInputs parameter override."""
    
    # Create ModelInputs with parameter override (no need to copy config)
    mi_override = ModelInputs(
        config=configurations.fund,  # Use base config
        mkt_cap_filter_threshold=200,  # Override specific parameter
        verbose=True,
        usage_mode='backtest',
        save_to_temp_cache=False,
        use_temp_cache=False
    )
    
    # Run backtest with the modified ModelInputs
    # Note: Using equal weights (1/4 each)
    return run_backtest_using_model_inputs(mi_override,
                                         transcript_weight=1/4, 
                                         revision_weight=1/4, 
                                         e_to_p_weight=1/4,
                                         eps_ntm_weight=1/4)

def run_backtest(
        model_inputs: ModelInputs,
        # Portfolio parameters - ALL REQUIRED
        num_ranks: int,
        num_long_ranks: int, 
        num_short_ranks: int,
        long_weight_total: float,
        short_weight_total: float,
        z_score_tolerance: float,
        # Factor weights - ALL REQUIRED
        factor_weights: dict[str, tuple[float, str | None]],  # Maps factor name to (weight, type)
        # Config parameters - ALL REQUIRED
        mkt_cap_filter_threshold: float,
        share_price_filter_threshold: float,
        dollar_volume_filter_threshold: float,
        # Output parameters - ALL REQUIRED
        signal_sets_path: str,
        results_manager: bt.ResultsManager  # Results manager is REQUIRED
    ) -> bt.Results:
    """
    EXPLICIT BACKTEST EXECUTION - NO OPTIONAL PARAMETERS
    
    This is the single source of truth for running backtests.
    All parameters must be explicitly provided to eliminate any ambiguity
    about which exact parameters produced which results.
    
    BUSINESS CRITICAL: Do not add optional parameters or fallbacks.
    """
    
    # Extract individual weights for BacktestDescription
    base_weights = {name: weight for name, (weight, type_) in factor_weights.items() if type_ is None}
    special_weights = {name: weight for name, (weight, type_) in factor_weights.items() if type_ is not None}
    
    # Create BacktestDescription object for consistent parameter tracking
    description_obj = BacktestDescription.from_run_backtest_parameters(
        num_ranks=num_ranks,
        num_long_ranks=num_long_ranks,
        num_short_ranks=num_short_ranks,
        long_weight_total=long_weight_total,
        short_weight_total=short_weight_total,
        z_score_tolerance=z_score_tolerance,
        transcript_weight=base_weights.get('transcript', 0.0),
        # revision_weight=base_weights.get('revision', 0.0),
        revision_weight=base_weights.get('revision_rank', 0.0),
        e_to_p_weight=base_weights.get('e_to_p', 0.0),
        eps_ntm_weight=special_weights.get('eps_ntm', 0.0),
        dividend_cutter_weight=special_weights.get('dividend_cutters', 0.0),
        mkt_cap_filter_threshold=mkt_cap_filter_threshold,
        share_price_filter_threshold=share_price_filter_threshold,
        dollar_volume_filter_threshold=dollar_volume_filter_threshold
    )
    
    # Set factor weights and get final scores
    model_inputs.factor_manager.weights = factor_weights
    final_scores = model_inputs.factor_manager.final_scores
    
    # Generate signal sets
    ss = bt.utilities.signal_sets.SignalSetsGenerator(
            final_scores, 
            model_inputs.context, 
            number_of_ranks=num_ranks,
            number_of_long_ranks=num_long_ranks,
            number_of_short_ranks=num_short_ranks,
            long_weight_total=long_weight_total,
            short_weight_total=short_weight_total,
            z_score_tolerance=z_score_tolerance
        )
    
    # Save signal set to Excel file using description object
    ss_full_path = os.path.normpath(os.path.join(signal_sets_path, f'{description_obj.file_name_description}.xlsx'))
    os.makedirs(os.path.dirname(ss_full_path), exist_ok=True)
    ss.da.to_pandas().to_excel(ss_full_path)

    # Calculate daily weighted returns
    dwr = bt.WeightedReturnsDaily(model_inputs.context, dailmodel_inputs.trd_da_raw, weights_param=ss.da)
    
    # Create results using description object (eliminates manual string construction risk)
    results = bt.Results(dwr.da, 
                         model_inputs.context.dates,
                         daily_total_rtns_da=model_inputs.trd_da_raw, 
                         file_name_description=description_obj.file_name_description,
                         description=description_obj.results_description)
    
    # Always add results to manager - it handles saving, comparison, and logging
    results_manager.add(results)
    
    # Store the results for tracking (if not already stored)
    if not hasattr(run_backtest, 'test_counter'):
        run_backtest.test_counter = 0
        run_backtest.best_results = None
        run_backtest.best_config = None
    
    run_backtest.test_counter += 1
    
    # Check if this is the best result so far
    is_best = False
    if run_backtest.best_results is None or results > run_backtest.best_results:
        is_best = True
        run_backtest.best_results = results
        run_backtest.best_config = {
            'num_ranks': num_ranks,
            'num_long_ranks': num_long_ranks,
            'num_short_ranks': num_short_ranks,
            'long_weight_total': long_weight_total,
            'short_weight_total': short_weight_total,
            'z_score_tolerance': z_score_tolerance,
            'factor_weights': factor_weights.copy(),
            'mkt_cap_filter_threshold': mkt_cap_filter_threshold,
            'share_price_filter_threshold': share_price_filter_threshold,
            'dollar_volume_filter_threshold': dollar_volume_filter_threshold,
            'sharpe': results.sharpe_ratio(),
            'return': results.time_weighted_return
        }
    
    # Print clear description of inputs and results
    print("\n" + "="*80)
    if is_best:
        print(f"🏆 NEW BEST RESULT! (Test #{run_backtest.test_counter})")
    else:
        print(f"📊 BACKTEST EXECUTION SUMMARY (Test #{run_backtest.test_counter})")
    print("="*80)
    
    # Show current vs best if not the best
    if not is_best and run_backtest.best_results:
        current_sharpe = results.sharpe_ratio()
        best_sharpe = run_backtest.best_config['sharpe']
        print(f"📊 Current Sharpe: {current_sharpe:.4f} | Best so far: {best_sharpe:.4f} (Test #{run_backtest.test_counter - 1})")
        print("-"*80)
    
    print("📥 INPUTS USED:")
    print_backtest_configuration(
        num_ranks=num_ranks,
        num_long_ranks=num_long_ranks,
        num_short_ranks=num_short_ranks,
        long_weight_total=long_weight_total,
        short_weight_total=short_weight_total,
        z_score_tolerance=z_score_tolerance,
        factor_weights=factor_weights,
        mkt_cap_filter_threshold=mkt_cap_filter_threshold,
        share_price_filter_threshold=share_price_filter_threshold,
        dollar_volume_filter_threshold=dollar_volume_filter_threshold,
        header="",  # No header since we already printed "INPUTS USED"
        prefix="   "
    )
    print("\n📈 RESULTS:")
    print(results)
    print("="*80)
    
    return results


def run_backtest_using_model_inputs(
        model_inputs: ModelInputs,
        # Override parameters (optional - will use ModelInputs config if not provided)
        num_ranks: int = None,
        num_long_ranks: int = None, 
        num_short_ranks: int = None,
        long_weight_total: float = None,
        short_weight_total: float = None,
        z_score_tolerance: float = None,
        factor_weights: dict[str, tuple[float, str | None]] = None
    ) -> bt.Results:
    """
    WRAPPER FUNCTION: Extracts parameters from ModelInputs and calls run_backtest()
    
    This function's job is to bridge between ModelInputs and the explicit run_backtest function.
    It extracts all required parameters from ModelInputs and passes them explicitly to run_backtest().
    """
    
    # Extract parameters from ModelInputs config, with overrides
    final_num_ranks = num_ranks if num_ranks is not None else getattr(model_inputs.config, 'number_of_ranks', 15)
    final_num_long_ranks = num_long_ranks if num_long_ranks is not None else getattr(model_inputs.config, 'number_of_long_ranks', 1)
    final_num_short_ranks = num_short_ranks if num_short_ranks is not None else getattr(model_inputs.config, 'number_of_short_ranks', 0)
    final_long_weight_total = long_weight_total if long_weight_total is not None else getattr(model_inputs.config, 'long_weight', 1.0)
    final_short_weight_total = short_weight_total if short_weight_total is not None else getattr(model_inputs.config, 'short_weight', 1.0)
    final_z_score_tolerance = z_score_tolerance if z_score_tolerance is not None else getattr(model_inputs.config, 'z_score_tolerance', 0.3)
    # Extract factor weights with overrides
    if factor_weights is None:
        # Get default weights from config
        factor_weights = {
            'transcript': (model_inputs.config.transcript_scores_weight, None),
            'revision_rank': (model_inputs.config.revision_rank_scores_weight, None),
            'e_to_p': (model_inputs.config.e_to_p_scores_weight, None),
            'eps_ntm_reward': (model_inputs.config.eps_ntm_scores_weight, 'reward'),
            'dividend_cutters': (model_inputs.config.dividend_cutter_weight, 'penalty')
        }
    
    # Call the explicit run_backtest function
    return run_backtest(
        model_inputs=model_inputs,
        num_ranks=final_num_ranks,
        num_long_ranks=final_num_long_ranks,
        num_short_ranks=final_num_short_ranks,
        long_weight_total=final_long_weight_total,
        short_weight_total=final_short_weight_total,
        z_score_tolerance=final_z_score_tolerance,
        factor_weights=factor_weights,
        mkt_cap_filter_threshold=model_inputs.config.mkt_cap_filter_threshold,
        share_price_filter_threshold=model_inputs.config.share_price_filter_threshold,
        dollar_volume_filter_threshold=model_inputs.config.dollar_volume_filter_threshold,
        signal_sets_path=model_inputs.signal_sets_path,
        results_manager=model_inputs.rm  # Always use the results manager
    )


best_results: bt.Results | None = None
best_params = None
config_vars = (f'{mi.config.mkt_cap_filter_threshold}_{mi.config.share_price_filter_threshold}_'
               f'{mi.config.dollar_volume_filter_threshold}')

def run_single_backtest(
        config: Configuration = None,
        num_ranks: int = None, 
        num_long_ranks: int = None, 
        num_short_ranks: int = None,
        z_score_tolerance: float = None,
        save_results: bool = None,
        long_weight_total: float = None,
        short_weight_total: float = None
    ):
    """
    DEPRECATED: Use run_backtest() or run_backtest_using_model_inputs() instead.
    
    This function is kept for backward compatibility but should not be used
    for new code. It has optional parameters that can create business risk
    through parameter ambiguity.
    """
    import warnings
    warnings.warn(
        "run_single_backtest() is deprecated. Use run_backtest() for explicit parameters "
        "or run_backtest_using_model_inputs() for ModelInputs-based parameters.", 
        DeprecationWarning, 
        stacklevel=2
    )
    
    # Use the wrapper function with appropriate parameter passing
    return run_backtest_using_model_inputs(
        model_inputs=mi,
        num_ranks=num_ranks,
        num_long_ranks=num_long_ranks,
        num_short_ranks=num_short_ranks,
        z_score_tolerance=z_score_tolerance,
        long_weight_total=long_weight_total,
        short_weight_total=short_weight_total
    )

def run_exclusion_tests(mi: ModelInputs, best_params: dict, best_results: bt.Results) -> tuple[bt.Results, dict, list]:
    """
    Run additional tests by excluding each factor one at a time to see if performance improves.
    
    Args:
        mi: ModelInputs instance with the best configuration
        best_params: Dictionary containing the best parameters found
        best_results: Best results object from previous testing
        
    Returns:
        tuple: (best_results, best_params, all_exclusion_results)
    """
    print("\n" + "=" * 80)
    print("🔍 RUNNING FACTOR EXCLUSION TESTS")
    print("=" * 80)
    
    # Get all factors (base, reward, and penalty)
    all_factors = best_params['factor_weights']
    
    all_exclusion_results = []
    
    # Test excluding each factor
    for factor_name, (weight, factor_type) in all_factors.items():
        print(f"\n🧪 Testing exclusion of {factor_name}")
        print("-" * 60)
        
        try:
            # Exclude the factor
            mi.factor_manager.exclude(factor_name)
            
            # Get original weights and types
            original_weights = best_params['factor_weights'].copy()
            excluded_weight = original_weights[factor_name][0]
            excluded_type = original_weights[factor_name][1]
            
            # Create new weights dictionary
            new_weights = {}
            
            if excluded_type is None:
                # For base factors, redistribute weight among other base factors
                remaining_base_factors = [
                    name for name, (_, type_) in original_weights.items()
                    if type_ is None and name != factor_name
                ]
                weight_addition = excluded_weight / len(remaining_base_factors)
                
                for name, (weight, type_) in original_weights.items():
                    if name == factor_name:
                        new_weights[name] = (0.0, type_)  # Zero weight for excluded factor
                    elif name in remaining_base_factors:
                        new_weights[name] = (weight + weight_addition, type_)  # Redistribute weight
                    else:
                        new_weights[name] = (weight, type_)  # Keep special factors unchanged
            else:
                # For special factors (reward/penalty), just set weight to 0
                for name, (weight, type_) in original_weights.items():
                    if name == factor_name:
                        new_weights[name] = (0.0, type_)  # Zero weight for excluded factor
                    else:
                        new_weights[name] = (weight, type_)  # Keep all other weights unchanged
            
            # Run backtest with new weights
            results = run_backtest(
                model_inputs=mi,
                num_ranks=best_params['number_of_ranks'],
                num_long_ranks=best_params['long_ranks'],
                num_short_ranks=best_params['short_ranks'],
                long_weight_total=best_params['long_weight_total'],
                short_weight_total=best_params['short_weight_total'],
                z_score_tolerance=getattr(configurations.fund, 'z_score_tolerance', 0.3),
                factor_weights=new_weights,
                mkt_cap_filter_threshold=mi.config.mkt_cap_filter_threshold,
                share_price_filter_threshold=mi.config.share_price_filter_threshold,
                dollar_volume_filter_threshold=mi.config.dollar_volume_filter_threshold,
                signal_sets_path=mi.signal_sets_path,
                results_manager=mi.rm
            )
            
            # Get metrics
            sharpe = results.sharpe_ratio()
            total_return = results.time_weighted_return
            
            print(f"   ✅ Without {factor_name}:")
            print(f"      Sharpe: {sharpe:.4f} (vs best: {best_params['sharpe_ratio']:.4f})")
            print(f"      Return: {total_return:.2%} (vs best: {best_params['total_return']:.2%})")
            
            # Store result
            exclusion_data = {
                'excluded_factor': factor_name,
                'sharpe': sharpe,
                'return': total_return,
                'results_obj': results,
                'weights': new_weights
            }
            all_exclusion_results.append(exclusion_data)
            
            # Check if this beats the best result
            if results > best_results:
                print(f"   🏆 NEW BEST RESULT FOUND BY EXCLUDING {factor_name}!")
                best_results = results
                best_params = {
                    **best_params,
                    'factor_weights': new_weights,
                    'excluded_factor': factor_name,
                    'sharpe_ratio': sharpe,
                    'total_return': total_return
                }
            
            # Re-include the factor for next test
            mi.factor_manager.include(factor_name)
            
        except Exception as e:
            print(f"   ❌ Error testing exclusion of {factor_name}: {e}")
            # Re-include the factor in case of error
            mi.factor_manager.include(factor_name)
            continue
    
    # Print exclusion test summary
    print("\n" + "=" * 80)
    print("📊 FACTOR EXCLUSION TEST SUMMARY")
    print("=" * 80)
    
    # Sort results by Sharpe ratio
    sorted_results = sorted(all_exclusion_results, key=lambda x: x['sharpe'], reverse=True)
    
    for idx, result in enumerate(sorted_results, 1):
        print(f"\n{idx}. Excluding {result['excluded_factor']}:")
        print(f"   • Sharpe: {result['sharpe']:.4f}")
        print(f"   • Return: {result['return']:.2%}")
        print("   • Weights:")
        for factor, (weight, factor_type) in result['weights'].items():
            if weight > 0:
                print(f"     - {factor}: {weight:.3f} ({factor_type or 'base'})")
    
    return best_results, best_params, all_exclusion_results

def run_loop(quick_test: bool = True):
    """
    Run machine learning optimization loop with immediate progress feedback.
    
    Args:
        quick_test: If True, run a smaller set of tests (~20). If False, run comprehensive tests (~24).
    """
    # Factor weight combinations for machine learning optimization
    if quick_test:
        factor_weight_combinations = [
            # (0.25, 0.25, 0.25, 0.25),  # Original fund config adjusted
            (0.80, 0.10, 0.10, 0.00),  # Heavy transcript focus
            (0.70, 0.15, 0.15, 0.00),  # Heavy transcript focus
            (0.80, 0.10, 0.10, 0.00),  # Heavy transcript focus
            (0.75, 0.15, 0.10, 0.00),  # Heavy transcript focus
            (0.75, 0.10, 0.15, 0.00),  # Heavy transcript focus
            (0.85, 0.05, 0.10, 0.00),  # Heavy transcript focus
            # (0.35, 0.25, 0.10, 0.30),  # Heavy revision focus  
            # (0.30, 0.25, 0.15, 0.30),  # Balanced transcript/revision
        ]
        # Quick test: 5 different portfolio parameter combinations
        param_combinations = [
            (16, 1, 1, 1.0, 1.0),   # ranks=21
            # (11, 1, 1, 1.0, 1.0),   # ranks=13
            # (12, 1, 1, 1.0, 1.0),   # ranks=15
            # (13, 1, 1, 1.0, 1.0),   # ranks=17
            # (14, 1, 1, 1.0, 1.0),   # ranks=19
        ]
    else:
        factor_weight_combinations = [
            (0.35, 0.25, 0.30, 0.10),  # Original fund config adjusted
            (0.35, 0.30, 0.25, 0.10),  # More revision weight
            (0.40, 0.25, 0.25, 0.10),  # More transcript weight
            (0.30, 0.35, 0.25, 0.10),  # Balanced transcript/revision
            (0.45, 0.20, 0.25, 0.10),  # Heavy transcript focus
            (0.25, 0.40, 0.25, 0.10),  # Heavy revision focus
        ]
        # Portfolio parameter combinations
        min_rank_num = 11
        max_rank_num = 15
        long_weight_totals = [1.0]
        short_weight_totals = [1.0]
        
        param_combinations = [
            (num_ranks, long_ranks, short_ranks, long_weight_total, short_weight_total)
            for num_ranks in range(min_rank_num, max_rank_num)  # 13 to 16 
            for long_ranks, short_ranks in [(1, 1)]  # Current rank patterns
            for long_weight_total in long_weight_totals
            for short_weight_total in short_weight_totals
        ]

    # Calculate total combinations
    total_combinations = len(factor_weight_combinations) * len(param_combinations)
    
    # Optimization loop
    test_type = "QUICK TEST" if quick_test else "COMPREHENSIVE"
    print("=" * 80)
    print(f"🤖 MACHINE LEARNING OPTIMIZATION - {test_type} FACTOR & PORTFOLIO TESTING")
    print("=" * 80)
    print(f"📊 Factor weight combinations: {len(factor_weight_combinations)}")
    print(f"📊 Portfolio combinations: {len(param_combinations)}")
    print(f"📊 Total tests: {total_combinations}")
    print(f"📊 Estimated time: ~{total_combinations * 0.5:.0f} seconds ({total_combinations * 0.5 / 60:.1f} minutes)")
    print("-" * 80)
    print(f"🎯 Testing factors: transcript, revision, e_to_p weights")
    if not quick_test:
        min_rank_num = min([p[0] for p in param_combinations])
        max_rank_num = max([p[0] for p in param_combinations]) + 1
        print(f"🎯 Testing portfolio: ranks {min_rank_num}-{max_rank_num-1}, long weights [1.0], short weights [1.0]")
    else:
        ranks_list = sorted(list(set([p[0] for p in param_combinations])))
        print(f"🎯 Testing portfolio: ranks {ranks_list}")
    print("=" * 80)
    
    # Track best results
    best_results: bt.Results | None = None
    best_params = None
    all_results = []
    test_count = 0
    best_test_number = 0

    for weights in factor_weight_combinations:
        # Convert tuple to factor weights dictionary
        factor_weights = {
            'transcript': (weights[0], None),
            'revision_rank': (weights[1], None),
            'e_to_p': (weights[2], None),
            'eps_ntm_reward': (weights[3], 'reward'),
            'dividend_cutters': (0.5, 'penalty')  # Keep consistent penalty weight
        }
        
        print(f"\n🧪 FACTOR WEIGHTS: " + ", ".join(f"{k}={v[0]:.2f} ({v[1] or 'base'})" for k, v in factor_weights.items()))
        print("-" * 60)
        
        # Set factor weights and get final scores
        print("🧮 Computing final scores with new factor weights...")
        mi.factor_manager.weights = factor_weights
        
        for num_ranks, long_ranks, short_ranks, long_weight_total, short_weight_total in param_combinations:
            test_count += 1
            
            # Validate combination
            if long_ranks + short_ranks > num_ranks:
                print(f"⏭️  Skipping invalid: ranks={num_ranks}, long={long_ranks}, short={short_ranks}")
                continue
            
            print(f"   🔬 Test {test_count}/{total_combinations}: ranks={num_ranks}, long={long_ranks}, short={short_ranks}, long_w={long_weight_total}, short_w={short_weight_total}")
            
            try:
                # Call run_backtest directly with explicit parameters for maximum business accuracy
                results = run_backtest(
                    model_inputs=mi,
                    num_ranks=num_ranks,
                    num_long_ranks=long_ranks,
                    num_short_ranks=short_ranks,
                    long_weight_total=long_weight_total,
                    short_weight_total=short_weight_total,
                    z_score_tolerance=getattr(configurations.fund, 'z_score_tolerance', 0.3),
                    factor_weights=factor_weights,
                    mkt_cap_filter_threshold=mi.config.mkt_cap_filter_threshold,
                    share_price_filter_threshold=mi.config.share_price_filter_threshold,
                    dollar_volume_filter_threshold=mi.config.dollar_volume_filter_threshold,
                    signal_sets_path=mi.signal_sets_path,
                    results_manager=mi.rm
                )
                
                # Get key metrics for display
                sharpe = results.sharpe_ratio()
                total_return = results.time_weighted_return
                
                config_str = print_config_summary(num_ranks, long_ranks, short_ranks, factor_weights)
                print(f"      ✅ Results: Sharpe={sharpe:.4f}, Return={total_return:.2%}")
                print(f"         Config: {config_str}")
                
                # Store result for tracking
                result_data = {
                    'test_num': test_count,
                    'factor_weights': factor_weights,
                    'portfolio': (num_ranks, long_ranks, short_ranks, long_weight_total, short_weight_total),
                    'sharpe': sharpe,
                    'return': total_return,
                    'results_obj': results
                }
                all_results.append(result_data)
                
                # Check if this is the best result so far
                if best_results is None or results > best_results:
                    if best_results is not None:
                        print(f"      🏆 *** NEW BEST RESULT *** Test #{test_count} - Sharpe: {sharpe:.4f} (prev: {best_params['sharpe_ratio']:.4f})")
                    else:
                        print(f"      🏆 *** FIRST RESULT - BEST SO FAR *** Test #{test_count} - Sharpe: {sharpe:.4f}")
                    
                    best_results = results
                    best_test_number = test_count
                    best_params = {
                        'factor_weights': factor_weights,
                        'number_of_ranks': num_ranks,
                        'long_ranks': long_ranks,
                        'short_ranks': short_ranks,
                        'long_weight_total': long_weight_total,
                        'short_weight_total': short_weight_total,
                        'sharpe_ratio': sharpe,
                        'total_return': total_return,
                        'test_number': test_count
                    }
                    print("         📊 Factors: " + ", ".join(f"{k}={v[0]:.2f} ({v[1] or 'base'})" for k, v in factor_weights.items()))
                    print(f"         📊 Portfolio: ranks={num_ranks}, long={long_ranks}, short={short_ranks}")
                    
            except Exception as e:
                print(f"      ❌ Error in test {test_count}: {e}")
                continue

        # Summary after each factor weight combination
        if best_params:
            # Find current factor group by matching base factor weights
            base_weights = tuple(v[0] for k, v in factor_weights.items() if k in ['transcript', 'revision_rank', 'e_to_p'])
            factor_group_num = next(i for i, w in enumerate(factor_weight_combinations, 1) if w[:3] == base_weights)
            print(f"\n   📊 CURRENT BEST AFTER FACTOR GROUP {factor_group_num}/{len(factor_weight_combinations)}:")
            print(f"      🥇 Best Sharpe: {best_params['sharpe_ratio']:.4f} (Return: {best_params['total_return']:.2%})")
            print("         Factors: " + ", ".join(f"{k}={v[0]:.2f} ({v[1] or 'base'})" for k, v in best_params['factor_weights'].items()))
            print(f"         Portfolio: ranks={best_params['number_of_ranks']}, long={best_params['long_ranks']}, short={best_params['short_ranks']}")
        print("-" * 60)

    # Print final summary
    print("\n" + "=" * 80)
    print(f"🎉 MACHINE LEARNING OPTIMIZATION COMPLETE - {test_type}")
    print("=" * 80)

    if best_params:
        print(f"📊 Total successful tests: {len(all_results)}/{total_combinations}")
        print(f"\n🏆 BEST CONFIGURATION FOUND:")
        print(f"   🎯 Factor Weights:")
        print("      • Factors:")
        for factor_name, (weight, factor_type) in best_params['factor_weights'].items():
            print(f"         - {factor_name}: {weight:.3f} ({factor_type or 'base'})")
        print(f"   🎯 Portfolio Parameters:")
        print(f"      • Number of ranks: {best_params['number_of_ranks']}")
        print(f"      • Long ranks: {best_params['long_ranks']}")
        print(f"      • Short ranks: {best_params['short_ranks']}")
        print(f"      • Long weight total: {best_params['long_weight_total']}")
        print(f"      • Short weight total: {best_params['short_weight_total']}")
        print(f"   🎯 Performance:")
        print(f"      • Sharpe Ratio: {best_params['sharpe_ratio']:.4f}")
        print(f"      • Total Return: {best_params['total_return']:.2%}")
        if not quick_test:
            print(f"\n📁 Results saved to: {mi.results_path}")
        print(f"\n🎯 Best Results Object:\n{best_results}")
    else:
        print("❌ No valid configurations found!")
    
    return best_results, best_params, all_results

# Initialize tracking variables for the entire script
total_tests = 0  # Will be updated after run_loop
all_exclusion_results = []  # Will be populated by run_exclusion_tests

# COMMENTED OUT - Multiple test loops for optimization
# Uncomment the block below to run multiple penalty weight tests
"""
# Test different penalty weights with best base factor weights
penalty_weights = [0.0, 0.25, 0.5, 0.75, 1.0]  # Range of penalty weights to test
base_weights = {
    'transcript': (0.85, None),
    'revision_rank': (0.05, None),
    'e_to_p': (0.10, None),
    'eps_ntm_reward': (0.0, 'reward')
}

print("\n" + "=" * 80)
print("🔍 TESTING DIFFERENT PENALTY WEIGHTS")
print("=" * 80)
print("Base factor weights:")
for name, (weight, type_) in base_weights.items():
    print(f"  • {name}: {weight:.2f} ({type_ or 'base'})")
print("\nTesting penalty weights:", ", ".join(f"{w:.2f}" for w in penalty_weights))
print("-" * 80)

best_results = None
best_params = None

for penalty_weight in penalty_weights:
    print(f"\n🧪 Testing penalty weight: {penalty_weight:.2f}")
    
    # Create factor weights dictionary with current penalty weight
    factor_weights = base_weights.copy()
    factor_weights['dividend_cutters'] = (penalty_weight, 'penalty')
    
    # Print the exact configuration being tested
    print(f"   📥 Inputs: ranks=16, L/S=1/1, penalty={penalty_weight:.2f}")
    print(f"      Factors: transcript={base_weights['transcript'][0]:.2f}, revision_rank={base_weights['revision_rank'][0]:.2f}, e_to_p={base_weights['e_to_p'][0]:.2f}")
    
    # Run backtest with these weights
    results = run_backtest(
        model_inputs=mi,
        num_ranks=16,
        num_long_ranks=1,
        num_short_ranks=1,
        long_weight_total=1.0,
        short_weight_total=1.0,
        z_score_tolerance=getattr(configurations.fund, 'z_score_tolerance', 0.3),
        factor_weights=factor_weights,
        mkt_cap_filter_threshold=mi.config.mkt_cap_filter_threshold,
        share_price_filter_threshold=mi.config.share_price_filter_threshold,
        dollar_volume_filter_threshold=mi.config.dollar_volume_filter_threshold,
        signal_sets_path=mi.signal_sets_path,
        results_manager=mi.rm
    )
    
    # Get performance metrics
    sharpe = results.sharpe_ratio()
    total_return = results.time_weighted_return
    
    print(f"   📈 Results: Sharpe={sharpe:.4f}, Return={total_return:.2%}")
    
    # Update best if this is better
    if best_results is None or results > best_results:
        if best_results is not None:
            print(f"   🏆 NEW BEST RESULT! (Previous best Sharpe: {best_params['sharpe_ratio']:.4f})")
        best_results = results
        best_params = {
            'factor_weights': factor_weights,
            'number_of_ranks': 16,
            'long_ranks': 1,
            'short_ranks': 1,
            'long_weight_total': 1.0,
            'short_weight_total': 1.0,
            'sharpe_ratio': sharpe,
            'total_return': total_return
        }

# Run initial backtest to get the baseline results
print("\n🔍 Running baseline test with best parameters...")
results = run_backtest(
        model_inputs=mi,
        num_ranks=best_params['number_of_ranks'],
        num_long_ranks=best_params['long_ranks'],
        num_short_ranks=best_params['short_ranks'],
        long_weight_total=best_params['long_weight_total'],
        short_weight_total=best_params['short_weight_total'],
        z_score_tolerance=getattr(configurations.fund, 'z_score_tolerance', 0.3),
        factor_weights=best_params['factor_weights'],
        mkt_cap_filter_threshold=mi.config.mkt_cap_filter_threshold,
        share_price_filter_threshold=mi.config.share_price_filter_threshold,
        dollar_volume_filter_threshold=mi.config.dollar_volume_filter_threshold,
        signal_sets_path=mi.signal_sets_path,
        results_manager=mi.rm
    )

# Update best_params with actual performance
best_params['sharpe_ratio'] = results.sharpe_ratio()
best_params['total_return'] = results.time_weighted_return
best_results = results

print("\n🔍 Testing if excluding any factors improves performance...")
best_results, best_params, all_exclusion_results = run_exclusion_tests(mi, best_params, best_results)

# Print final conclusion with complete configuration
print("\n" + "="*80)
print("🏆 FINAL OPTIMIZATION SUMMARY")
print("="*80)

# Calculate total tests performed
total_tests_performed = len(penalty_weights) + 1 + len(all_exclusion_results)  # +1 for baseline test
print(f"\n📊 TESTS PERFORMED:")
print(f"   • Penalty weight tests: {len(penalty_weights)}")
print(f"   • Baseline test: 1")
print(f"   • Factor exclusion tests: {len(all_exclusion_results)}")
print(f"   • TOTAL: {total_tests_performed}")

print("\n🏆 BEST CONFIGURATION FOUND:")
if 'excluded_factor' in best_params:
    print(f"   ⚠️  Best performance achieved by EXCLUDING the {best_params['excluded_factor']} factor!")

print("\n📥 WINNING INPUTS:")
print_backtest_configuration(
    num_ranks=best_params['number_of_ranks'],
    num_long_ranks=best_params['long_ranks'],
    num_short_ranks=best_params['short_ranks'],
    long_weight_total=best_params['long_weight_total'],
    short_weight_total=best_params['short_weight_total'],
    z_score_tolerance=getattr(configurations.fund, 'z_score_tolerance', 0.3),
    factor_weights=best_params['factor_weights'],
    mkt_cap_filter_threshold=mi.config.mkt_cap_filter_threshold,
    share_price_filter_threshold=mi.config.share_price_filter_threshold,
    dollar_volume_filter_threshold=mi.config.dollar_volume_filter_threshold,
    header="",
    show_reproducible_code=True,
    signal_sets_path=mi.signal_sets_path
)

print("\n📈 BEST RESULTS ACHIEVED:")
print(f"   • Sharpe Ratio: {best_params['sharpe_ratio']:.4f}")
print(f"   • Total Return: {best_params['total_return']:.2%}")
if 'test_number' in best_params:
    print(f"   • Found in: Test #{best_params['test_number']} of {total_tests_performed}")

# Show comparison to baseline if available
if 'baseline_sharpe' in locals():
    improvement = ((best_params['sharpe_ratio'] - baseline_sharpe) / baseline_sharpe) * 100
    print(f"\n📊 IMPROVEMENT OVER BASELINE:")
    print(f"   • Baseline Sharpe: {baseline_sharpe:.4f}")
    print(f"   • Best Sharpe: {best_params['sharpe_ratio']:.4f}")
    print(f"   • Improvement: {improvement:+.1f}%")

print("="*80)
"""

# Uncomment to run full comprehensive optimization (24 tests):
# best_results, best_params, all_results = run_loop(quick_test=False)
# if best_results and best_params:
#     best_results, best_params, exclusion_results = run_exclusion_tests(mi, best_params, best_results)

# ============================================================================
# FACTOR WEIGHT OPTIMIZATION - TESTING MULTIPLE COMBINATIONS
# ============================================================================

# Initialize the factor manager before running tests
_ = mi.factor_manager  # Trigger lazy initialization

print("\n" + "=" * 80)
print("🔬 FACTOR WEIGHT OPTIMIZATION - COMPREHENSIVE TESTING")
print("=" * 80)

# Define base factor weight combinations (must sum to 1.0)
# Testing various allocations between transcript, revision, and e_to_p
base_factor_combinations = [
    # Heavy transcript focus
    (0.80, 0.10, 0.10),  # 80% transcript, 10% revision, 10% e_to_p
    (0.70, 0.15, 0.15),  # 70% transcript, balanced others
    (0.60, 0.20, 0.20),  # 60% transcript, balanced others
    
    # Balanced approaches
    (0.50, 0.25, 0.25),  # 50% transcript, balanced others
    (0.40, 0.30, 0.30),  # Fairly balanced
    (0.34, 0.33, 0.33),  # Equal weights
    
    # Heavy revision focus
    (0.20, 0.60, 0.20),  # 60% revision
    (0.25, 0.50, 0.25),  # 50% revision
    (0.30, 0.40, 0.30),  # 40% revision
    
    # Heavy e_to_p focus
    (0.20, 0.20, 0.60),  # 60% e_to_p
    (0.25, 0.25, 0.50),  # 50% e_to_p
    (0.30, 0.30, 0.40),  # 40% e_to_p
    
    # Mixed strategies
    (0.45, 0.35, 0.20),  # Transcript + revision focus
    (0.45, 0.20, 0.35),  # Transcript + e_to_p focus
    (0.35, 0.45, 0.20),  # Revision + transcript focus
]

# EPS NTM reward weights to test
eps_ntm_weights = [0.0, 0.1, 0.2, 0.3]

# Dividend cutter penalty weights to test
dividend_penalty_weights = [0.0, 0.25, 0.5, 0.75, 1.0]

# Calculate total combinations
total_combinations = len(base_factor_combinations) * len(eps_ntm_weights) * len(dividend_penalty_weights)

print(f"\n📊 Test Configuration:")
print(f"   • Base factor combinations: {len(base_factor_combinations)}")
print(f"   • EPS NTM reward weights: {len(eps_ntm_weights)} values")
print(f"   • Dividend penalty weights: {len(dividend_penalty_weights)} values")
print(f"   • Total tests to run: {total_combinations}")
print(f"   • Estimated time: ~{total_combinations * 0.5:.0f} seconds ({total_combinations * 0.5 / 60:.1f} minutes)")
print("=" * 80)

# Track best results
best_results = None
best_params = None
all_results = []
test_count = 0

# Run all combinations
for base_weights in base_factor_combinations:
    transcript_w, revision_w, e_to_p_w = base_weights
    
    for eps_ntm_w in eps_ntm_weights:
        for dividend_penalty_w in dividend_penalty_weights:
            test_count += 1
            
            # Create factor weights dictionary
            factor_weights = {
                'transcript': (transcript_w, None),
                # 'revision': (revision_w, None),
                'revision_rank': (revision_w, None),
                'e_to_p': (e_to_p_w, None),
                'eps_ntm_reward': (eps_ntm_w, 'reward'),
                'dividend_cutters': (dividend_penalty_w, 'penalty')
            }
            
            print(f"\n🧪 Test {test_count}/{total_combinations}:")
            print(f"   Base: T={transcript_w:.2f}, R={revision_w:.2f}, E={e_to_p_w:.2f}")
            print(f"   Special: EPS_NTM={eps_ntm_w:.2f}, DIV_PEN={dividend_penalty_w:.2f}")
            
            try:
                # Run backtest with these weights
                results = run_backtest(
                    model_inputs=mi,
                    num_ranks=16,  # Using default from config
                    num_long_ranks=1,
                    num_short_ranks=0,
                    long_weight_total=1.0,
                    short_weight_total=0.0,
                    z_score_tolerance=0.6,
                    factor_weights=factor_weights,
                    mkt_cap_filter_threshold=mi.config.mkt_cap_filter_threshold,
                    share_price_filter_threshold=mi.config.share_price_filter_threshold,
                    dollar_volume_filter_threshold=mi.config.dollar_volume_filter_threshold,
                    signal_sets_path=mi.signal_sets_path,
                    results_manager=mi.rm
                )
                
                # Get metrics
                sharpe = results.sharpe_ratio()
                total_return = results.time_weighted_return
                
                print(f"   ✅ Results: Sharpe={sharpe:.4f}, Return={total_return:.2%}")
                
                # Store result
                result_data = {
                    'test_num': test_count,
                    'transcript': transcript_w,
                    # 'revision': revision_w,
                    'revision_rank': revision_w,
                    'e_to_p': e_to_p_w,
                    'eps_ntm': eps_ntm_w,
                    'dividend_penalty': dividend_penalty_w,
                    'sharpe': sharpe,
                    'return': total_return,
                    'results_obj': results
                }
                all_results.append(result_data)
                
                # Check if this is the best result
                if best_results is None or results > best_results:
                    print(f"   🏆 NEW BEST RESULT! Sharpe: {sharpe:.4f}")
                    best_results = results
                    best_params = result_data.copy()
                    
            except Exception as e:
                print(f"   ❌ Error in test {test_count}: {e}")
                continue

# Print final summary
print("\n" + "=" * 80)
print("📊 OPTIMIZATION COMPLETE - FINAL SUMMARY")
print("=" * 80)
print(f"\n✅ Successfully completed {len(all_results)}/{total_combinations} tests")

if best_params:
    print(f"\n🏆 BEST CONFIGURATION FOUND:")
    print(f"   📥 Factor Weights:")
    print(f"      • Transcript: {best_params['transcript']:.2%}")
    # print(f"      • Revision: {best_params['revision']:.2%}")
    print(f"      • Revision Rank: {best_params['revision_rank']:.2%}")
    print(f"      • E-to-P: {best_params['e_to_p']:.2%}")
    print(f"      • EPS NTM Reward: {best_params['eps_ntm']:.2f}")
    print(f"      • Dividend Penalty: {best_params['dividend_penalty']:.2f}")
    print(f"   📈 Performance:")
    print(f"      • Sharpe Ratio: {best_params['sharpe']:.4f}")
    print(f"      • Total Return: {best_params['return']:.2%}")
    print(f"      • Test Number: {best_params['test_num']} of {total_combinations}")
    
    # Show top 5 configurations
    print("\n📊 TOP 5 CONFIGURATIONS BY SHARPE RATIO:")
    sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)[:5]
    for idx, config in enumerate(sorted_results, 1):
        print(f"\n{idx}. Sharpe: {config['sharpe']:.4f}, Return: {config['return']:.2%}")
        # print(f"   Base: T={config['transcript']:.2f}, R={config['revision']:.2f}, E={config['e_to_p']:.2f}")
        print(f"   Base: T={config['transcript']:.2f}, R={config['revision_rank']:.2f}, E={config['e_to_p']:.2f}")
        print(f"   Special: EPS={config['eps_ntm']:.2f}, DIV={config['dividend_penalty']:.2f}")

print("\n" + "=" * 80)
print(f"📁 Results saved to: {mi.results_path}")
print("=" * 80)
