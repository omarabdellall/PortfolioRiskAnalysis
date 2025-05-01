#functions for calculating risk metrics
import numpy as np
import pandas as pd
from datetime import datetime
from arch import arch_model

from .config import HISTORICAL_STRESS_SCENARIOS, PROSPECTIVE_SCENARIOS

def calculate_portfolio_returns(portfolio_df, historical_data, currency_data=None, 
                                fx_rates=None, fx_tickers=None, base_currency='USD'):
    print("\nCalculating daily portfolio value and returns...")
    if fx_rates is not None and not fx_rates.empty:
        print(f"Applying FX adjustments to base currency: {base_currency}")
        # align fx_rates to historical_data index, forward fill any new NaNs
        fx_rates_aligned = fx_rates.reindex(historical_data.index).ffill()
        # backfill any initial NaNs, just in case fx starts later than stock data
        fx_rates_aligned = fx_rates_aligned.bfill() 
    else:
        fx_rates_aligned = None
        
    try:
        required_tickers = list(portfolio_df['Ticker'])
        if not all(ticker in historical_data.columns for ticker in required_tickers):
            missing = [t for t in required_tickers if t not in historical_data.columns]
            print(f"Error: Price data missing for tickers in portfolio: {missing}")
            return None, None

        # dictionary of holdings: {Ticker: Holding}
        holdings_dict = portfolio_df.set_index('Ticker')['Holding'].to_dict()
        portfolio_prices = historical_data[required_tickers]

        # calculate value over time
        portfolio_values = portfolio_prices.copy()
        for ticker, holding in holdings_dict.items():
            # local currency first
            local_value = portfolio_prices[ticker] * holding
            if fx_rates_aligned is not None and currency_data and ticker in currency_data:
                stock_curr = currency_data[ticker].upper()
                if stock_curr != base_currency.upper() and stock_curr in fx_tickers:
                    fx_pair = fx_tickers[stock_curr]
                    if fx_pair in fx_rates_aligned.columns:
                        fx_rate_series = fx_rates_aligned[fx_pair]
                        combined_value = local_value * fx_rate_series
                        portfolio_values[ticker] = combined_value
                    else:
                        print(f"Warning: FX rate for {fx_pair} not found, cannot adjust {ticker}.")
                        portfolio_values[ticker] = np.nan
                else:
                    portfolio_values[ticker] = local_value
            else:
                portfolio_values[ticker] = local_value

        # calculate total portfolio value 
        portfolio_total_value = portfolio_values.sum(axis=1)
        # calculate daily percentage returns
        portfolio_returns = portfolio_total_value.pct_change().dropna() # dropna removes the first NaN
        # combine into a results DataFrame
        results_df = pd.DataFrame({
            'Total Value': portfolio_total_value,
            'Daily Return': portfolio_returns
        })
        asset_values_base_ccy = portfolio_values
        asset_returns_base_ccy = asset_values_base_ccy.pct_change()

        print("Portfolio value and returns calculated successfully.")
        return results_df, asset_returns_base_ccy

    except Exception as e:
        print(f"Error calculating portfolio returns: {e}")
        return None, None

def calculate_covariance_matrix(asset_returns):
    if asset_returns is None or asset_returns.empty:
        print("Warning: Cannot calculate covariance matrix from empty asset returns.")
        return None
    
    try:
        asset_returns = asset_returns.dropna(axis=1, how='all')
        if asset_returns.empty:
            print("Warning: Asset returns DataFrame is empty after dropping all-NaN columns.")
            return None
            
        asset_returns_clean = asset_returns.dropna()
        if len(asset_returns_clean) < 2:
             print("Warning: Not enough non-NaN rows to calculate covariance matrix.")
             return None
             
        cov_matrix_daily = asset_returns_clean.cov()
        
        # annualize the covariance matrix
        trading_days = 252
        cov_matrix_annualized = cov_matrix_daily * trading_days
        
        return cov_matrix_annualized

    except Exception as e:
        print(f"Error calculating covariance matrix: {e}")
        return None

def calculate_historical_var(portfolio_returns_series, confidence_level=0.95, horizon_days=5):
    """Calculates VaR using the historical simulation method.
    
    Args:
        portfolio_returns_series (pd.Series): Series of daily portfolio returns.
        confidence_level (float): Confidence level for VaR (e.g., 0.95 for 95%).
        horizon_days (int): Time horizon for VaR in days.

    Returns:
        float: Calculated VaR value (as a negative number representing loss).
               Returns np.nan if calculation fails.
    """
    print(f"Calculating {horizon_days}-day VaR at {confidence_level*100}% confidence...")
    if not isinstance(portfolio_returns_series, pd.Series):
        print("Error: Input must be a pandas Series of returns.")
        return np.nan
        
    if portfolio_returns_series.empty or portfolio_returns_series.isnull().all():
        print("Warning: Portfolio returns data is empty or all NaN. Cannot calculate VaR.")
        return np.nan
    
    daily_returns = portfolio_returns_series.dropna()
    if daily_returns.empty:
         print("Warning: No valid daily returns after dropping NaN. Cannot calculate VaR.")
         return np.nan
         
    # sort returns and find the percentile corresponding to the confidence level
    sorted_returns = np.sort(daily_returns)
    var_percentile_index = int((1.0 - confidence_level) * len(sorted_returns))
    
    # ensure index is within bounds
    if var_percentile_index >= len(sorted_returns):
        print(f"Warning: Not enough data points ({len(sorted_returns)}) to calculate VaR at {confidence_level*100}% confidence. Returning worst loss.")
        var_1_day = sorted_returns[0] if len(sorted_returns) > 0 else np.nan
    else:
        var_1_day = sorted_returns[var_percentile_index]
    
    # scale VaR to the desired horizon
    var_horizon = var_1_day * np.sqrt(horizon_days)
    print(f"Calculated VaR: {var_horizon:.4f} (This represents the potential loss)")
    return var_horizon

def calculate_historical_cvar(portfolio_returns_series, confidence_level=0.95, horizon_days=5):
    print(f"Calculating {horizon_days}-day CVaR at {confidence_level*100}% confidence...")
    if not isinstance(portfolio_returns_series, pd.Series):
        print("Error: Input must be a pandas Series of returns.")
        return np.nan
        
    if portfolio_returns_series.empty or portfolio_returns_series.isnull().all():
        print("Warning: Portfolio returns data is empty or all NaN. Cannot calculate CVaR.")
        return np.nan

    daily_returns = portfolio_returns_series.dropna()
    if daily_returns.empty:
         print("Warning: No valid daily returns after dropping NaN. Cannot calculate CVaR.")
         return np.nan

    sorted_returns = np.sort(daily_returns)
    var_percentile_index = int((1.0 - confidence_level) * len(sorted_returns))
    if var_percentile_index >= len(sorted_returns):
        print(f"Warning: Not enough data points ({len(sorted_returns)}) to calculate CVaR at {confidence_level*100}% confidence. Returning worst loss if available.")
        cvar_1_day = sorted_returns[0] if len(sorted_returns) > 0 else np.nan
    elif var_percentile_index < 0:
         print("Error: Invalid index calculated for CVaR tail.")
         return np.nan
    else:
        tail_returns = sorted_returns[:var_percentile_index + 1]
        if len(tail_returns) == 0:
             print("Warning: No returns found in the tail for CVaR calculation.")
             return np.nan
        cvar_1_day = np.mean(tail_returns)
    if np.isnan(cvar_1_day):
         return np.nan

    cvar_horizon = cvar_1_day * np.sqrt(horizon_days)
    print(f"Calculated CVaR: {cvar_horizon:.4f}")
    return cvar_horizon

def run_stress_test(portfolio_df, historical_data, scenario_key):
    if scenario_key not in HISTORICAL_STRESS_SCENARIOS:
        print(f"Error: Unknown historical stress scenario key: {scenario_key}")
        return None

    scenario = HISTORICAL_STRESS_SCENARIOS[scenario_key]
    start = scenario['start_date']
    end = scenario['end_date']
    description = scenario['description']

    print(f"\nRunning Stress Test: {description} ({start} to {end})...")

    try:
        if not isinstance(historical_data.index, pd.DatetimeIndex):
             historical_data.index = pd.to_datetime(historical_data.index)
             
        scenario_data = historical_data.loc[start:end]
        if scenario_data.empty:
            print(f"Warning: No historical price data available for the period {start} to {end}. Cannot run stress test.")
            return {"scenario": description, "start": start, "end": end, "impact_pct": np.nan, "error": "No data for period"}

        required_tickers = list(portfolio_df['Ticker'])
        problematic_tickers = []
        for ticker in required_tickers:
            if ticker not in scenario_data.columns:
                problematic_tickers.append(f"{ticker} (Missing column)")
            elif scenario_data[ticker].isnull().any():
                problematic_tickers.append(f"{ticker} (NaN values)")
        
        if problematic_tickers:
            error_msg = f"Invalid/Missing data within period for: {', '.join(problematic_tickers)}"
            print(f"Warning: {error_msg}")
            return {"scenario": description, "start": start, "end": end, "impact_pct": np.nan, "error": error_msg}
            
        holdings_dict = portfolio_df.set_index('Ticker')['Holding'].to_dict()
        start_prices = scenario_data.iloc[0][required_tickers]
        end_prices = scenario_data.iloc[-1][required_tickers]
        
        if start_prices.isnull().any() or end_prices.isnull().any():
             print("Warning: NaN values found in start/end prices for the stress period. Cannot reliably calculate impact.")
             return {"scenario": description, "start": start, "end": end, "impact_pct": np.nan, "error": "NaN prices in period"}
             
        start_value = sum(start_prices[ticker] * holdings_dict[ticker] for ticker in required_tickers)
        end_value = sum(end_prices[ticker] * holdings_dict[ticker] for ticker in required_tickers)

        if start_value == 0:
            impact_pct = 0.0 if end_value == 0 else np.inf
        else:
            impact_pct = ((end_value - start_value) / start_value) 

        print(f"Stress test completed. Start Value: {start_value:.2f}, End Value: {end_value:.2f}, Impact: {impact_pct*100:.2f}%")
        return {"scenario": description, "start": start, "end": end, "impact_pct": impact_pct}

    except KeyError as e:
         print(f"Error running stress test {description}: Data likely missing for date range {start} to {end}. Error: {e}")
         return {"scenario": description, "start": start, "end": end, "impact_pct": np.nan, "error": "Date range likely missing"}
    except Exception as e:
        print(f"Error running stress test {description}: {e}")
        return {"scenario": description, "start": start, "end": end, "impact_pct": np.nan, "error": str(e)}

def run_scenario_analysis(portfolio_df, latest_prices, beta_data, sector_data, scenario_key):
    if scenario_key not in PROSPECTIVE_SCENARIOS:
        print(f"Error: Unknown prospective scenario key: {scenario_key}")
        return None

    scenario = PROSPECTIVE_SCENARIOS[scenario_key]
    description = scenario['description']
    scenario_type = scenario['type']

    print(f"\nRunning Scenario Analysis: {description}...")

    try:
        holdings_dict = portfolio_df.set_index('Ticker')['Holding'].to_dict()
        required_tickers = list(portfolio_df['Ticker'])

        # check if latest prices are available for all tickers
        if latest_prices.isnull().any():
             missing_prices = latest_prices[latest_prices.isnull()].index.tolist()
             print(f"Warning: Latest price data missing for tickers: {missing_prices}. Cannot run scenario.")
             return {"scenario": description, "impact_pct": np.nan, "error": f"Missing latest prices for {missing_prices}"}

        # calculate initial portfolio value
        initial_value = sum(latest_prices[ticker] * holdings_dict[ticker] for ticker in required_tickers)
        if initial_value == 0:
            print("Warning: Initial portfolio value is zero. Cannot calculate percentage impact.")
            return {"scenario": description, "impact_pct": 0.0 if initial_value == 0 else np.nan, "error": "Initial value is zero"}

        stressed_value = 0
        missing_info_tickers = []

        for ticker in required_tickers:
            original_holding_value = latest_prices[ticker] * holdings_dict[ticker]
            stressed_holding_value = original_holding_value
            if scenario_type == "uniform":
                shock = scenario['shock_pct']
                stressed_holding_value *= (1 + shock)

            elif scenario_type == "beta_weighted":
                beta = beta_data.get(ticker)
                if beta is None or pd.isna(beta):
                    print(f"Warning: Beta missing or NaN for {ticker}. Cannot apply beta-weighted shock. Excluding from stressed value change.")
                    missing_info_tickers.append(ticker)
                elif not isinstance(beta, (int, float)):
                    print(f"Warning: Beta value '{beta}' for {ticker} is not numeric. Cannot apply beta-weighted shock. Excluding from stressed value change.")
                    missing_info_tickers.append(ticker)
                else:
                    index_shock = scenario['index_shock_pct']
                    ticker_shock = beta * index_shock
                    stressed_holding_value *= (1 + ticker_shock)

            elif scenario_type == "sector_specific":
                sector = sector_data.get(ticker)
                target_sector = scenario['target_sector']
                if sector is None or sector == 'Unknown':
                     print(f"Warning: Sector unknown for {ticker}. Cannot apply sector shock.")
                     missing_info_tickers.append(ticker)
                elif sector == target_sector:
                    shock = scenario['shock_pct']
                    stressed_holding_value *= (1 + shock)

            stressed_value += stressed_holding_value
        # overall impact
        impact_pct = (stressed_value - initial_value) / initial_value
        result = {"scenario": description, "impact_pct": impact_pct}
        if missing_info_tickers:
            result["warning"] = f"Calculation incomplete due to missing info (beta/sector) for: {list(set(missing_info_tickers))}"
            print(f"Scenario Warning: {result['warning']}")

        print(f"Scenario analysis completed. Initial Value: {initial_value:.2f}, Stressed Value: {stressed_value:.2f}, Impact: {impact_pct*100:.2f}%")
        return result

    except Exception as e:
        print(f"Error running scenario analysis {description}: {e}")
        return {"scenario": description, "impact_pct": np.nan, "error": str(e)}


def calculate_portfolio_beta(portfolio_returns, market_returns):
    print("\nCalculating Portfolio Beta...")
    try:
        if isinstance(portfolio_returns, pd.DataFrame):
            portfolio_returns = portfolio_returns.squeeze()
        if isinstance(market_returns, pd.DataFrame):
             market_returns = market_returns.squeeze()
             
        if not isinstance(portfolio_returns, pd.Series) or not isinstance(market_returns, pd.Series):
             print("Error: Inputs to calculate_portfolio_beta must be pandas Series.")
             return np.nan
             
        aligned_data = pd.DataFrame({'portfolio': portfolio_returns, 'market': market_returns}).dropna()
        
        if len(aligned_data) < 2:
            print("Warning: Not enough overlapping data points to calculate beta.")
            return np.nan
            
        covariance = aligned_data['portfolio'].cov(aligned_data['market'], ddof=1)
        market_variance = aligned_data['market'].var(ddof=1)
        if market_variance is None or market_variance == 0 or pd.isna(market_variance) or pd.isna(covariance):
            print("Warning: Could not calculate beta due to zero market variance or NaN results.")
            return np.nan
        portfolio_beta = covariance / market_variance
        print(f"Calculated Portfolio Beta: {portfolio_beta:.3f}")
        return portfolio_beta
    except Exception as e:
        print(f"Error calculating portfolio beta: {e}")
        return np.nan

def fit_garch_model(returns_series):
    print("\nFitting GARCH(1,1) model...")
    if returns_series is None or returns_series.empty or returns_series.isnull().all():
        print("Warning: Cannot fit GARCH model on empty or NaN returns.")
        return np.nan
    scaled_returns = returns_series.dropna() * 100
    
    if scaled_returns.empty or len(scaled_returns) < 5: #need enough data
         print("Warning: Not enough valid data points to fit GARCH model after scaling/dropna.")
         return np.nan
         
    if scaled_returns.var() < 1e-12:
        print("Warning: Returns variance is near zero. Cannot fit GARCH. Returning 0 volatility.")
        return 0.0

    try:
        model = arch_model(scaled_returns, mean='Constant', vol='Garch', p=1, q=1, rescale=False)
        results = model.fit(disp='off')
        forecast = results.forecast(horizon=1, reindex=False)
        cond_variance = forecast.variance.iloc[0, 0]
        cond_volatility_scaled = np.sqrt(cond_variance)
        cond_volatility = cond_volatility_scaled / 100.0
        
        print(f"GARCH Forecasted Conditional Volatility (1-day): {cond_volatility:.6f}")
        return cond_volatility

    except Exception as e:
        print(f"Error fitting GARCH model or forecasting: {e}")
        return np.nan

def calculate_monte_carlo_var_cvar(portfolio_returns_series, 
                                     forecasted_volatility, 
                                     num_simulations=10000,
                                     confidence_level=0.95, 
                                     horizon_days=5,
                                     seed=42):
    print(f"\nCalculating Monte Carlo VaR & CVaR ({horizon_days}-day, {confidence_level*100}% confidence, {num_simulations} sims)...")
    
    if portfolio_returns_series is None or portfolio_returns_series.empty or portfolio_returns_series.isnull().all():
        print("Warning: Historical returns needed for drift calculation are missing.")
        return np.nan, np.nan
    
    if forecasted_volatility is None or np.isnan(forecasted_volatility) or forecasted_volatility < 0:
        print("Warning: Forecasted volatility is missing or invalid.")
        return np.nan, np.nan
        
    np.random.seed(seed)
    
    try:
        #calculate drift
        daily_returns = portfolio_returns_series.dropna()
        if daily_returns.empty:
             print("Warning: No valid historical returns for drift calculation.")
             return np.nan, np.nan
        mean_return = daily_returns.mean()

        # simulate returns for the horizon
        random_shocks = np.random.standard_normal(size=(num_simulations, horizon_days))
        
        # simulate daily returns for each path over the horizon
        simulated_daily_returns = mean_return + forecasted_volatility * random_shocks
        
        # cap loss at -99.99% to avoid 1+r <= 0 
        simulated_daily_returns = np.maximum(simulated_daily_returns, -0.9999)

        # calculate cumulative return over the horizon for each simulation path
        cumulative_returns = np.prod(1 + simulated_daily_returns, axis=1) - 1
                
        # VaR
        sorted_sim_returns = np.sort(cumulative_returns)
        var_index = int((1.0 - confidence_level) * num_simulations)
        
        if var_index >= num_simulations:
             print("Warning: Simulation index out of bounds for VaR.")
             mc_var = sorted_sim_returns[0] if num_simulations > 0 else np.nan
        else:
             mc_var = sorted_sim_returns[var_index]
             
        # CVaR
        if var_index >= num_simulations:
             print("Warning: Simulation index out of bounds for CVaR.")
             mc_cvar = sorted_sim_returns[0] if num_simulations > 0 else np.nan
        elif var_index < 0:
             print("Error: Invalid index for CVaR tail.")
             mc_cvar = np.nan
        else:
             tail_returns = sorted_sim_returns[:var_index + 1]
             if len(tail_returns) == 0:
                 print("Warning: No returns in simulation tail for CVaR.")
                 mc_cvar = np.nan
             else:
                 mc_cvar = np.mean(tail_returns)
                 
        print(f"Monte Carlo VaR: {mc_var:.4f}")
        print(f"Monte Carlo CVaR: {mc_cvar:.4f}")
        return mc_var, mc_cvar

    except Exception as e:
        print(f"Error during Monte Carlo simulation: {e}")
        return np.nan, np.nan

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    if weights is None or mean_returns is None or cov_matrix is None:
         return np.nan, np.nan, np.nan
    
    weights = np.array(weights)
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    risk_free_rate = 0.0
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else np.nan
    return portfolio_return, portfolio_volatility, sharpe_ratio

def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    _ , _ , sharpe = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    if pd.isna(sharpe) or not np.isfinite(sharpe):
         return 1e6
    return -sharpe

def optimize_portfolio(asset_returns, objective='max_sharpe'):
    print(f"\nOptimizing portfolio for objective: {objective}...")
    if asset_returns is None or asset_returns.empty:
        print("Warning: Cannot optimize portfolio with empty asset returns.")
        return None, np.nan, np.nan, np.nan
        
    asset_returns_clean = asset_returns.dropna(axis=0, how='any')
    if len(asset_returns_clean) < 2:
        print("Warning: Not enough data after dropping NaNs for optimization.")
        return None, np.nan, np.nan, np.nan
    num_assets = asset_returns_clean.shape[1]
    if num_assets == 0:
         print("Warning: No assets remaining after cleaning returns data.")
         return None, np.nan, np.nan, np.nan

    try:
        mean_returns_daily = asset_returns_clean.mean()
        cov_matrix = calculate_covariance_matrix(asset_returns_clean)
        if cov_matrix is None:
            print("Error: Failed to calculate covariance matrix for optimization.")
            return None, np.nan, np.nan, np.nan

        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        
        initial_weights = np.array(num_assets * [1. / num_assets])

        if objective == 'max_sharpe':
            opt_func = neg_sharpe_ratio
            args = (mean_returns_daily, cov_matrix)
        elif objective == 'min_variance':
            def portfolio_variance(weights, cov_matrix):
                return calculate_portfolio_performance(weights, mean_returns_daily, cov_matrix)[1]**2
            opt_func = portfolio_variance
            args = (cov_matrix,)
        else:
            print(f"Error: Unknown optimization objective '{objective}'")
            return None, np.nan, np.nan, np.nan

        from scipy.optimize import minimize # local import
        optimization_result = minimize(
            fun=opt_func, 
            x0=initial_weights, 
            args=args, 
            method='SLSQP',
            bounds=bounds, 
            constraints=constraints
        )

        if not optimization_result.success:
            print(f"Warning: Portfolio optimization failed. Message: {optimization_result.message}")
            return None, np.nan, np.nan, np.nan
        
        optimal_weights_array = optimization_result.x
        optimal_weights_array[np.abs(optimal_weights_array) < 1e-6] = 0
        optimal_weights_array /= np.sum(optimal_weights_array) 

        tickers = asset_returns_clean.columns
        optimal_weights_dict = dict(zip(tickers, optimal_weights_array))

        opt_return, opt_volatility, opt_sharpe = calculate_portfolio_performance(
            optimal_weights_array, mean_returns_daily, cov_matrix
        )

        print("Portfolio optimization successful.")
        print("Optimal Weights:")
        for ticker, weight in optimal_weights_dict.items():
            if weight > 1e-4:
                 print(f"  {ticker}: {weight*100:.2f}%")
        print(f"Expected Annual Return: {opt_return*100:.2f}%\nExpected Annual Volatility: {opt_volatility*100:.2f}%\nExpected Sharpe Ratio: {opt_sharpe:.3f}")
        
        return optimal_weights_dict, opt_return, opt_volatility, opt_sharpe

    except Exception as e:
        print(f"Error during portfolio optimization: {e}")
        import traceback
        traceback.print_exc()
        return None, np.nan, np.nan, np.nan
