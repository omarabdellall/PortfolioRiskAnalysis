# use this script to run a risk analysis through the command line instead of website!

import argparse
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(SCRIPT_DIR, 'web_app', 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from risk_engine import (
    load_portfolio, 
    fetch_stock_data,
    fetch_fx_rates,
    fetch_market_index_data,
    calculate_portfolio_returns,
    optimize_portfolio,
    calculate_historical_var, 
    calculate_historical_cvar,
    calculate_portfolio_beta,
    fit_garch_model,
    calculate_monte_carlo_var_cvar,
    run_stress_test,         
    run_scenario_analysis,  
    calculate_risk_level,    
    generate_heatmap,        
    generate_summary_report, 
)
from risk_engine.config import HISTORICAL_STRESS_SCENARIOS, PROSPECTIVE_SCENARIOS 

def main():
    parser = argparse.ArgumentParser(description="Run Portfolio Risk Analysis")
    parser.add_argument("--portfolio", required=True, help="Path to the portfolio CSV file.")
    parser.add_argument("--output_report", required=True, help="Path to save the analysis summary report.")
    parser.add_argument("--output_heatmap", default="risk_heatmap.png", help="Path to save the risk heatmap image.")
    parser.add_argument("--hist_years", type=int, default=25, help="Number of years for historical data.")
    parser.add_argument("--var_confidence", type=float, default=0.95, help="Confidence level for VaR (e.g., 0.95).")
    parser.add_argument("--var_horizon", type=int, default=5, help="Time horizon in days for VaR (e.g., 5).")
    parser.add_argument("--base_currency", type=str, default="USD", help="Base currency for reporting (e.g., USD, EUR).")
    parser.add_argument("--market_index", type=str, default="^GSPC", help="Ticker for market index (for beta calc).")
    parser.add_argument("--optimize_objective", type=str, default="max_sharpe", 
                        choices=['max_sharpe', 'min_variance'], 
                        help="Portfolio optimization objective ('max_sharpe' or 'min_variance').")
    args = parser.parse_args()
    print(f"Starting analysis for portfolio: {args.portfolio}")
    print(f"Reporting Base Currency: {args.base_currency}")
    
    # load portfolio
    portfolio_df = load_portfolio(args.portfolio)
    if portfolio_df is None:
        print("Exiting due to portfolio loading error.")
        sys.exit(1)
    print("\nLoaded Portfolio:")
    print(portfolio_df)
    print("-"*20)

    # fetch historical data
    tickers = portfolio_df['Ticker'].tolist()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=args.hist_years * 365.25) # use 365.25 for better accuracy
    
    print(f"\nFetching {args.hist_years} years of historical data and info ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})...")
    # now fetches currency data as well
    historical_prices, beta_data, sector_data, currency_data = fetch_stock_data( 
        tickers=tickers, 
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    # check if fetching was successful (at least prices)
    if historical_prices is None or historical_prices.empty:
        print("Exiting due to historical price data fetching error.")
        sys.exit(1)
    
    # warnings if beta/sector data is partially missing but proceed
    if not beta_data or not sector_data:
         print("\nWarning: Could not retrieve complete beta/sector information. Prospective scenarios might be affected.")
         # initialize empty dicts if fetch failed entirely to avoid downstream errors
         if beta_data is None: beta_data = {} 
         if sector_data is None: sector_data = {}
         # currency data defaults to USD in fetcher if info fails, so should exist
         if currency_data is None: currency_data = {t: 'USD' for t in tickers} # fallback just in case

    print("\nFetched Historical Prices (showing first 5 rows):")
    print(historical_prices.head())

    # fetch fx rates
    base_currency = args.base_currency.upper()
    required_fx_pairs = set()
    fx_tickers = {}
    if currency_data:
        for ticker, stock_currency in currency_data.items():
            stock_currency = stock_currency.upper()
            if stock_currency != base_currency:
                # construct ticker like EURUSD=X or CADUSD=X
                pair_ticker = f"{stock_currency}{base_currency}=X"
                required_fx_pairs.add(pair_ticker)
                fx_tickers[stock_currency] = pair_ticker # map stock currency to its FX ticker vs base

    fx_rates = None
    if required_fx_pairs:
        fx_rates = fetch_fx_rates(
            currency_pairs=list(required_fx_pairs), 
            start_date=start_date.strftime('%Y-%m-%d'), 
            end_date=end_date.strftime('%Y-%m-%d')
        )
        if fx_rates is None:
            print("Warning: Failed to fetch some/all FX rates. Proceeding without FX adjustments.")
            # ensure fx_rates is an empty DataFrame if fetch failed, to avoid errors downstream
            fx_rates = pd.DataFrame()
    else:
        print("\nNo FX rate fetching required (all holdings in base currency or no currency info).")
        fx_rates = pd.DataFrame() # create empty DataFrame if no pairs needed

    # fetch market index data
    market_index_ticker = args.market_index
    market_data = fetch_market_index_data(
        index_ticker=market_index_ticker, 
        start_date=start_date.strftime('%Y-%m-%d'), 
        end_date=end_date.strftime('%Y-%m-%d')
    )

    # calculate portfolio value & returns; pass FX data to calculator
    portfolio_analysis, asset_returns_base_ccy = calculate_portfolio_returns(
        portfolio_df,
        historical_prices,
        currency_data=currency_data,
        fx_rates=fx_rates,
        fx_tickers=fx_tickers,
        base_currency=base_currency
    )
    
    if portfolio_analysis is None:
        print("Exiting due to portfolio calculation error.")
        sys.exit(1)
        
    print("\nCalculated Portfolio Analysis (showing last 5 rows):")
    #print(portfolio_analysis.head())
    print(portfolio_analysis.tail())
    print("-"*20)
    
    # fit garch model
    garch_forecasted_vol = np.nan # initialize
    if 'Daily Return' in portfolio_analysis.columns:
        garch_forecasted_vol = fit_garch_model(portfolio_analysis['Daily Return'])
    else:
        print("\nSkipping GARCH model fitting as daily returns are not available.")
    # initialize optimization variables
    optimal_weights = None 
    optimal_portfolio_metrics = {"return": np.nan, "volatility": np.nan, "sharpe": np.nan}

    # run portfolio optimization
    if asset_returns_base_ccy is not None:
        # call optimization function, passing the objective from args
        opt_weights_result, opt_ret, opt_vol, opt_sharpe = optimize_portfolio(
            asset_returns=asset_returns_base_ccy, 
            objective=args.optimize_objective
        )
        # store results only if optimization was successful
        if opt_weights_result is not None:
             optimal_weights = opt_weights_result # assign the successful result
             optimal_portfolio_metrics = {"return": opt_ret, "volatility": opt_vol, "sharpe": opt_sharpe}
    else:
        print("\nSkipping portfolio optimization due to missing asset return data.")
    print("-"*20)

    # calculate portfolio beta
    portfolio_beta = np.nan # initialize
    if market_data is not None and portfolio_analysis is not None and 'Daily Return' in portfolio_analysis.columns:
        market_returns = market_data[market_index_ticker].pct_change().dropna()
        portfolio_returns_for_beta = portfolio_analysis['Daily Return'].dropna()
        portfolio_beta = calculate_portfolio_beta(portfolio_returns_for_beta, market_returns)
    else:
        print("\nSkipping Portfolio Beta calculation due to missing market or portfolio return data.")

    # calculate value at risk
    print(f"\nCalculating VaR ({args.var_horizon}-day, {args.var_confidence*100:.0f}% confidence)...")
    daily_returns = portfolio_analysis['Daily Return'].dropna()

    var_result_pct = calculate_historical_var(
        portfolio_returns_series=daily_returns,
        confidence_level=args.var_confidence,
        horizon_days=args.var_horizon
    )

    var_amount = np.nan # initialize
    last_portfolio_value = np.nan 
    cvar_result_pct = np.nan 
    cvar_amount = np.nan 
    mc_var_pct = np.nan 
    mc_cvar_pct = np.nan 
    mc_var_amount = np.nan 
    mc_cvar_amount = np.nan 
    
    if var_result_pct is None or np.isnan(var_result_pct):
        print("Error calculating VaR. Skipping further analysis dependent on VaR.")
    else:
        # get the most recent portfolio value for context
        if 'Total Value' in portfolio_analysis.columns and not portfolio_analysis.empty:
            last_portfolio_value = portfolio_analysis['Total Value'].iloc[-1]
            var_amount = var_result_pct * last_portfolio_value 
            print(f"Calculated {args.var_horizon}-Day VaR ({args.var_confidence*100:.0f}% Confidence):")
            print(f"  Return: {var_result_pct:.4f} ({var_result_pct*100:.2f}%)")
            print(f"  Amount: ${var_amount:,.2f} (potential loss on current value of ${last_portfolio_value:,.2f})")

            # calculate conditional value at risk
            print(f"\nCalculating CVaR ({args.var_horizon}-day, {args.var_confidence*100:.0f}% confidence)...")
            cvar_result_pct = calculate_historical_cvar(
                portfolio_returns_series=daily_returns,
                confidence_level=args.var_confidence,
                horizon_days=args.var_horizon
            )

            if cvar_result_pct is None or np.isnan(cvar_result_pct):
                print("Error calculating CVaR.")
            else:
                cvar_amount = cvar_result_pct * last_portfolio_value 
                print(f"Calculated {args.var_horizon}-Day CVaR ({args.var_confidence*100:.0f}% Confidence):")
                print(f"  Return: {cvar_result_pct:.4f} ({cvar_result_pct*100:.2f}%)")
                print(f"  Amount: ${cvar_amount:,.2f} (expected loss beyond VaR on current value of ${last_portfolio_value:,.2f})")

            # calculate monte carlo VaR/CVaR (using GARCH vol if available)
            if not np.isnan(garch_forecasted_vol):
                mc_var_pct, mc_cvar_pct = calculate_monte_carlo_var_cvar(
                    portfolio_returns_series=daily_returns, 
                    forecasted_volatility=garch_forecasted_vol,
                    confidence_level=args.var_confidence,
                    horizon_days=args.var_horizon,
                    num_simulations=10000 # Can make this an argument later
                )
                
                if mc_var_pct is not None and not np.isnan(mc_var_pct):
                    mc_var_amount = mc_var_pct * last_portfolio_value
                    print(f"Calculated {args.var_horizon}-Day Monte Carlo VaR ({args.var_confidence*100:.0f}% Confidence):")
                    print(f"  Return: {mc_var_pct:.4f} ({mc_var_pct*100:.2f}%)")
                    print(f"  Amount: ${mc_var_amount:,.2f}")
                else:
                    print("Monte Carlo VaR calculation failed.")

                if mc_cvar_pct is not None and not np.isnan(mc_cvar_pct):
                    mc_cvar_amount = mc_cvar_pct * last_portfolio_value
                    print(f"Calculated {args.var_horizon}-Day Monte Carlo CVaR ({args.var_confidence*100:.0f}% Confidence):")
                    print(f"  Return: {mc_cvar_pct:.4f} ({mc_cvar_pct*100:.2f}%)")
                    print(f"  Amount: ${mc_cvar_amount:,.2f}")
                else:
                    print("Monte Carlo CVaR calculation failed.")
            else:
                print("\nSkipping Monte Carlo VaR/CVaR calculation as GARCH volatility is unavailable.")

        else:
            print("Warning: Cannot calculate VaR/CVaR amount as portfolio value is unavailable.")

    # run historical stress tests
    print("\nRunning Historical Stress Tests...")
    stress_test_results = []
    for scenario_key in HISTORICAL_STRESS_SCENARIOS.keys(): 
        result = run_stress_test(portfolio_df, historical_prices, scenario_key) 
        if result:
            stress_test_results.append(result)
            print(f"  Scenario: {result.get('scenario', 'N/A')}")
            impact = result.get('impact_pct', np.nan)
            if not np.isnan(impact):
                print(f"    Impact: {impact*100:.2f}% change in value ({result.get('start', 'N/A')} to {result.get('end', 'N/A')})")
            else:
                print(f"    Impact: Calculation failed ({result.get('error', 'Unknown error')})")
        else:
            print(f"  Scenario: {scenario_key} - Failed to execute.")
            stress_test_results.append({"scenario": scenario_key, "impact_pct": np.nan, "error": "Execution failed"})
    print("-"*20)

    # run prospective scenario analysis
    print("\nRunning Prospective Scenario Analysis...")
    prospective_scenario_results = []
    # need latest prices for prospective scenarios
    latest_prices = None
    if historical_prices is not None and not historical_prices.empty:
        latest_prices = historical_prices.iloc[-1]
        
    if latest_prices is None or latest_prices.isnull().all():
        print("Warning: Could not determine latest prices. Skipping prospective scenario analysis.")
    else:
        for scenario_key in PROSPECTIVE_SCENARIOS.keys(): 
            result = run_scenario_analysis(
                portfolio_df=portfolio_df, 
                latest_prices=latest_prices, 
                beta_data=beta_data, 
                sector_data=sector_data, 
                scenario_key=scenario_key
            )
            if result:
                prospective_scenario_results.append(result)
                print(f"  Scenario: {result.get('scenario', 'N/A')}")
                impact = result.get('impact_pct', np.nan)
                warning = result.get('warning', None)
                if not np.isnan(impact):
                    print(f"    Impact: {impact*100:.2f}% change in value")
                    if warning: print(f"    Warning: {warning}")
                else:
                    print(f"    Impact: Calculation failed ({result.get('error', 'Unknown error')})")
            else:
                 print(f"  Scenario: {scenario_key} - Failed to execute.")
                 prospective_scenario_results.append({"scenario": scenario_key, "impact_pct": np.nan, "error": "Execution failed"})
    print("-" * 20)

    # calculate overall risk level
    print("\nCalculating Overall Risk Level...")
    # use the reporter's function for consistency (it was imported)
    # pass scenario results as well now
    risk_level, risk_rationale = calculate_risk_level(
        var_result_pct,
        stress_test_results,
        prospective_scenario_results 
    )
    print(f"Determined Risk Level: {risk_level}")
    print("-" * 20)
    
    # generate risk heatmap
    try:
        print(f"\nGenerating Risk Heatmap to {args.output_heatmap}...")
        generate_heatmap(risk_level, save_path=args.output_heatmap)
        print("Heatmap generated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to generate heatmap: {e}")

    # generate summary report
    print(f"\nGenerating Summary Report to {args.output_report}...")
    analysis_results = {
        "portfolio_file": args.portfolio,
        "base_currency": base_currency,
        "analysis_date": end_date.strftime('%Y-%m-%d %H:%M:%S'),
        "historical_years": args.hist_years,
        "var_confidence": args.var_confidence,
        "var_horizon": args.var_horizon,
        "last_portfolio_value": last_portfolio_value if not np.isnan(last_portfolio_value) else None,
        "portfolio_beta": portfolio_beta if not np.isnan(portfolio_beta) else None,
        "market_index": market_index_ticker,
        "var_pct": var_result_pct if not np.isnan(var_result_pct) else None,
        "var_amount": var_amount if not np.isnan(var_amount) else None,
        "cvar_pct": cvar_result_pct if not np.isnan(cvar_result_pct) else None, 
        "cvar_amount": cvar_amount if not np.isnan(cvar_amount) else None,  
        "garch_volatility": garch_forecasted_vol if not np.isnan(garch_forecasted_vol) else None, 
        "mc_var_pct": mc_var_pct if not np.isnan(mc_var_pct) else None,      
        "mc_var_amount": mc_var_amount if not np.isnan(mc_var_amount) else None, 
        "mc_cvar_pct": mc_cvar_pct if not np.isnan(mc_cvar_pct) else None,    
        "mc_cvar_amount": mc_cvar_amount if not np.isnan(mc_cvar_amount) else None, 
        "optimal_weights": optimal_weights, 
        "optimal_portfolio_metrics": optimal_portfolio_metrics, 
        "optimize_objective": args.optimize_objective, 
        "stress_tests": stress_test_results,
        "scenario_analysis": prospective_scenario_results,
        "risk_level": risk_level, 
        "portfolio_df": portfolio_df, 
        "latest_prices": latest_prices, 
        "beta_data": beta_data 
    }

    # generate and save report
    generate_summary_report(analysis_results, args.output_report)
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main() 