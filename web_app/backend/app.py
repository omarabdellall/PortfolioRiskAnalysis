from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import sys
import os
import tempfile
import base64
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, 'data')
ALLOWED_EXAMPLES = [
    'portfolio_tech_giants_usd.csv',
    'portfolio_large_cap_diversified_usd.csv',
    'portfolio_global_mix.csv',
    'portfolio_low_beta_usd.csv',
    'portfolio_emerging_markets.csv',
    'portfolio_healthcare_focus_usd.csv',
    'portfolio_energy_focus_usd.csv',
    'portfolio_esg_focus_mix.csv',
    'portfolio_small_cap_focus_usd.csv',
    'portfolio_high_dividend_usd.csv',
    'portfolio_nvda_only.csv'
]
sys.path.insert(0, BACKEND_DIR)

risk_engine_available = False
try:
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
        generate_mitigation_suggestions
    )
    from risk_engine.config import HISTORICAL_STRESS_SCENARIOS, PROSPECTIVE_SCENARIOS
    risk_engine_available = True
except ImportError as e:
    print(f"Error importing risk_engine modules: {e}", file=sys.stderr)

app = Flask(__name__)
CORS(app)

# helper functions
def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(i) for i in data]
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, (np.float64, np.float16, np.float32)):
        if np.isnan(data):
            return None
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return convert_numpy_types(data.tolist())
    elif pd.isna(data):
         return None
    return data

# endpoints
@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    if not risk_engine_available:
         return jsonify({"error": "Risk engine modules not available."}), 500

    temp_file_path = None
    portfolio_file_path = None
    is_temp_file = False

    try:
        # get parameters
        try:
            base_currency = request.form.get('base_currency', 'USD').upper()
            hist_years = int(request.form.get('hist_years', 25))
            var_confidence = float(request.form.get('var_confidence', 0.95))
            var_horizon = int(request.form.get('var_horizon', 5))
            market_index = request.form.get('market_index', '^GSPC')
            optimize_objective = request.form.get('optimize_objective', 'max_sharpe')
            example_filename = request.form.get('example_filename')

            if optimize_objective not in ['max_sharpe', 'min_variance']:
                 optimize_objective = 'max_sharpe'
        except ValueError as e:
             return jsonify({"error": f"Invalid parameter format: {e}"}), 400

        if example_filename:
            if example_filename not in ALLOWED_EXAMPLES:
                return jsonify({"error": "Invalid or disallowed example file requested."}), 400
            portfolio_file_path = os.path.join(DATA_DIR, example_filename)
            if not os.path.exists(portfolio_file_path):
                 return jsonify({"error": f"Example file '{example_filename}' not found on server."}), 404
            print(f"Using example portfolio: {portfolio_file_path}")
        elif 'portfolio_file' in request.files:
            file = request.files['portfolio_file']
            if file.filename == '':
                return jsonify({"error": "No selected file for upload."}), 400
            _, temp_file_path = tempfile.mkstemp(suffix='.csv')
            file.save(temp_file_path)
            portfolio_file_path = temp_file_path
            is_temp_file = True
            print(f"Using uploaded portfolio: {portfolio_file_path}")
        else:
             return jsonify({"error": "No portfolio file uploaded and no example file specified."}), 400

        portfolio_df = load_portfolio(portfolio_file_path)
        if portfolio_df is None or portfolio_df.empty:
            return jsonify({"error": "Failed to load or invalid portfolio file format."}), 400

        # data fetching
        tickers = portfolio_df['Ticker'].tolist()
        end_date = datetime.today()
        start_date = end_date - timedelta(days=hist_years * 365.25)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        historical_prices, beta_data, sector_data, currency_data = fetch_stock_data(
            tickers=tickers, start_date=start_date_str, end_date=end_date_str
        )
        if historical_prices is None or historical_prices.empty:
            return jsonify({"error": "Failed to fetch historical price data. Cannot proceed."}), 500
        if beta_data is None: beta_data = {}
        if sector_data is None: sector_data = {}
        if currency_data is None: currency_data = {t: 'USD' for t in tickers}

        required_fx_pairs = set()
        fx_tickers = {}
        for ticker, stock_currency in currency_data.items():
            stock_currency = stock_currency.upper()
            if stock_currency != base_currency:
                pair_ticker = f"{stock_currency}{base_currency}=X"
                required_fx_pairs.add(pair_ticker)
                fx_tickers[stock_currency] = pair_ticker

        fx_rates = pd.DataFrame()
        if required_fx_pairs:
            fx_rates = fetch_fx_rates(
                currency_pairs=list(required_fx_pairs), start_date=start_date_str, end_date=end_date_str
            )
            if fx_rates is None: fx_rates = pd.DataFrame()

        market_data = fetch_market_index_data(
            index_ticker=market_index, start_date=start_date_str, end_date=end_date_str
        )
        if market_data is None:
             print(f"Warning: Failed to fetch market index data for {market_index}", file=sys.stderr)

        # calculate returns & basic metrics
        portfolio_analysis, asset_returns_base_ccy = calculate_portfolio_returns(
            portfolio_df, historical_prices, currency_data, fx_rates, fx_tickers, base_currency
        )
        if portfolio_analysis is None or 'Daily Return' not in portfolio_analysis.columns:
            return jsonify({"error": "Failed to calculate portfolio returns. Cannot proceed."}), 500

        daily_returns = portfolio_analysis['Daily Return'].dropna()
        latest_prices = historical_prices.iloc[-1] if not historical_prices.empty else pd.Series(dtype=float)
        last_portfolio_value = portfolio_analysis['Total Value'].iloc[-1] if 'Total Value' in portfolio_analysis.columns and not portfolio_analysis.empty else np.nan

        # garch
        garch_forecasted_vol = np.nan
        if not daily_returns.empty:
             try:
                 garch_forecasted_vol = fit_garch_model(daily_returns)
             except Exception as e:
                  print(f"[ERROR] GARCH model fitting failed: {e}", file=sys.stderr)
                  garch_forecasted_vol = np.nan

        # optimization
        optimal_weights = None
        optimal_portfolio_metrics = {"return": np.nan, "volatility": np.nan, "sharpe": np.nan}
        if asset_returns_base_ccy is not None and not asset_returns_base_ccy.empty:
             try:
                 opt_weights_result, opt_ret, opt_vol, opt_sharpe = optimize_portfolio(
                     asset_returns=asset_returns_base_ccy,
                     objective=optimize_objective
                 )
                 if opt_weights_result is not None:
                     optimal_weights = opt_weights_result
                     optimal_portfolio_metrics = {"return": opt_ret, "volatility": opt_vol, "sharpe": opt_sharpe}
             except Exception as e:
                  print(f"[ERROR] Optimization failed: {e}", file=sys.stderr)

        # beta
        portfolio_beta = np.nan
        if market_data is not None and not market_data.empty and not daily_returns.empty:
            try:
                market_prices_series = market_data[market_index]
                if isinstance(market_prices_series, pd.DataFrame):
                     market_prices_series = market_prices_series.squeeze()
                if isinstance(market_prices_series, pd.Series):
                    market_returns = market_prices_series.pct_change().dropna()
                    common_index = daily_returns.index.intersection(market_returns.index)
                    if not common_index.empty:
                         portfolio_beta = calculate_portfolio_beta(daily_returns[common_index], market_returns[common_index])
                    else:
                         print("Warning: No overlapping dates between portfolio and market returns for Beta calculation.", file=sys.stderr)
                else:
                    print(f"[ERROR] Market data for index {market_index} is not a Series, cannot calculate returns.", file=sys.stderr)
            except KeyError:
                 print(f"[ERROR] Market index ticker '{market_index}' not found in fetched market data columns.", file=sys.stderr)
            except Exception as e:
                 print(f"[ERROR] Beta calculation failed: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)

        # var / cvar (historical)
        var_hist_pct = np.nan
        cvar_hist_pct = np.nan
        var_hist_amount = np.nan
        cvar_hist_amount = np.nan
        if not daily_returns.empty:
            try:
                var_hist_pct = calculate_historical_var(daily_returns, var_confidence, var_horizon)
                cvar_hist_pct = calculate_historical_cvar(daily_returns, var_confidence, var_horizon)
                if not np.isnan(var_hist_pct) and not np.isnan(last_portfolio_value):
                    var_hist_amount = var_hist_pct * last_portfolio_value
                if not np.isnan(cvar_hist_pct) and not np.isnan(last_portfolio_value):
                    cvar_hist_amount = cvar_hist_pct * last_portfolio_value
            except Exception as e:
                 print(f"[ERROR] Historical VaR/CVaR calculation failed: {e}", file=sys.stderr)

        # var / cvar (monte carlo)
        var_mc_pct = np.nan
        cvar_mc_pct = np.nan
        var_mc_amount = np.nan
        cvar_mc_amount = np.nan
        if not daily_returns.empty and not np.isnan(garch_forecasted_vol):
             try:
                 var_mc_pct, cvar_mc_pct = calculate_monte_carlo_var_cvar(
                     portfolio_returns_series=daily_returns,
                     forecasted_volatility=garch_forecasted_vol,
                     confidence_level=var_confidence,
                     horizon_days=var_horizon,
                     num_simulations=10000
                 )
                 if not np.isnan(var_mc_pct) and not np.isnan(last_portfolio_value):
                     var_mc_amount = var_mc_pct * last_portfolio_value
                 if not np.isnan(cvar_mc_pct) and not np.isnan(last_portfolio_value):
                     cvar_mc_amount = cvar_mc_pct * last_portfolio_value
             except Exception as e:
                  print(f"[ERROR] Monte Carlo VaR/CVaR calculation failed: {e}", file=sys.stderr)

        # run stress tests & scenario analysis individually
        stress_test_results = {}
        print("\n--- Running Historical Stress Tests ---")
        for scenario_key, scenario_config in HISTORICAL_STRESS_SCENARIOS.items():
            print(f"Running Stress Test: {scenario_key}...")
            try:
                result = run_stress_test(
                    portfolio_df=portfolio_df,
                    historical_data=historical_prices,
                    scenario_key=scenario_key
                )
                stress_test_results[scenario_key] = result
            except Exception as e:
                print(f"[ERROR] Stress test '{scenario_key}' failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                stress_test_results[scenario_key] = {"scenario": scenario_key, "impact_pct": np.nan, "error": str(e)}
        print(f"Completed Stress Tests. Results keys: {list(stress_test_results.keys())}")

        scenario_analysis_results = {}
        print("\n--- Running Prospective Scenario Analysis ---")
        for scenario_key, scenario_config in PROSPECTIVE_SCENARIOS.items():
             print(f"Running Scenario: {scenario_key}...")
             try:
                 result = run_scenario_analysis(
                     portfolio_df=portfolio_df,
                     latest_prices=latest_prices,
                     beta_data=beta_data,
                     sector_data=sector_data,
                     scenario_key=scenario_key
                 )
                 scenario_analysis_results[scenario_key] = result
             except Exception as e:
                 print(f"[ERROR] Scenario analysis '{scenario_key}' failed: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
                 scenario_analysis_results[scenario_key] = {"scenario": scenario_key, "impact_pct": np.nan, "error": str(e)}
        print(f"Completed Scenario Analysis. Results keys: {list(scenario_analysis_results.keys())}")


        # risk level calculation
        risk_level = "Unavailable"
        risk_rationale = "Could not be determined."
        var_for_risk_level = var_hist_pct if not np.isnan(var_hist_pct) else np.nan

        if not np.isnan(var_for_risk_level):
            try:
                print(f"Calculating risk level with VaR={var_for_risk_level}, stress keys={list(stress_test_results.keys())}, scenario keys={list(scenario_analysis_results.keys())}")
                risk_level, risk_rationale = calculate_risk_level(
                    var_pct_result=var_for_risk_level,
                    stress_test_results=stress_test_results,
                    scenario_results=scenario_analysis_results
                )
            except TypeError as te:
                print(f"[ERROR] Risk Level TypeError (check arguments): {te}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                risk_level = "Error"
                risk_rationale = f"Calculation failed due to argument mismatch: {te}"
            except Exception as e:
                 print(f"[ERROR] Risk level calculation failed: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
                 risk_level = "Error"
                 risk_rationale = f"Calculation failed: {e}"
        else:
             risk_rationale = "Risk level could not be determined because VaR calculation failed."
        print(f"Risk Level Result: {risk_level}")


        # heatmap generation
        heatmap_base64 = None
        try:
            fig = generate_heatmap(risk_level, save_path=None)
            if fig:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                heatmap_base64 = base64.b64encode(buf.read()).decode('utf-8')
            else:
                 print("Warning: generate_heatmap did not return a figure object.", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Heatmap generation failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        # mitigation suggestions
        mitigation_suggestions = []
        if risk_level in ['Moderate', 'High']:
            try:
                if latest_prices is not None and not latest_prices.empty:
                    print(f"Calling mitigation with risk={risk_level}, beta_data keys={list(beta_data.keys())}, latest_prices available")
                    mitigation_suggestions = generate_mitigation_suggestions(
                        portfolio_df=portfolio_df,
                        latest_prices=latest_prices,
                        beta_data=beta_data,
                        risk_level=risk_level,
                        risk_rationale=risk_rationale
                    )
                else:
                    print("Warning: Skipping mitigation suggestions due to missing latest price data.")
                    mitigation_suggestions.append("Suggestions unavailable due to missing price data.")
            except TypeError as te:
                print(f"[ERROR] Mitigation suggestion TypeError (check arguments): {te}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                mitigation_suggestions.append(f"Suggestion generation failed: {te}")
            except Exception as e:
                 print(f"[ERROR] Mitigation suggestion generation failed: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
                 mitigation_suggestions.append(f"Suggestion generation failed: {e}")


        # structure results
        results = {
            "summary": {
                "riskLevel": risk_level,
                "riskRationale": risk_rationale,
                "portfolioBeta": portfolio_beta,
                "lastPortfolioValue": last_portfolio_value,
                "baseCurrency": base_currency,
                "marketIndex": market_index,
                "optimizationObjective": optimize_objective
            },
            "varCvar": {
                "historical": {"varPct": var_hist_pct, "varAmount": var_hist_amount, "cvarPct": cvar_hist_pct, "cvarAmount": cvar_hist_amount},
                "monteCarlo": {"varPct": var_mc_pct, "varAmount": var_mc_amount, "cvarPct": cvar_mc_pct, "cvarAmount": cvar_mc_amount, "garchForecastVol": garch_forecasted_vol},
                "parameters": {"confidence": var_confidence, "horizonDays": var_horizon}
            },
            "stressTests": stress_test_results,
            "scenarioAnalysis": scenario_analysis_results,
            "optimization": {
                "objective": optimize_objective,
                "optimalWeights": optimal_weights,
                "metrics": optimal_portfolio_metrics
            },
            "mitigationSuggestions": mitigation_suggestions,
            "visualizations": {"heatmapBase64": heatmap_base64}
        }

        results_serializable = convert_numpy_types(results)
        print("Analysis completed successfully. Returning JSON.")
        return jsonify(results_serializable)

    except Exception as e:
        error_timestamp = datetime.now().isoformat()
        print(f"[{error_timestamp}] Unhandled exception in /api/analyze: {type(e).__name__} - {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": f"An unexpected internal server error occurred ({type(e).__name__}). Please check server logs."}), 500
    finally:
        # clean up temp file
        if is_temp_file and temp_file_path and os.path.exists(temp_file_path):
             try:
                 os.remove(temp_file_path)
                 print(f"Cleaned up temporary file: {temp_file_path}")
             except Exception as e_clean:
                  print(f"Error cleaning up temp file {temp_file_path}: {e_clean}", file=sys.stderr)

# endpoint to list available example portfolios
@app.route('/api/example-portfolios', methods=['GET'])
def list_example_portfolios():
    # for simplicity and security, return the allowed list
    return jsonify({"examples": ALLOWED_EXAMPLES})

# endpoint for example portfolio content
@app.route('/api/example-portfolio', methods=['GET'])
def get_example_portfolio():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"error": "Missing filename parameter."}), 400

    print(f"[DEBUG] Received request for example filename: '{filename}'", file=sys.stderr)
    print(f"[DEBUG] Comparing against ALLOWED_EXAMPLES: {ALLOWED_EXAMPLES}", file=sys.stderr)
    
    if filename not in ALLOWED_EXAMPLES:
        print(f"[ERROR] Filename '{filename}' not found in ALLOWED_EXAMPLES.", file=sys.stderr)
        return jsonify({"error": "Invalid or disallowed example file requested."}), 400

    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": f"Example file '{filename}' not found on server."}), 404

    try:
        df = pd.read_csv(file_path)
        content_json = df.replace({np.nan: None}).to_dict(orient='records')
        return jsonify(content_json)
    except pd.errors.ParserError:
         return jsonify({"error": f"Failed to parse CSV file '{filename}'."}), 500
    except Exception as e:
        print(f"Error reading example file {filename}: {e}", file=sys.stderr)
        return jsonify({"error": f"An internal error occurred while reading the example file."}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Backend is running", "risk_engine_available": risk_engine_available})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
