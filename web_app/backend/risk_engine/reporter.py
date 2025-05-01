#functions for generating reports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .config import RISK_THRESHOLDS
from .mitigation import generate_mitigation_suggestions 

def calculate_risk_level(var_pct_result, stress_test_results, scenario_results):
    risk_level = 'Low' # default
    rationale = ["Overall risk assessed as Low."]
    reasons_moderate = []
    reasons_high = []

    # evaluate VaR
    if not np.isnan(var_pct_result):
        if var_pct_result <= RISK_THRESHOLDS['var_pct']['red']:
            risk_level = 'High'
            reasons_high.append(f"VaR ({var_pct_result*100:.2f}%) exceeds Red threshold ({RISK_THRESHOLDS['var_pct']['red']*100:.1f}%)")
        elif var_pct_result <= RISK_THRESHOLDS['var_pct']['yellow']:
            if risk_level == 'Low': risk_level = 'Moderate'
            reasons_moderate.append(f"VaR ({var_pct_result*100:.2f}%) exceeds Yellow threshold ({RISK_THRESHOLDS['var_pct']['yellow']*100:.1f}%)")
    else:
        reasons_moderate.append("VaR could not be calculated.")
        if risk_level == 'Low': risk_level = 'Moderate' # uncertainty increases risk

    # evaluate stress tests
    worst_stress_impact = 0.0
    worst_scenario_name = "N/A"
    # ensure valid stress test results
    valid_stress_results = [r for r in stress_test_results if isinstance(r, dict) and 'impact_pct' in r and not np.isnan(r['impact_pct'])]
    
    if valid_stress_results:
        worst_stress_impact = min(r['impact_pct'] for r in valid_stress_results)
        worst_scenario = min(valid_stress_results, key=lambda x: x['impact_pct'])
        worst_scenario_name = worst_scenario.get('scenario', 'Unknown Scenario')
        
        if worst_stress_impact <= RISK_THRESHOLDS['stress_pct']['red']:
            risk_level = 'High' # stress test alone can trigger high
            reasons_high.append(f"Worst stress test ({worst_scenario_name}: {worst_stress_impact*100:.2f}%) exceeds Red threshold ({RISK_THRESHOLDS['stress_pct']['red']*100:.1f}%)")
        elif worst_stress_impact <= RISK_THRESHOLDS['stress_pct']['yellow']:
             if risk_level == 'Low': risk_level = 'Moderate'
             reasons_moderate.append(f"Worst stress test ({worst_scenario_name}: {worst_stress_impact*100:.2f}%) exceeds Yellow threshold ({RISK_THRESHOLDS['stress_pct']['yellow']*100:.1f}%)")
    else:
        reasons_moderate.append("No valid stress test results available.")
        if risk_level == 'Low': risk_level = 'Moderate'

    # evaluate prospective scenario
    index_drop_impact = np.nan
    index_drop_scenario_name = "-10% Major Index Drop (Modeled)"
    if scenario_results:
        for r in scenario_results:
            if isinstance(r, dict) and r.get('scenario', '').lower() == index_drop_scenario_name.lower() and 'impact_pct' in r and pd.notna(r['impact_pct']): 
                index_drop_impact = r['impact_pct']
                break
                
    if not np.isnan(index_drop_impact):
        if index_drop_impact <= RISK_THRESHOLDS['scenario_index_drop_pct']['red']:
            risk_level = 'High' # this scenario alone can trigger high
            reasons_high.append(f"Prospective Index Drop ({index_drop_impact*100:.2f}%) exceeds Red threshold ({RISK_THRESHOLDS['scenario_index_drop_pct']['red']*100:.1f}%)")
        elif index_drop_impact <= RISK_THRESHOLDS['scenario_index_drop_pct']['yellow']:
             if risk_level == 'Low': risk_level = 'Moderate'
             reasons_moderate.append(f"Prospective Index Drop ({index_drop_impact*100:.2f}%) exceeds Yellow threshold ({RISK_THRESHOLDS['scenario_index_drop_pct']['yellow']*100:.1f}%)")
    else:
        if scenario_results:
             reasons_moderate.append(f"Prospective Index Drop scenario result unavailable.")
             if risk_level == 'Low': risk_level = 'Moderate'

    # combine rationales
    if risk_level == 'High':
        rationale = ["Overall risk assessed as High."] + reasons_high + reasons_moderate
    elif risk_level == 'Moderate':
         rationale = ["Overall risk assessed as Moderate."] + reasons_moderate

    return risk_level, rationale

def generate_heatmap(risk_level, save_path=None):
    print(f"\nGenerating Risk Heatmap...")
    color_map = {'Low': 'green', 'Moderate': 'yellow', 'High': 'red'}
    label_map = {'Low': 'Low Risk', 'Moderate': 'Moderate Risk', 'High': 'High Risk'}
    
    level = risk_level if risk_level in color_map else 'Moderate'
    color = color_map[level]
    label = label_map[level]

    print(f"Assessed Risk Level: {label}")

    fig, ax = plt.subplots(figsize=(3, 1.5)) 
    ax.set_facecolor(color)
    ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=14, 
            color='white' if color != 'yellow' else 'black', weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Overall Portfolio Risk Level', fontsize=12)
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        except Exception as e:
            print(f"Error saving heatmap to {save_path}: {e}")
        finally:
            plt.close(fig)
        return None
    else:
        return fig 

def generate_summary_report(analysis_results, output_file): 
    print(f"\nGenerating summary report to {output_file}...")
    risk_level = analysis_results.get('risk_level', 'Unknown')
    var_pct = analysis_results.get('var_pct')
    var_amount = analysis_results.get('var_amount')
    cvar_pct = analysis_results.get('cvar_pct') 
    cvar_amount = analysis_results.get('cvar_amount') 
    portfolio_beta = analysis_results.get('portfolio_beta')
    garch_volatility = analysis_results.get('garch_volatility') 
    mc_var_pct = analysis_results.get('mc_var_pct') 
    mc_var_amount = analysis_results.get('mc_var_amount') 
    mc_cvar_pct = analysis_results.get('mc_cvar_pct') 
    mc_cvar_amount = analysis_results.get('mc_cvar_amount') 
    optimal_weights = analysis_results.get('optimal_weights') 
    optimal_metrics = analysis_results.get('optimal_portfolio_metrics', {}) 
    stress_test_results = analysis_results.get('stress_tests', [])
    scenario_results = analysis_results.get('scenario_analysis', [])
    var_confidence = analysis_results.get('var_confidence', 0.95)
    var_horizon = analysis_results.get('var_horizon', 5)
    analysis_date = analysis_results.get('analysis_date', 'N/A')
    portfolio_file = analysis_results.get('portfolio_file', 'N/A')
    base_currency = analysis_results.get('base_currency', 'USD')
    _, risk_rationale = calculate_risk_level(var_pct, stress_test_results, scenario_results)

    with open(output_file, 'w') as f:
        f.write(f"========== Portfolio Risk Analysis Summary ==========\n")
        f.write(f"Portfolio File: {portfolio_file}\n")
        f.write(f"Analysis Date: {analysis_date}\n")
        f.write(f"Reporting Currency: {base_currency}\n\n")
        f.write(f"Overall Risk Level: {risk_level}\n")
        f.write("Rationale:\n")
        f.write(f"- {risk_rationale}\n\n") 
        f.write(f"--- Value at Risk & CVaR ({var_horizon}-Day, {var_confidence*100:.0f}% Confidence) ---\n")
        if var_pct is not None and not np.isnan(var_pct):
            f.write(f"-> VaR Return: {var_pct*100:.2f}%\n") 
            if var_amount is not None and not np.isnan(var_amount):
                 f.write(f"-> VaR Amount: ${var_amount:,.2f}\n")  
            else:
                 f.write("-> VaR Amount: Not Available\n")
        else:
            f.write("-> VaR: Calculation Failed\n") 
        if cvar_pct is not None and not np.isnan(cvar_pct):
            f.write(f"-> CVaR Return (Expected Shortfall): {cvar_pct*100:.2f}%\n") 
            if cvar_amount is not None and not np.isnan(cvar_amount):
                 f.write(f"-> CVaR Amount: ${cvar_amount:,.2f}\n")  
            else:
                 f.write("-> CVaR Amount: Not Available\n")
        else:
             f.write("-> CVaR: Calculation Failed\n")
        f.write("\n")
        f.write(f"--- Monte Carlo VaR & CVaR ({var_horizon}-Day, {var_confidence*100:.0f}% Confidence) ---\n")
        if garch_volatility is not None and not np.isnan(garch_volatility):
             f.write(f"(Based on GARCH Volatility Forecast: {garch_volatility:.6f})\n")
             if mc_var_pct is not None and not np.isnan(mc_var_pct):
                 f.write(f"-> MC VaR Return: {mc_var_pct*100:.2f}%\n")
                 if mc_var_amount is not None and not np.isnan(mc_var_amount):
                     f.write(f"-> MC VaR Amount: ${mc_var_amount:,.2f}\n")
                 else:
                     f.write("-> MC VaR Amount: Not Available\n")
             else:
                 f.write("-> MC VaR: Calculation Failed\n")
             if mc_cvar_pct is not None and not np.isnan(mc_cvar_pct):
                 f.write(f"-> MC CVaR Return: {mc_cvar_pct*100:.2f}%\n")
                 if mc_cvar_amount is not None and not np.isnan(mc_cvar_amount):
                     f.write(f"-> MC CVaR Amount: ${mc_cvar_amount:,.2f}\n")
                 else:
                     f.write("-> MC CVaR Amount: Not Available\n")
             else:
                 f.write("-> MC CVaR: Calculation Failed\n")
        else:
             f.write("Monte Carlo simulation skipped (GARCH volatility unavailable).\n")
        f.write("\n")
        f.write(f"--- Portfolio Beta (vs Market) ---\n") 
        if portfolio_beta is not None and not np.isnan(portfolio_beta):
            f.write(f"-> Beta: {portfolio_beta:.3f}\n\n")
        else:
            f.write("Portfolio Beta calculation failed or not applicable.\n\n")
        f.write("--- Historical Stress Test Results ---\n")
        if stress_test_results:
            for result in stress_test_results:
                scenario_name = result.get('scenario', 'Unknown')
                impact_pct = result.get('impact_pct', np.nan)
                impact_amt = result.get('impact_amount', np.nan)
                
                if not np.isnan(impact_pct):
                    f.write(f"-> Scenario: {scenario_name}\n")
                    f.write(f"   - Estimated Impact: {impact_pct*100:.2f}%\n")
                    if not np.isnan(impact_amt):
                         f.write(f"   - Estimated Amount: ${impact_amt:,.2f}\n")
                    else:
                         f.write("   - Estimated Amount: Not Available\n")
                else:
                    f.write(f"-> Scenario: {scenario_name} - Calculation Failed\n")
        else:
            f.write("No stress test results available.\n")
        f.write("\n")
        f.write("--- Prospective Scenario Analysis Results ---\n")
        if scenario_results:
            for result in scenario_results:
                scenario_name = result.get('scenario', 'Unknown')
                impact_pct = result.get('impact_pct', np.nan)
                impact_amt = result.get('impact_amount', np.nan)
                
                if not np.isnan(impact_pct):
                    f.write(f"-> Scenario: {scenario_name}\n")
                    f.write(f"   - Estimated Impact: {impact_pct*100:.2f}%\n")
                    if not np.isnan(impact_amt):
                         f.write(f"   - Estimated Amount: ${impact_amt:,.2f}\n")
                    else:
                         f.write("   - Estimated Amount: Not Available\n")
                else:
                    f.write(f"-> Scenario: {scenario_name} - Calculation Failed\n")
        else:
            f.write("No scenario analysis results available.\n")
        f.write("\n")
        f.write(f"--- Portfolio Optimization Suggestion ({analysis_results.get('optimization_objective', 'N/A')}) ---\n")
        if optimal_weights is not None and isinstance(optimal_weights, pd.Series) and not optimal_weights.empty:
            f.write("Optimized Weights:\n")
            sorted_weights = optimal_weights.sort_values(ascending=False)
            for ticker, weight in sorted_weights.items():
                if weight > 1e-4:
                    f.write(f"  - {ticker}: {weight*100:.2f}%\n")
            f.write("Estimated Performance of Optimized Portfolio:\n")
            f.write(f"  - Expected Annual Return: {optimal_metrics.get('return', np.nan)*100:.2f}%\n")
            f.write(f"  - Expected Annual Volatility: {optimal_metrics.get('volatility', np.nan)*100:.2f}%\n")
            opt_ret = optimal_metrics.get('return', np.nan)
            opt_vol = optimal_metrics.get('volatility', np.nan)
            sharpe = opt_ret / opt_vol if opt_vol != 0 and not np.isnan(opt_ret) and not np.isnan(opt_vol) else np.nan
            f.write(f"  - Expected Sharpe Ratio: {sharpe:.3f}\n")
        else:
            f.write("Optimization was not performed or failed.\n")
        f.write("\n")
        f.write("--- Risk Mitigation Suggestions ---\n")
        if risk_level in ['Moderate', 'High']:
            portfolio_df = analysis_results.get('portfolio_df')
            latest_prices = analysis_results.get('latest_prices')
            beta_data = analysis_results.get('beta_data')
            if portfolio_df is not None and latest_prices is not None and beta_data is not None:
                mitigation_suggestions = generate_mitigation_suggestions(
                    portfolio_df, latest_prices, beta_data, risk_level, risk_rationale
                )
                if mitigation_suggestions:
                    for suggestion in mitigation_suggestions:
                        f.write(f"- {suggestion}\n")
                else:
                    f.write("No specific suggestions generated, but review based on risk level is advised.\n")
            else:
                f.write("Could not generate specific suggestions due to missing input data (portfolio, prices, or beta).\n")
                f.write("Consider general strategies based on High/Moderate risk level (diversification, hedging, position sizing).\n")
        else:
            f.write("Risk level is Low. No specific mitigation actions suggested. Continue monitoring.\n")
        f.write("\n")
        
        f.write("========== End of Report ==========\n")

    print("Summary report generated successfully.")