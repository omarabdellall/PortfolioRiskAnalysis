#functions for generating risk mitigation suggestions
import pandas as pd
import numpy as np

def generate_mitigation_suggestions(portfolio_df, latest_prices, beta_data, risk_level, risk_rationale, top_n=3):
    suggestions = []

    if risk_level == 'Low':
        suggestions.append("Current risk profile appears acceptable based on defined thresholds. Continue regular monitoring.")
        return suggestions
        
    if risk_level == 'Moderate':
        suggestions.append("Risk Level is Moderate. Specific factors contributing to review:")
        moderate_risk_drivers = []
        
        # join rationale list into single string for searching
        rationale_text = " ".join(risk_rationale) if isinstance(risk_rationale, list) else risk_rationale if risk_rationale else ""
        rationale_lower = rationale_text.lower()
        if "var" in rationale_lower and "yellow" in rationale_lower:
            moderate_risk_drivers.append("VaR")
        if "stress test" in rationale_lower and "yellow" in rationale_lower:
            moderate_risk_drivers.append("StressTest")
        if "index drop" in rationale_lower and "yellow" in rationale_lower:
             moderate_risk_drivers.append("IndexDrop")

        # check for concentration/beta even for moderate
        try:
            holdings_dict = portfolio_df.set_index('Ticker')['Holding'].to_dict()
            required_tickers = list(portfolio_df['Ticker'])
            
            if latest_prices is not None and not latest_prices.isnull().any():
                current_values = {} 
                total_value = 0
                valid_tickers_for_value = []
                for ticker in required_tickers:
                    if ticker in latest_prices and pd.notna(latest_prices[ticker]) and ticker in holdings_dict:
                        value = latest_prices[ticker] * holdings_dict[ticker]
                        current_values[ticker] = value
                        total_value += value
                        valid_tickers_for_value.append(ticker)
                
                if total_value > 0:
                    # concentration check
                    weights = {ticker: (current_values[ticker] / total_value) * 100 for ticker in valid_tickers_for_value}
                    sorted_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)
                    top_concentrated = sorted_weights[:top_n]
                    concen_details = [f"{ticker} ({weight:.1f}%)" for ticker, weight in top_concentrated]
                    concen_text = ", ".join(concen_details)

                    # beta check 
                    beta_details = "N/A"
                    portfolio_beta = np.nan # placeholder
                    top_beta_stocks = []
                    if beta_data:
                        valid_betas = {ticker: beta for ticker, beta in beta_data.items() if ticker in valid_tickers_for_value and pd.notna(beta)}
                        if valid_betas:
                            # attempt to calculate portfolio beta
                            try:
                                weights_series = pd.Series({t: w/100.0 for t, w in weights.items()})
                                beta_series = pd.Series(valid_betas)
                                common_idx = weights_series.index.intersection(beta_series.index)
                                if not common_idx.empty:
                                    portfolio_beta = np.sum(weights_series[common_idx] * beta_series[common_idx])
                            except Exception:
                                portfolio_beta = np.nan
                                
                            sorted_betas = sorted(valid_betas.items(), key=lambda item: item[1], reverse=True)
                            top_beta_stocks = sorted_betas[:top_n]
                            beta_details = [f"{ticker} (Beta: {beta:.2f})" for ticker, beta in top_beta_stocks] 
                            beta_text = ", ".join(beta_details)

                    # generate suggestions based on drivers
                    if "VaR" in moderate_risk_drivers:
                        suggestions.append(f" - Moderate VaR: Review volatility contribution from top holdings ({concen_text}) and high beta stocks (Top: {beta_text if beta_text else 'N/A'}).")
                    if "StressTest" in moderate_risk_drivers:
                        beta_display = f"{portfolio_beta:.2f}" if not np.isnan(portfolio_beta) else "N/A"
                        suggestions.append(f" - Moderate Stress Test Performance: Review concentration (Top: {concen_text}) and market sensitivity (Beta: {beta_display}) in light of historical downturns.")
                    if "IndexDrop" in moderate_risk_drivers:
                        suggestions.append(f" - Moderate Sensitivity to Index Drops: Review exposure to high beta stocks (Top: {beta_text if beta_text else 'N/A'}). Consider if >1.0 is appropriate.")

                    if not moderate_risk_drivers:
                        suggestions.append(" - General: Review portfolio concentration and market sensitivity (beta) for potential adjustments.")
                else:
                    suggestions.append(" - Could not analyze factors due to zero portfolio value.")
            else:
                 suggestions.append(" - Could not analyze factors due to missing price data.")
        except Exception as e:
            print(f"Error calculating factors for Moderate mitigation: {e}")
            suggestions.append(" - An error occurred during factor analysis for suggestions.")

        # final generic advice for moderate
        suggestions.append("Consider targeted adjustments (e.g., slight rebalancing, reviewing specific high-beta/volatile positions) if these factors align negatively with your market outlook.")
        return suggestions

    # calculations & suggestions for high risk
    suggestions.append("Risk Level is High. Specific factors contributing:")
    high_risk_drivers = []
    
    rationale_text = " ".join(risk_rationale) if isinstance(risk_rationale, list) else risk_rationale if risk_rationale else ""
    
    rationale_lower = rationale_text.lower()
    if "var" in rationale_lower and ("red" in rationale_lower or "yellow" in rationale_lower):
        high_risk_drivers.append("VaR")
    if "stress test" in rationale_lower and ("red" in rationale_lower or "yellow" in rationale_lower):
        high_risk_drivers.append("StressTest")
    if "index drop" in rationale_lower and ("red" in rationale_lower or "yellow" in rationale_lower):
         high_risk_drivers.append("IndexDrop")
         
    if not high_risk_drivers:
         suggestions.append(" - (Could not isolate specific driver from rationale text, review rationale details)")

    try:
        holdings_dict = portfolio_df.set_index('Ticker')['Holding'].to_dict()
        required_tickers = list(portfolio_df['Ticker'])
        
        # check for missing latest prices
        if latest_prices is None or latest_prices.isnull().any():
            pass 
            
        else: # proceed only if we have prices
            # calculate current values and total portfolio value
            current_values = {} 
            total_value = 0
            valid_tickers_for_value = []
            for ticker in required_tickers:
                if ticker in latest_prices and pd.notna(latest_prices[ticker]) and ticker in holdings_dict:
                    value = latest_prices[ticker] * holdings_dict[ticker]
                    current_values[ticker] = value
                    total_value += value
                    valid_tickers_for_value.append(ticker)
                else:
                     print(f"Warning: Excluding ticker {ticker} from mitigation value calc due to missing price/holding.")

            if total_value > 0:
                # 1. concentration risk
                weights = {ticker: (current_values[ticker] / total_value) * 100 for ticker in valid_tickers_for_value}
                sorted_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)
                
                top_concentrated = sorted_weights[:top_n]
                if top_concentrated:
                    concen_str = f"High Concentration. Top {len(top_concentrated)} holdings: "
                    concen_details = [f"{ticker} ({weight:.1f}%)" for ticker, weight in top_concentrated]
                    concen_str += ", ".join(concen_details) + "."
                    suggestions.append(f" - {concen_str}")
                    suggestions.append("   -> Suggestion: Review concentration. Consider rebalancing towards a maximum position size (e.g., <20-25%) if single stocks dominate risk.")

                # 2. high beta risk
                high_beta_found = False
                if beta_data:
                    valid_betas = {ticker: beta for ticker, beta in beta_data.items() if ticker in valid_tickers_for_value and pd.notna(beta)}
                    if valid_betas:
                        sorted_betas = sorted(valid_betas.items(), key=lambda item: item[1], reverse=True)
                        top_beta_stocks = sorted_betas[:top_n]
                        if top_beta_stocks and top_beta_stocks[0][1] > 1.2: # Only flag if highest beta is reasonably high
                            beta_str = f"High Market Sensitivity. Top {len(top_beta_stocks)} beta stocks: "
                            beta_details = [f"{ticker} (Beta: {beta:.2f})" for ticker, beta in top_beta_stocks]
                            beta_str += ", ".join(beta_details) + "."
                            suggestions.append(f" - {beta_str}")
                            suggestions.append("   -> Suggestion: High beta increases sensitivity to market moves. Consider reducing exposure to these stocks, especially if market downturns are a concern or if Stress Tests/Index Drop scenarios are key risk drivers.")
                            high_beta_found = True

                # 3. generic hedge suggestion based on drivers
                hedge_suggestion = "General Suggestion: "
                if "StressTest" in high_risk_drivers or "IndexDrop" in high_risk_drivers or high_beta_found:
                    hedge_suggestion += "Given sensitivity to market downturns (Stress Tests, Index Drop, High Beta), consider broad market hedges (e.g., index put options, inverse ETFs) to mitigate overall market risk."
                elif "VaR" in high_risk_drivers and not high_beta_found:
                     hedge_suggestion += "Consider diversification or targeted hedges for specific high-volatility positions contributing to VaR."
                else: #fallback
                     hedge_suggestion += "Consider portfolio adjustments like diversification or position sizing based on identified risk factors."
                suggestions.append(hedge_suggestion)
            
            else:
                suggestions.append(" - Cannot calculate specific factors as total portfolio value is zero.")

    except Exception as e:
        print(f"Error generating mitigation suggestions: {e}")
        suggestions = ["Risk Level is High. An error occurred generating specific suggestions.", 
                       "Review overall portfolio risk and consider general mitigation strategies."]

    # ensure there's at least some suggestion if high risk was determined but calculation failed
    if risk_level == 'High' and len(suggestions) <= 1:
         suggestions.append("Could not generate specific suggestions. Review portfolio risk and consider general mitigation strategies (diversification, hedging, position sizing).")

    return suggestions 