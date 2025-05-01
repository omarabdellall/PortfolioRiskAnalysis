#functions for fetching financial data
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def fetch_stock_data(tickers, start_date, end_date):
    if not tickers:
        print("Error: No tickers provided for data fetching.")
        return None, None, None, None

    cleaned_tickers = sorted(list(set([t.upper() for t in tickers])))
    print(f"Fetching data for tickers: {cleaned_tickers} from {start_date} to {end_date}...")

    historical_prices = None
    failed_price_tickers = []
    try:
        data = yf.download(cleaned_tickers, 
                           start=start_date, 
                           end=end_date, 
                           auto_adjust=True, 
                           progress=False,
                           group_by='ticker') # group by ticker to isolate failures
        
        if data.empty:
            print("Warning: No historical price data downloaded.")
            failed_price_tickers = cleaned_tickers #mark all tickers failed
        else:
            all_adj_close_data = {}
            successful_tickers = []
            for ticker in cleaned_tickers:
                try:
                    ticker_data = data[ticker]
                    if not ticker_data.empty and 'Close' in ticker_data.columns:
                        adj_close = ticker_data[['Close']].rename(columns={'Close': ticker})
                        if not adj_close.isnull().all().iloc[0]: #check if series is not all none
                            all_adj_close_data[ticker] = adj_close[ticker]
                            successful_tickers.append(ticker)
                        else:
                            print(f"Warning: Price data for {ticker} is all NaN.")
                            failed_price_tickers.append(ticker)
                    else:
                        print(f"Warning: No valid price data structure found for {ticker}.")
                        failed_price_tickers.append(ticker)
                except KeyError:
                    print(f"Warning: Could not retrieve price data column for {ticker}. Download likely failed.")
                    failed_price_tickers.append(ticker)
                except Exception as e:
                    print(f"Warning: Error processing price data for {ticker}: {e}")
                    failed_price_tickers.append(ticker)

            if successful_tickers:
                historical_prices = pd.DataFrame(all_adj_close_data)
                historical_prices.index = pd.to_datetime(historical_prices.index)
            else:
                 print("Warning: Failed to fetch valid price data for any requested ticker.")

    except Exception as e:
        print(f"Error during yfinance price download call: {e}")
        failed_price_tickers = cleaned_tickers #mark all as failed if download fails

    ticker_info = {}
    beta_data = {}
    sector_data = {}
    failed_info_tickers = []
    currency_data = {}

    print(f"Fetching additional info (beta, sector, currency) for tickers: {cleaned_tickers}...")
    yf_tickers = yf.Tickers(cleaned_tickers)

    for ticker in cleaned_tickers:
        try:
            info = yf_tickers.tickers[ticker].info
            beta = info.get('beta', np.nan) #get beta, default to NaN if missing
            sector = info.get('sector', 'Unknown') #get sector, default unknown if missing
            currency = info.get('currency', 'USD') #get currency, default to USD if missing

            if pd.isna(beta):
                print(f"Warning: Beta not available for {ticker}.")
            if sector == 'Unknown':
                print(f"Warning: Sector not available for {ticker}.")
            
            beta_data[ticker] = beta
            sector_data[ticker] = sector
            currency_data[ticker] = currency

        except Exception as e:
            print(f"Warning: Could not retrieve info for {ticker}: {e}")
            failed_info_tickers.append(ticker)
            beta_data[ticker] = np.nan 
            sector_data[ticker] = 'Unknown'
            currency_data[ticker] = 'USD'
    print("Data fetching process completed.")

    if failed_price_tickers:
        print(f"Warning: Failed to fetch or process price data completely for: {failed_price_tickers}")
    if failed_info_tickers:
        print(f"Warning: Failed to fetch info (beta/sector/currency) for: {failed_info_tickers}")

    if historical_prices is None and not beta_data:
         print("Error: Failed to fetch any useful data.")
         return None, None, None, None

    return historical_prices, beta_data, sector_data, currency_data

def fetch_fx_rates(currency_pairs, start_date, end_date):
    if not currency_pairs:
        print("No currency pairs provided for FX fetching.")
        return pd.DataFrame() 
    print(f"\nFetching FX rates for: {currency_pairs} from {start_date} to {end_date}...")
    try:
        fx_data = yf.download(currency_pairs, 
                              start=start_date, 
                              end=end_date, 
                              auto_adjust=True, 
                              progress=False)
        
        if fx_data.empty:
            print("Warning: No FX rate data downloaded.")
            return None

        # select only 'close' prices
        if isinstance(fx_data.columns, pd.MultiIndex):
            fx_close_prices = fx_data['Close']
        elif len(currency_pairs) == 1:
             if 'Close' in fx_data.columns:
                 fx_close_prices = fx_data[['Close']].rename(columns={'Close': currency_pairs[0]})
             else:
                  print(f"Warning: 'Close' column not found for FX pair {currency_pairs[0]}.")
                  return None
        else:
             print("Warning: Unexpected FX data structure.")
             return None

        # forward fill missing values (common for weekends/holidays)
        fx_close_prices = fx_close_prices.ffill()
        
        # check if all requested pairs were fetched successfully
        missing_pairs = [p for p in currency_pairs if p not in fx_close_prices.columns]
        if missing_pairs:
            print(f"Warning: Failed to fetch FX data for pairs: {missing_pairs}")
            for pair in missing_pairs:
                fx_close_prices[pair] = np.nan 

        print("FX rate fetching completed.")
        return fx_close_prices

    except Exception as e:
        print(f"Error during yfinance FX download call: {e}")
        return None

def fetch_market_index_data(index_ticker, start_date, end_date):
    print(f"\nFetching market index data for: {index_ticker} from {start_date} to {end_date}...")
    try:
        index_data = yf.download(index_ticker, 
                                 start=start_date, 
                                 end=end_date, 
                                 auto_adjust=True, 
                                 progress=False)
        
        if index_data.empty:
            print(f"Warning: No data downloaded for market index {index_ticker}.")
            return None

        # select only 'close' prices and rename column
        if 'Close' in index_data.columns:
             index_close = index_data[['Close']].rename(columns={'Close': index_ticker})
             print(f"Market index data for {index_ticker} fetched successfully.")
             return index_close
        else:
            print(f"Warning: 'Close' column not found for market index {index_ticker}.")
            return None

    except Exception as e:
        print(f"Error during yfinance market index download call: {e}")
        return None