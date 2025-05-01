#functions for loading portfolio data
import pandas as pd

def load_portfolio(csv_path):
    try:
        portfolio_df = pd.read_csv(csv_path)
        # basic validation
        if 'Ticker' not in portfolio_df.columns or 'Holding' not in portfolio_df.columns:
            print(f"Error: CSV file {csv_path} must contain 'Ticker' and 'Holding' columns.")
            return None
        # ensure holding is numeric
        portfolio_df['Holding'] = pd.to_numeric(portfolio_df['Holding'], errors='coerce')
        if portfolio_df['Holding'].isnull().any():
            print(f"Error: Non-numeric values found in 'Holding' column of {csv_path}.")
            return None
            
        print(f"Portfolio loaded successfully from {csv_path}.")
        return portfolio_df
    except FileNotFoundError:
        print(f"Error: Portfolio file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading portfolio from {csv_path}: {e}")
        return None 