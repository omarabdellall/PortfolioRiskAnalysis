"""
Configuration settings for the Risk Analysis System.
"""

RISK_THRESHOLDS = {
    # VaR thresholds
    'var_pct': {
        'yellow': -0.05,
        'red': -0.08
    },
    # stress test thresholds
    'stress_pct': {
        'yellow': -0.25,
        'red': -0.40
    },
    # prospective scenario thresholds
    'scenario_index_drop_pct': {
        'yellow': -0.15,
        'red': -0.20
    }
}

HISTORICAL_STRESS_SCENARIOS = {
    "2008_financial_crisis": {
        "start_date": "2008-01-01",
        "end_date": "2009-03-09",
        "description": "Global Financial Crisis (GFC)"
    },
    "2020_covid_crash": {
        "start_date": "2020-02-19",
        "end_date": "2020-03-23",
        "description": "COVID-19 Market Crash"
    },
    "dot_com_bubble": {
        "start_date": "2000-03-10",
        "end_date": "2002-10-09",
        "description": "Dot-com Bubble Burst"
    },
    "us_debt_ceiling_crisis_2011": {
        "start_date": "2011-05-01",
        "end_date": "2011-10-04",
        "description": "2011 US Debt Ceiling Crisis"
    }
}

PROSPECTIVE_SCENARIOS = {
    "interest_rate_shock": {
        "type": "uniform",
        "shock_pct": -0.05,
        "description": "+2% Interest Rate Shock"
    },
    "index_drop": {
        "type": "beta_weighted",
        "index_shock_pct": -0.10,
        "description": "-10% Major Index Drop"
    },
    "sector_downturn_tech": {
        "type": "sector_specific",
        "target_sector": "Technology",
        "shock_pct": -0.15,
        "description": "-15% Technology Sector Downturn"
    },
    "sector_downturn_financial": {
        "type": "sector_specific",
        "target_sector": "Financial Services",
        "shock_pct": -0.12,
        "description": "-12% Financial Sector Downturn"
    },
    "oil_price_shock": {
        "type": "uniform",
        "shock_pct": -0.03,
        "description": "+30% Oil Price Shock"
    }
} 