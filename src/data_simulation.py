import numpy as np
import pandas as pd

def generate_data(n=365, adstock=True):
    """Generate synthetic media mix data with realistic patterns"""
    np.random.seed(42)
    
    # Generate date index
    dates = pd.date_range(start="2023-01-01", periods=n)
    
    # Media spends with different distributions
    tv_spend = np.abs(np.random.normal(100, 30, n))
    digital_spend = np.abs(np.random.gamma(shape=2, scale=50, size=n))
    
    # Apply adstock transformation if needed
    if adstock:
        tv_spend = adstock_transform(tv_spend, decay=0.75)
        digital_spend = adstock_transform(digital_spend, decay=0.5)
    
    # Confounding variables
    competitor_promo = np.random.binomial(1, 0.3, n)
    holiday = seasonality(n, periods=[30, 90, 180])
    
    # Sales response with diminishing returns
    tv_effect = 2.7 * np.log1p(tv_spend)
    digital_effect = 1.8 * np.log1p(digital_spend)
    base_sales = 500 + tv_effect + digital_effect
    sales = base_sales - 100 * competitor_promo + 200 * holiday
    sales = np.abs(sales + np.random.normal(0, 50, n))
    
    return pd.DataFrame({
        "date": dates,
        "tv_spend": tv_spend,
        "digital_spend": digital_spend,
        "competitor_promo": competitor_promo,
        "holiday": holiday,
        "sales": sales
    })

def adstock_transform(x, decay=0.8):
    """Vectorized adstock transformation with carryover effect"""
    x = np.asarray(x)
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]
    
    for i in range(1, len(x)):
        adstocked[i] = x[i] + decay * adstocked[i-1]
    
    return adstocked

def seasonality(n, periods=[90]):
    seasonality = np.zeros(n)
    for p in periods:
        seasonality += 0.5 * np.sin(2 * np.pi * np.arange(n) / p)
    return (seasonality > 0.3).astype(int)