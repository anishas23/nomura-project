import pandas as pd

# -----------------------------
# 1. LOAD DATA (MUST BE FIRST)
# -----------------------------
stocks = pd.read_csv("data/raw_data/nifty50_ohlcv_2019_2024.csv")
vix = pd.read_csv("data/raw_data/india_vix.csv")
inflation = pd.read_csv("data/raw_data/india_inflation.csv")
rates = pd.read_csv("data/raw_data/interest_rates.csv")


# -----------------------------
# 2. CLEAN COLUMN NAMES
# -----------------------------
def clean_columns(df):
    df.columns = df.columns.str.lower().str.strip()
    return df

stocks = clean_columns(stocks)
vix = clean_columns(vix)
inflation = clean_columns(inflation)
rates = clean_columns(rates)

# Rename columns
stocks = stocks.rename(columns={"ticker": "symbol"})
vix = vix.rename(columns={"india_vix": "vix"})


# -----------------------------
# 3. FIX DATE FORMAT
# -----------------------------
stocks['date'] = pd.to_datetime(stocks['date'], format='mixed')
vix['date'] = pd.to_datetime(vix['date'])
inflation['date'] = pd.to_datetime(inflation['date'])
rates['date'] = pd.to_datetime(rates['date'])

# Sort data
stocks = stocks.sort_values(['symbol', 'date'])
vix = vix.sort_values('date')
inflation = inflation.sort_values('date')
rates = rates.sort_values('date')

# -----------------------------
# 5. HANDLE MISSING VALUES (STOCKS)
# -----------------------------

# Forward fill within each stock
stocks = stocks.sort_values(['symbol', 'date'])
stocks[['open', 'high', 'low', 'close', 'volume']] = (
    stocks.groupby('symbol')[['open', 'high', 'low', 'close', 'volume']]
    .transform(lambda x: x.ffill())
)

# Reset index (important after groupby)
stocks = stocks.reset_index(drop=True)

# Drop any remaining missing values
stocks = stocks.dropna()

print("\nAFTER HANDLING MISSING VALUES:")
print(stocks.isnull().sum())

# -----------------------------
# 6. MERGE MACRO DATA
# -----------------------------

# Merge VIX (daily, easy)
data = pd.merge(stocks, vix, on='date', how='left')

# Merge inflation (monthly)
data = pd.merge(data, inflation, on='date', how='left')

# Merge interest rates (monthly)
data = pd.merge(data, rates, on='date', how='left')


# -----------------------------
# 7. FILL MACRO MISSING VALUES
# -----------------------------

data[['inflation', 'interest_rate', 'vix']] = (
    data[['inflation', 'interest_rate', 'vix']]
    .ffill()
    .bfill()
)


# -----------------------------
# 8. CHECK FINAL DATA
# -----------------------------
print("\nFINAL DATA CHECK:")
print(data.isnull().sum())
print(data.head())
# -----------------------------
# -----------------------------
# 4. CHECK OUTPUT
# -----------------------------
print("\nAFTER CLEANING:")
print(stocks.head())
print("\nDATA TYPES:")
print(stocks.dtypes)
# -----------------------------  # -----------------------------
# 9. CALCULATE RETURNS
# -----------------------------

import numpy as np

# Simple returns
data['return'] = data.groupby('symbol')['close'].pct_change()

# Log returns (better for finance)
data['log_return'] = data.groupby('symbol')['close'].transform(
    lambda x: np.log(x / x.shift(1))
)

# -----------------------------
# 10. REMOVE INITIAL NaNs
# -----------------------------

data = data.dropna(subset=['return', 'log_return'])

print("\nRETURNS CHECK:")
print(data[['symbol', 'date', 'close', 'return', 'log_return']].head())
print("\nMissing values after returns:")
print(data[['return', 'log_return']].isnull().sum())
# -----------------------------# -----------------------------
# 11. CREATE MATRICES
# -----------------------------

print("\nREACHED MATRIX STEP")

price_matrix = data.pivot(index='date', columns='symbol', values='close')

# Keep only dates where ALL stocks have data
price_matrix = price_matrix.dropna()

# Recalculate returns from clean prices
return_matrix = price_matrix.pct_change().dropna()

print("\nPRICE MATRIX:")
print(price_matrix.head())

print("\nRETURN MATRIX:")
print(return_matrix.head())


# -----------------------------
# 12. SAVE DATA
# -----------------------------

import os

os.makedirs("data/processed_data", exist_ok=True)

data.to_csv("data/processed_data/cleaned_data.csv", index=False)
price_matrix.to_csv("data/processed_data/price_matrix.csv")
return_matrix.to_csv("data/processed_data/return_matrix.csv")

print("\n✅ Data saved successfully!")