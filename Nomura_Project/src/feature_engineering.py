import pandas as pd

# Load data
price_matrix = pd.read_csv("data/processed_data/price_matrix.csv", index_col='date', parse_dates=True)
return_matrix = pd.read_csv("data/processed_data/return_matrix.csv", index_col='date', parse_dates=True)

print("Data loaded!")
print(price_matrix.shape)
print(return_matrix.shape)

# -----------------------------
# STEP 1: Moving Average (MA20)
# -----------------------------

ma_20 = price_matrix.rolling(window=20).mean()

print("MA20 calculated")
print(ma_20.head())

print("\nAfter 25 days:")
print(ma_20.iloc[25].head())
ma_20.to_csv("data/features/ma_20.csv")

# -----------------------------
# STEP: Bollinger Bands
# -----------------------------

# Rolling std (same window as MA20)
rolling_std = price_matrix.rolling(window=20).std()

# Upper band
bb_upper = ma_20 + (2 * rolling_std)

# Lower band
bb_lower = ma_20 - (2 * rolling_std)

print("\nBollinger Bands calculated")
print(bb_upper.head())

# Save (optional)
bb_upper.to_csv("data/features/bb_upper.csv")
bb_lower.to_csv("data/features/bb_lower.csv")
# -----------------------------
# STEP: Normalized Bollinger Width
# -----------------------------

bb_width = (bb_upper - bb_lower) / ma_20

print("\nBB Width calculated")
print(bb_width.head())

bb_width.to_csv("data/features/bb_width.csv")
# -----------------------------
# STEP: Moving Average (MA50)
# -----------------------------

ma_50 = price_matrix.rolling(window=50).mean()

print("MA50 calculated")
print(ma_50.head())

ma_50.to_csv("data/features/ma_50.csv")

# -----------------------------
# STEP: Moving Average (MA200)
# -----------------------------

ma_200 = price_matrix.rolling(window=200).mean()

print("MA200 calculated")
print(ma_200.head())

ma_200.to_csv("data/features/ma_200.csv")



# -----------------------------
# STEP 2: RSI
# -----------------------------

delta = price_matrix.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

rsi = 100 - (100 / (1 + rs))

print("\nRSI calculated")
print(rsi.head())
print("\nRSI after 20 days:")
print(rsi.iloc[20].head())
rsi.to_csv("data/features/rsi.csv")

# -----------------------------
# STEP 3: Volatility
# -----------------------------

volatility = return_matrix.rolling(window=20).std()

print("\nVolatility calculated")
print(volatility.head())
print("\nVolatility after 25 days:")
print(volatility.iloc[25].head())
volatility.to_csv("data/features/volatility.csv")

# -----------------------------
# STEP: Value at Risk (VaR)
# -----------------------------

# 95% VaR (5th percentile)
var_95 = return_matrix.rolling(window=20).quantile(0.05)

print("\nVaR (95%) calculated")
print(var_95.head())

# Save
var_95.to_csv("data/features/var_95.csv")

# -----------------------------
# STEP: CVaR (Expected Shortfall)
# -----------------------------

def compute_cvar(x):
    var = x.quantile(0.05)
    return x[x <= var].mean()

cvar_95 = return_matrix.rolling(window=20).apply(compute_cvar, raw=False)

print("\nCVaR (95%) calculated")
print(cvar_95.head())

# Save
cvar_95.to_csv("data/features/cvar_95.csv")

# -----------------------------
# STEP: Maximum Drawdown
# -----------------------------

# Rolling max price
rolling_max = price_matrix.cummax()

# Drawdown
drawdown = (price_matrix - rolling_max) / rolling_max

# Save
drawdown.to_csv("data/features/drawdown.csv")

# -----------------------------
# STEP: Momentum (12-month return)
# -----------------------------

momentum = price_matrix.pct_change(periods=252, fill_method=None)

print("\nMomentum calculated")
print(momentum.head())

# Save
momentum.to_csv("data/features/momentum.csv")

print("\nDrawdown calculated")
print(drawdown.head())

# -----------------------------
# STEP: MARKET TREND (BULL/BEAR)
# -----------------------------

# Market proxy = average of all stocks
market_price = price_matrix.mean(axis=1)

# 50-day moving average of market
market_ma50 = market_price.rolling(50).mean()

# Bull = 1, Bear = 0
market_trend = (market_price > market_ma50).astype(int)

print("\nMarket trend calculated")
print(market_trend.head())

# -----------------------------
# STEP: VOLATILITY REGIME (VIX)
# -----------------------------

# Load VIX (processed one)
vix = pd.read_csv("data/raw_data/india_vix.csv", parse_dates=['Date'])
vix.columns = ['date', 'vix']

# Sort
vix = vix.sort_values('date')

# -----------------------------
# STEP: VIX Regime
# -----------------------------

# 75th percentile threshold
vix_threshold = vix['vix'].quantile(0.75)

# High-volatility regime
vix['vix_regime'] = (vix['vix'] > vix_threshold).astype(int)

print("\nVIX Regime calculated")
print(vix.head())

print("\nVIX Threshold:")
print(vix_threshold)

print("\nVIX regime calculated")
print(vix.head())

# -----------------------------
# STEP: Market Trend Regime
# -----------------------------

market_trend = (price_matrix > ma_200).astype(int)

print("\nMarket Trend Regime calculated")
print(market_trend.head())

market_trend.to_csv("data/features/market_trend.csv")

# -----------------------------
# STEP 4: Convert to Long Format
# -----------------------------

returns_long = return_matrix.stack().reset_index()
returns_long.columns = ['date', 'symbol', 'return']

ma20_long = ma_20.stack().reset_index()
ma20_long.columns = ['date', 'symbol', 'ma20']

rsi_long = rsi.stack().reset_index()
rsi_long.columns = ['date', 'symbol', 'rsi']

vol_long = volatility.stack().reset_index()
vol_long.columns = ['date', 'symbol', 'volatility']

bb_upper_long = bb_upper.stack().reset_index()
bb_upper_long.columns = ['date', 'symbol', 'bb_upper']

bb_lower_long = bb_lower.stack().reset_index()
bb_lower_long.columns = ['date', 'symbol', 'bb_lower']

bb_width_long = bb_width.stack().reset_index()
bb_width_long.columns = ['date', 'symbol', 'bb_width']

var_long = var_95.stack().reset_index()
var_long.columns = ['date', 'symbol', 'var_95']

cvar_long = cvar_95.stack().reset_index()
cvar_long.columns = ['date', 'symbol', 'cvar_95']

drawdown_long = drawdown.stack().reset_index()
drawdown_long.columns = ['date', 'symbol', 'drawdown']

momentum_long = momentum.stack().reset_index()
momentum_long.columns = ['date', 'symbol', 'momentum']

# Market trend to long format
market_trend_long = market_trend.stack().reset_index()

market_trend_long.columns = [
    'date',
    'symbol',
    'market_trend'
]

# Keep only needed columns
vix_regime_long = vix[['date', 'vix_regime']]

# Market trend to dataframe
market_trend_df = market_trend.reset_index()
# -----------------------------
# Convert Market Trend to Long Format
# -----------------------------

market_trend_long = market_trend.stack().reset_index()

market_trend_long.columns = [
    'date',
    'symbol',
    'market_trend'
]

print("\nMarket Trend Long Format")
print(market_trend_long.head())

# VIX regime already in df → just keep needed cols
vix_regime_df = vix[['date', 'vix_regime']]

print("\nConverted to long format")
print(returns_long.head())

# -----------------------------
# STEP 5: Merge All Features
# -----------------------------

# Start with returns
final_data = returns_long.copy()

# Merge MA20
final_data = final_data.merge(ma20_long, on=['date', 'symbol'], how='left')

# Merge RSI
final_data = final_data.merge(rsi_long, on=['date', 'symbol'], how='left')

# Merge Volatility
final_data = final_data.merge(vol_long, on=['date', 'symbol'], how='left')

final_data = final_data.merge(bb_upper_long, on=['date', 'symbol'], how='left')
final_data = final_data.merge(bb_lower_long, on=['date', 'symbol'], how='left')

final_data = final_data.merge(var_long, on=['date', 'symbol'], how='left')

final_data = final_data.merge(cvar_long, on=['date', 'symbol'], how='left')

final_data = final_data.merge(drawdown_long, on=['date', 'symbol'], how='left')

final_data = final_data.merge(momentum_long, on=['date', 'symbol'], how='left')

final_data = final_data.merge(
    market_trend_long,
    on=['date', 'symbol'],
    how='left'
)

final_data = final_data.merge(
    vix_regime_long,
    on='date',
    how='left'
)



final_data = final_data.merge(
    bb_width_long,
    on=['date', 'symbol'],
    how='left'
)

print("\nFinal dataset created")
print(final_data.head())


print(final_data.head())
print("Shape:", final_data.shape)


# -----------------------------
# ADD MA50 & MA200 TO FINAL DATASET
# -----------------------------

# Convert to long format
ma50_long = ma_50.stack().reset_index()
ma50_long.columns = ['date', 'symbol', 'ma50']

ma200_long = ma_200.stack().reset_index()
ma200_long.columns = ['date', 'symbol', 'ma200']

# Merge into final dataset
final_data = final_data.merge(ma50_long, on=['date', 'symbol'], how='left')
final_data = final_data.merge(ma200_long, on=['date', 'symbol'], how='left')

print("✅ MA50 & MA200 added to final dataset")
print(final_data.head())


# -----------------------------
# FINAL CLEANING (CORRECT PLACE)
# -----------------------------

final_data = final_data.dropna()

print("\nAfter final dropna:")
print(final_data.head())
print("Shape:", final_data.shape)
final_data.to_csv("data/features/final_dataset.csv", index=False)


