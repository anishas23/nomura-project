import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
price_matrix = pd.read_csv("data/processed_data/price_matrix.csv", index_col='date', parse_dates=True)
return_matrix = pd.read_csv("data/processed_data/return_matrix.csv", index_col='date', parse_dates=True)

print("Data loaded successfully!")

# -----------------------------
# 1. PRICE TRENDS
# -----------------------------

import os
os.makedirs("reports/visualizations", exist_ok=True)

sample_stocks = price_matrix.columns[:5]

plt.figure(figsize=(12,6))

for stock in sample_stocks:
    plt.plot(price_matrix.index, price_matrix[stock], label=stock)

plt.title("Price Trends of Sample Stocks")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()

# Save instead of show
plt.savefig("reports/visualizations/price_trends.png")
plt.close()

print("✅ Price trends plot saved!")

# -----------------------------
# 2. RETURN DISTRIBUTION
# -----------------------------

plt.figure(figsize=(12,6))

# Plot histogram for 5 stocks
for stock in return_matrix.columns[:5]:
    sns.histplot(return_matrix[stock], bins=50, kde=True, label=stock, alpha=0.5)

plt.title("Return Distribution of Sample Stocks")
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.legend()

# Save plot
plt.savefig("reports/visualizations/return_distribution.png")
plt.close()

print("✅ Return distribution plot saved!")

# -----------------------------
# 3. CORRELATION HEATMAP
# -----------------------------

plt.figure(figsize=(12,10))

# Compute correlation matrix
corr_matrix = return_matrix.corr()

# Plot heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)

plt.title("Stock Correlation Heatmap")

# Save plot
plt.savefig("reports/visualizations/correlation_heatmap.png")
plt.close()

print("✅ Correlation heatmap saved!")

# -----------------------------
# 4. VOLATILITY COMPARISON
# -----------------------------

# Calculate volatility (std of returns)
volatility = return_matrix.std()

# Sort values
volatility = volatility.sort_values(ascending=False)

# Take top 10 most volatile stocks
top_vol = volatility.head(10)

plt.figure(figsize=(12,6))

top_vol.plot(kind='bar')

plt.title("Top 10 Most Volatile Stocks")
plt.xlabel("Stocks")
plt.ylabel("Volatility")

# Save plot
plt.savefig("reports/visualizations/volatility_comparison.png")
plt.close()

print("✅ Volatility comparison plot saved!")