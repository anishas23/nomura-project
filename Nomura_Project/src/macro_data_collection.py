import yfinance as yf
import pandas as pd
import pandas_datareader.data as web

start_date = "2019-01-01"
end_date = "2024-01-01"

# -----------------------------
# 1. INDIA VIX
# -----------------------------
print("Downloading India VIX...")

vix = yf.download(
    "^INDIAVIX",
    start=start_date,
    end=end_date,
    auto_adjust=False
)

# Fix multi-index columns
vix.columns = vix.columns.get_level_values(0)

vix = vix.reset_index()
vix["Date"] = pd.to_datetime(vix["Date"])

vix = vix[["Date","Close"]]
vix.rename(columns={"Close":"India_VIX"}, inplace=True)

vix.to_csv("india_vix.csv", index=False)

print("India VIX data saved")
print(vix.head())


# -----------------------------
# 2. INTEREST RATES (India 10Y Bond Yield)
# -----------------------------
print("\nDownloading Interest Rate Data...")

rates = web.DataReader(
    "INDIRLTLT01STM",   # India 10Y Government Bond Yield
    "fred",
    start=start_date,
    end=end_date
)

rates = rates.reset_index()

rates.rename(columns={
    "DATE": "Date",
    "INDIRLTLT01STM": "Interest_Rate"
}, inplace=True)

rates["Date"] = pd.to_datetime(rates["Date"])

rates.to_csv("interest_rates.csv", index=False)

print("Interest rate data saved")
print(rates.head())

# -----------------------------
# 3. INFLATION (CPI)
# -----------------------------
print("\nDownloading Inflation Data...")

inflation = web.DataReader(
    "INDCPIALLMINMEI",
    "fred",
    start=start_date,
    end=end_date
)

inflation = inflation.reset_index()

# Rename columns correctly
inflation.rename(columns={
    "DATE":"Date",
    "INDCPIALLMINMEI":"Inflation"
}, inplace=True)

inflation["Date"] = pd.to_datetime(inflation["Date"])

inflation = inflation[["Date","Inflation"]]

inflation.to_csv("india_inflation.csv", index=False)

print("Inflation data saved")
print(inflation.head())


print("\nMacroeconomic data collection completed!")
