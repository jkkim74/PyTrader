import FinanceDataReader as fdr

df_kospi = fdr.StockListing('KOSPI')
print(df_kospi.head(10))
df_kosdaq = fdr.StockListing('KOSDAQ')
print(df_kosdaq.head(10))