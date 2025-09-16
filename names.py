# names.py
import pandas as pd
import FinanceDataReader as fdr

def get_name_map_krx(markets=("KOSPI","KOSDAQ")) -> dict:
    krx = fdr.StockListing("KRX")[["Code","Name","Market"]]
    krx = krx[krx["Market"].isin(markets)]
    # 상장폐지 제외 컬럼이 있다면 필터
    if "DelistingDate" in krx.columns:
        krx = krx[krx["DelistingDate"].isna()]

    mp = {}
    for _, r in krx.iterrows():
        suffix = ".KS" if r["Market"] == "KOSPI" else ".KQ"
        mp[r["Code"] + suffix] = r["Name"]    # 005930.KS -> 삼성전자
        mp[r["Code"]] = r["Name"]             # 005930     -> 삼성전자
    return mp
