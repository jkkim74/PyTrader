# --- robust_data.py ----------------------------------------------------------
# KRX는 FDR/pykrx 우선, 그 외는 yfinance 사용
import pandas as pd
import numpy as np

try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None

try:
    from pykrx import stock as krx
except Exception:
    krx = None

import yfinance as yf

def _standardize_ohlcv(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    df = df.rename(columns=colmap).copy()
    need = ["open","high","low","close","volume"]
    df = df[ [c for c in need if c in df.columns] ]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def fetch_daily_any(ticker: str, start="2015-01-01", end=None, quiet=True) -> pd.DataFrame:
    """
    ticker 예:
      - KRX: '005930.KS', '000660.KS', '035420.KQ' 또는 '005930' 같은 코드만도 OK
      - 해외: 'AAPL', 'NVDA' 등은 yfinance
    """
    is_krx = ticker.endswith(".KS") or ticker.endswith(".KQ") or ticker.isdigit()
    code = ticker.split(".")[0] if is_krx else ticker
    errs = []

    # 1) FDR 우선 (KRX일 때)
    if is_krx and fdr is not None:
        try:
            df = fdr.DataReader(code, start, end)
            if not df.empty:
                return _standardize_ohlcv(df, {
                    "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
                })
        except Exception as e:
            errs.append(f"FDR:{e}")

    # 2) PyKrx 폴백
    if is_krx and krx is not None:
        try:
            s = (pd.Timestamp(start).strftime("%Y%m%d") if start else "19900101")
            e = (pd.Timestamp(end).strftime("%Y%m%d") if end else pd.Timestamp.today().strftime("%Y%m%d"))
            df = krx.get_market_ohlcv_by_date(s, e, code)
            if not df.empty:
                df.columns = [c.strip() for c in df.columns]
                return _standardize_ohlcv(df, {
                    "시가":"open","고가":"high","저가":"low","종가":"close","거래량":"volume"
                })
        except Exception as e:
            errs.append(f"pykrx:{e}")

    # 3) yfinance (해외 또는 최종 폴백)
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if not df.empty:
            return _standardize_ohlcv(df, {
                "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
            })
    except Exception as e:
        errs.append(f"yfinance:{e}")

    if not quiet:
        print(f"[DATA-EMPTY] {ticker} -> {' | '.join(errs)}")
    return pd.DataFrame()

def get_krx_universe(markets=("KOSPI","KOSDAQ"), limit=None):
    """
    현재 상장 종목만 반환. 결과는 yfinance 스타일 접미사(.KS/.KQ)를 붙여 반환.
    """
    if fdr is None:
        raise RuntimeError("FinanceDataReader가 필요합니다: pip install FinanceDataReader")
    krx_df = fdr.StockListing("KRX")
    krx_df = krx_df[krx_df["Market"].isin(markets)].copy()

    # 상장폐지 제외
    if "DelistingDate" in krx_df.columns:
        krx_df = krx_df[krx_df["DelistingDate"].isna()]

    # 보통주만
    if "Type" in krx_df.columns:
        krx_df = krx_df[krx_df["Type"].isin(["Stock","Common Stock","보통주"])]

    def add_suffix(row):
        suffix = ".KS" if row["Market"] == "KOSPI" else ".KQ"
        return f"{row['Code']}{suffix}"

    tickers = krx_df.apply(add_suffix, axis=1).dropna().unique().tolist()
    if limit:
        tickers = tickers[:limit]
    return tickers

# --- weekly helper (미완성 주봉 제외 옵션 포함) -------------------------------
def to_weekly(df_daily: pd.DataFrame, include_partial_week=True,
              tz="Asia/Seoul", market_close_hour=15, market_close_min=30) -> pd.DataFrame:
    o = df_daily["open"].resample("W-FRI").first()
    h = df_daily["high"].resample("W-FRI").max()
    l = df_daily["low"].resample("W-FRI").min()
    c = df_daily["close"].resample("W-FRI").last()
    v = df_daily["volume"].resample("W-FRI").sum()
    w = pd.concat([o,h,l,c,v], axis=1)
    w.columns = ["open","high","low","close","volume"]
    w = w.dropna()

    if not include_partial_week and len(w) > 0:
        now = pd.Timestamp.now(tz=tz)
        if now.weekday() <= 4:
            fri_date = now.normalize() + pd.Timedelta(days=(4 - now.weekday()))
        else:
            fri_date = now.normalize() - pd.Timedelta(days=(now.weekday() - 4))
        cutoff = fri_date + pd.Timedelta(hours=market_close_hour, minutes=market_close_min)
        if now < cutoff:
            w = w.iloc[:-1]
    return w
