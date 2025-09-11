# weekly_w_breakout_screener.py
import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from tqdm import tqdm

# ---------- 설정 파라미터 (필요시 조정) ----------
MARKETS = ["KOSPI", "KOSDAQ"]  # 스캔할 시장
YEARS_BACK = 2                  # 몇 년치 데이터로 주봉 생성할지
LOOKBACK_WEEKS = 26             # W자형/평균을 계산할 때 사용할 주 수 (예: 26주 = 약 6개월)
VOLUME_MULTIPLIER = 3.0         # 돌파주간의 거래량이 평균 대비 몇 배 이상인지
WEEKLY_AVG_VOL_WINDOW = 12      # 거래량 평균 계산 주수
BODY_MULTIPLIER = 1.5           # 장대양봉 기준: 이번 주 몸통이 과거 평균 몸통의 몇 배 이상
MIN_CLOSE_ABOVE_BREAK = 0.005   # 돌파 레벨을 얼마나 넘었을지 (예: 0.5% 이상)
# ------------------------------------------------

def get_ticker_list():
    tickers = []
    for m in MARKETS:
        if m == "KOSPI":
            tickers += stock.get_market_ticker_list(market="KOSPI")
        else:
            tickers += stock.get_market_ticker_list(market="KOSDAQ")
    return tickers

def fetch_daily_ohlcv(ticker, start_date, end_date):
    # pykrx returns DataFrame indexed by date (YYYYMMDD)
    df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={"시가":"open", "고가":"high", "저가":"low", "종가":"close", "거래량":"volume", "거래대금":"value"})
    return df[["open","high","low","close","volume"]]

def make_weekly_from_daily(df_daily):
    # 리샘플: 주 단위 종가 등. 주의: 'W-FRI'로 주간을 마감(금요일 기준)
    df = df_daily.copy()
    df_weekly = pd.DataFrame()
    df_weekly['open']  = df['open'].resample('W-FRI').first()
    df_weekly['high']  = df['high'].resample('W-FRI').max()
    df_weekly['low']   = df['low'].resample('W-FRI').min()
    df_weekly['close'] = df['close'].resample('W-FRI').last()
    df_weekly['volume']= df['volume'].resample('W-FRI').sum()
    df_weekly = df_weekly.dropna()
    return df_weekly

def detect_w_bottom_breakout(df_weekly):
    """
    휴리스틱 설명:
    - lookback = 최근 LOOKBACK_WEEKS 주
    - 두 개의 trough(저점)를 첫/두번째 반구간에서 각각 찾음
    - 두 저점(low1, low2)가 서로 비슷(예: 8% 이내)하고, 두 저점 사이에 peak(중간봉의 고점)이 존재하면
      그 peak의 high를 돌파레벨로 간주.
    - 현재(마지막) 주의 close가 돌파레벨보다 MIN_CLOSE_ABOVE_BREAK 이상 높고,
      volume 조건, 장대양봉 조건을 만족하면 True 반환
    """
    if len(df_weekly) < LOOKBACK_WEEKS + 1:
        return False, None

    recent = df_weekly.iloc[-(LOOKBACK_WEEKS+1):-1]  # 마지막 주(현재주 일전까지) 분석용
    current = df_weekly.iloc[-1]  # 이번 주(검사 대상, 금요일 마감후의 주)
    # split into two halves
    mid = len(recent) // 2
    first = recent.iloc[:mid]
    second = recent.iloc[mid:]

    if first.empty or second.empty:
        return False, None

    # find troughs (lowest low)
    low1_idx = first['low'].idxmin()
    low1 = first.loc[low1_idx, 'low']
    low2_idx = second['low'].idxmin()
    low2 = second.loc[low2_idx, 'low']

    # middle peak between the two troughs (use slices between indices)
    start_idx = min(low1_idx, low2_idx)
    end_idx = max(low1_idx, low2_idx)
    between = recent.loc[start_idx:end_idx]
    if between.empty:
        return False, None
    peak_high = between['high'].max()

    # check troughs similarity (within 12% by default)
    if low1 == 0 or low2 == 0:
        return False, None
    trough_similarity = abs(low1 - low2) / max(low1, low2)
    if trough_similarity > 0.12:
        return False, None

    # breakout if current close > peak_high * (1 + MIN_CLOSE_ABOVE_BREAK)
    breakout_level = peak_high
    if current['close'] <= breakout_level * (1 + MIN_CLOSE_ABOVE_BREAK):
        return False, None

    # volume spike check: current volume >= VOLUME_MULTIPLIER * avg(volume of last WEEKLY_AVG_VOL_WINDOW weeks)
    vol_avg = recent['volume'].iloc[-WEEKLY_AVG_VOL_WINDOW:].mean() if len(recent) >= WEEKLY_AVG_VOL_WINDOW else recent['volume'].mean()
    if vol_avg == 0 or np.isnan(vol_avg):
        return False, None
    vol_ratio = current['volume'] / vol_avg
    if vol_ratio < VOLUME_MULTIPLIER:
        return False, None

    # 장대양봉 체크: 이번 주는 양봉이고 몸통 크기가 과거 평균 몸통의 BODY_MULTIPLIER 배 이상
    body = current['close'] - current['open']
    if body <= 0:
        return False, None
    # 과거 주의 몸통(절대값)
    past_bodies = (recent['close'] - recent['open']).abs()
    past_avg_body = past_bodies.mean() if len(past_bodies) > 0 else 0.0
    if past_avg_body == 0:
        # 과거 몸통이 매우 작다면, 절대 기준으로도 장대양봉 판단 (예: 종가-시가 > 2% )
        if (body / current['open']) < 0.02:
            return False, None
    else:
        if body < BODY_MULTIPLIER * past_avg_body:
            return False, None

    # 모두 통과 -> 반환
    info = {
        "breakout_level": breakout_level,
        "vol_ratio": vol_ratio,
        "body": body,
        "body_pct": body / current['open']
    }
    return True, info

def scan_all():
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * YEARS_BACK)).strftime("%Y%m%d")

    tickers = get_ticker_list()
    results = []

    for t in tqdm(tickers, desc="Scanning tickers"):
        try:
            df_daily = fetch_daily_ohlcv(t, start_date, end_date)
            if df_daily is None or df_daily.empty:
                continue
            df_weekly = make_weekly_from_daily(df_daily)
            ok, info = detect_w_bottom_breakout(df_weekly)
            if ok:
                last_week = df_weekly.iloc[-1]
                results.append({
                    "ticker": t,
                    "name": stock.get_market_ticker_name(t),
                    "date": df_weekly.index[-1].strftime("%Y-%m-%d"),
                    "close": last_week['close'],
                    "breakout_level": info['breakout_level'],
                    "vol_ratio": round(info['vol_ratio'],2),
                    "body_pct": round(info['body_pct']*100,2)
                })
        except Exception as e:
            # 개별 종목 오류는 무시하지만 로그를 남길 수 있음
            # print(f"Error {t}: {e}")
            continue

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by="vol_ratio", ascending=False)
    return df_res

if __name__ == "__main__":
    df_found = scan_all()
    if df_found.empty:
        print("조건을 만족하는 종목이 없습니다.")
    else:
        print(df_found.to_string(index=False))
