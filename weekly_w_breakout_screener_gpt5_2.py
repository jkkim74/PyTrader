# -*- coding: utf-8 -*-
"""
Weekly W Breakout Screener (with Entry/Stop/Targets)
- 20주선 위 W형 전고 돌파/대기/되돌림 스캐너
- KRX: FinanceDataReader/pykrx 우선, 해외/폴백: yfinance
- 진행바(tqdm), CSV/HTML 저장, 이름 매핑, 레이트리밋/재시도/타임아웃
- 적정 매수가(entry), 손절가(stop), 목표가(target1/2), 손익비(rr1) 계산 포함

Author: gpt-5
"""

from __future__ import annotations

import socket, time, random, threading, logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

# Optional providers
try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None

try:
    from pykrx import stock as krx
except Exception:
    krx = None

import yfinance as yf

# tqdm (progress bar)
try:
    from tqdm.auto import tqdm
except Exception:
    class tqdm:  # fallback dummy
        def __init__(self, total=None, desc="", unit=""): self.n=0
        def update(self, n=1): self.n += n
        def set_postfix(self, **kw): pass
        def close(self): pass

# ----------------------
# Networking safety
# ----------------------
socket.setdefaulttimeout(10)  # avoid infinite waits

RATE_LIMIT_PER_SEC = 2.0  # KRX는 1.5~3 권장
_rate_lock = threading.Lock()
_last_call = 0.0
def rate_limit():
    global _last_call
    with _rate_lock:
        now = time.perf_counter()
        min_gap = 1.0 / max(1e-9, RATE_LIMIT_PER_SEC)
        wait = min_gap - (now - _last_call)
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.05))
        _last_call = time.perf_counter()

def _retry(func, tries=3, base_sleep=0.6, max_sleep=3.0):
    last_err = None
    for i in range(tries):
        try:
            return func()
        except Exception as e:
            last_err = e
            time.sleep(min(max_sleep, base_sleep * (2 ** i)) + random.uniform(0, 0.2))
    raise last_err

# ----------------------
# Logging
# ----------------------
def setup_logger(level=logging.INFO):
    logger = logging.getLogger("scanner")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger
logger = setup_logger(logging.INFO)

# ----------------------
# Data utils
# ----------------------
def _standardize_ohlcv(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=colmap).copy()
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    df = df[keep]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna()
    return df

def fetch_daily_any(ticker: str, start="2015-01-01", end=None, quiet=True) -> pd.DataFrame:
    """
    - KRX: '005930.KS' / '035420.KQ' / '005930'
    - Global: 'AAPL', 'NVDA' (yfinance)
    """
    is_krx = ticker.endswith(".KS") or ticker.endswith(".KQ") or ticker.isdigit()
    code = ticker.split(".")[0] if is_krx else ticker
    errs = []

    if is_krx and fdr is not None:
        try:
            def _fdr():
                rate_limit()
                df = fdr.DataReader(code, start, end)
                return _standardize_ohlcv(df, {
                    "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
                })
            df = _retry(_fdr)
            if not df.empty:
                return df
        except Exception as e:
            errs.append(f"FDR:{e}")

    if is_krx and krx is not None:
        try:
            def _pykrx():
                rate_limit()
                s = (pd.Timestamp(start).strftime("%Y%m%d") if start else "19900101")
                e = (pd.Timestamp(end).strftime("%Y%m%d") if end else pd.Timestamp.today().strftime("%Y%m%d"))
                df = krx.get_market_ohlcv_by_date(s, e, code)
                if df.empty: return pd.DataFrame()
                df.columns = [c.strip() for c in df.columns]
                return _standardize_ohlcv(df, {
                    "시가":"open","고가":"high","저가":"low","종가":"close","거래량":"volume"
                })
            df = _retry(_pykrx)
            if not df.empty:
                return df
        except Exception as e:
            errs.append(f"pykrx:{e}")

    try:
        def _yf():
            rate_limit()
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
            return _standardize_ohlcv(df, {
                "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
            })
        df = _retry(_yf)
        if not df.empty:
            return df
    except Exception as e:
        errs.append(f"yfinance:{e}")

    if not quiet:
        logger.warning(f"[DATA-EMPTY] {ticker} -> {' | '.join(errs)}")
    return pd.DataFrame()

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """0/음수 가격행 제거, 인덱스 중복 제거"""
    if df.empty: return df
    df = df.copy()
    df = df[~df.index.duplicated(keep="last")]
    pos = (df[["open","high","low","close"]] > 0).all(axis=1)
    df = df[pos]
    df = df[df["high"] >= df["low"]]
    return df

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
    w = w[(w[["open","high","low","close"]] > 0).all(axis=1)]
    if not include_partial_week and len(w) > 0:
        now = pd.Timestamp.now(tz=tz)
        if now.weekday() <= 4:
            fri = now.normalize() + pd.Timedelta(days=(4 - now.weekday()))
        else:
            fri = now.normalize() - pd.Timedelta(days=(now.weekday() - 4))
        cutoff = fri + pd.Timedelta(hours=market_close_hour, minutes=market_close_min)
        if now < cutoff:
            w = w.iloc[:-1]
    return w

def get_krx_universe(markets=("KOSPI","KOSDAQ"), limit=None) -> List[str]:
    if fdr is None:
        raise RuntimeError("FinanceDataReader 필요: pip install FinanceDataReader")
    krx_df = fdr.StockListing("KRX")
    krx_df = krx_df[krx_df["Market"].isin(markets)].copy()
    if "DelistingDate" in krx_df.columns:
        krx_df = krx_df[krx_df["DelistingDate"].isna()]
    if "Type" in krx_df.columns:
        krx_df = krx_df[krx_df["Type"].isin(["Stock","Common Stock","보통주"])]

    def suff(row):
        return ".KS" if row["Market"] == "KOSPI" else ".KQ"
    tickers = (krx_df["Code"] + krx_df.apply(suff, axis=1)).tolist()
    if limit:
        tickers = tickers[:limit]
    return tickers

def get_name_map_krx(markets=("KOSPI","KOSDAQ")) -> Dict[str, str]:
    if fdr is None:
        return {}
    df = fdr.StockListing("KRX")[["Code","Name","Market"]]
    df = df[df["Market"].isin(markets)]
    if "DelistingDate" in df.columns:
        df = df[df["DelistingDate"].isna()]
    mp = {}
    for _, r in df.iterrows():
        suffix = ".KS" if r["Market"] == "KOSPI" else ".KQ"
        mp[r["Code"] + suffix] = r["Name"]
        mp[r["Code"]] = r["Name"]
    return mp

# ----------------------
# Indicators
# ----------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).rolling(n).mean()

def add_indicators(w: pd.DataFrame) -> pd.DataFrame:
    out = w.copy()
    out["sma5"] = sma(out["close"], 5)
    out["sma10"] = sma(out["close"], 10)
    out["sma20"] = sma(out["close"], 20)
    out["atr14"] = atr(out, 14)
    out["avgvol10"] = out["volume"].rolling(10).mean()
    out["sma20_slope3"] = out["sma20"] - out["sma20"].shift(3)
    return out

def has_vol_contraction(w: pd.DataFrame, lookback: int = 6) -> bool:
    if len(w) < 30:
        return False
    atr_now = w["atr14"].iloc[-1]
    atr_prev = w["atr14"].shift(lookback).iloc[-1]
    vol3 = w["volume"].rolling(3).mean().iloc[-1]
    vol10 = w["volume"].rolling(10).mean().iloc[-1]
    if np.isnan([atr_now, atr_prev, vol3, vol10]).any():
        return False
    return (atr_now < atr_prev * 0.9) and (vol3 < vol10)

# ----------------------
# Swing points (local extremes)
# ----------------------
@dataclass
class Pivot:
    idx: int
    date: pd.Timestamp
    price: float
    kind: str  # "H" | "L"

def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[Pivot]:
    pivots: List[Pivot] = []
    for i in range(left, len(df) - right):
        w = df.iloc[i-left:i+right+1]
        if df["low"].iloc[i] == w["low"].min():
            pivots.append(Pivot(i, df.index[i], float(df["low"].iloc[i]), "L"))
        if df["high"].iloc[i] == w["high"].max():
            pivots.append(Pivot(i, df.index[i], float(df["high"].iloc[i]), "H"))
    pivots.sort(key=lambda p: p.idx)
    cleaned: List[Pivot] = []
    for p in pivots:
        if not cleaned:
            cleaned.append(p); continue
        last = cleaned[-1]
        if p.idx == last.idx:
            if p.kind == last.kind:
                if (p.kind == "H" and p.price >= last.price) or (p.kind == "L" and p.price <= last.price):
                    cleaned[-1] = p
            else:
                if last.kind == "L" and p.kind == "H":
                    cleaned.append(p)
                elif last.kind == "H" and p.kind == "L":
                    cleaned[-1] = p
                    cleaned.append(last)
            continue
        if p.kind == last.kind:
            if (p.kind == "H" and p.price >= last.price) or (p.kind == "L" and p.price <= last.price):
                cleaned[-1] = p
        else:
            cleaned.append(p)
    return cleaned

def filter_pivots_min_move(pivots: List[Pivot], min_move_pct: float = 0.10) -> List[Pivot]:
    pivots = [p for p in pivots if p.price and p.price > 0]
    if len(pivots) < 2:
        return pivots
    res: List[Pivot] = [pivots[0]]
    for p in pivots[1:]:
        last = res[-1]
        if p.kind == last.kind:
            if (p.kind == "H" and p.price >= last.price) or (p.kind == "L" and p.price <= last.price):
                res[-1] = p
            continue
        change = abs(p.price - last.price) / max(last.price, 1e-9)
        if np.isnan(change) or np.isinf(change): continue
        if change < min_move_pct: continue
        res.append(p)
    return res

# ----------------------
# W Pattern detection
# ----------------------
@dataclass
class WPattern:
    L1: Pivot
    H: Pivot
    L2: Pivot
    neckline: float
    depth_pct: float
    length_weeks: int
    valid: bool

def find_recent_W(pivots: List[Pivot],
                  w: pd.DataFrame,
                  lookback_weeks: int = 60,
                  min_depth: float = 0.10,
                  max_depth: float = 0.35,
                  l2_v_l1_min: float = 0.97,
                  min_span: int = 6,
                  max_span: int = 30) -> Optional[WPattern]:
    if len(pivots) < 3: return None
    last_cut = max(0, len(w) - lookback_weeks)
    pivots = [p for p in pivots if p.idx >= last_cut]
    n = len(pivots)
    if n < 3: return None

    for i in range(n - 3, -1, -1):
        a, b, c = pivots[i], pivots[i+1], pivots[i+2]
        if a.kind == "L" and b.kind == "H" and c.kind == "L":
            H = b.price
            if H <= 0: continue
            mL = min(a.price, c.price)
            depth = (H - mL) / H
            span = c.idx - a.idx
            if not (min_span <= span <= max_span): continue
            if not (min_depth <= depth <= max_depth): continue
            if c.price < a.price * l2_v_l1_min: continue
            return WPattern(L1=a, H=b, L2=c, neckline=H, depth_pct=depth,
                            length_weeks=span, valid=True)
    return None

# ----------------------
# Triggers & Scoring
# ----------------------
@dataclass
class Signal:
    ticker: str
    status: str   # "pre_breakout" | "breakout" | "pullback"
    date: pd.Timestamp
    close: float
    neckline: float
    dist_to_H_pct: float  # signed (%)
    vol_ratio: float
    sma20: float
    sma20_slope3: float
    atr14: float          # weekly ATR14 (for info)
    extra: Dict

def check_triggers(ticker: str, w: pd.DataFrame, wp: WPattern,
                   near_pct: float = 0.03,
                   breakout_min_pct: float = 0.02,
                   breakout_atr_mult: float = 0.5,
                   vol_mult: float = 1.5) -> Optional[Signal]:
    last = w.iloc[-1]
    H = wp.neckline
    close = float(last["close"])
    low = float(last["low"])
    high = float(last["high"])
    open_ = float(last["open"])
    atr_ = float(last["atr14"])
    avgvol10 = float(last["avgvol10"])
    vol = float(last["volume"])
    sma20 = float(last["sma20"])
    slope = float(last["sma20_slope3"])
    if np.isnan([atr_, avgvol10, sma20, slope]).any():
        return None

    if not (close > sma20 and slope > 0):
        return None

    dist_signed_pct = (close - H) / H * 100.0
    vol_ratio = vol / avgvol10 if avgvol10 > 0 else np.nan

    if close < H and (H - close) / H <= near_pct and has_vol_contraction(w):
        return Signal(ticker, "pre_breakout", w.index[-1], close, H,
                      dist_to_H_pct=float(dist_signed_pct),
                      vol_ratio=float(vol_ratio), sma20=sma20, sma20_slope3=slope,
                      atr14=atr_, extra={"note": "vol_contraction"})

    breakout_buffer = max(breakout_atr_mult * atr_, breakout_min_pct * H)
    if close >= H + breakout_buffer and vol_ratio >= vol_mult:
        return Signal(ticker, "breakout", w.index[-1], close, H,
                      dist_to_H_pct=float(dist_signed_pct),
                      vol_ratio=float(vol_ratio), sma20=sma20, sma20_slope3=slope,
                      atr14=atr_, extra={"buffer": breakout_buffer})

    is_bullish = close > open_ and close >= (low + (high - low) * 0.5)
    if (low <= H + atr_) and (close > sma20) and is_bullish:
        return Signal(ticker, "pullback", w.index[-1], close, H,
                      dist_to_H_pct=float(dist_signed_pct),
                      vol_ratio=float(vol_ratio), sma20=sma20, sma20_slope3=slope,
                      atr14=atr_, extra={"note": "H+ATR retest"})

    return None

def score_signal(sig: Signal) -> float:
    score = 0.0
    if sig.status == "pre_breakout":
        score += 2.5 - min(2.5, abs(sig.dist_to_H_pct) / 1.0)
    elif sig.status == "breakout":
        score += 3.0
    elif sig.status == "pullback":
        score += 2.0
    score += 1.0 if sig.sma20_slope3 > 0 else 0.0
    score += min(3.0, max(0.0, sig.vol_ratio - 0.5))
    return float(max(0.0, score))

# ----------------------
# Trade plan (entry/stop/targets)
# ----------------------
def last_swing_low_daily(d: pd.DataFrame, lookback=30, left=2, right=2) -> Optional[float]:
    piv = swing_points(d, left=left, right=right)
    cut = max(0, len(d) - lookback)
    cands = [p for p in piv if p.kind == "L" and p.idx >= cut]
    if cands:
        return float(cands[-1].price)
    if len(d) >= 5:
        return float(d["low"].iloc[-min(lookback, len(d)):].min())
    return None

def compute_trade_plan(d_daily: pd.DataFrame, w_weekly: pd.DataFrame, wp: WPattern, sig: Signal) -> Dict:
    atr14d = true_range(d_daily).rolling(14).mean().iloc[-1]
    H = float(wp.neckline)
    depth = float(H - min(wp.L1.price, wp.L2.price))
    target1 = H + depth * 1.0
    target2 = H + depth * 1.5

    pct1 = 0.01 * H  # 1%
    entry = None
    stop = None
    entry_low = None
    entry_high = None

    if sig.status == "breakout":
        buffer = max(0.25 * atr14d, pct1)
        entry = H + buffer
        stop = H - max(1.0 * atr14d, pct1)
    elif sig.status == "pre_breakout":
        buffer = max(0.25 * atr14d, pct1)
        entry = H + buffer
        swl = last_swing_low_daily(d_daily, lookback=30, left=2, right=2)
        atr_stop = H - max(1.0 * atr14d, pct1)
        stop = max(float(swl) if swl is not None else atr_stop, atr_stop)
    else:  # pullback
        entry = H
        entry_low = H - 0.5 * atr14d
        entry_high = H + 0.5 * atr14d
        stop = H - max(1.0 * atr14d, pct1)

    if entry is not None and stop is not None and stop >= entry:
        stop = entry * 0.99  # guard

    rr1 = None
    if entry and stop and (entry - stop) > 1e-9:
        rr1 = (target1 - entry) / (entry - stop)

    return {
        "atr14d": float(atr14d),
        "entry": float(entry) if entry is not None else None,
        "stop": float(stop) if stop is not None else None,
        "target1": float(target1),
        "target2": float(target2),
        "entry_low": float(entry_low) if entry_low is not None else None,
        "entry_high": float(entry_high) if entry_high is not None else None,
        "rr1": float(rr1) if rr1 is not None else None
    }

# ----------------------
# Analyze per ticker
# ----------------------
def analyze_ticker(ticker: str,
                   start="2015-01-01",
                   lookback_weeks: int = 60,
                   swing_left_right: int = 3,
                   min_move_pct: float = 0.10,
                   include_partial_week=True) -> Tuple[Optional[Signal], Optional[WPattern]]:
    d = fetch_daily_any(ticker, start=start, end=None, quiet=True)
    if d.empty or len(d) < 200:
        return None, None
    d = clean_ohlcv(d)
    if d.empty or len(d) < 200:
        return None, None

    w = to_weekly(d, include_partial_week=include_partial_week)
    if w.empty or len(w) < 40:
        return None, None
    w = add_indicators(w)

    piv = swing_points(w, left=swing_left_right, right=swing_left_right)
    piv = filter_pivots_min_move(piv, min_move_pct=min_move_pct)
    if len(piv) < 3:
        return None, None

    wp = find_recent_W(piv, w, lookback_weeks=lookback_weeks)
    if not wp or not wp.valid:
        return None, None

    sig = check_triggers(ticker, w, wp)
    if not sig:
        return None, None

    plan = compute_trade_plan(d, w, wp, sig)
    sig.extra.update(plan)
    return sig, wp

# ----------------------
# HTML rendering (manual, 링크 활성/가독성 개선)
# ----------------------
from datetime import datetime

def quote_link(ticker: str) -> str:
    if ticker.endswith(".KS") or ticker.endswith(".KQ") or ticker.isdigit():
        code = ticker.split(".")[0]
        return f"https://finance.naver.com/item/main.nhn?code={code}"
    return f"https://finance.yahoo.com/quote/{ticker}"

def render_html(df: pd.DataFrame, html_path: str, title="W Breakout Scanner", note=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    base_css = """
    .wrap { max-width: 1200px; margin: 10px auto; padding: 4px 8px; }
    table { width: 100%; table-layout: fixed; border-collapse: collapse; }
    th { position: sticky; top: 0; z-index: 2; background: #222; color: #fff; padding: 8px; }
    td { padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: middle; }
    tbody tr:nth-child(even) { background: #fafafa; }
    tbody tr:hover { background: #f1f5ff; }
    a { color: #1565c0; text-decoration: none; }
    a:hover { text-decoration: underline; }
    td.col-name { max-width: 220px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    td.col-ticker, td.col-status { text-align: center; white-space: nowrap; width: 110px; }
    td.num { text-align: right; white-space: nowrap; }
    .badge { display:inline-block; padding: 2px 8px; border-radius: 10px; color:#000; font-weight:600; }
    .b-breakout { background:#06d6a0; }
    .b-pre { background:#ffd166; }
    .b-pullback { background:#118ab2; color:#fff; }
    .sub { color:#666; font-size: 12px; }
    """

    if df.empty:
        html = f"""<html><head><meta charset="utf-8"><title>{title}</title>
        <style>{base_css}</style></head>
        <body><div class="wrap">
          <h2>{title}</h2><div class="sub">Updated: {now} {note}</div>
          <p>No signals.</p>
        </div></body></html>"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return

    df2 = df.copy()
    df2["name"] = df2["name"].fillna("")
    df2["종목"] = df2.apply(
        lambda r: f'<a href="{quote_link(r["ticker"])}" title="{r["name"] or r["ticker"]}" target="_blank">{r["name"] or r["ticker"]}</a>',
        axis=1
    )
    def badge(s):
        if s == "breakout": return '<span class="badge b-breakout">breakout</span>'
        if s == "pre_breakout": return '<span class="badge b-pre">pre</span>'
        return '<span class="badge b-pullback">pullback</span>'
    df2["status_badge"] = df2["status"].map(badge)

    # 표시 컬럼
    show_cols = [
        "종목","ticker","status_badge",
        "entry","stop","target1","target2","rr1",
        "score","close","neckline","dist_to_H_pct",
        "vol_ratio","sma20","sma20_slope3","atr14","date"
    ]
    show_cols = [c for c in show_cols if c in df2.columns]
    df2 = df2[show_cols].rename(columns={"status_badge":"status"})

    num_cols = {"entry","stop","target1","target2","rr1",
                "score","close","neckline","dist_to_H_pct","vol_ratio","sma20","sma20_slope3","atr14"}

    # 헤더
    thead = "<tr>" + "".join(f"<th>{c}</th>" for c in df2.columns) + "</tr>"

    # 바디
    body_rows = []
    for _, r in df2.iterrows():
        tds = []
        for c, v in r.items():
            if c == "종목":
                tds.append(f'<td class="col-name">{v}</td>')
            elif c in {"ticker","status"}:
                tds.append(f'<td class="col-ticker">{v}</td>')
            elif c in num_cols:
                fmt = "{:,.2f}"
                if c in {"score","vol_ratio","rr1"}: fmt = "{:.2f}"
                if c == "sma20_slope3": fmt = "{:+.2f}"
                if c == "dist_to_H_pct": fmt = "{:+.2f}%"
                try:
                    v = fmt.format(float(v))
                except Exception:
                    pass
                tds.append(f'<td class="num">{v}</td>')
            else:
                tds.append(f"<td>{v}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "\n".join(body_rows)

    table_html = f"<table><thead>{thead}</thead><tbody>{tbody}</tbody></table>"

    html = f"""<html><head><meta charset="utf-8"><title>{title}</title>
    <style>{base_css}</style></head>
    <body>
      <div class="wrap">
        <h2 style="margin:6px 0 2px 0">{title}</h2>
        <div class="sub">Updated: {now} {note}</div>
        <div style="overflow-x:auto;">{table_html}</div>
      </div>
    </body></html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

# ----------------------
# Scanning with progress/checkpoints
# ----------------------
from concurrent.futures import ThreadPoolExecutor, as_completed

def scan_universe(tickers: List[str],
                  workers=3,
                  show_progress=True,
                  checkpoint_path: Optional[str]=None,
                  checkpoint_interval: int=200,
                  checkpoint_html_path: Optional[str]=None,
                  name_map: Optional[Dict[str,str]]=None,
                  include_partial_week=True) -> pd.DataFrame:
    rows = []
    counters = {"total": len(tickers), "done": 0, "signals": 0, "errors": 0}
    t0 = time.perf_counter()
    bar = tqdm(total=len(tickers), desc="Scanning", unit="stk") if show_progress else None

    def _task(t):
        try:
            sig, wp = analyze_ticker(t, include_partial_week=include_partial_week)
            if sig:
                score = score_signal(sig)
                ex = sig.extra or {}
                row = {
                    "ticker": t,
                    "date": sig.date,
                    "status": sig.status,
                    "score": round(score, 3),
                    "close": round(sig.close, 3),
                    "neckline": round(sig.neckline, 3),
                    "dist_to_H_pct": round(sig.dist_to_H_pct, 2),
                    "vol_ratio": round(sig.vol_ratio, 2),
                    "sma20": round(sig.sma20, 3),
                    "sma20_slope3": round(sig.sma20_slope3, 3),
                    "atr14": round(sig.atr14, 3),
                    # trade plan
                    "entry": round(ex.get("entry", float("nan")), 3),
                    "stop": round(ex.get("stop", float("nan")), 3),
                    "target1": round(ex.get("target1", float("nan")), 3),
                    "target2": round(ex.get("target2", float("nan")), 3),
                    "rr1": round(ex.get("rr1", float("nan")), 2),
                    "extra": ex
                }
                return ("ok_signal", row)
            else:
                return ("ok_no_signal", None)
        except Exception as e:
            return ("error", (t, str(e)))

    def _dump_checkpoint():
        if not checkpoint_path and not checkpoint_html_path:
            return
        df = pd.DataFrame(rows)
        if not df.empty:
            if name_map:
                df.insert(1, "name", df["ticker"].map(name_map).fillna(""))
            df = df.sort_values(["status","score"], ascending=[True, False]).reset_index(drop=True)
            if checkpoint_path:
                df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
            if checkpoint_html_path:
                render_html(df, checkpoint_html_path, note="(checkpoint)")
        else:
            if checkpoint_html_path:
                render_html(pd.DataFrame(), checkpoint_html_path, note="(checkpoint)")

    def _on_progress(kind, payload):
        counters["done"] += 1
        if kind == "ok_signal":
            counters["signals"] += 1
        elif kind == "error":
            counters["errors"] += 1
            t, msg = payload
            logger.warning(f"{t}: {msg}")
        if bar:
            bar.set_postfix(signals=counters["signals"], err=counters["errors"])
            bar.update(1)
        if (checkpoint_path or checkpoint_html_path) and counters["done"] % checkpoint_interval == 0:
            _dump_checkpoint()

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_task, t): t for t in tickers}
            for fut in as_completed(fut_map):
                kind, payload = fut.result()
                if kind == "ok_signal" and payload:
                    rows.append(payload)
                _on_progress(kind, payload)
    else:
        for t in tickers:
            kind, payload = _task(t)
            if kind == "ok_signal" and payload:
                rows.append(payload)
            _on_progress(kind, payload)

    if bar:
        bar.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        if name_map:
            df.insert(1, "name", df["ticker"].map(name_map).fillna(""))
        df = df.sort_values(["status","score"], ascending=[True, False]).reset_index(drop=True)

    elapsed = time.perf_counter() - t0
    avg = elapsed / max(1, counters["done"])
    logger.info(f"Scan done. total={counters['total']} done={counters['done']} "
                f"signals={counters['signals']} errors={counters['errors']} "
                f"time={elapsed:.1f}s avg={avg:.2f}s/stock")

    if checkpoint_path and not df.empty:
        df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
    if checkpoint_html_path:
        render_html(df, checkpoint_html_path)

    return df

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    try:
        tickers = get_krx_universe(markets=("KOSPI","KOSDAQ"))
        name_map = get_name_map_krx()
    except Exception as e:
        logger.warning(f"KRX universe failed ({e}), fallback to sample tickers.")
        tickers = ["005930.KS", "000660.KS", "035420.KS", "068270.KS",
                   "AAPL", "MSFT", "NVDA", "TSLA"]
        name_map = {}

    result = scan_universe(
        tickers=tickers,
        workers=6,                         # KRX는 2~3 권장
        show_progress=True,
        checkpoint_path="scan_progress.csv",
        checkpoint_interval=200,
        checkpoint_html_path="scan_progress.html",
        name_map=name_map,
        include_partial_week=True          # 오늘까지 진행 주봉 포함
    )

    if result.empty:
        print("No signals.")
    else:
        print(result.head(20))
        print(f"Total signals: {len(result)} -> saved to scan_progress.csv / scan_progress.html")
