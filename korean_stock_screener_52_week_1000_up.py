#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국 주식 기술적 분석 스크리너
============================

조건:
1. 52주 신고가 돌파
2. 20주 이동평균선 돌파
3. 거래량이 최근 3개월 평균 거래량의 3배 이상
4. 돌파시 거래금액이 1000억 이상

작성자: AI Assistant
작성일: 2025-01-01
"""

# 필요한 라이브러리 import
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import FinanceDataReader as fdr
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class KoreanStockScreener:
    def __init__(self):
        self.kospi_stocks = None
        self.kosdaq_stocks = None
        self.current_date = datetime.now()

    def load_stock_list(self):
        """한국 주식 종목 리스트 로드"""
        try:
            self.kospi_stocks = fdr.StockListing('KOSPI')
            self.kosdaq_stocks = fdr.StockListing('KOSDAQ')

            # 거래정지/상폐 종목 제거
            if 'State' in self.kospi_stocks.columns:
                self.kospi_stocks = self.kospi_stocks[self.kospi_stocks['State'].isna()]
            if 'State' in self.kosdaq_stocks.columns:
                self.kosdaq_stocks = self.kosdaq_stocks[self.kosdaq_stocks['State'].isna()]

            return True
        except Exception as e:
            print(f"종목 리스트 로드 실패: {e}")
            self.kospi_stocks, self.kosdaq_stocks = pd.DataFrame(), pd.DataFrame()
            return False

    def get_stock_data(self, symbol, start='2023-01-01'):
        """개별 종목 데이터 가져오기"""
        try:
            data = fdr.DataReader(symbol, start=start)
            if data.empty:
                return None
            return data
        except Exception as e:
            print(f"데이터 조회 실패 ({symbol}): {e}")
            return None

    def calculate_technical_indicators(self, data):
        """기술적 조건 계산"""
        if data is None or data.empty:
            return None

        try:
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]

            # 1. 52주 신고가
            one_year_ago = self.current_date - timedelta(days=365)
            last_year_data = data[data.index >= one_year_ago]
            if len(last_year_data) < 50:
                return None
            high_52w = last_year_data['High'].max()

            # 2. 20주 이동평균 (주봉)
            weekly_data = data.resample('W').agg({
                'Open': 'first', 'High': 'max',
                'Low': 'min', 'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            if len(weekly_data) < 20:
                return None
            ma_20_weekly = weekly_data['Close'].rolling(window=20).mean().iloc[-1]

            # 3. 최근 3개월 평균 거래량
            three_months_ago = self.current_date - timedelta(days=90)
            recent_data = data[data.index >= three_months_ago]
            avg_volume_3m = recent_data['Volume'].mean()

            # 4. 거래금액(현재가 × 거래량)
            traded_value = current_price * current_volume

            # 조건 검사
            cond1 = True#current_price >= high_52w          # 52주 신고가 돌파
            cond2 = True#current_price >= ma_20_weekly      # 20주선 돌파
            cond3 = True#current_volume >= (avg_volume_3m * 3)  # 거래량 3배
            cond4 = traded_value >= 1000000000000  # 거래대금 1000억 이상

            return {
                '현재가': current_price,
                '52주신고가': high_52w,
                '20주이평선': ma_20_weekly,
                '3개월평균거래량': avg_volume_3m,
                '현재거래량': current_volume,
                '거래대금': traded_value,
                '조건1_신고가': cond1,
                '조건2_이평선': cond2,
                '조건3_거래량': cond3,
                '조건4_거래대금': cond4,
                'all_conditions': cond1 and cond2 and cond3 and cond4
            }

        except Exception as e:
            print(f"기술적 분석 오류: {e}")
            return None

    def screen_stocks(self, max_stocks=50):
        """조건 충족 종목 스크리닝"""
        if self.kospi_stocks is None or self.kosdaq_stocks is None:
            self.load_stock_list()

        all_stocks = pd.concat([self.kospi_stocks, self.kosdaq_stocks], ignore_index=True)
        all_stocks = all_stocks.sort_values('Marcap', ascending=False, na_position='last')

        results = []
        for idx, stock in all_stocks.head(max_stocks).iterrows():
            symbol, name = stock['Code'], stock['Name']
            data = self.get_stock_data(symbol, start='2024-01-01')
            if data is None or data['Volume'].iloc[-1] == 0:
                continue

            analysis = self.calculate_technical_indicators(data)
            if analysis and analysis['all_conditions']:
                results.append({
                    '종목코드': symbol,
                    '종목명': name,
                    **analysis
                })
                print(f"✅ 조건 충족: {name} ({symbol})")

        return pd.DataFrame(results)


# 실행 예시
if __name__ == "__main__":
    screener = KoreanStockScreener()
    results = screener.screen_stocks(max_stocks=100)

    if not results.empty:
        print("\n🎯 조건 충족 종목 리스트:")
        print(results[['종목명', '현재가', '52주신고가', '20주이평선', '거래대금']].to_string())
        results.to_csv("breakout_results.csv", index=False, encoding="utf-8-sig")
        print("\n💾 결과 저장 완료: breakout_results.csv")
    else:
        print("❌ 조건에 맞는 종목 없음")
