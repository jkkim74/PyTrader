#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국 주식 기술적 분석 스크리너
============================

조건:
1. 최근 3개월 최고가 돌파
2. 20주 이동평균선 돌파  
3. 거래량이 최근 3개월 평균 거래량의 3배 이상

작성자: AI Assistant
작성일: 2025-01-01
"""

# 필요한 라이브러리 import
import pandas as pd
import numpy as np
import yfinance as yf
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class KoreanStockScreener:
    """
    한국 주식시장 스크리너 클래스
    - 기술적 분석을 통한 종목 스크리닝
    - 최고가 돌파, 이동평균 돌파, 거래량 급증 조건 검사
    """

    def __init__(self):
        self.kospi_stocks = None
        self.kosdaq_stocks = None
        self.current_date = datetime.now()

    def load_stock_list(self):
        """한국 주식 종목 리스트 로드"""
        try:
            # KOSPI 종목 리스트
            self.kospi_stocks = fdr.StockListing('KOSPI')
            print(f"KOSPI 종목 수: {len(self.kospi_stocks)}")

            # KOSDAQ 종목 리스트  
            self.kosdaq_stocks = fdr.StockListing('KOSDAQ')
            print(f"KOSDAQ 종목 수: {len(self.kosdaq_stocks)}")

            return True
        except Exception as e:
            print(f"종목 리스트 로드 실패: {e}")
            return False

    def get_stock_data(self, symbol, period='1y'):
        """
        특정 종목의 주식 데이터 조회

        Args:
            symbol (str): 주식 종목코드
            period (str): 조회 기간 ('1y', '6mo', '3mo' 등)

        Returns:
            DataFrame: 주식 데이터 (Open, High, Low, Close, Volume)
        """
        try:
            # FinanceDataReader로 한국 주식 데이터 조회
            data = fdr.DataReader(symbol, start='2023-01-01')

            if data.empty:
                return None

            # 컬럼명 표준화
            data = data.rename(columns={
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })

            return data

        except Exception as e:
            print(f"데이터 조회 실패 ({symbol}): {e}")
            return None

    def calculate_technical_indicators(self, data):
        """
        기술적 분석 지표 계산

        Args:
            data (DataFrame): 주식 데이터

        Returns:
            dict: 분석 결과
        """
        if data is None or data.empty:
            return None

        try:
            # 최근 데이터 (현재가)
            current_price = data['Close'].iloc[-1]

            # 1. 최근 3개월 최고가 계산
            three_months_ago = self.current_date - timedelta(days=90)
            recent_data = data[data.index >= three_months_ago]

            if len(recent_data) < 30:  # 충분한 데이터가 없는 경우
                return None

            three_month_high = recent_data['High'].max()

            # 2. 20주 이동평균 계산 (주봉 기준)
            # 일봉 데이터를 주봉으로 변환
            weekly_data = data.resample('W').agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            if len(weekly_data) < 20:
                return None

            ma_20_weekly = weekly_data['Close'].rolling(window=20).mean().iloc[-1]

            # 3. 최근 3개월 평균 거래량 계산
            avg_volume_3m = recent_data['Volume'].mean()
            current_volume = data['Volume'].iloc[-1]

            # 4. 조건 검사
            condition_1 = current_price >= three_month_high  # 최고가 돌파
            condition_2 = current_price >= ma_20_weekly      # 20주봉 돌파  
            condition_3 = current_volume >= (avg_volume_3m * 3)  # 거래량 3배 이상

            return {
                'current_price': current_price,
                'three_month_high': three_month_high,
                'ma_20_weekly': ma_20_weekly,
                'avg_volume_3m': avg_volume_3m,
                'current_volume': current_volume,
                'condition_1': condition_1,
                'condition_2': condition_2, 
                'condition_3': condition_3,
                'all_conditions': condition_1 and condition_2 and condition_3
            }

        except Exception as e:
            print(f"기술적 분석 계산 오류: {e}")
            return None

    def screen_stocks(self, max_stocks=50):
        """
        조건에 맞는 주식 스크리닝

        Args:
            max_stocks (int): 검사할 최대 종목 수

        Returns:
            DataFrame: 조건에 맞는 종목 리스트
        """
        if self.kospi_stocks is None:
            self.load_stock_list()

        # 결과 저장용 리스트
        results = []

        # KOSPI + KOSDAQ 종목 합치기
        all_stocks = pd.concat([self.kospi_stocks, self.kosdaq_stocks], ignore_index=True)

        # 시가총액 기준으로 정렬하여 상위 종목부터 검사 (시가총액이 큰 종목이 더 안정적)
        all_stocks = all_stocks.sort_values('Marcap', ascending=False, na_last=True)

        print(f"총 {len(all_stocks)} 종목 중 상위 {max_stocks}개 종목을 검사합니다...")

        count = 0
        for idx, stock in all_stocks.head(max_stocks).iterrows():
            count += 1
            symbol = stock['Code']
            name = stock['Name']

            if count % 10 == 0:
                print(f"진행상황: {count}/{max_stocks} 종목 검사 완료")

            # 주식 데이터 조회
            data = self.get_stock_data(symbol)

            if data is None:
                continue

            # 기술적 분석 수행
            analysis = self.calculate_technical_indicators(data)

            if analysis is None:
                continue

            # 모든 조건을 만족하는 종목만 결과에 추가
            if analysis['all_conditions']:
                results.append({
                    '종목코드': symbol,
                    '종목명': name,
                    '현재가': analysis['current_price'],
                    '3개월최고가': analysis['three_month_high'],
                    '20주이평선': analysis['ma_20_weekly'],
                    '3개월평균거래량': analysis['avg_volume_3m'],
                    '현재거래량': analysis['current_volume'],
                    '거래량비율': analysis['current_volume'] / analysis['avg_volume_3m'],
                    '최고가돌파': analysis['condition_1'],
                    '이평선돌파': analysis['condition_2'],
                    '거래량급증': analysis['condition_3']
                })
                print(f"✅ 조건 만족 종목 발견: {name} ({symbol})")

        if results:
            result_df = pd.DataFrame(results)
            return result_df
        else:
            print("조건을 만족하는 종목이 없습니다.")
            return pd.DataFrame()

    def visualize_results(self, results_df, stock_symbol=None):
        """
        결과 시각화

        Args:
            results_df (DataFrame): 스크리닝 결과
            stock_symbol (str): 특정 종목 차트 (옵션)
        """
        if results_df.empty:
            print("시각화할 데이터가 없습니다.")
            return

        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        # 1. 결과 요약 차트
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Korean Stock Screening Results', fontsize=16, fontweight='bold')

        # 거래량 비율 분포
        axes[0,0].hist(results_df['거래량비율'], bins=10, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Volume Ratio Distribution')
        axes[0,0].set_xlabel('Volume Ratio (Current/3M Avg)')
        axes[0,0].set_ylabel('Frequency')

        # 현재가 vs 3개월 최고가
        axes[0,1].scatter(results_df['3개월최고가'], results_df['현재가'], alpha=0.7, color='orange')
        axes[0,1].plot([results_df['3개월최고가'].min(), results_df['3개월최고가'].max()],
                       [results_df['3개월최고가'].min(), results_df['3개월최고가'].max()], 
                       'r--', alpha=0.5)
        axes[0,1].set_title('Current Price vs 3-Month High')
        axes[0,1].set_xlabel('3-Month High')
        axes[0,1].set_ylabel('Current Price')

        # 현재가 vs 20주 이평선
        axes[1,0].scatter(results_df['20주이평선'], results_df['현재가'], alpha=0.7, color='green')
        axes[1,0].plot([results_df['20주이평선'].min(), results_df['20주이평선'].max()],
                       [results_df['20주이평선'].min(), results_df['20주이평선'].max()], 
                       'r--', alpha=0.5)
        axes[1,0].set_title('Current Price vs 20-Week MA')
        axes[1,0].set_xlabel('20-Week Moving Average')
        axes[1,0].set_ylabel('Current Price')

        # 거래량 비교
        axes[1,1].bar(range(len(results_df)), results_df['거래량비율'], alpha=0.7, color='purple')
        axes[1,1].axhline(y=3, color='r', linestyle='--', alpha=0.7, label='3x Threshold')
        axes[1,1].set_title('Volume Ratio by Stock')
        axes[1,1].set_xlabel('Stock Index')
        axes[1,1].set_ylabel('Volume Ratio')
        axes[1,1].legend()

        plt.tight_layout()
        plt.show()

        # 2. 특정 종목 상세 차트
        if stock_symbol and stock_symbol in results_df['종목코드'].values:
            self.plot_stock_detail(stock_symbol)

    def plot_stock_detail(self, symbol):
        """특정 종목 상세 차트"""
        data = self.get_stock_data(symbol)
        if data is None:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Stock Analysis: {symbol}', fontsize=14, fontweight='bold')

        # 가격 차트
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=1)
        ax1.plot(data.index, data['High'], alpha=0.5, label='High', linewidth=0.5)

        # 20주 이동평균선
        weekly_data = data.resample('W').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        ma_20 = weekly_data['Close'].rolling(window=20).mean()
        ax1.plot(weekly_data.index, ma_20, 'r--', label='20-Week MA', alpha=0.7)

        ax1.set_title('Price Chart')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 거래량 차트
        ax2.bar(data.index, data['Volume'], alpha=0.6, width=1)
        ax2.set_title('Volume Chart')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# 메인 실행 함수
def main():
    """
    한국 주식 스크리너 메인 실행 함수

    사용법:
    1. 스크리너 객체 생성
    2. 종목 스크리닝 실행
    3. 결과 확인 및 시각화
    """

    print("="*60)
    print("🚀 한국 주식 기술적 분석 스크리너 시작")
    print("="*60)
    print("📋 검색 조건:")
    print("  1️⃣ 최근 3개월 최고가 돌파")
    print("  2️⃣ 20주 이동평균선 돌파") 
    print("  3️⃣ 거래량이 최근 3개월 평균의 3배 이상")
    print("="*60)

    # 스크리너 객체 생성
    screener = KoreanStockScreener()

    try:
        # 주식 스크리닝 실행 (상위 50개 종목 검사)
        results = screener.screen_stocks(max_stocks=50)

        if not results.empty:
            print(f"\n🎯 총 {len(results)}개 종목이 조건을 만족합니다!")
            print("\n📊 스크리닝 결과:")
            print(results.round(2))

            # 결과를 CSV 파일로 저장
            output_file = 'korean_stock_screening_results.csv'
            results.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n💾 결과가 저장되었습니다: {output_file}")

            # 시각화 (결과가 있는 경우만)
            screener.visualize_results(results)

            # 첫 번째 종목의 상세 차트 표시 (예시)
            if len(results) > 0:
                first_stock = results.iloc[0]['종목코드']
                print(f"\n📈 첫 번째 종목 상세 분석: {first_stock}")
                screener.plot_stock_detail(first_stock)

        else:
            print("\n❌ 조건을 만족하는 종목이 없습니다.")
            print("💡 다음을 시도해보세요:")
            print("   - 검사 종목 수 늘리기 (max_stocks 파라미터)")
            print("   - 조건 완화하기") 

    except Exception as e:
        print(f"\n❌ 스크리닝 중 오류 발생: {e}")
        print("💡 인터넷 연결 및 API 접근을 확인해주세요.")

    print("\n" + "="*60)
    print("🏁 스크리닝 완료!")
    print("="*60)


# 사용 예제 함수들
def example_single_stock_analysis(symbol='005930'):  # 삼성전자
    """단일 종목 분석 예제"""
    print(f"\n🔍 단일 종목 분석 예제: {symbol}")

    screener = KoreanStockScreener()
    data = screener.get_stock_data(symbol)

    if data is not None:
        analysis = screener.calculate_technical_indicators(data)

        if analysis:
            print(f"현재가: {analysis['current_price']:,.0f}원")
            print(f"3개월 최고가: {analysis['three_month_high']:,.0f}원")
            print(f"20주 이평선: {analysis['ma_20_weekly']:,.0f}원")
            print(f"현재거래량: {analysis['current_volume']:,.0f}주")
            print(f"3개월 평균거래량: {analysis['avg_volume_3m']:,.0f}주")
            print(f"거래량 비율: {analysis['current_volume']/analysis['avg_volume_3m']:.2f}배")

            print(f"\n📋 조건 충족 여부:")
            print(f"  최고가 돌파: {'✅' if analysis['condition_1'] else '❌'}")
            print(f"  이평선 돌파: {'✅' if analysis['condition_2'] else '❌'}")
            print(f"  거래량 급증: {'✅' if analysis['condition_3'] else '❌'}")
            print(f"  전체 조건: {'✅' if analysis['all_conditions'] else '❌'}")

            # 상세 차트 표시
            screener.plot_stock_detail(symbol)
        else:
            print("❌ 기술적 분석 실패")
    else:
        print("❌ 데이터 조회 실패")


def example_custom_screening(max_stocks=100):
    """커스텀 스크리닝 예제"""
    print(f"\n🎯 커스텀 스크리닝 예제: 상위 {max_stocks}개 종목 검사")

    screener = KoreanStockScreener()
    results = screener.screen_stocks(max_stocks=max_stocks)

    if not results.empty:
        print(f"\n결과: {len(results)}개 종목 발견")

        # 거래량 비율 기준으로 정렬
        results_sorted = results.sort_values('거래량비율', ascending=False)
        print("\n🔥 거래량 급증 상위 5종목:")
        print(results_sorted[['종목명', '현재가', '거래량비율']].head().to_string())

    return results


if __name__ == "__main__":
    # 프로그램 실행
    main()

    # 또는 개별 함수 실행
    # example_single_stock_analysis('005930')  # 삼성전자 분석
    # example_custom_screening(100)  # 커스텀 스크리닝
