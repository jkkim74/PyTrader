import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import FinanceDataReader as fdr
from pykrx import stock
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class AdvancedStockScreener:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.screening_results = []
        self.last_trading_day = None
        self.debug_mode = True

    def get_last_trading_day(self) -> str:
        """최근 거래일 조회"""
        try:
            today = datetime.now()

            for i in range(7):
                check_date = today - timedelta(days=i)
                date_str = check_date.strftime('%Y%m%d')

                if check_date.weekday() >= 5:
                    continue

                try:
                    test_list = stock.get_market_ticker_list(date_str, market='KOSPI')
                    if len(test_list) > 0:
                        self.logger.info(f"✅ 최근 거래일: {check_date.strftime('%Y-%m-%d (%A)')}")
                        return date_str
                except:
                    continue

            return today.strftime('%Y%m%d')

        except Exception as e:
            self.logger.error(f"거래일 조회 실패: {e}")
            return datetime.now().strftime('%Y%m%d')

    async def screen_stocks(self, market='ALL') -> List[Dict]:
        """고급 주식 스크리닝 메인 함수 (병렬 처리)"""
        try:
            self.last_trading_day = self.get_last_trading_day()

            self.logger.info("종목 리스트 수집 중...")
            stock_list = self.get_stock_list(market)

            if len(stock_list) == 0:
                self.logger.error("종목 리스트가 비어있습니다.")
                return []

            self.logger.info(f"총 {len(stock_list)}개 종목을 스크리닝합니다.")

            # 디버깅: 첫 번째 종목으로 컬럼명 확인
            if self.debug_mode and len(stock_list) > 0:
                await self.debug_first_stock(stock_list[0])

            results = []
            batch_size = 10
            total_batches = (len(stock_list) + batch_size - 1) // batch_size

            for i in range(0, len(stock_list), batch_size):
                batch_num = i // batch_size + 1
                batch = stock_list[i:i + batch_size]

                tasks = [self.screen_single_stock(code) for code in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if result is not None and not isinstance(result, Exception):
                        results.append(result)

                if batch_num % 10 == 0 or batch_num == total_batches:
                    self.logger.info(f"진행률: {batch_num}/{total_batches} 배치 완료 "
                                     f"({i + len(batch)}/{len(stock_list)} 종목) - "
                                     f"조건 만족 종목: {len(results)}개")

                await asyncio.sleep(0.3)

            sorted_results = sorted(
                results,
                key=lambda x: x['smart_money_score'],
                reverse=True
            )

            self.logger.info(f"✅ 총 {len(sorted_results)}개 종목이 조건을 만족합니다.")
            return sorted_results

        except Exception as e:
            self.logger.error(f"스크리닝 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def debug_first_stock(self, stock_code: str):
        """첫 번째 종목으로 디버깅"""
        print(f"\n{'=' * 100}")
        print(f"🔍 디버깅: {stock_code} 종목의 실제 데이터 구조 확인")
        print(f"{'=' * 100}\n")

        try:
            df_volume = stock.get_market_trading_volume_by_date(
                self.last_trading_day, self.last_trading_day, stock_code
            )

            print("📊 get_market_trading_volume_by_date() 결과:")
            print(f"컬럼명: {df_volume.columns.tolist()}")
            print(f"\n데이터:\n{df_volume}\n")

        except Exception as e:
            print(f"디버깅 실패: {e}")

        print(f"{'=' * 100}\n")

    async def screen_single_stock(self, stock_code: str) -> Dict:
        """단일 종목 스크리닝"""
        try:
            end_date_str = self.last_trading_day
            end_date = datetime.strptime(end_date_str, '%Y%m%d')
            start_date = end_date - timedelta(days=400)

            price_df = fdr.DataReader(stock_code, start_date, end_date)
            if price_df is None or len(price_df) < 250:
                return None

            breakout_250d, gain_from_low, high_250d, low_250d = self.check_250d_breakout(price_df)
            if not breakout_250d:
                return None

            if gain_from_low > 200:
                return None

            volume_surge, surge_ratio, avg_value_20d, current_value = self.check_volume_surge(price_df)
            if not volume_surge:
                return None

            smart_money = self.check_smart_money(stock_code, end_date_str)
            if smart_money['total_net_buy'] <= 0:
                return None

            stock_info = self.get_stock_info(stock_code, end_date_str)

            current_price = int(price_df['Close'].iloc[-1])

            market_cap = stock_info.get('market_cap', 0)
            if market_cap > 0:
                turnover_ratio = (current_value / market_cap) * 100
            else:
                turnover_ratio = 0

            self.logger.info(f"✅ 조건 만족: {stock_info.get('name', 'Unknown')}({stock_code})")

            return {
                'code': stock_code,
                'name': stock_info.get('name', 'Unknown'),
                'current_price': current_price,
                'high_250d': int(high_250d),
                'low_250d': int(low_250d),
                'breakout': breakout_250d,
                'gain_from_low': round(gain_from_low, 2),
                'volume_surge_ratio': round(surge_ratio, 2),
                'avg_value_20d': round(avg_value_20d / 100000000, 2),
                'current_value': round(current_value / 100000000, 2),
                'market_cap': round(market_cap / 100000000, 2),
                'turnover_ratio': round(turnover_ratio, 2),
                'inst_net_buy': round(smart_money['inst_net_buy'] / 100000000, 2),
                'foreign_net_buy': round(smart_money['foreign_net_buy'] / 100000000, 2),
                'smart_money_score': smart_money['total_net_buy'],
                'inst_volume': smart_money.get('inst_volume', 0),
                'foreign_volume': smart_money.get('foreign_volume', 0),
                'trading_date': end_date_str
            }

        except Exception as e:
            return None

    def check_250d_breakout(self, df: pd.DataFrame) -> tuple:
        """250일 신고가 돌파 확인"""
        try:
            current_close = df['Close'].iloc[-1]
            high_250d = df['High'].iloc[:-1].rolling(window=250).max().iloc[-1]

            breakout = current_close > high_250d

            low_250d = df['Low'].rolling(window=250).min().iloc[-1]
            gain_from_low = (current_close - low_250d) / low_250d * 100

            return breakout, gain_from_low, high_250d, low_250d
        except Exception as e:
            return False, 0, 0, 0

    def check_volume_surge(self, df: pd.DataFrame,
                           surge_threshold: float = 2.0) -> tuple:
        """거래대금 급증 확인"""
        try:
            df = df.copy()
            df['trading_value'] = df['Volume'] * df['Close']
            df['avg_value_20d'] = df['trading_value'].rolling(window=20).mean()

            current_value = df['trading_value'].iloc[-1]
            avg_value = df['avg_value_20d'].iloc[-1]

            if avg_value == 0 or pd.isna(avg_value):
                return False, 0, 0, 0

            surge_ratio = current_value / avg_value
            volume_surge = surge_ratio >= surge_threshold

            return volume_surge, surge_ratio, avg_value, current_value
        except Exception as e:
            return False, 0, 0, 0

    def check_smart_money(self, stock_code: str, date_str: str) -> Dict:
        """기관/외국인 순매수 확인"""
        try:
            df_volume = stock.get_market_trading_volume_by_date(
                date_str, date_str, stock_code
            )

            df_price = stock.get_market_ohlcv(date_str, date_str, stock_code)

            if df_volume is None or len(df_volume) == 0 or df_price is None or len(df_price) == 0:
                return {
                    'inst_net_buy': 0,
                    'foreign_net_buy': 0,
                    'total_net_buy': 0,
                    'inst_volume': 0,
                    'foreign_volume': 0
                }

            if '종가' in df_price.columns:
                close_price = df_price['종가'].iloc[-1]
            elif 'Close' in df_price.columns:
                close_price = df_price['Close'].iloc[-1]
            else:
                close_price = df_price.iloc[-1, 3]

            last_row = df_volume.iloc[-1]

            inst_volume = 0
            foreign_volume = 0

            for col in df_volume.columns:
                if '기관합계' in col or '기관' == col:
                    val = last_row[col]
                    if not pd.isna(val):
                        inst_volume = int(val)
                        if self.debug_mode:
                            print(f"기관 컬럼 발견: {col} = {inst_volume}")

                if '외국인' in col and '기타' not in col:
                    val = last_row[col]
                    if not pd.isna(val):
                        foreign_volume += int(val)
                        if self.debug_mode:
                            print(f"외국인 컬럼 발견: {col} = {int(val)}")

                if '기타외국인' in col:
                    val = last_row[col]
                    if not pd.isna(val):
                        foreign_volume += int(val)
                        if self.debug_mode:
                            print(f"기타외국인 컬럼 발견: {col} = {int(val)}")

            inst_net_buy = float(inst_volume * close_price)
            foreign_net_buy = float(foreign_volume * close_price)

            self.debug_mode = False

            return {
                'inst_net_buy': inst_net_buy,
                'foreign_net_buy': foreign_net_buy,
                'total_net_buy': inst_net_buy + foreign_net_buy,
                'inst_volume': inst_volume,
                'foreign_volume': foreign_volume
            }

        except Exception as e:
            return {
                'inst_net_buy': 0,
                'foreign_net_buy': 0,
                'total_net_buy': 0,
                'inst_volume': 0,
                'foreign_volume': 0
            }

    def get_stock_info(self, stock_code: str, date_str: str) -> Dict:
        """종목 정보 조회"""
        try:
            ticker_name = stock.get_market_ticker_name(stock_code)

            try:
                cap_df = stock.get_market_cap(date_str, date_str, stock_code)
                if cap_df is not None and len(cap_df) > 0:
                    if '시가총액' in cap_df.columns:
                        market_cap = cap_df['시가총액'].iloc[-1]
                    else:
                        market_cap = cap_df.iloc[-1, 0]
                else:
                    market_cap = 0
            except:
                market_cap = 0

            return {
                'name': ticker_name if ticker_name else stock_code,
                'market': 'KOSPI/KOSDAQ',
                'market_cap': market_cap
            }

        except Exception as e:
            return {'name': stock_code, 'market': 'Unknown', 'market_cap': 0}

    def get_stock_list(self, market: str) -> List[str]:
        """시장별 종목 리스트 조회"""
        try:
            trade_date = self.last_trading_day

            if market == 'KOSPI':
                stock_list = stock.get_market_ticker_list(trade_date, market='KOSPI')
            elif market == 'KOSDAQ':
                stock_list = stock.get_market_ticker_list(trade_date, market='KOSDAQ')
            else:
                kospi_list = stock.get_market_ticker_list(trade_date, market='KOSPI')
                kosdaq_list = stock.get_market_ticker_list(trade_date, market='KOSDAQ')
                stock_list = kospi_list + kosdaq_list

            self.logger.info(f"총 {len(stock_list)}개 종목을 가져왔습니다.")

            return stock_list

        except Exception as e:
            self.logger.error(f"종목 리스트 조회 실패: {e}")
            import traceback
            traceback.print_exc()
            return []

    def generate_html_report(self, results: List[Dict], elapsed_time: float) -> str:
        """통합 HTML 리포트 생성"""
        if len(results) == 0:
            return """
            <html>
            <head>
                <meta charset="UTF-8">
                <title>주식 스크리닝 결과</title>
            </head>
            <body>
                <h1>⚠️ 조건을 만족하는 종목이 없습니다.</h1>
            </body>
            </html>
            """

        df = pd.DataFrame(results)

        # 날짜 포맷
        trade_date = datetime.strptime(self.last_trading_day, '%Y%m%d')
        date_str = trade_date.strftime('%Y년 %m월 %d일')

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>주식 스크리닝 결과 - {date_str}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}

        .container {{
            max-width: 98%;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}

        .header .info {{
            font-size: 1.1em;
            opacity: 0.95;
            margin-top: 10px;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}

        .stat-card .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            font-weight: 500;
        }}

        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}

        .content {{
            padding: 30px;
            overflow-x: auto;
        }}

        .table-wrapper {{
            overflow-x: auto;
            margin-top: 20px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }}

        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        th {{
            padding: 15px 8px;
            text-align: center;
            font-weight: 600;
            position: sticky;
            top: 0;
            white-space: nowrap;
            font-size: 0.85em;
            z-index: 10;
        }}

        td {{
            padding: 12px 8px;
            text-align: center;
            border-bottom: 1px solid #eee;
            white-space: nowrap;
        }}

        tbody tr:hover {{
            background: linear-gradient(to right, #f8f9fa, #e9ecef);
            cursor: pointer;
            transition: all 0.3s;
        }}

        tbody tr:nth-child(even) {{
            background: #fafbfc;
        }}

        .rank {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
        }}

        .stock-name {{
            font-weight: 600;
            color: #2c3e50;
            text-align: left;
            min-width: 120px;
        }}

        .stock-code {{
            color: #7f8c8d;
            font-size: 0.9em;
            font-family: 'Courier New', monospace;
        }}

        .positive {{
            color: #e74c3c;
            font-weight: 600;
        }}

        .negative {{
            color: #3498db;
            font-weight: 600;
        }}

        .neutral {{
            color: #95a5a6;
        }}

        .price {{
            font-weight: 600;
            color: #2c3e50;
            font-family: 'Courier New', monospace;
        }}

        .volume {{
            color: #34495e;
            font-family: 'Courier New', monospace;
        }}

        .footer {{
            background: #2c3e50;
            color: white;
            padding: 25px;
            text-align: center;
            font-size: 0.9em;
        }}

        .footer p {{
            margin: 5px 0;
        }}

        .section-title {{
            font-size: 1.8em;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            color: #667eea;
            font-weight: 600;
        }}

        /* 컬럼 그룹 헤더 스타일 */
        .group-header {{
            background: rgba(255, 255, 255, 0.2) !important;
            font-size: 0.9em;
            font-weight: 700;
            border-left: 2px solid rgba(255, 255, 255, 0.5);
        }}

        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
            tbody tr:hover {{
                background: transparent !important;
            }}
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            th, td {{
                padding: 8px 4px;
                font-size: 0.75em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 주식 스크리닝 결과</h1>
            <div class="info">
                <p><strong>기준일:</strong> {date_str}</p>
                <p><strong>스크리닝 조건:</strong> 250일 신고가 돌파 + 거래대금 2배 증가 + 세력 순매수</p>
                <p><strong>소요 시간:</strong> {elapsed_time:.1f}초 ({elapsed_time / 60:.1f}분)</p>
            </div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="label">📌 발견 종목 수</div>
                <div class="value">{len(results)}</div>
            </div>
            <div class="stat-card">
                <div class="label">📈 평균 상승률</div>
                <div class="value">{df['gain_from_low'].mean():.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="label">💹 평균 거래대금 증가</div>
                <div class="value">{df['volume_surge_ratio'].mean():.1f}배</div>
            </div>
            <div class="stat-card">
                <div class="label">💰 총 세력 순매수</div>
                <div class="value">{(df['inst_net_buy'].sum() + df['foreign_net_buy'].sum()):.0f}억</div>
            </div>
        </div>

        <div class="content">
            <h2 class="section-title">🎯 전체 스크리닝 결과</h2>

            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th rowspan="2">순위</th>
                            <th rowspan="2">종목명</th>
                            <th rowspan="2">코드</th>
                            <th colspan="5" class="group-header">가격 정보</th>
                            <th colspan="4" class="group-header">거래대금 정보</th>
                            <th colspan="4" class="group-header">수급 정보</th>
                        </tr>
                        <tr>
                            <!-- 가격 정보 -->
                            <th>현재가</th>
                            <th>250일<br>신고가</th>
                            <th>250일<br>최저가</th>
                            <th>250일<br>상승률</th>
                            <th>거래대금<br>증가</th>

                            <!-- 거래대금 정보 -->
                            <th>당일<br>거래대금</th>
                            <th>20일<br>평균</th>
                            <th>시가총액</th>
                            <th>회전율</th>

                            <!-- 수급 정보 -->
                            <th>기관<br>순매수(억)</th>
                            <th>외국인<br>순매수(억)</th>
                            <th>기관<br>거래량</th>
                            <th>외국인<br>거래량</th>
                        </tr>
                    </thead>
                    <tbody>
"""

        for i, row in df.head(50).iterrows():
            inst_class = 'positive' if row['inst_net_buy'] > 0 else 'negative' if row['inst_net_buy'] < 0 else 'neutral'
            foreign_class = 'positive' if row['foreign_net_buy'] > 0 else 'negative' if row[
                                                                                            'foreign_net_buy'] < 0 else 'neutral'
            turnover_class = 'positive' if row['turnover_ratio'] > 10 else 'neutral'

            html += f"""
                        <tr>
                            <td class="rank">{i + 1}</td>
                            <td class="stock-name">{row['name']}</td>
                            <td class="stock-code">{row['code']}</td>

                            <!-- 가격 정보 -->
                            <td class="price">{row['current_price']:,}</td>
                            <td class="price">{row['high_250d']:,}</td>
                            <td class="price">{row['low_250d']:,}</td>
                            <td class="positive">{row['gain_from_low']:.1f}%</td>
                            <td class="positive">{row['volume_surge_ratio']:.1f}배</td>

                            <!-- 거래대금 정보 -->
                            <td class="volume">{row['current_value']:.1f}억</td>
                            <td class="volume">{row['avg_value_20d']:.1f}억</td>
                            <td class="volume">{row['market_cap']:,.0f}억</td>
                            <td class="{turnover_class}">{row['turnover_ratio']:.2f}%</td>

                            <!-- 수급 정보 -->
                            <td class="{inst_class}">{row['inst_net_buy']:.1f}</td>
                            <td class="{foreign_class}">{row['foreign_net_buy']:.1f}</td>
                            <td class="volume">{row['inst_volume']:,}</td>
                            <td class="volume">{row['foreign_volume']:,}</td>
                        </tr>
"""

        html += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="footer">
            <p><strong>Generated by Advanced Stock Screener</strong></p>
            <p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="margin-top: 10px; font-size: 0.85em; opacity: 0.8;">
                ⚠️ 이 정보는 투자 참고용이며, 투자 결과에 대한 책임은 투자자 본인에게 있습니다.
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html


# 사용 예시
async def main():
    screener = AdvancedStockScreener()

    print(f"\n{'=' * 100}")
    print(f"{'주식 스크리닝 시작':^100}")
    print(f"{'=' * 100}\n")

    start_time = datetime.now()
    results = await screener.screen_stocks(market='ALL')
    end_time = datetime.now()

    elapsed_time = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 100}")
    print(f"스크리닝 완료 - 소요 시간: {elapsed_time:.1f}초 ({elapsed_time / 60:.1f}분)")
    print(f"기준일: {screener.last_trading_day}")
    print(f"총 {len(results)}개 종목 발견")
    print(f"{'=' * 100}\n")

    if len(results) > 0:
        # HTML 리포트 생성
        html_content = screener.generate_html_report(results, elapsed_time)

        # HTML 파일 저장
        html_file = f"stock_screening_{screener.last_trading_day}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ HTML 리포트가 '{html_file}' 파일로 저장되었습니다.")

        # CSV 파일도 저장
        df_results = pd.DataFrame(results)
        csv_file = f"stock_screening_{screener.last_trading_day}.csv"

        columns_order = [
            'code', 'name', 'current_price', 'high_250d', 'low_250d',
            'gain_from_low', 'volume_surge_ratio',
            'current_value', 'avg_value_20d', 'market_cap', 'turnover_ratio',
            'inst_net_buy', 'foreign_net_buy',
            'inst_volume', 'foreign_volume',
            'trading_date'
        ]

        df_results[columns_order].to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 파일이 '{csv_file}' 파일로 저장되었습니다.")

        # 브라우저에서 열기
        import webbrowser
        import os
        webbrowser.open('file://' + os.path.realpath(html_file))
        print(f"\n🌐 브라우저에서 HTML 리포트를 여는 중...")

    else:
        print("⚠️ 조건을 만족하는 종목이 없습니다.")


if __name__ == "__main__":
    asyncio.run(main())