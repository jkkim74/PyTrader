import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def get_latest_trading_date():
    """
    가장 최근 거래일을 찾아 반환하는 함수
    """
    today = datetime.now()

    # 최대 10일 전까지 확인
    for i in range(10):
        check_date = (today - timedelta(days=i)).strftime('%Y%m%d')
        try:
            # 대표 종목(삼성전자)으로 거래일 확인
            test_data = stock.get_market_ohlcv_by_date(check_date, check_date, '005930')
            if not test_data.empty:
                return check_date
        except:
            continue

    raise ValueError("최근 거래일을 찾을 수 없습니다.")


def calculate_change_rate(ticker, trading_date):
    """
    특정 종목의 전일 대비 상승률을 계산하는 함수
    """
    try:
        # 충분한 기간의 데이터를 가져와서 최근 2일 데이터 확보
        start_date = (datetime.strptime(trading_date, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d')
        ohlcv_data = stock.get_market_ohlcv_by_date(start_date, trading_date, ticker)

        if len(ohlcv_data) >= 2:
            current_close = ohlcv_data['종가'].iloc[-1]
            previous_close = ohlcv_data['종가'].iloc[-2]

            if previous_close != 0:
                return round(((current_close - previous_close) / previous_close) * 100, 2)

        return 0.0
    except:
        return 0.0


def detect_net_buy_column(df):
    """
    pykrx 버전에 따른 기관 순매수 컬럼명 자동 감지
    """
    possible_columns = [
        '기관합계', '기관', '순매수', '순매수금액',
        '순매수거래대금', '기관_순매수', '기관합계_순매수'
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    # 기관이나 순매수가 포함된 컬럼 찾기
    for col in df.columns:
        if '기관' in str(col) and ('순매수' in str(col) or '합계' in str(col)):
            return col
        elif '순매수' in str(col):
            return col

    raise KeyError(f"기관 순매수 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df.columns)}")


def get_institutional_top20(sort_by_trading_value=False):
    """
    기관 투자자 순매수 상위 20개 종목을 조회하는 메인 함수

    Args:
        sort_by_trading_value (bool): True시 거래대금 기준으로 정렬

    Returns:
        pandas.DataFrame: 조회 결과
    """
    print("📊 기관 투자자 순매수 상위 종목 분석을 시작합니다...")

    # 1. 최근 거래일 확인
    try:
        trading_date = "20251030"##get_latest_trading_date()
        print(f"✅ 분석 기준일: {trading_date}")
    except Exception as e:
        print(f"❌ 오류: {e}")
        return pd.DataFrame()

    # 2. 투자자별 거래 데이터 조회 (올바른 함수 사용)
    try:
        print("📈 투자자별 거래 데이터를 가져오는 중...")

        # 방법 1: get_market_trading_value_by_investor 시도
        try:
            institutional_data = stock.get_market_trading_value_by_investor(
                trading_date, trading_date, market="ALL", investor="기관"
            )

            if not institutional_data.empty:
                net_buy_col = detect_net_buy_column(institutional_data)
                print(f"✅ 기관 데이터 컬럼 '{net_buy_col}' 사용")
            else:
                raise ValueError("데이터가 비어있음")

        except Exception:
            # 방법 2: get_market_net_purchases_of_equities_by_ticker 시도
            try:
                print("   대체 방법으로 데이터 조회 중...")
                institutional_data = stock.get_market_net_purchases_of_equities_by_ticker(
                    trading_date, trading_date, investor="기관합계", market="ALL"
                )
                net_buy_col = detect_net_buy_column(institutional_data)
                print(f"✅ 기관 데이터 컬럼 '{net_buy_col}' 사용")

            except Exception:
                # 방법 3: 전체 투자자 데이터에서 기관 부분 추출
                print("   전체 투자자 데이터에서 기관 정보 추출 중...")
                all_investor_data = stock.get_market_trading_value_and_volume_by_investor(
                    trading_date, trading_date, market="ALL"
                )

                # 기관 관련 컬럼 찾기
                institutional_cols = [col for col in all_investor_data.columns if '기관' in str(col)]
                if not institutional_cols:
                    raise ValueError("기관 투자자 데이터를 찾을 수 없습니다")

                # 첫 번째 기관 컬럼 사용
                net_buy_col = institutional_cols[0]
                institutional_data = all_investor_data[[net_buy_col]].copy()
                institutional_data.rename(columns={net_buy_col: '기관순매수'}, inplace=True)
                net_buy_col = '기관순매수'
                print(f"✅ 전체 데이터에서 기관 컬럼 '{institutional_cols[0]}' 추출")

        # 3. 거래대금 데이터 조회
        print("💰 거래대금 데이터를 가져오는 중...")
        try:
            trading_value_data = stock.get_market_ohlcv_by_ticker(trading_date, market="ALL")
            trading_value_data = trading_value_data[['거래대금']].copy()
        except Exception:
            # 대체 방법
            trading_value_data = stock.get_market_trading_value_by_date(
                trading_date, trading_date, market="ALL"
            )
            trading_value_data = trading_value_data[['거래대금']].copy()

        if trading_value_data.empty:
            print("❌ 거래대금 데이터가 없습니다.")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ 데이터 조회 중 오류 발생: {e}")
        print("💡 pykrx 라이브러리 버전을 확인하거나 업데이트해보세요: pip install -U pykrx")
        return pd.DataFrame()

    # 4. 데이터 병합 및 필터링
    try:
        # 두 데이터프레임을 종목코드 기준으로 병합
        merged_data = pd.merge(
            institutional_data, trading_value_data,
            left_index=True, right_index=True, how='inner'
        )

        # 기관 순매수가 양수인 종목만 필터링
        positive_buys = merged_data[merged_data[net_buy_col] > 0].copy()

        if positive_buys.empty:
            print("❌ 기관 순매수 종목이 없습니다.")
            return pd.DataFrame()

        # 순매수 금액 기준으로 정렬하여 상위 20개 선택
        top_20 = positive_buys.sort_values(net_buy_col, ascending=False).head(20)

        print(f"📋 상위 {len(top_20)}개 종목의 상세 정보를 수집 중...")

    except Exception as e:
        print(f"❌ 데이터 처리 중 오류: {e}")
        return pd.DataFrame()

    # 5. 결과 데이터 구성
    results = []

    for rank, (ticker, row) in enumerate(top_20.iterrows(), 1):
        try:
            # 종목명 조회
            stock_name = stock.get_market_ticker_name(ticker)

            # 기본 데이터
            net_buy_amount = int(row[net_buy_col])
            trading_value = int(row['거래대금'])

            # 상승률 계산
            change_rate = calculate_change_rate(ticker, trading_date)

            results.append({
                '순위': rank,
                '종목코드': ticker,
                '종목명': stock_name,
                '기관순매수금액': net_buy_amount,
                '거래대금': trading_value,
                '상승률(%)': change_rate
            })

            # 진행률 표시
            if rank % 5 == 0:
                print(f"   진행률: {rank}/{len(top_20)} 완료")

        except Exception as e:
            print(f"⚠️  종목 {ticker} 처리 중 오류: {e}")
            continue

    # 6. DataFrame 생성
    df = pd.DataFrame(results)

    if df.empty:
        print("❌ 결과 데이터가 없습니다.")
        return pd.DataFrame()

    # 7. 거래대금 기준 정렬 옵션
    if sort_by_trading_value:
        df = df.sort_values('거래대금', ascending=False).reset_index(drop=True)
        df['순위'] = range(1, len(df) + 1)
        print("\n💰 거래대금 기준으로 재정렬되었습니다.")

    return df


def display_results(df, title="기관 투자자 순매수 상위 20개 종목"):
    """
    결과를 보기 좋게 출력하는 함수
    """
    if df.empty:
        print("표시할 데이터가 없습니다.")
        return

    print(f"\n{'=' * 80}")
    print(f"🏢 {title}")
    print(f"{'=' * 80}")

    # 금액을 억원 단위로 변환하여 표시
    display_df = df.copy()
    display_df['기관순매수금액(억원)'] = (display_df['기관순매수금액'] / 100000000).round(1)
    display_df['거래대금(억원)'] = (display_df['거래대금'] / 100000000).round(1)

    # 출력용 컬럼 선택
    output_df = display_df[['순위', '종목코드', '종목명', '기관순매수금액(억원)', '거래대금(억원)', '상승률(%)']]

    print(output_df.to_string(index=False))
    print(f"{'=' * 80}")

    # 요약 통계
    total_net_buy = display_df['기관순매수금액(억원)'].sum()
    avg_change_rate = display_df['상승률(%)'].mean()
    positive_count = len(display_df[display_df['상승률(%)'] > 0])

    print(f"\n📊 요약 통계:")
    print(f"   • 총 순매수 금액: {total_net_buy:,.1f}억원")
    print(f"   • 평균 상승률: {avg_change_rate:.2f}%")
    print(f"   • 상승 종목 수: {positive_count}개 / {len(display_df)}개")


def create_visualization(df):
    """
    상위 10개 종목의 순매수 금액을 막대 그래프로 시각화
    """
    if df.empty or len(df) == 0:
        print("시각화할 데이터가 없습니다.")
        return

    # 한글 폰트 설정
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
    except:
        try:
            plt.rcParams['font.family'] = 'AppleGothic'  # macOS
        except:
            print("⚠️  한글 폰트 설정 실패. 그래프에서 한글이 깨질 수 있습니다.")

    plt.rcParams['axes.unicode_minus'] = False

    # 상위 10개 종목 선택
    top10 = df.head(10).copy()
    top10['순매수_억원'] = top10['기관순매수금액'] / 100000000

    # 그래프 생성
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(top10)), top10['순매수_억원'],
                   color='steelblue', alpha=0.7, edgecolor='navy')

    # 그래프 설정
    plt.title('기관 투자자 순매수 상위 10개 종목', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('종목', fontsize=12)
    plt.ylabel('순매수 금액 (억원)', fontsize=12)

    # x축 레이블 (종목명)
    plt.xticks(range(len(top10)), top10['종목명'], rotation=45, ha='right')

    # 막대 위에 값 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{height:.0f}억', ha='center', va='bottom', fontsize=10)

    # 그리드 추가
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def main():
    """
    메인 실행 함수
    """
    print("🚀 한국 주식 시장 기관 투자자 분석 프로그램")
    print("=" * 60)

    try:
        # 1. 기본 조회 (기관순매수 기준)
        df_result = get_institutional_top20(sort_by_trading_value=False)

        if df_result.empty:
            print("❌ 데이터 조회에 실패했습니다.")
            return

        display_results(df_result)

        # 2. 거래대금 기준 정렬 옵션
        print("\n" + "=" * 60)
        user_input = input("거래대금 기준으로 정렬해서 보시겠습니까? (y/n): ").lower().strip()

        if user_input in ['y', 'yes', 'ㅇ']:
            df_by_trading = get_institutional_top20(sort_by_trading_value=True)
            display_results(df_by_trading, "거래대금 기준 상위 20개 종목")

        # 3. 시각화 옵션
        print("\n" + "=" * 60)
        viz_input = input("순매수 금액 상위 10개 종목을 차트로 보시겠습니까? (y/n): ").lower().strip()

        if viz_input in ['y', 'yes', 'ㅇ']:
            create_visualization(df_result)

        print("\n✅ 프로그램이 완료되었습니다!")

    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()