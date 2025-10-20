import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional
import logging
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import shutil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockScreenerConfig:
    """주식 검색 조건 상수 정의"""

    # 검색 조건 (완화된 버전)
    MIN_TRADING_VALUE = 10_000_000_000  # 100억원 (500억 → 100억)
    HIGH_PERIOD_DAYS = 200  # 200일 신고가 (250일 → 200일)
    TRADING_VALUE_MULTIPLIER = 1.5  # 20일 평균 대비 1.5배 (2.0배 → 1.5배)
    MAX_PRICE_RATIO = 3.0  # 200일 최저가 대비 300% (200% → 300%)
    AVG_TRADING_PERIOD = 20  # 평균 거래대금 계산 기간

    # 파일 경로
    STOCK_LIST_CSV = 'stock_list.csv'
    RESULT_CSV = 'screening_result.csv'

    # 디버깅 설정
    DEBUG_MODE = False  # 전체 실행으로 변경
    DEBUG_STOCK_CODES = ['005930', '000660', '035420']

    # 캐시 관리
    CLEAR_CACHE_ON_START = False  # 캐시 유지 (속도 향상)

    # 성능 설정
    USE_MULTIPROCESSING = True
    MAX_WORKERS = 5  # 워커 수 증가
    DATA_FETCH_DAYS = 400
    API_RETRY_DELAY = 0.2  # 지연 시간 감소
    PROGRESS_REPORT_INTERVAL = 100

    # 캐싱 설정
    USE_CACHE = True
    CACHE_DIR = 'cache'

    # 결과 필터링
    MIN_RESULTS_TO_SHOW = 20  # 최소 이 개수만큼 결과 표시


class StockDataManager:
    """주식 기본 데이터 관리 클래스"""

    def __init__(self, config: StockScreenerConfig):
        self.config = config
        self.stock_list_df = None

        # 캐시 디렉토리 관리
        if config.CLEAR_CACHE_ON_START and os.path.exists(config.CACHE_DIR):
            logger.info(f"기존 캐시 삭제 중: {config.CACHE_DIR}")
            shutil.rmtree(config.CACHE_DIR)

        if config.USE_CACHE and not os.path.exists(config.CACHE_DIR):
            os.makedirs(config.CACHE_DIR)

    def load_or_fetch_stock_list(self) -> pd.DataFrame:
        """CSV에서 주식 목록 로드 또는 새로 가져오기"""

        if os.path.exists(self.config.STOCK_LIST_CSV):
            logger.info(f"기존 CSV 파일에서 주식 목록 로드: {self.config.STOCK_LIST_CSV}")
            self.stock_list_df = pd.read_csv(self.config.STOCK_LIST_CSV)
            logger.info(f"총 {len(self.stock_list_df)}개 종목 로드 완료")
        else:
            logger.info("주식 목록을 새로 가져옵니다...")
            self.stock_list_df = self._fetch_stock_list()
            self.stock_list_df.to_csv(self.config.STOCK_LIST_CSV, index=False, encoding='utf-8-sig')
            logger.info(f"주식 목록 CSV 저장 완료: {len(self.stock_list_df)}개 종목")

        return self.stock_list_df

    def _fetch_stock_list(self) -> pd.DataFrame:
        """KRX에서 전체 주식 목록 가져오기"""
        try:
            from pykrx import stock
            today = datetime.now().strftime('%Y%m%d')

            logger.info("코스피 종목 목록 조회 중...")
            kospi_tickers = stock.get_market_ticker_list(today, market="KOSPI")
            logger.info(f"코스피: {len(kospi_tickers)}개")

            logger.info("코스닥 종목 목록 조회 중...")
            kosdaq_tickers = stock.get_market_ticker_list(today, market="KOSDAQ")
            logger.info(f"코스닥: {len(kosdaq_tickers)}개")

            all_tickers = kospi_tickers + kosdaq_tickers

            stock_data = []
            for idx, ticker in enumerate(all_tickers):
                if idx % 500 == 0 and idx > 0:
                    logger.info(f"  진행: {idx}/{len(all_tickers)}")

                try:
                    name = stock.get_market_ticker_name(ticker)
                    stock_data.append({'code': ticker, 'name': name})
                except Exception:
                    pass

            return pd.DataFrame(stock_data)

        except Exception as e:
            logger.error(f"주식 목록 가져오기 실패: {e}")
            return pd.DataFrame(columns=['code', 'name'])


class ProgressMonitor:
    """실시간 진행률 모니터"""

    def __init__(self, total: int):
        self.total = total
        self.processed = 0
        self.success = 0
        self.failed = 0
        self.error_reasons = {}
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update(self, success: bool = True, reason: str = None):
        with self.lock:
            self.processed += 1
            if success:
                self.success += 1
            else:
                self.failed += 1
                if reason:
                    self.error_reasons[reason] = self.error_reasons.get(reason, 0) + 1

    def get_status(self) -> str:
        with self.lock:
            elapsed = time.time() - self.start_time
            progress_pct = (self.processed / self.total * 100) if self.total > 0 else 0
            speed = self.processed / elapsed if elapsed > 0 else 0

            if self.processed > 0 and speed > 0:
                eta = (self.total - self.processed) / speed
                eta_str = self._format_time(eta)
            else:
                eta_str = "계산중"

            return (
                f"진행: {self.processed}/{self.total} ({progress_pct:.1f}%) | "
                f"성공: {self.success} | 실패: {self.failed} | "
                f"속도: {speed:.1f}개/초 | ETA: {eta_str}"
            )

    def get_error_summary(self) -> str:
        with self.lock:
            if not self.error_reasons:
                return "에러 없음"

            summary = "\n실패 이유 통계:\n"
            sorted_errors = sorted(self.error_reasons.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_errors[:10]:
                pct = (count / self.failed * 100) if self.failed > 0 else 0
                summary += f"  - {reason}: {count}건 ({pct:.1f}%)\n"
            return summary

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}초"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}분"
        else:
            return f"{seconds / 3600:.1f}시간"


class StockScreener:
    """주식 스크리닝 실행 클래스"""

    def __init__(self, config: StockScreenerConfig):
        self.config = config
        self.today = datetime.now().strftime('%Y%m%d')
        self.start_date = (datetime.now() - timedelta(days=config.DATA_FETCH_DAYS)).strftime('%Y%m%d')

    def _get_cache_path(self, code: str) -> str:
        return os.path.join(self.config.CACHE_DIR, f"{code}_{self.today}.pkl")

    def _load_from_cache(self, code: str) -> Optional[pd.DataFrame]:
        if not self.config.USE_CACHE:
            return None

        cache_path = self._get_cache_path(code)
        if os.path.exists(cache_path):
            try:
                return pd.read_pickle(cache_path)
            except Exception:
                return None
        return None

    def _save_to_cache(self, code: str, df: pd.DataFrame):
        if not self.config.USE_CACHE:
            return

        try:
            cache_path = self._get_cache_path(code)
            df.to_pickle(cache_path)
        except Exception:
            pass

    def screen_stock(self, code: str, name: str, verbose: bool = False) -> tuple[Optional[Dict], str]:
        """개별 종목 스크리닝"""

        try:
            # 캐시 확인
            df = self._load_from_cache(code)

            if df is None:
                try:
                    from pykrx import stock
                    time.sleep(self.config.API_RETRY_DELAY)
                    df = stock.get_market_ohlcv(self.start_date, self.today, code)
                except Exception as e:
                    return None, f"API_ERROR"

                if df is None or len(df) == 0:
                    return None, "NO_DATA"

                self._save_to_cache(code, df)

            # 거래대금 계산
            if '거래대금' not in df.columns:
                df['거래대금'] = df['종가'] * df['거래량']

            latest = df.iloc[-1]
            current_price = latest['종가']
            current_volume = latest['거래량']
            current_value = latest['거래대금']

            # 조건 검사
            if current_value == 0 or pd.isna(current_value):
                return None, "ZERO_TRADING_VALUE"

            if current_value < self.config.MIN_TRADING_VALUE:
                return None, "LOW_TRADING_VALUE"

            if len(df) < self.config.HIGH_PERIOD_DAYS:
                return None, "INSUFFICIENT_DATA"

            period_high = df['고가'].iloc[-self.config.HIGH_PERIOD_DAYS:].max()
            if current_price < period_high:
                return None, "NOT_NEW_HIGH"

            if len(df) < self.config.AVG_TRADING_PERIOD:
                return None, "INSUFFICIENT_AVG_DATA"

            avg_trading_value = df['거래대금'].iloc[-self.config.AVG_TRADING_PERIOD:].mean()
            if avg_trading_value == 0 or pd.isna(avg_trading_value):
                return None, "ZERO_AVG_TRADING_VALUE"

            trading_value_ratio = current_value / avg_trading_value
            if trading_value_ratio < self.config.TRADING_VALUE_MULTIPLIER:
                return None, "LOW_TRADING_RATIO"

            period_low = df['저가'].iloc[-self.config.HIGH_PERIOD_DAYS:].min()
            if period_low == 0 or pd.isna(period_low):
                return None, "ZERO_PERIOD_LOW"

            price_ratio = current_price / period_low
            if price_ratio > self.config.MAX_PRICE_RATIO:
                return None, "HIGH_PRICE_RATIO"

            # 조건 통과
            result = {
                'code': code,
                'name': name,
                'current_price': current_price,
                'current_value': current_value,
                'avg_trading_value': avg_trading_value,
                'trading_value_ratio': trading_value_ratio,
                'period_high': period_high,
                'period_low': period_low,
                'price_ratio': price_ratio,
                'volume': current_volume
            }

            return result, "SUCCESS"

        except Exception as e:
            return None, f"EXCEPTION"

    def run_screening(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """전체 종목 스크리닝 실행"""

        results = []

        # 디버그 모드
        if self.config.DEBUG_MODE:
            logger.info(f"=== 디버그 모드 ===")
            logger.info(f"테스트 종목: {', '.join(self.config.DEBUG_STOCK_CODES)}")

            for idx, code in enumerate(self.config.DEBUG_STOCK_CODES):
                stock_info = stock_list[stock_list['code'] == code]

                if len(stock_info) == 0:
                    logger.warning(f"종목 {code}를 찾을 수 없습니다.")
                    continue

                name = stock_info.iloc[0]['name']
                logger.info(f"\n{'=' * 60}")
                logger.info(f"[{idx + 1}/{len(self.config.DEBUG_STOCK_CODES)}] {code} {name}")
                logger.info(f"{'=' * 60}")

                result, reason = self.screen_stock(code, name, verbose=True)

                if result:
                    results.append(result)

            logger.info(f"\n디버그 테스트 완료: {len(results)}/{len(self.config.DEBUG_STOCK_CODES)}개 통과")
            return pd.DataFrame(results)

        # 일반 모드
        total = len(stock_list)
        logger.info(f"=== 전체 {total}개 종목 스크리닝 시작 ===")

        if self.config.USE_MULTIPROCESSING:
            logger.info(f"병렬 처리 모드: {self.config.MAX_WORKERS}개 워커")
            results = self._run_parallel(stock_list)
        else:
            logger.info("순차 처리 모드")
            results = self._run_sequential(stock_list)

        return pd.DataFrame(results)

    def _run_sequential(self, stock_list: pd.DataFrame) -> List[Dict]:
        results = []
        monitor = ProgressMonitor(len(stock_list))

        for idx, row in stock_list.iterrows():
            result, reason = self.screen_stock(row['code'], row['name'])

            if result:
                results.append(result)
                monitor.update(success=True)
                logger.info(f"✓ [{row['code']}] {row['name']}")
            else:
                monitor.update(success=False, reason=reason)

            if (idx + 1) % self.config.PROGRESS_REPORT_INTERVAL == 0:
                logger.info(monitor.get_status())

        logger.info(monitor.get_status())
        logger.info(monitor.get_error_summary())
        return results

    def _run_parallel(self, stock_list: pd.DataFrame) -> List[Dict]:
        results = []
        monitor = ProgressMonitor(len(stock_list))

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            future_to_stock = {
                executor.submit(self.screen_stock, row['code'], row['name']): (row['code'], row['name'])
                for _, row in stock_list.iterrows()
            }

            for future in as_completed(future_to_stock):
                code, name = future_to_stock[future]

                try:
                    result, reason = future.result()

                    if result:
                        results.append(result)
                        monitor.update(success=True)
                        logger.info(f"✓ [{code}] {name}")
                    else:
                        monitor.update(success=False, reason=reason)

                    if monitor.processed % self.config.PROGRESS_REPORT_INTERVAL == 0:
                        logger.info(monitor.get_status())

                except Exception as e:
                    monitor.update(success=False, reason="FUTURE_ERROR")

        logger.info(monitor.get_status())
        logger.info(monitor.get_error_summary())
        return results


def main():
    """메인 실행 함수"""

    logger.info("=" * 60)
    logger.info("주식 자동화 스크리닝 프로그램 시작")
    logger.info("=" * 60)

    config = StockScreenerConfig()

    logger.info(f"검색 조건 (완화됨):")
    logger.info(f"  - 최소 거래대금: {config.MIN_TRADING_VALUE:,}원 (100억)")
    logger.info(f"  - 신고가 기간: {config.HIGH_PERIOD_DAYS}일 (200일)")
    logger.info(f"  - 거래대금 배수: {config.TRADING_VALUE_MULTIPLIER}배 (1.5배)")
    logger.info(f"  - 최대 가격 비율: {config.MAX_PRICE_RATIO}배 (300%)")
    logger.info(f"")
    logger.info(f"성능 설정:")
    logger.info(f"  - 병렬 처리: {config.USE_MULTIPROCESSING} ({config.MAX_WORKERS}개 워커)")
    logger.info(f"  - 캐시 사용: {config.USE_CACHE}")

    try:
        start_time = time.time()

        data_manager = StockDataManager(config)
        stock_list = data_manager.load_or_fetch_stock_list()

        if len(stock_list) == 0:
            logger.error("주식 목록을 불러올 수 없습니다.")
            return

        screener = StockScreener(config)
        results_df = screener.run_screening(stock_list)

        if len(results_df) > 0:
            results_df = results_df.sort_values('trading_value_ratio', ascending=False)
            results_df.to_csv(config.RESULT_CSV, index=False, encoding='utf-8-sig')

            logger.info("=" * 60)
            logger.info(f"스크리닝 결과: {len(results_df)}개 종목 발견!")
            logger.info("=" * 60)

            # 상위 결과만 표시
            display_count = min(len(results_df), config.MIN_RESULTS_TO_SHOW)

            for idx, row in results_df.head(display_count).iterrows():
                logger.info(f"\n[{idx + 1}] {row['name']} ({row['code']})")
                logger.info(f"  현재가: {row['current_price']:,.0f}원")
                logger.info(f"  당일 거래대금: {row['current_value']:,.0f}원")
                logger.info(f"  거래대금 비율: {row['trading_value_ratio']:.2f}배")
                logger.info(f"  가격 상승률: {row['price_ratio']:.2f}배")

            if len(results_df) > display_count:
                logger.info(f"\n... 외 {len(results_df) - display_count}개 종목")

            logger.info(f"\n전체 결과 저장: {config.RESULT_CSV}")
        else:
            logger.info("\n조건을 만족하는 종목이 없습니다.")
            logger.info("추가 완화 제안:")
            logger.info("  - MIN_TRADING_VALUE를 50억으로 낮추기")
            logger.info("  - HIGH_PERIOD_DAYS를 150일로 낮추기")
            logger.info("  - TRADING_VALUE_MULTIPLIER를 1.3배로 낮추기")

        elapsed = time.time() - start_time
        logger.info(f"\n총 소요 시간: {elapsed:.2f}초")

    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}", exc_info=True)

    logger.info("=" * 60)
    logger.info("프로그램 종료")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()