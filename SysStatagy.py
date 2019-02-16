#!/usr/bin/env python
# -*- coding: utf-8 -*-
import FinanceDataReader as fdr
from datetime import datetime
import util
class SysStratagy():

    def get_maedo_price(self, price, rate):
        s_price = int(price * rate)
        if (1000 <= s_price < 5000):
            r_price = round(s_price, -1) + 5
        elif (5000 <= s_price < 10000):
            dif = s_price % 5
            r_price = s_price - dif
        elif (10000 <= s_price < 50000):
            r_price = round(s_price, -2)
        elif (50000 <= s_price < 100000):
            r_price = round(s_price, -2)
        elif (100000 <= s_price < 500000):
            dif = s_price % 500
            r_price = s_price - dif
        elif (s_price >= 500000):
            r_price = round(s_price, -3)
        else:
            r_price = s_price
        return r_price

    def isBuyStockAvailable(self, buy_stock_code, code_nm , cur_price, start_price, s_year_date):
        print('----------------------------------------------------------------------')
        # 금일날짜
        today = datetime.today().strftime("%Y%m%d")
        today_f = datetime.today().strftime("%Y%m%d")
        prev_bus_day = util.get_prev_date(1, 2, today)
        if prev_bus_day == None:
            print('매수일이 아닙니다.')
            prev_bus_day = util.get_prev_date(1, 2, str(int(today) - 1))
            if prev_bus_day == None:
                prev_bus_day = util.get_prev_date(1, 2, str(int(today) - 2))
        s_standard_date = prev_bus_day[1]
        e_standard_date = prev_bus_day[0]
        # 대상종목의 매수가 산정을 위한 가격데이타 수집
        df = fdr.DataReader(buy_stock_code, s_year_date)
        print('* 종목코드 : ', buy_stock_code, ', 종목명 : ',code_nm)
        print('* 5%이상상승당일 종가 : ', df['Close'][s_standard_date], '시가갭날 시가 : ', df['Open'][e_standard_date],
              '시가갭날 종가 : ',
              df['Close'][e_standard_date])  # 매수전날 시가

        # 매수가능 구간 가격 조회
        s_buy_close_price_t = df['Close'][s_standard_date]
        e_buy_open_price_t = df['Open'][e_standard_date]
        e_buy_close_price_t = df['Close'][e_standard_date]
        if (e_buy_open_price_t > e_buy_close_price_t):
            e_buy_price = int(e_buy_close_price_t)
        else:
            e_buy_price = int(e_buy_open_price_t)

        if (s_buy_close_price_t > e_buy_price):
            s_buy_price = int(e_buy_price)
            e_buy_price = int(s_buy_close_price_t)
        else:
            s_buy_price = int(s_buy_close_price_t)
            e_buy_price = int(e_buy_price)

        if start_price[0] == '-' or start_price[0] == '+':
            start_price = start_price[1:]

        if s_buy_price > int(start_price):
            print("금일 시가가 매수시작가보다 낮아 매수 불가합니다.")
            return False
        if cur_price[0] == '-' or cur_price[0] == '+':
            cur_price = cur_price[1:]
        if (e_buy_price >= int(cur_price) >= s_buy_price):
            bBuyStock = True
        else:
            bBuyStock = False
        print('* 매수범위 : ', s_buy_price, ' ~ ', e_buy_price, ' 현재가 : ', int(cur_price), ' 시가 : ', start_price,
              "매수가능여부 :",
              bBuyStock)  # 매수전날 시가
        return bBuyStock

    def isTimeAvalable(self, maesu_start_time, maesu_end_time):
        now_time = int(datetime.now().strftime('%H%M%S'))
        if (maesu_end_time >= now_time >= maesu_start_time):
            return True
        else:
            print("매수가능한 시간이 아닙니다.", maesu_start_time, "~", maesu_end_time, " : ", now_time)
            return False

    # 매수가 및 매수수량 구해오는 메소드
    def get_buy_num_price(self, total_buy_money, high_price, cur_price):
        high_price = int(high_price)
        if cur_price[0] == '-' or cur_price[0] == '+':
            buy_price = int(cur_price[1:])
        else:
            buy_price = int(cur_price)
        num = int(total_buy_money / buy_price)
        price = high_price
        return (num, price)






