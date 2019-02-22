import sys
import json
from PyQt5.QtWidgets import *
from Kiwoom import *
import time
import pickle
from datetime import datetime
import pandas as pd
SEL_CONDITION_NAME = '스캘퍼_시가갭'
from SysStatagy import *
class PyMon:
    def __init__(self):
        self.kiwoom = Kiwoom()
        self.kiwoom.comm_connect()
        self.stratagy = SysStratagy()

    def run(self):
        # self.run_pbr_per_screener()
        # self.run_condition_data()
        # self.get_codition_stock_list()
        self.init_maedo_proc()

    def init_maedo_proc(self):
        self.check_balance()
        # Item list
        item_count = len(self.kiwoom.opw00018_output['multi'])
        if item_count == 0:
            print("보유종목이 없습니다. [",item_count,"]")
            pass
        # 한 종목에 대한 종목명, 보유량, 매입가, 현재가, 평가손익, 수익률(%)은 출력
        stratagy = SysStratagy()
        for j in range(item_count):
            row = self.kiwoom.opw00018_output['multi'][j]
            boyou_cnt   = int(row[1].replace(',',''))
            maeip_price = int(row[2].replace(',',''))
            stock_code = row[6]
            mado_price = stratagy.get_maedo_price(maeip_price,1.02)
            self.add_stock_sell_info(stock_code,mado_price,boyou_cnt)

    def check_balance(self):
        self.kiwoom.reset_opw00018_output()
        account_number = self.kiwoom.get_login_info("ACCNO")
        account_number = account_number.split(';')[0]# 첫번째 계좌번호 호출

        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 0, "2000")

        while self.kiwoom.remained_data:
            time.sleep(2)
            self.kiwoom.set_input_value("계좌번호", account_number)
            self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 2, "2000")

        # opw00001
        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.comm_rq_data("opw00001_req", "opw00001", 0, "2000")

    def add_stock_sell_info(self,code, sell_price, sell_qty):
        dm = ';'
        b_gubun = "매도"
        b_status = "매도전"
        b_price = sell_price
        b_method = "지정가"
        b_qty = sell_qty
        included = False
        code_info = self.kiwoom.get_master_code_name(code)
        mste_info = self.kiwoom.get_master_construction(code)
        stock_state = self.kiwoom.get_master_stock_state(code)
        print(code_info, mste_info, stock_state)

        f = open(self.kiwoom.sell_loc, 'rt', encoding='UTF-8')
        sell_list = f.readlines()
        f.close()

        if self.stratagy.isTimeAvalable(self.kiwoom.maesu_start_time,self.kiwoom.maesu_end_time):
            if len(sell_list) > 0:
                write_mode = 'a' # 추가
            else:
                write_mode = 'wt'

            for stock in sell_list:
                if code in stock:
                    included = True
                else:
                    included = False


            if included == False:
                f = open(self.kiwoom.sell_loc, write_mode, encoding='UTF-8')
                stock_info = b_gubun + dm + code + dm + b_method + dm + str(b_qty) + dm + str(b_price) + dm + b_status
                f.write(stock_info + '\n')
                f.close()
        else:
            f = open(self.kiwoom.sell_loc, 'wt', encoding='UTF-8')
            stock_info = b_gubun + dm + code + dm + b_method + dm + str(b_qty) + dm + str(b_price) + dm + b_status
            f.write(stock_info + '\n')
            f.close()

app = QApplication(sys.argv)
pymon = PyMon()
pymon.run()
