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

    def run(self):
        # self.run_pbr_per_screener()
        # self.run_condition_data()
        # self.get_codition_stock_list()
        self.init_maedo_proc()

    def init_maedo_proc(self):
        self.check_balance()
        # Item list
        item_count = len(self.kiwoom.opw00018_output['multi'])
        # 한 종목에 대한 종목명, 보유량, 매입가, 현재가, 평가손익, 수익률(%)은 출력
        stratagy = SysStratagy()
        for j in range(item_count):
            row = self.kiwoom.opw00018_output['multi'][j]
            boyou_cnt   = int(row[1].replace(',',''))
            maeip_price = int(row[2].replace(',',''))
            stock_code = row[6]
            mado_price = stratagy.get_maedo_price(maeip_price)
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

        f = open("sell_list.txt", 'rt', encoding='UTF-8')
        sell_list = f.readlines()
        f.close()
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
            f = open("sell_list.txt", write_mode, encoding='UTF-8')
            stock_info = b_gubun + dm + code + dm + b_method + dm + str(b_qty) + dm + str(b_price) + dm + b_status
            f.write(stock_info + '\n')
            f.close()

    def get_codition_stock_list(self):
        self.kiwoom.get_condition_load()
        # self.kiwoom.get_condition_name_list()
        self.kiwoom.send_condition("0150", SEL_CONDITION_NAME, "011", 1)
        # print(self.kiwoom.condition_code_list[:-1])
        code_list = self.kiwoom.condition_code_list[:-1]
        print("조건검색결과 주식 : ", code_list, len(code_list))
        f = open("buy_list.txt", 'wt', encoding='UTF-8')
        dm = ';'
        b_gubun = "매수"
        b_status = "매수전"
        b_price = 0
        b_method = "시장가"
        b_qty = 0
        for code in code_list:
            code_info = self.kiwoom.get_master_code_name(code)
            mste_info = self.kiwoom.get_master_construction(code)
            stock_state = self.kiwoom.get_master_stock_state(code)
            print(code_info, mste_info, stock_state)
            stock_info = b_gubun+dm+code+dm+b_method+dm+str(b_qty)+dm+str(b_price)+dm+b_status
            f.write(stock_info+'\n')
        f.close()
        if len(code_list) == 0:
            print("해당하는 조건검색의 결과 주식이 없습니다.")
            pass
        # 로직구현 필요함.
        # result = []
        # for i, code in enumerate(code_list):
        #     print("%d : %d" % (i, len(code_list)))
        #     if i > 100:
        #         break
        #
        #     (per, pbr) = self.get_per_pbr(code)
        #     if 2.5 <= per <= 10:
        #         result.append((code, per, pbr))
        #
        # data = sorted(result, key=lambda x:x[2])
        # self.dump_data(code_list)
        # TEST
        # code_list = ['066590','006920','005690']
        # self.dump_data_json(code_list)

    def run_condition_data(self):
        self.kiwoom.get_condition_load()
        #self.kiwoom.get_condition_name_list()
        self.kiwoom.send_condition("0150", "스캘퍼_시가갭", "011", 1)
        #print(self.kiwoom.condition_code_list[:-1])
        code_list = self.kiwoom.condition_code_list[:-1]
        # 금일날짜
        today = datetime.today().strftime("%Y%m%d")
        r_price = self.get_condition_param(code_list[1], today)
        print(r_price)
        # 영업일 하루전날짜
        df_hdays = pd.read_excel("data.xls")
        hdays = df_hdays['일자 및 요일'].str.extract('(\d{4}-\d{2}-\d{2})', expand=False)
        hdays = pd.to_datetime(hdays)
        hdays.name = '날짜'
        mdays = pd.date_range('2019-01-01', '2019-12-31', freq='B')
        #print(mdays)
        mdays = mdays.drop(hdays)
        #f_mdays = mdays.to_frame(index=True)
        #print(f_mdays)
        # 개장일을 index로 갖는 DataFrame
        #data = {'values': range(1, 31)}
        #df_sample = pd.DataFrame(data, index=pd.date_range('2019-01-01', '2019-01-31'))
        df_mdays = pd.DataFrame({'date':mdays})
        df_mdays_list = df_mdays['date'].tolist()
        for i, df_day in enumerate(df_mdays_list):
            if(df_day.__format__('%Y%m%d') == today):
                self.prev_bus_day_1 = df_mdays_list[i - 1].__format__('%Y-%m-%d')
                self.prev_bus_day_2 = df_mdays_list[i - 2].__format__('%Y-%m-%d')

        print(self.prev_bus_day_1,self.prev_bus_day_2)
        # 두 DataFrame (df_sample, df_mdays)의 인덱스를 기준으로 합친다(merge)
        #df = pd.merge(df_sample, df_mdays, right_index=True, left_index=True)
        #df.head(10)


    def run_pbr_per_screener(self):
        code_list = self.kiwoom.get_code_list_by_market(0) + self.kiwoom.get_code_list_by_market(10)

        # result = []
        # for i, code in enumerate(code_list):
        #     print("%d : %d" % (i, len(code_list)))
        #     if i > 100:
        #         break
        #
        #     (per, pbr) = self.get_per_pbr(code)
        #     if 2.5 <= per <= 10:
        #         result.append((code, per, pbr))
        #
        # data = sorted(result, key=lambda x:x[2])
        # self.dump_data(data[:30])
    def get_condition_param(self, code, s_date):
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.set_input_value("시작일자", s_date)
        self.kiwoom.comm_rq_data("opt10086_req", "opt10086", 0, "0101")
        return (self.kiwoom.s_price, self.kiwoom.e_price)

    def get_per_pbr(self, code):
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.comm_rq_data("opt10001_req", "opt10001", 0, "0101")
        time.sleep(0.2)
        if(self.kiwoom.per == ''):
            self.kiwoom.per = 0
        if(self.kiwoom.pbr == ''):
            self.kiwoom.pbr = 0
        print(self.kiwoom.per, self.kiwoom.pbr)
        return (float(self.kiwoom.per), float(self.kiwoom.pbr))

    def dump_data(self, data):
        f = open("./database.db", "wb")
        pickle.dump(data, f)
        f.close()
    def dump_data_json(self,data):
        with open('buy_stock.json','w',encoding='utf-8') as stock_file:
            json.dump(data, stock_file, ensure_ascii=False, indent="\t")


app = QApplication(sys.argv)
pymon = PyMon()
pymon.run()
