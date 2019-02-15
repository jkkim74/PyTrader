import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time
import pandas as pd
import sqlite3
import datetime
import jk_util, util
from SysStatagy import *

TR_REQ_TIME_INTERVAL = 0.2

class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self._create_kiwoom_instance()
        self._set_signal_slots()
        self.jangoInfo = {}  # { 'jongmokCode': { '이익실현가': 222, ...}}
        self.michegyeolInfo = {}
        self.chegyeolInfo = {}  # { '날짜' : [ [ '주문구분', '매도', '주문/체결시간', '체결가' , '체결수량', '미체결수량'] ] }
        self.currentTime = datetime.now()

    @staticmethod
    def change_format(data):
        strip_data = data.lstrip('-0')
        if strip_data == '':
            strip_data = '0'
        try:
            format_data = format(int(strip_data), ',d')
        except ValueError:
            format_data = format(float(strip_data))
        if data.startswith('-'):
            format_data = '-' + format_data

        return format_data

    @staticmethod
    def change_format2(data):
        strip_data = data.lstrip('-0')

        if strip_data == '':
            strip_data = '0'

        if strip_data.startswith('.'):
            strip_data = '0' + strip_data

        if data.startswith('-'):
            strip_data = '-' + strip_data

        return strip_data

    def get_server_gubun(self):
        ret = self.dynamicCall("KOA_Functions(QString, QString)", "GetServerGubun", "")
        return ret

    def _create_kiwoom_instance(self):
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

    def _set_signal_slots(self):
        self.OnEventConnect.connect(self._event_connect)
        self.OnReceiveTrData.connect(self._receive_tr_data)
        self.OnReceiveChejanData.connect(self._receive_chejan_data)
        self.OnReceiveConditionVer.connect(self._receive_condition_ver)
        self.OnReceiveTrCondition.connect(self._receive_tr_condition)

    def comm_connect(self):
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def _event_connect(self, err_code):
        if err_code == 0:
            print("connected")
        else:
            print("disconnected")

        self.login_event_loop.exit()

    def get_code_list_by_market(self, market):
        code_list = self.dynamicCall("GetCodeListByMarket(QString)", market)
        code_list = code_list.split(';')
        return code_list[:-1]

    def get_master_code_name(self, code):
        code_name = self.dynamicCall("GetMasterCodeName(QString)", code)
        return code_name

    def get_master_construction(self, code):
        ret = self.dynamicCall("GetMasterConstruction(QString)", code)
        return ret

    def get_master_stock_state(self, code):
        ret = self.dynamicCall("GetMasterStockState(QString)", code)
        return ret

    def get_connect_state(self):
        ret = self.dynamicCall("GetConnectState()")
        return ret

    def set_input_value(self, id, value):
        self.dynamicCall("SetInputValue(QString, QString)", id, value)

    def comm_rq_data(self, rqname, trcode, next, screen_no):
        self.dynamicCall("CommRqData(QString, QString, int, QString)", rqname, trcode, next, screen_no)
        self.tr_event_loop = QEventLoop()
        self.tr_event_loop.exec_()

    def _comm_get_data(self, code, real_type, field_name, index, item_name):
        ret = self.dynamicCall("CommGetData(QString, QString, QString, int, QString)", code,
                               real_type, field_name, index, item_name)
        return ret.strip()

    def _get_repeat_cnt(self, trcode, rqname):
        ret = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        return ret

    def _receive_tr_data(self, screen_no, rqname, trcode, record_name, next, unused1, unused2, unused3, unused4):
        if next == '2':
            self.remained_data = True
        else:
            self.remained_data = False

        if rqname == "opt10081_req":
            self._opt10081(rqname, trcode)
        elif rqname == "opw00001_req":
            self._opw00001(rqname, trcode)
        elif rqname == "opw00018_req":
            self._opw00018(rqname, trcode)
        elif rqname == "opt10001_req":
            self._opt10001(rqname, trcode)

        try:
            self.tr_event_loop.exit()
        except AttributeError:
            pass

    def _opt10081(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)

        for i in range(data_cnt):
            date = self._comm_get_data(trcode, "", rqname, i, "일자")
            open = self._comm_get_data(trcode, "", rqname, i, "시가")
            high = self._comm_get_data(trcode, "", rqname, i, "고가")
            low = self._comm_get_data(trcode, "", rqname, i, "저가")
            close = self._comm_get_data(trcode, "", rqname, i, "현재가")
            volume = self._comm_get_data(trcode, "", rqname, i, "거래량")

            self.ohlcv['date'].append(date)
            self.ohlcv['open'].append(int(open))
            self.ohlcv['high'].append(int(high))
            self.ohlcv['low'].append(int(low))
            self.ohlcv['close'].append(int(close))
            self.ohlcv['volume'].append(int(volume))

    def _opw00001(self, rqname, trcode):
        self.d2_deposit = self._comm_get_data(trcode, "", rqname, 0, "d+2추정예수금")
        self.d2_deposit = Kiwoom.change_format(self.d2_deposit)

    def reset_opw00018_output(self):
        self.opw00018_output = {'single': [], 'multi': []}

    def _opw00018(self, rqname, trcode):
        total_purchase_price = self._comm_get_data(trcode, "", rqname, 0, "총매입금액")
        total_eval_price = self._comm_get_data(trcode, "", rqname, 0, "총평가금액")
        total_eval_profit_loss_price = self._comm_get_data(trcode, "", rqname, 0, "총평가손익금액")
        total_earning_rate = self._comm_get_data(trcode, '', rqname, 0, "총수익률(%)")
        estimated_deposit = self._comm_get_data(trcode, '', rqname, 0, "추정예탁자산")

        self.opw00018_output['single'].append(Kiwoom.change_format(total_purchase_price))
        self.opw00018_output['single'].append(Kiwoom.change_format(total_eval_price))
        self.opw00018_output['single'].append(Kiwoom.change_format(total_eval_profit_loss_price))

        # _opw00018 메서드에서 모의투자일 때는 총수익률(%)의 값을 100으로 나눈 후 출력 되도록 다음과 같이 코드 수정.
        # if self.get_server_gubun():
        #     total_earning_rate = float(total_earning_rate) / 100
        #     total_earning_rate = str(total_earning_rate)

        self.opw00018_output['single'].append(Kiwoom.change_format(total_earning_rate))
        self.opw00018_output['single'].append(Kiwoom.change_format(estimated_deposit))

        # multi data
        rows = self._get_repeat_cnt(trcode, rqname)
        for i in range(rows):
            name = self._comm_get_data(trcode, "", rqname, i, "종목명")
            quantity = self._comm_get_data(trcode, "", rqname, i, "보유수량")
            purchase_price = self._comm_get_data(trcode, "", rqname, i, "매입가")
            current_price = self._comm_get_data(trcode, "", rqname, i, "현재가")
            eval_profit_loss_price = self._comm_get_data(trcode, "", rqname, i, "평가손익")
            earning_rate = self._comm_get_data(trcode, "", rqname, i, "수익률(%)")
            code = self._comm_get_data(trcode, "", rqname, i, "종목번호")[1:]

            quantity = Kiwoom.change_format(quantity)
            purchase_price = Kiwoom.change_format(purchase_price)
            current_price = Kiwoom.change_format(current_price)
            eval_profit_loss_price = Kiwoom.change_format(eval_profit_loss_price)
            earning_rate = Kiwoom.change_format2(earning_rate)

            self.opw00018_output['multi'].append([name, quantity, purchase_price, current_price, eval_profit_loss_price, earning_rate, code])

    def _opt10001(self, rqname, trcode):
        self.cur_price  = self._comm_get_data(trcode,"", rqname, 0, "현재가")
        self.open_price = self._comm_get_data(trcode,"", rqname, 0, "시가")
        self.high_price = self._comm_get_data(trcode, "", rqname, 0, "상한가")

    def get_chejan_data(self, fid):
        ret = self.dynamicCall("GetChejanData(int)", fid)
        return ret

    def _receive_chejan_data(self, gubun, item_cnt, fid_list):
        # print(gubun)
        # print(self.get_chejan_data(9203))
        # print(self.get_chejan_data(302))
        # print(self.get_chejan_data(900))
        # print(self.get_chejan_data(901))
        print('gubun :', gubun)
        # print(util.whoami() + 'gubun: {}, itemCnt: {}, fidList: {}'
        #         .format(gubun, itemCnt, fidList))
        if (gubun == "1"):  # 잔고 정보
            print('##################### : ', gubun)
            # 잔고 정보에서는 매도/매수 구분이 되지 않음

            self.jongmok_code = self.get_chejan_data(jk_util.name_fid['종목코드'])[1:]
            self.boyou_suryang = int(self.get_chejan_data(jk_util.name_fid['보유수량']))
            self.jumun_ganeung_suryang = int(self.get_chejan_data(jk_util.name_fid['주문가능수량']))
            self.maeip_danga = int(self.get_chejan_data(jk_util.name_fid['매입단가']))
            jongmok_name = self.get_chejan_data(jk_util.name_fid['종목명']).strip()
            current_price = abs(int(self.get_chejan_data(jk_util.name_fid['현재가'])))
            print('종목코드 : ', self.jongmok_code)
            print('보유수량 : ', self.boyou_suryang)
            print('주문가능수량 : ', self.jumun_ganeung_suryang)
            print('매입단가 : ', self.maeip_danga)
            print('종목명 : ', jongmok_name)
            print('현재가 : ', current_price)
            # 미체결 수량이 있는 경우 잔고 정보 저장하지 않도록 함
            if (self.jongmok_code  in self.michegyeolInfo):
                if (self.michegyeolInfo[self.jongmok_code ]['미체결수량']):
                    return
                    # 미체결 수량이 없으므로 정보 삭제
            del (self.michegyeolInfo[self.jongmok_code ])
            if (self.boyou_suryang == 0):
                # 보유 수량이 0 인 경우 매도 수행
                if (self.jongmok_code  not in self.todayTradedCodeList):
                    self.todayTradedCodeList.append(self.jongmok_code )
                self.jangoInfo.pop(self.jongmok_code )
                # self.removeConditionOccurList(jongmok_code)
                # else:
                # 보유 수량이 늘었다는 것은 매수수행했다는 소리임
                #   self.sigBuy.emit()

                # 아래 잔고 정보의 경우 TR:계좌평가잔고내역요청 필드와 일치하게 만들어야 함
                current_jango = {}
                current_jango['보유수량'] = self.boyou_suryang
                current_jango['매매가능수량'] = self.jumun_ganeung_suryang  # TR 잔고에서 매매가능 수량 이란 이름으로 사용되므로
                current_jango['매입가'] = self.maeip_danga
                current_jango['종목번호'] = self.jongmok_code
                current_jango['종목명'] = jongmok_name.strip()
                chegyeol_info = util.cur_date_time('%Y%m%d%H%M%S') + ":" + str(current_price)

                if (self.jongmok_code  not in self.jangoInfo):
                    current_jango['주문/체결시간'] = [util.cur_date_time('%Y%m%d%H%M%S')]
                    current_jango['체결가/체결시간'] = [chegyeol_info]
                    current_jango['최근매수가'] = [current_price]
                    current_jango['매수횟수'] = 1

                    self.jangoInfo[self.jongmok_code] = current_jango

                else:
                    chegyeol_time_list = self.jangoInfo[self.jongmok_code]['주문/체결시간']
                    chegyeol_time_list.append(util.cur_date_time('%Y%m%d%H%M%S'))
                    current_jango['주문/체결시간'] = chegyeol_time_list

                    last_chegyeol_info = self.jangoInfo[self.jongmok_code]['체결가/체결시간'][-1]
                    if (int(last_chegyeol_info.split(':')[1]) != current_price):
                        chegyeol_info_list = self.jangoInfo[self.jongmok_code]['체결가/체결시간']
                        chegyeol_info_list.append(chegyeol_info)
                        current_jango['체결가/체결시간'] = chegyeol_info_list

                    price_list = self.jangoInfo[self.jongmok_code]['최근매수가']
                    last_price = price_list[-1]
                    if (last_price != current_price):
                        # 매수가 나눠져서 진행 중이므로 자료 매수횟수 업데이트 안함
                        price_list.append(current_price)
                    current_jango['최근매수가'] = price_list

                    chumae_count = self.jangoInfo[self.jongmok_code]['매수횟수']
                    if (last_price != current_price):
                        current_jango['매수횟수'] = chumae_count + 1
                    else:
                        current_jango['매수횟수'] = chumae_count

                    self.jangoInfo[self.jongmok_code].update(current_jango)

            # self.makeEtcJangoInfo(jongmok_code)
            # self.makeJangoInfoFile()
            # 미체결수량이 0이고 매수인 경우, 확정매도 처리
            if self.michegyeol_suryang == 0 and self.maedo_maesu_gubun == "2":
                sysStatagy = SysStratagy()
                sell_price = sysStatagy.get_maedo_price(self.maeip_danga)
                sell_qty = self.boyou_suryang
                self.add_stock_sell_info(self.jongmok_code, sell_price, sell_qty)

            pass
        elif (gubun == "0"):
            print('##################### : ', gubun)
            self.jumun_sangtae = self.get_chejan_data(jk_util.name_fid['주문상태'])
            self.jongmok_code = self.get_chejan_data(jk_util.name_fid['종목코드'])[1:]
            self.michegyeol_suryang = int(self.get_chejan_data(jk_util.name_fid['미체결수량']))
            self.maedo_maesu_gubun = self.get_chejan_data(jk_util.name_fid['매도매수구분'])
            if self.maedo_maesu_gubun == "1":
                print('주문상태 : ', '매도', self.jumun_sangtae)
            else:
                print('주문상태 : ', '매수', self.jumun_sangtae)
            print('종목코드 : ', self.jongmok_code)
            print('미체결수량 : ', self.michegyeol_suryang)
            # 주문 상태
            # 매수 시 접수(gubun-0) - 체결(gubun-0) - 잔고(gubun-1)
            # 매도 시 접수(gubun-0) - 잔고(gubun-1) - 체결(gubun-0) - 잔고(gubun-1)   순임
            # 미체결 수량 정보를 입력하여 잔고 정보 처리시 미체결 수량 있는 경우에 대한 처리를 하도록 함
            if (self.jongmok_code not in self.michegyeolInfo):
                self.michegyeolInfo[self.jongmok_code] = {}
            self.michegyeolInfo[self.jongmok_code]['미체결수량'] = self.michegyeol_suryang

            # if (jumun_sangtae == "체결"):
            #     self.makeChegyeolInfo(self.jongmok_code, fid_list)
            #     self.makeChegyeolInfoFile()
            #     pass

            pass



    def add_stock_sell_info(self,code, sell_price, sell_qty):
        dm = ';'
        b_gubun = "매도"
        b_status = "매도전"
        b_price = sell_price
        b_method = "지정가"
        b_qty = sell_qty
        code_info = self.get_master_code_name(code)
        mste_info = self.get_master_construction(code)
        stock_state = self.get_master_stock_state(code)
        print(code_info, mste_info, stock_state)

        f = open("sell_list.txt", 'a', encoding='UTF-8')
        stock_info = b_gubun + dm + code + dm + b_method + dm + str(b_qty) + dm + str(b_price) + dm + b_status
        f.write(stock_info + '\n')
        f.close()

    def send_order(self, rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no):
        self.dynamicCall("SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
                         [rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no])

    def get_login_info(self, tag):
        ret = self.dynamicCall("GetLoginInfo(QString)", tag)
        return ret

    def get_condition_load(self):
        self.dynamicCall("GetConditionLoad()")
        self.condition_load_loop = QEventLoop()
        self.condition_load_loop.exec_()

    def _receive_condition_ver(self, ret, msg):
        if ret == 1:
            print("조건식 저장 성공")
        else:
            print("조건식 저장 실패")

        self.condition_load_loop.exit()

    def get_condition_name_list(self):
        ret =  self.dynamicCall("GetConditionNameList()")
        print(ret)

    def send_condition(self, screen_no, condition_name, index, search):
        self.dynamicCall("SendCondition(QString, QString, int, int)",
                         screen_no, condition_name, index, search)
        self.condition_tr_loop = QEventLoop()
        self.condition_tr_loop.exec_()

    def _receive_tr_condition(self, screen_no, code_list, condition_name, index, next ):
        print(condition_name)
        self.condition_code_list = code_list.split(";")
        #print(code_list)
        self.condition_tr_loop.exit()
        #return code_list




if __name__ == "__main__":
    app = QApplication(sys.argv)
    kiwoom = Kiwoom()
    kiwoom.comm_connect()

    account_number = kiwoom.get_login_info("ACCNO")
    account_number = account_number.split(';')[0]

    kiwoom.set_input_value("계좌번호", account_number)
    kiwoom.comm_rq_data("opw00018_req", "opw00018", 0, "2000")
