import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
from Kiwoom import *
from SysStatagy import *

form_class = uic.loadUiType("pytrader.ui")[0]
test_invest = True
if test_invest:
    total_buy_money = 10000000
else:
    total_buy_money = 50000
s_year_date = '2019-01-01';

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.trade_stocks_done = False # 자동 trading False이면 작동

        self.kiwoom = Kiwoom()
        self.kiwoom.comm_connect()
        # 전략
        self.stratagy = SysStratagy()

        # Timer1
        self.timer = QTimer(self)
        self.timer.start(1000*5)

        # Timer2
        self.timer2 = QTimer(self)
        self.timer2.start(1000 * 10)

        # Timer3
        self.timer3 = QTimer(self)
        self.timer3.start(1000 * 10)

        # Timer4 손절처리를 위한 Timer 설정
        self.timer4 = QTimer(self)
        self.timer4.start(1000 * 4)

        # 계좌정보 넣어줌
        accouns_num = int(self.kiwoom.get_login_info("ACCOUNT_CNT"))
        accounts = self.kiwoom.get_login_info("ACCNO")
        accounts_list = accounts.split(';')[0:accouns_num]
        self.comboBox.addItems(accounts_list)

        # 매도/매수 리스트 조회
        self.load_buy_sell_list()

        # 이벤트 정보 연결
        self.timer.timeout.connect(self.timeout)
        self.timer2.timeout.connect(self.timeout2)
        self.timer3.timeout.connect(self.timeout3)
        self.timer4.timeout.connect(self.timeout4) # stop loss 처리
        self.lineEdit.textChanged.connect(self.code_changed)
        self.pushButton.clicked.connect(self.send_order)
        self.pushButton_2.clicked.connect(self.check_balance_Widget) # pushButton_2 라는 객체가 클릭될 때 check_balance라는 메서드가 호출

    def timeout(self):
        market_start_time = QTime(9, 0, 0)
        current_time = QTime.currentTime()

        if current_time > market_start_time and self.trade_stocks_done is False:
            self.trade_stocks()
            #self.trade_stocks_done = True

        text_time = current_time.toString("hh:mm:ss")
        time_msg = "현재시간: " + text_time

        state = self.kiwoom.get_connect_state()
        if state == 1:
            state_msg = "서버 연결 중"
        else:
            state_msg = "서버 미 연결 중"

        self.statusbar.showMessage(state_msg + " | " + time_msg)

    def timeout2(self):
        if self.checkBox.isChecked():
            self.check_balance_Widget()

    def timeout3(self):
        if self.checkBox_2.isChecked():
            self.load_buy_sell_list()

    def timeout4(self):
        if self.checkBox_3.isChecked():
            self.stock_stop_loss()

    def code_changed(self):
        code = self.lineEdit.text()
        name = self.kiwoom.get_master_code_name(code)
        self.lineEdit_2.setText(name)

    # 이익을 위한 매도주문(즉시 매도처리 이므로)을 취소하고 손실을 중지하기 위한 주문처리를 함.
    def stock_stop_loss(self):
        print("손실에 대한 loss 처리 설정했습니다.")
        self.check_balance()
        # Item list
        item_count = len(self.kiwoom.opw00018_output['multi'])
        if item_count == 0:
            print("보유종목이 없습니다.")

        # 한 종목에 대한 종목명, 보유량, 매입가, 현재가, 평가손익, 수익률(%)은 출력
        for j in range(item_count):
            row = self.kiwoom.opw00018_output[j]
            boyou_cnt = int(row[1].replace(',', ''))
            maeip_price = int(row[2].replace(',', ''))
            cur_price = int(row[3].replace(',', ''))
            stock_code = row[6]
            # 보유종목의 이익매도 주문이 있는 경우 이익매도주문 취소후 익절처리
            self.check_michegyoel_joomoon(stock_code)
            row2 = self.kiwoom.opw00007_output[j]
            if len(row2) > 0:
                orgJoomoonNo  = int(row2[4]) # 원주문번호 정보를 가져온다.
                self._file_line_delete(self.sell_loc, stock_code) # stor파일에 해당 종목을 삭제한다.
            else:
                orgJoomoonNo = ""
            print("종목코드 :", stock_code, " 원주문번호 : ", orgJoomoonNo)
            mado_price = self.stratagy.get_maedo_price(maeip_price, 0.95) # 4% 익절가처리
            # 해당주식의 (이익을 얻기 위한)매도 주문 취소 처리
            # 아침에 자동 매도주문 처리가 됐을것이고 그것에 대해 취소처리를 하는 것..
            if not self._item_stock_exist(stock_code):
                if cur_price < mado_price: # 익절가보다 작으면 매도처리
                    self.kiwoom.add_stock_sell_info(stock_code, mado_price, boyou_cnt, orgJoomoonNo)

    def send_order(self):
        order_type_lookup = {'신규매수': 1, '신규매도': 2, '매수취소': 3, '매도취소': 4}
        hoga_lookup = {'지정가': "00", '시장가': "03"}

        account = self.comboBox.currentText()
        order_type = self.comboBox_2.currentText()
        code = self.lineEdit.text()
        hoga = self.comboBox_3.currentText()
        num = self.spinBox.value()
        price = self.spinBox_2.value()

        self.kiwoom.send_order("send_order_req", "0101", account, order_type_lookup[order_type], code, num, price,
                               hoga_lookup[hoga], "")

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

    def check_michegyoel_joomoon(self, code):
        self.kiwoom.reset_opw00007_output()
        account_number = self.kiwoom.get_login_info("ACCNO")
        account_number = account_number.split(';')[0]  # 첫번째 계좌번호 호출
        joomoondate = util.cur_date('%y%m%d')
        self.kiwoom.set_input_value("주문일자", joomoondate)
        self.kiwoom.set_input_value("계좌번호", account_number)
        self.kiwoom.set_input_value("비밀번호", "3051")
        self.kiwoom.set_input_value("비밀번호매체구분", "00")
        self.kiwoom.set_input_value("조회구분", "3")
        self.kiwoom.set_input_value("주식채권구분", "1")
        self.kiwoom.set_input_value("매도매수구분", "1")
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.set_input_value("시작주문번호", "")
        self.kiwoom.comm_rq_data("opw00007_req", "opw00007", 0, "2000")

    def check_balance_Widget(self):
        self.check_balance()
        # balance
        item = QTableWidgetItem(self.kiwoom.d2_deposit)
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.tableWidget.setItem(0, 0, item)

        # 총매입, 총평가, 총손익, 총수익률(%), 추정자산을 QTableWidget의 칼럼에 추가하는 코드.
        # 데이터는 self.kiwoom.opw00018_output['single']을 통해 얻어올 수 있음.
        for i in range(1, 6):
            print('Debug', self.kiwoom.opw00018_output['single'][i-1])
            item = QTableWidgetItem(self.kiwoom.opw00018_output['single'][i - 1])
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.tableWidget.setItem(0, i, item)

        # resizeRowsToContents 메서드를 호출해서 아이템의 크기에 맞춰 행의 높이를 조절
        self.tableWidget.resizeRowsToContents()

        # Item list
        item_count = len(self.kiwoom.opw00018_output['multi'])
        self.tableWidget_2.setRowCount(item_count)

        # 한 종목에 대한 종목명, 보유량, 매입가, 현재가, 평가손익, 수익률(%)은 출력
        for j in range(item_count):
            row = self.kiwoom.opw00018_output['multi'][j]
            for i in range(len(row)):
                item = QTableWidgetItem(row[i])
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.tableWidget_2.setItem(j, i, item)

        # resizeRowsToContents 메서드를 호출해서 아이템의 크기에 맞춰 행의 높이를 조절
        self.tableWidget.resizeRowsToContents()

    def load_buy_sell_list(self):
        f = open(self.kiwoom.buy_loc, 'rt', encoding='UTF-8')
        buy_list = f.readlines()
        f.close()

        f = open(self.kiwoom.sell_loc, 'rt', encoding='UTF-8')
        sell_list = f.readlines()
        f.close()

        row_count = len(buy_list) + len(sell_list)
        self.tableWidget_3.setRowCount(row_count)

        # buy list
        for j in range(len(buy_list)):
            row_data = buy_list[j]
            split_row_data = row_data.split(';')
            split_row_data[1] = self.kiwoom.get_master_code_name(split_row_data[1].rsplit())

            for i in range(len(split_row_data)):
                item = QTableWidgetItem(split_row_data[i].rstrip())
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
                self.tableWidget_3.setItem(j, i, item)

        # sell list
        for j in range(len(sell_list)):
            row_data = sell_list[j]
            split_row_data = row_data.split(';')
            split_row_data[1] = self.kiwoom.get_master_code_name(split_row_data[1].rstrip())

            for i in range(len(split_row_data)):
                item = QTableWidgetItem(split_row_data[i].rstrip())
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
                self.tableWidget_3.setItem(len(buy_list) + j, i, item)

        self.tableWidget_3.resizeRowsToContents()
    # buy_list는 애초에 모니터링시 기본정보 목록에서 추출
    # 매매전략에 해당하는 종목을 buy_list_txt에 저장 
    def trade_buy_stratagic(self,code):
        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.comm_rq_data("opt10001_req", "opt10001", 0, "2000")
        name = self.kiwoom.get_master_code_name(code)
        # cur_price = self.kiwoom.jangoInfo[code]['현재가']
        # if cur_price[0] == '-' or cur_price[0] == '+':
        #     cur_price = cur_price[1:]
        # open_price = self.kiwoom.jangoInfo[code]['시가']
        # if cur_price[0] == '-' or cur_price[0] == '+':
        #     open_price = open_price[1:]
        # print(name, ",현재가 : ", self.kiwoom.cur_price)
        result = self.stratagy.isBuyStockAvailable(code, name, self.kiwoom.cur_price, self.kiwoom.open_price, s_year_date)
        return result
        # return True

    def _file_update(self,fileName,code,pre_status,chg_status):
        stock_list = []
        f = open(fileName, 'rt', encoding='UTF-8')
        stock_list = f.readlines()
        f.close()

        for i, row_data in enumerate(stock_list):
            if code in stock_list[i]:
                stock_list[i] = stock_list[i].replace(pre_status, chg_status)

        # file update
        f = open(fileName, 'wt', encoding='UTF-8')
        for row_data in stock_list:
            f.write(row_data)
        f.close()

    def _file_line_delete(self,fileName,code):
        stock_list = []
        f = open(fileName, 'rt', encoding='UTF-8')
        stock_list = f.readlines()
        f.close()

        for i, row_data in enumerate(stock_list):
            if code in stock_list[i]:
                stock_list[i+1].remove()


        # file update
        f = open(fileName, 'wt', encoding='UTF-8')
        for row_data in stock_list:
            f.write(row_data)
        f.close()

    def _item_stock_exist(self, fileName, code):
        stock_list = []
        f = open(fileName, 'rt', encoding='UTF-8')
        stock_list = f.readlines()
        f.close()
        b_exist = False
        for i, row_data in enumerate(stock_list):
            if code in stock_list[i]:
                b_exist = True

        return b_exist


    def trade_stocks(self):
        if self.stratagy.isTimeAvalable(self.kiwoom.maesu_start_time,self.kiwoom.maesu_end_time):
            hoga_lookup = {'지정가': "00", '시장가': "03"}
            f = open(self.kiwoom.buy_loc, 'rt', encoding='UTF-8')
            buy_list = f.readlines()
            f.close()

            f = open(self.kiwoom.sell_loc, 'rt', encoding='UTF-8')
            sell_list = f.readlines()
            f.close()

            account = self.comboBox.currentText()

            if len(buy_list) == 0:
                print("매수 대상 종목이 존재하지 않습니다.")

            # buy list
            for row_data in buy_list:
                split_row_data = row_data.split(';')
                hoga = split_row_data[2]
                code = split_row_data[1]
                num = split_row_data[3]
                price = split_row_data[4]

                if split_row_data[-1].rstrip() == '매수전':
                    if self.trade_buy_stratagic(code):  # * 매수전략 적용 *
                        buy_num_info = self.stratagy.get_buy_num_price(total_buy_money, self.kiwoom.high_price, self.kiwoom.cur_price)
                        num = buy_num_info[0]
                        price = buy_num_info[1]
                        print("매수수량 : ", num, " 매수상한가 : ", price)
                        self.kiwoom.send_order("send_order_req", "0101", account, 1, code, num, price, hoga_lookup[hoga], "") # 1: 매수, 2: 매도
                        if self.kiwoom.order_result == 0:
                            self._file_update(self.kiwoom.buy_loc, code, '매수전', '주문완료')
                        else:
                            print(self.kiwoom.order_result, ': 매수 처리 못했습니다.')

            if len(sell_list) == 0:
                print("매도 대상 종목이 존재하지 않습니다.")
            # sell list
            for row_data in sell_list:
                split_row_data = row_data.split(';')
                hoga = split_row_data[2]
                code = split_row_data[1]
                num = split_row_data[3]
                price = split_row_data[4]

                if split_row_data[-2].rstrip() == '매도전':
                    self.kiwoom.send_order("send_order_req", "0101", account, 2, code, num, price, hoga_lookup[hoga], "") # 1: 매수, 2: 매도
                    print('결과 : ',self.kiwoom.order_result)
                    if self.kiwoom.order_result == 0:
                        self._file_update(self.kiwoom.sell_loc, code, '매도전', '주문완료')
                    else:
                        print(self.kiwoom.order_result,': 매도 처리 못했습니다.')
        # # buy list
        # for i, row_data in enumerate(buy_list):
        #     buy_list[i] = buy_list[i].replace("매수전", "주문완료")
        #
        # # file update
        # f = open("buy_list.txt", 'wt', encoding='UTF-8')
        # for row_data in buy_list:
        #     f.write(row_data)
        # f.close()

        # # sell list
        # for i, row_data in enumerate(sell_list):
        #     sell_list[i] = sell_list[i].replace("매도전", "주문완료")
        #
        # # file update
        # f = open("sell_list.txt", 'wt', encoding='UTF-8')
        # for row_data in sell_list:
        #     f.write(row_data)
        # f.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()