#파일명 realtimekiwoom.py 시작
import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *


class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self._create_kiwoom_instance()
        self._set_signal_slots()
    def _create_kiwoom_instance(self):
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

    def _set_signal_slots(self):
        self.OnReceiveRealData.connect(self._receive_real_data)
        self.OnEventConnect.connect(self._event_connect)
    def _comm_connect(self):
        self.dynamicCall("CommConnect()")
        self.login_event_loop=QEventLoop()
        self.login_event_loop.exec_()
    def _event_connect(self,err_code):
        if err_code==0:
            print("connected")
        else:
            print("disconnected")
        self.login_event_loop.exit()
    def _get_connect_state(self):
        ret=self.dynamicCall("GetConnectState()")
        return ret
# 이벤트 처리하는곳
    def _receive_real_data(self, sRealKey, sRealType, sRealData):
        print(sRealData)
        pass #kiwoom,py 에서 실시간처리를 하고싶으면 main에 real_check부분 내용을 여기로 가져오면 됨
#기본함수 정의
    def _get_comm_real_data(self,strRealType,nFid):
        ret=self.dynamicCall("GetCommRealData(QString,int)",strRealType,nFid)
        return ret.strip()
    def _set_real_reg(self, strScreenNo, strCodeList, strFidList, strRealType):
        ret=self.dynamicCall("SetRealReg(QString,QString,QString,QString)", strScreenNo, strCodeList, strFidList,strRealType)
        return ret
    def _set_real_remove(self,strScrNo,strDelCode):
        ret=self.dynamicCall("SetRealRemove(QString,QString)",strScrNo,strDelCode)
        return ret
#############       Tr 정의
    def res_set_real(self,ScrNo):
        ret=self._set_real_reg(ScrNo, "005930;039490", "27;28", "0")
        ###더 많은 종목과 받고싶은 데이터 형태를 추가 가능
        #키움증권과 삼성증권의 최우선매수호가와 최우선매도호가를 실시간으로 받아오는 것을 등록
        return ret

if __name__ == "__main__":
    app = QApplication(sys.argv)
    kiwoom = Kiwoom()
    kiwoom._comm_connect()
    kiwoom.res_set_real('0101')
    app.exec_()