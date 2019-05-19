import json
from datetime import datetime
from logger import *
boyou_stock_file = 'stor/boyouStock.json'
boyou_stock_list = []# [['123456','naver','40000','10','41200','38000'],['890678','daum','40000','10','41200','38000'],['123654','samsung','40000','10','41200','38000']]

class BoyouStock:
    def updateBoyouStockInfo(self,boyou_stock_list):
        order_stock = {}
        if len(boyou_stock_list) > 0:
            order_stock['s_date'] = datetime.today().strftime('%Y%m%d')
            order_stock['t_number'] = len(boyou_stock_list)
            order_stock['s_datail'] = boyou_stock_list
            with open(boyou_stock_file,'w',encoding='utf-8') as stock_data:
                json.dump(order_stock,stock_data,ensure_ascii=False,indent="\t")
        else:
            print('보유 종목이 없습니다.')
            order_stock['s_date'] = datetime.today().strftime('%Y%m%d')
            order_stock['t_number'] = len(boyou_stock_list)
            order_stock['s_datail'] = boyou_stock_list
            with open(boyou_stock_file, 'w', encoding='utf-8') as stock_data:
                json.dump(order_stock, stock_data, ensure_ascii=False, indent="\t")

    def readBoyouStockInfo(self):
        with open(boyou_stock_file,'rt',encoding='utf-8') as stock_json:
            stock_data = json.load(stock_json)
        logger.debug('boyouStock Info : ' + str(stock_data))
        return stock_data['s_datail']


    def stock_buy(self,maesu_stock):
        with open(boyou_stock_file,'rt',encoding='utf-8') as stock_json:
            stock_data = json.load(stock_json)
        stock_list = stock_data['s_datail']
        stock_list.append(maesu_stock)
        self.updateBoyouStockInfo(stock_list)

    def stock_sell(self,maedo_stock):
        with open(boyou_stock_file,'rt',encoding='utf-8') as stock_json:
            stock_data = json.load(stock_json)
        if len(stock_data) > 0:
            stock_list = stock_data['s_datail']
            del stock_list[stock_list.index(maedo_stock)]
            self.updateBoyouStockInfo(stock_list)
        else:
            print('보유 종목이 없습니다.')



    # stock_buy(['123456','naver','40000','10','41200','38000'])
    # stock_buy(['890678','daum','40000','10','41200','38000'])
    # # stock_sell(['123456','naver','40000','10','41200','38000'])
    # # stock_sell(['890678','daum','40000','10','41200','38000'])
    # # boyou_stock_list=[]
    # # updateBoyouStockInfo(boyou_stock_list)
    # readBoyouStockInfo(boyou_stock_file)