import sys
from PyQt5.QtWidgets import *
import json
from datetime import datetime
from logger import *
boyou_stock_file = 'stor/boyouStock.json'
boyou_stock_list = []# [['123456','naver','40000','10','41200','38000'],['890678','daum','40000','10','41200','38000'],['123654','samsung','40000','10','41200','38000']]


def updateBoyouStockInfo(self, boyou_stock_list):
    order_stock = {}
    if len(boyou_stock_list) > 0:
        order_stock['s_date'] = datetime.today().strftime('%Y%m%d')
        order_stock['t_number'] = len(boyou_stock_list)
        order_stock['s_datail'] = boyou_stock_list
        with open(boyou_stock_file, 'w', encoding='utf-8') as stock_data:
            json.dump(order_stock, stock_data, ensure_ascii=False, indent="\t")
    else:
        print('보유 종목이 없습니다.')
        order_stock['s_date'] = datetime.today().strftime('%Y%m%d')
        order_stock['t_number'] = len(boyou_stock_list)
        order_stock['s_datail'] = boyou_stock_list
        with open(boyou_stock_file, 'w', encoding='utf-8') as stock_data:
            json.dump(order_stock, stock_data, ensure_ascii=False, indent="\t")


def readBoyouStockInfo(self):
    with open(boyou_stock_file, 'rt', encoding='utf-8') as stock_json:
        stock_data = json.load(stock_json)
    logger.debug('########################## boyouStock Info start ####################### ')
    logger.debug(str(stock_data))
    logger.debug('########################## boyouStock Info end   ####################### ')
    return stock_data['s_datail']


def stock_buy(maesu_stock):
    with open(boyou_stock_file, 'rt', encoding='utf-8') as stock_json:
        stock_data = json.load(stock_json)
    stock_list = stock_data['s_datail']
    stock_list.append(maesu_stock)
    updateBoyouStockInfo(stock_list)


def stock_sell(maedo_stock):
    with open(boyou_stock_file, 'rt', encoding='utf-8') as stock_json:
        stock_data = json.load(stock_json)
    if len(stock_data) > 0:
        stock_list = stock_data['s_datail']
        if len(stock_list) > 0:
            for key in range(len(stock_list)):
                if stock_list[key][0] == maedo_stock[0]:
                    del stock_list[key]
        updateBoyouStockInfo(stock_list)
    else:
        print('보유 종목이 없습니다.')



stock_info = ['121890', '에스디시스템', 3747, 4000, 3825, 3565]
stock_sell(stock_info)