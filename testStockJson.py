import json
from datetime import datetime

def updateBoyouStockInfo(boyouStockFileInfo,boyou_stock_list):
    order_stock = {}
    if len(boyou_stock_list) > 0:
        order_stock['s_date'] = datetime.today().strftime('%Y%m%d')
        order_stock['t_number'] = len(boyou_stock_list)
        order_stock['s_datail'] = boyou_stock_list
        with open(boyouStockFileInfo,'w',encoding='utf-8') as stock_data:
            json.dump(order_stock,stock_data,ensure_ascii=False,indent="\t")
    else:
        print('보유 종목이 없습니다.')
        with open(boyouStockFileInfo, 'w', encoding='utf-8') as stock_data:
            json.dump(order_stock, stock_data, ensure_ascii=False, indent="\t")

def readBoyouStockInfo(boyouStockFileInfo):
    with open(boyouStockFileInfo) as stock_json:
        stock_data = json.load(stock_json)
    print(stock_data)
    for value in stock_data.values():
        print(value)

boyou_stock_file = 'stor/boyouStock.json'
boyou_stock_list = [['naver','40000','10'],['daum','40000','10'],['samsung','40000','10'],['abc','40000','10'],['cde','40000','10'],['fge','40000','10'],['grerg','40000','10']]
# boyou_stock_list=[]
updateBoyouStockInfo(boyou_stock_file,boyou_stock_list)
readBoyouStockInfo(boyou_stock_file)