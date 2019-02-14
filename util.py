# -*-coding: utf-8 -*-
import inspect
from datetime import datetime
import os
import os.path
import pandas as pd

def save_log(contents, subject="None", folder=""):
    current_dir = os.getcwd()
    filePath = current_dir + os.sep + folder + os.sep + cur_month() + ".txt"
    openMode = ""
    if (os.path.isfile(filePath)):
        openMode = 'a'
    else:
        openMode = 'w'
    line = '[{0:<8}][{1:<10}] {2}\n'.format(cur_date_time(), subject, contents)

    with open(filePath, openMode, encoding='utf8') as f:
        f.write(line)
    pass

def whoami():
    return '* ' + cur_time_msec() + ' ' + inspect.stack()[1][3] + ' '
    
def whosdaddy():
    return '*' + cur_time_msec() + ' ' + inspect.stack()[2][3] + ' '
    
def cur_date_time(time_string = '%y-%m-%d %H:%M:%S'):
    cur_time = datetime.now().strftime(time_string)
    return cur_time

def cur_time_msec(time_string ='%H:%M:%S.%f'):
    cur_time = datetime.now().strftime(time_string) 
    return cur_time

def cur_date(time_string = '%y-%m-%d'):
    cur_time = datetime.now().strftime(time_string)
    return cur_time

def cur_month(time_string ='%y-%m'):
    cur_time = datetime.now().strftime(time_string)
    return cur_time

def cur_time(time_string ='%H:%M:%S' ):
    cur_time = datetime.now().strftime(time_string)
    return cur_time

# 주식 실거래일 구하기
def get_prev_date(dif1, dif2, today):
    # 금일날짜
    # today = datetime.today().strftime("%Y%m%d")
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
            prev_bus_day_1 = df_mdays_list[i - dif1].__format__('%Y-%m-%d')
            prev_bus_day_2 = df_mdays_list[i - dif2].__format__('%Y-%m-%d')
            return (prev_bus_day_1, prev_bus_day_2)

# --------------------------------------------------------
# 변환 관련 유틸
# --------------------------------------------------------
def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default

if __name__ == "__main__":
    print(cur_time())
    print(cur_date())
    print(cur_date_time() )
    save_log("한글", "한글", "log")