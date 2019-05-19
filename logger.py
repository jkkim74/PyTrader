import os
import logging
from logging import FileHandler

# 로그 파일 핸들러
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
fh_log = FileHandler(os.path.join(BASE_DIR, 'logs/buy_debug.log'), encoding='utf-8')
fh_log.setLevel(logging.DEBUG)

# 로거 생성 및 핸들러 등록
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(fh_log)