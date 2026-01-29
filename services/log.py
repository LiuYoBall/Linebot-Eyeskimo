import logging
import sys
from config import settings

class Logger:
    def __init__(self):
        # 設定日誌格式
        # 格式範例: [2026-01-28 10:00:00] [INFO] [image.py:50] 訊息內容
        log_format = (
            "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s"
        )
        
        # 決定 Log 等級 (Debug 模式下顯示詳細資訊，生產環境只顯示 Info 以上)
        level = logging.DEBUG if settings.DEBUG_MODE else logging.INFO

        # 設定 Handler (輸出到標準輸出 stdout，這是 Cloud Run 抓 Log 的標準方式)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format))

        # 初始化 Root Logger
        self.logger = logging.getLogger("eye_project")
        self.logger.setLevel(level)
        
        # 避免重複添加 Handler (防止 Log 重複印兩次)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger

# 初始化並供外部使用
# 使用方式: from services.log import logger
logger = Logger().get_logger()