# tests/test_logger_helper.py
import pytest
from src.utils.logger_helper import logger
from unittest.mock import patch

def test_logger():
    logger.info("haha")

# # 如果你想真正检查日志文件：
# def test_log_file_exists():
#     import os
#     assert os.path.isfile('debug.log'), "Log file should be created"
