import os
import sys
import logging
import logging.config
import logging.handlers
import datetime
import shutil

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure log directory exists
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

# Clear log file
log_file_path = os.path.join(DEFAULT_LOG_DIR, "app.log")
with open(log_file_path, "w") as f:
    pass  # Clear log file content

# 定义日志文件路径
APP_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "app.log")

# 在模块加载时生成日志文件路径
TRAIN_LOG_FILENAME = "train.log"
TRAIN_PROCESS_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, TRAIN_LOG_FILENAME)

# 检查train.log是否存在，如果存在则重命名
if os.path.exists(TRAIN_PROCESS_LOG_FILE):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"train_{timestamp}.log"
    backup_path = os.path.join(DEFAULT_LOG_DIR, backup_filename)
    shutil.move(TRAIN_PROCESS_LOG_FILE, backup_path)
    print(f"Existing train.log renamed to {backup_filename}")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": APP_LOG_FILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "train_process_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": TRAIN_PROCESS_LOG_FILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "train_process": {
            "level": "INFO",
            "handlers": ["train_process_file"],
            "propagate": False,
        },
    },
    "root": {  # root logger configuration
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}

# 初始化日志配置
def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)

# 获取训练过程日志器
def get_train_process_logger():
    return logging.getLogger("train_process")
