import os
import logging
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}.log"

LOG_FILE_DIR = os.path.join(os.getcwd(), "logs")

os.makedirs(LOG_FILE_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)