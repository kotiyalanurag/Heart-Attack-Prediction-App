import os
import sys
import logging

log_dir = "logs"
file_path = os.path.join(log_dir, "logs.log")
os.makedirs(log_dir, exist_ok = True)

logging_str = "[%(asctime)s - %(levelname)s - %(message)s]"

logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("MLClassifierLogger")

# logger.info("It's working!")