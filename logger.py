import logging

logging.basicConfig(level = logging.INFO, 
                    filename = "logs.log", 
                    filemode = "w",
                    format = "%(asctime)s - %(levelname)s - %(message)s",
                    )

logger = logging.getLogger("MLClassifier")

# logger.info("It's working!")