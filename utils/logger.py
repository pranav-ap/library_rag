import sys
import os
import logging
import warnings
from loguru import logger
from config import config


class MyLogger:
    @staticmethod
    def init_loguru():
        logger.remove()  # Remove the default handler
        logger.add(
            sys.stdout,
            format="<level>{level: <8}</level> | "
                   "<cyan>{function}</cyan> | "
                   "<level>{message}</level>",
            level=config.log_level,
        )

    @staticmethod
    def init_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="tensorboard")
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def setup_logging():
    MyLogger.init_loguru()
    MyLogger.init_warnings()


setup_logging()
