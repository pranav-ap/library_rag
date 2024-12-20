import sys
import os
import logging
import warnings
import neptune
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

        # logging.getLogger("neptune").setLevel(logging.CRITICAL)

        class _FilterCallback(logging.Filterer):
            def filter(self, record: logging.LogRecord):
                return not (
                    record.name == "neptune"
                    and record.getMessage().startswith(
                        "Error occurred during asynchronous operation processing: X-coordinates (step) must be strictly increasing for series attribute"
                    )
                )

        neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
            _FilterCallback()
        )


def setup_logging():
    MyLogger.init_loguru()
    MyLogger.init_warnings()


setup_logging()
