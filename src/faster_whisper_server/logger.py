import logging


def setup_logger(log_level: str) -> None:
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level.upper())
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s")
