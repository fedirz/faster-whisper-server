import logging

from speaches.config import config

# Disables all but `speaches` logger

root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(config.log_level.upper())
logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s"
)
