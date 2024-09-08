import logging

from faster_whisper_server.config import config

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(config.log_level.upper())
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s")
