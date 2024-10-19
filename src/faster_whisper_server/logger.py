import logging

from faster_whisper_server.dependencies import get_config


def setup_logger() -> None:
    config = get_config()  # HACK
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s", level=config.log_level.upper())
