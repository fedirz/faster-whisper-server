import logging

disable_loggers = ["multipart.multipart", "faster_whisper"]


def pytest_configure():
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
