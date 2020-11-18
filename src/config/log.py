import logging
import logging.handlers
import os
import sys
from logging.config import dictConfig


DEFAULT_LOGGING = {"version": 1, "disable_existing_loggers": False}


def get_logging_handlers(logfile_path: str = "logs/logs.txt"):
    """
    Initialize logging defaults for Project.
    :param logfile_path: logfile used to the logfile
    :type logfile_path: string
    This function does:
    - Assign INFO and DEBUG level to logger file handler and console handler
    """
    if not os.path.exists("logs"):
        os.makedirs("logs")

    dictConfig(DEFAULT_LOGGING)

    default_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)8s] [%(name)s] [%(funcName)s():%(lineno)s] [PID:%(process)d "
        "TID:%(thread)d] %(message)s",
        "%d/%m/%Y %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        logfile_path, maxBytes=10485760, backupCount=300, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    file_handler.setFormatter(default_formatter)
    console_handler.setFormatter(default_formatter)

    return [file_handler, console_handler]


def configure_logging():
    handlers = get_logging_handlers()
    logging.getLogger("").handlers.extend(handlers)
    logging.root.setLevel(logging.DEBUG)


def disable_flask_logging(flask_app):
    flask_app.logger.handlers = []
    flask_app.logger.propagate = False
