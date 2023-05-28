import os
import logging
import logging.config

from logging_tree import printout

# Logging Config
# More on Logging Configuration
# https://docs.python.org/3/library/logging.config.html
# Setting up a config
LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        # "simple": {"format": "%(message)s"},
    },
    "handlers": {
        "handler": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
        }
    },
    "root": {"level": "DEBUG"},
}


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a ew logger object will be created from root.
            cfg: dict()
                    Configuration of the logging to be implemented by default
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """
    if not cfg:
        cfg = LOGGING_DEFAULT_CONFIG  # assign default configuration if not provided
    logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)
            try:
                formatter = logging.Formatter(
                    fmt=cfg["formatters"]["default"]["format"],
                    datefmt=cfg["formatters"]["default"]["datefmt"],
                )
            except:
                formatter = logging.Formatter(
                    "%(ascitime)s %(name)-12s %(levelname)-8s %(message)s"
                )
            fh.setFormatter(formatter)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)
            try:
                formatter = logging.Formatter(
                    fmt=cfg["formatters"]["default"]["format"],
                    datefmt=cfg["formatters"]["default"]["datefmt"],
                )
            except:
                formatter = logging.Formatter(
                    "%(ascitime)s %(name)-12s %(levelname)-8s %(message)s"
                )
            sh.setFormatter(formatter)

    return logger


if __name__ == "__main__":
    # configuring and assigning in the logger can be done by the below function
    logger = configure_logger(
        log_file=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "custom_config.log"
        )
    )
    logger.info(f"Logging Test - Start")
    logger.info(f"Logging Test - Test 1 Done")
    logger.warning("Watch out!")

    # printing out the current loging confiurations being used
    printout()
