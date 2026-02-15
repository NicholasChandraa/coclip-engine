import logging
import os
from rich.logging import RichHandler


def setup_logger(name: str = "coclip"):
    """
    Setup rich logger with both console AND file output.

    - Console: Rich formatted output (colored, tracebacks)
    - File: Plain text log mirroring console output
    """
    # Create logs directory
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "coclip.log")

    # Clear any existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Console handler (Rich)
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_path=True,
        enable_link_path=True,
        show_time=True,
        show_level=True,
        markup=True,
    )
    console_handler.setLevel(logging.DEBUG)

    # File handler (plain text, same content as console)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[console_handler, file_handler],
    )

    # Suppress noisy external logs
    logging.getLogger("torio").setLevel(logging.WARNING)
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Create app logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    logger.info(f"📝 Logging to file: {log_file}")

    return logger


# Singleton logger instance
logger = setup_logger()
