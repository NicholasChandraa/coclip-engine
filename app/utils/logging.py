import logging
from rich.logging import RichHandler

def setup_logger(name: str = "coclip"):
    """
    Setup rich logger with filename and line number support.
    """
    # Konfigurasi basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_path=True,          # Tampilkan nama file + line number
                enable_link_path=True,   # Bikin path Ctrl+clickable di terminal
                show_time=True,
                show_level=True,
                markup=True
            )
        ]
    )

    # Buat logger instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    return logger

# Singleton logger instance yang bisa di-import di mana saja
logger = setup_logger()