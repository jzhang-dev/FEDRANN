from typing import Literal
import logging
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

class ColoredFormatter(logging.Formatter):
    # Define color mappings for log levels
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)  # Default to white
        record.msg = f"{log_color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

# Create logger
logger = logging.getLogger(__name__)

# Create console handler and set level
console_handler = logging.StreamHandler()

# Create and set formatter
console_formatter = ColoredFormatter(LOG_FORMAT)
console_handler.setFormatter(console_formatter)

# Add the handler to the logger
logger.addHandler(console_handler)


def set_logging_level(level: Literal['debug', 'info', 'error', 'critical']) -> None:
    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO
    elif level == 'error':
        _level = logging.ERROR
    elif level == 'critical':
        _level = logging.CRITICAL
    else:
        raise ValueError()
        
    logger.setLevel(_level)  
    for handler in logger.handlers:
        handler.setLevel(_level)


def add_log_file(path: str) -> None:
    file_handler = logging.FileHandler(path, mode='w')
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)

    # Add file handler to the front to avoid writing color codes to the file
    logger.handlers = [file_handler] + logger.handlers 

def test_logging() -> None:
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")