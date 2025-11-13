import logging
import sys
import os

def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger that logs to both a file and the console.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create file handler
    file_handler = logging.FileHandler(log_file)        
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times if the logger is re-used
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def close_logger(logger):
    """
    Closes all handlers associated with a logger to free up file resources.
    """
    if logger:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
