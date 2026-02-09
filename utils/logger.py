"""
Simple logger for tracking progress throughout the project.
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_dir='logs'):
    """
    Create a logger that writes to both console and file.
    
    Args:
        name: Logger name (e.g., 'tokenizer', 'training')
        log_dir: Directory to save log files
    
    Returns:
        Logger instance
    """
    # Create logs directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Format: timestamp | logger_name | level | message
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console output
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'{name}_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f'Logger initialized. Saving to {log_file}')
    return logger
