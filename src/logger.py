import logging
from datetime import datetime
from config import PROJECT_ROOT

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger that writes to a timestamped file in the logs/ directory.
    
    Args:
        name: The name of the module requesting the logger.
        
    Returns:
        A configured logging.Logger instance.
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"asr_pipeline_{date_str}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logger is called multiple times for the same module
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger
