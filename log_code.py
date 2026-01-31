import logging

def setup_logging(script_name):
    logger = logging.getLogger(script_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(
            f'C:\\Users\\harsh\\OneDrive\\Desktop\\ml_project\\ml_pipeline\\logs\\{script_name}.log',
            mode='w'
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger