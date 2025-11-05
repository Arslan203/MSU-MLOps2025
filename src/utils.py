import logging
import sys

def setup_logging():
    """Настраивает базовую конфигурацию логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
