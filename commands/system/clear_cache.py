import traceback

import click
from loguru import logger

from utils import get_cache


@click.command("clear-cache")
def clear_cache():
    """
    Clears disk cache.
    """
    try:
        get_cache().clear()
        logger.info("Disk cache was cleared.")
    except Exception:
        logger.debug(traceback.format_exc())
        logger.error("Could not clear disk cache.")
        exit(1)
