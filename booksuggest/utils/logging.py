"""Logger setup functions.
"""
import logging


def setup_root_logger():
    """Sets up the root logger with a custom configuration.
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
