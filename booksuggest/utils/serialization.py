"""Functions used for serializing python objects.
"""
import pickle

from typing import Any


def save_object(obj, filename: str):
    """Saves the given object in a file specified
    by filename in pickle format.

    Args:
        obj: Object to be saved.
        filename: Path to the file in which the object
            should be saved.
    """
    with open(filename, 'wb') as save_file:
        pickle.dump(obj, save_file)


def read_object(filename: str) -> Any:
    """Reads an object saved in pickle format in
    the given file.

    Args:
        filename: File from which data should be read.

    Returns:
        Object that was saved in the given filename.
    """
    with open(filename, 'rb') as read_file:
        obj = pickle.load(read_file)

    return obj
