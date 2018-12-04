import pickle


def save_object(obj, filename: str):
    """Saves the given object in a file specified
    by filename in pickle format.
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_object(filename: str):
    """Reads an object saved in pickle format in
    the given file.
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj
