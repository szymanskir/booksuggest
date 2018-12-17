import pickle


def save_object(obj, filename: str):
    """Saves the given object in a file specified
    by filename in pickle format.
    """
    with open(filename, 'wb') as save_file:
        pickle.dump(obj, save_file)


def read_object(filename: str):
    """Reads an object saved in pickle format in
    the given file.
    """
    with open(filename, 'rb') as read_file:
        obj = pickle.load(read_file)

    return obj
