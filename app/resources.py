import pandas as pd

from os.path import dirname, join, realpath, splitext, basename
from booksuggest.models.load_models import load_model
from glob import glob

_MINI_LOGO = 'http://sfinks.fizyka.pw.edu.pl/img/logo_mini.png'
GOODREADS_URL = 'https://www.goodreads.com/book/show'

# Data sources
CURRENT_DIR = dirname(realpath(__file__))
BOOK_DATA = pd.read_csv(
    join(CURRENT_DIR, 'assets/book.csv'), index_col='book_id'
)
USER_DATA = pd.read_csv(join(CURRENT_DIR, 'assets/ratings-train.csv')).dropna()


# Model sources
def get_model(filename: str) -> str:
    return load_model(join(CURRENT_DIR, 'assets/models/', filename))


def read_models_from_dir(models_dir: str):
    pkl_files = glob(join(models_dir, '*.pkl'))
    return {splitext(basename(path))[0]: get_model(path) for path in pkl_files}


CB_MODELS = read_models_from_dir(join(CURRENT_DIR, 'assets/models/cb'))
CF_MODELS = read_models_from_dir(join(CURRENT_DIR, 'assets/models/cf'))
