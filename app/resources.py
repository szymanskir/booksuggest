import pandas as pd

from os.path import dirname, join, realpath
from src.models.load_models import load_model

_MINI_LOGO = 'http://sfinks.fizyka.pw.edu.pl/img/logo_mini.png'
GOODREADS_URL = 'https://www.goodreads.com/book/show'

# Data sources
CURRENT_DIR = dirname(realpath(__file__))
BOOK_DATA = pd.read_csv(
    join(CURRENT_DIR, 'assets/book.csv'), index_col='book_id'
)
USER_DATA = pd.read_csv(join(CURRENT_DIR, 'assets/ratings.csv')).dropna()


# Model sources
def get_model(filename: str) -> str:
    return load_model(join(CURRENT_DIR, 'assets/models/', filename))


CF_MODELS = {
    'cf-dummy': get_model('dummy_model.pkl'),
    'cf-svd': get_model('basic-svd-model.pkl')
}

CB_MODELS = {
    'cb-dummy': get_model('dummy_model.pkl'),
    'basic-tf-idf': get_model('basic-tf-idf-model.pkl')
}
