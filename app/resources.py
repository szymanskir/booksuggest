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
    'cf-dummy': get_model('cf_dummy_model.pkl'),
    'cf-svd': get_model('basic-svd-model.pkl')
}

CB_MODELS = {
    'cb-dummy': get_model('cb_dummy_model.pkl'),
    'tf-idf-nouns': get_model('tf-idf-nouns-model.pkl'),
    'tf-idf-no-nouns': get_model('tf-idf-no-nouns-model.pkl'),
    'tf-idf-nouns-2grams': get_model('tf-idf-nouns-2grams-model.pkl'),
    'tf-idf-no-nouns-2grams': get_model('tf-idf-no-nouns-2grams-model.pkl'),
    'tf-idf-nouns-3grams': get_model('tf-idf-nouns-3grams-model.pkl'),
    'tf-idf-no-nouns-3grams': get_model('tf-idf-no-nouns-3grams-model.pkl'),
    'count-nouns': get_model('count-nouns-model.pkl'),
    'count-no-nouns': get_model('count-no-nouns-model.pkl'),
    'count-nouns-2grams': get_model('count-nouns-2grams-model.pkl'),
    'count-no-nouns-2grams': get_model('count-no-nouns-2grams-model.pkl'),
    'count-nouns-3grams': get_model('count-nouns-3grams-model.pkl'),
    'count-no-nouns-3grams': get_model('count-no-nouns-3grams-model.pkl'),
}
