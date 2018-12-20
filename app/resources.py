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
CF_MODELS = {
    'cf-dummy': load_model('dummy_model')
}

CB_MODELS = {
    'cb-dummy': load_model('dummy_model'),
    'tf-idf-nouns': load_model('tf-idf-nouns-model'),
    'tf-idf-no-nouns': load_model('tf-idf-no-nouns-model'),
    'tf-idf-nouns-2grams': load_model('tf-idf-nouns-2grams-model'),
    'tf-idf-no-nouns-2grams': load_model('tf-idf-no-nouns-2grams-model'),
    'tf-idf-nouns-3grams': load_model('tf-idf-nouns-3grams-model'),
    'tf-idf-no-nouns-3grams': load_model('tf-idf-no-nouns-3grams-model')
}
