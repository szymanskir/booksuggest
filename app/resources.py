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
    'basic-tf-idf': load_model('basic-tf-idf-model')
}
