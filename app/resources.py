import pandas as pd
from src.models.load_models import load_dummy_model

_MINI_LOGO = 'http://sfinks.fizyka.pw.edu.pl/img/logo_mini.png'

DATA = pd.read_csv('assets/book.csv', index_col='book_id')

CF_MODELS = {
    'cf-dummy': load_dummy_model(),
}

CB_MODELS = {
    'cb-dummy': load_dummy_model(),
}
