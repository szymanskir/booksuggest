import pandas as pd

_MINI_LOGO = 'http://sfinks.fizyka.pw.edu.pl/img/logo_mini.png'

DATA = pd.read_csv('assets/book.csv')

CF_MODELS = {
    'cf-matrix-factorization': None,
    'cf-item-item': None
}

CB_MODELS = {
    'cb-tf-idf': None,
    'cb-nn': None
}
