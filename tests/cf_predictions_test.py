import pytest
from src.models.cf_recommend_models import SlopeOneRecommendationModel
from src.models.cf_predict_models import _chunk_users, predict_model
from os.path import dirname, join, realpath
import pandas as pd

test_case_dir = join(dirname(realpath(__file__)), 'data')


@pytest.mark.parametrize("ratings_filepath, expected", [
    (join(test_case_dir, "ratings-simple.csv"), [(1, 13, 4.25), (2, 11, 4.25)])
])
def test_generate_antitest(ratings_filepath, expected):
    model = SlopeOneRecommendationModel(ratings_filepath)
    assert list(model.generate_antitest_set(model.users)) == expected


@pytest.mark.parametrize("ratings_filepath, chunks_count, expected", [
    (join(test_case_dir, "ratings-simple.csv"), 1, [[1, 2]]),
    (join(test_case_dir, "ratings-simple.csv"), 2, [[1], [2]]),
    (join(test_case_dir, "ratings-simple.csv"), 4, [[1], [2], [], []]),
])
def test_users_chunking(ratings_filepath, chunks_count, expected):
    users = list(SlopeOneRecommendationModel(ratings_filepath).users)
    assert _chunk_users(users, chunks_count) == expected


@pytest.mark.parametrize("ratings_filepath, expected", [
    (join(test_case_dir, "ratings-simple.csv"), [(1, 13), (2, 11)]),
])
def test_users_chunking_pipeline(ratings_filepath, expected):
    model = SlopeOneRecommendationModel(ratings_filepath)
    model.train()
    df1 = predict_model(model, 1, 2, 0)
    df2 = predict_model(model, 1, 2, 1)
    df = df1.append(df2)
    results = [(x.user_id, x.book_id)
               for x in df[['user_id', 'book_id']].itertuples()]
    assert results == expected
