import pytest
from src.models.cf_recommend_models import SvdRecommendationModel
from os.path import dirname, join, realpath

test_case_dir = join(dirname(realpath(__file__)), 'data')


@pytest.mark.parametrize("ratings_filepath, expected", [
    (join(test_case_dir, "ratings-simple.csv"), [(1, 13, 4.25), (2, 11, 4.25)])
])
def test_generate_antitest(ratings_filepath, expected):
    model = SvdRecommendationModel(ratings_filepath)
    assert list(model.generate_antitest_set()) == expected
