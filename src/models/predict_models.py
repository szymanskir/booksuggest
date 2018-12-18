import click
import logging
import pandas as pd

from typing import List

from .recommendation_models import IRecommendationModel

from ..utils.serialization import read_object
from ..utils.csv_utils import save_csv


def _read_test_cases(test_cases_filepath: str):
    test_cases_data = pd.read_csv(test_cases_filepath)
    test_cases = test_cases_data['book_id'].sort_values().unique()

    return test_cases.tolist()


def predict_model(model: IRecommendationModel,
                  test_cases: List[int]) -> pd.DataFrame:
    def recommend_helper(model, test_case_id):
        logging.debug(f'Computing {test_case_id}')
        recommendations = list(model.recommend({test_case_id: 5}).keys())
        return [{'book_id': test_case_id,
                 'similar_book_id': recommended_book}
                for recommended_book in recommendations]

    predicted_similar_books = sum([recommend_helper(model, test_case_id)
                                   for test_case_id in test_cases], [])

    return predicted_similar_books


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('test_cases_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(model_filepath: str, test_cases_filepath: str, output_filepath: str):
    logger = logging.getLogger(__name__)

    logger.info('Loading model and test cases...')
    model = read_object(model_filepath)
    test_cases = _read_test_cases(test_cases_filepath)

    logger.info('Calculating predictions...')
    predictions = predict_model(model, test_cases)

    logger.info(f'Saving results to {output_filepath}...')
    save_csv(predictions, output_filepath, ['book_id', 'similar_book_id'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
