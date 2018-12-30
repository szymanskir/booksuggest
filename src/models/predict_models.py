import logging

from typing import List

import click
import pandas as pd

from .cb_recommend_models import ICbRecommendationModel

from ..utils.serialization import read_object
from ..utils.csv_utils import save_csv


def _read_test_cases(test_cases_filepath: str) -> List[int]:
    """Reads test_cases from the given file.

    Similar books test cases are composed of book ids
    for which similar books are calculated.

    Args:
        test_cases_filepath: file containing a data frame with
        the 'book_id' column.

    Returns:
        list of book ids used for similar books calculations.
    """
    test_cases_data = pd.read_csv(test_cases_filepath)
    test_cases = test_cases_data['book_id'].sort_values().unique()

    return test_cases.tolist()


def predict_model(model: ICbRecommendationModel,
                  test_cases: List[int]) -> pd.DataFrame:
    """Uses the given model to calculate similar books.

    Each test case is a book id for which similar books
    are calculated using the given model.

    Args:
        model: model used for recommending similar books
        test_cases: list of book ids for which similar books are
        calculated.

    Returns:
    data frame containing the book_id and the similar_book_id columns,
    grouping is needed in order to retrieve all similar books of a specific
    book.
    """
    def recommend_helper(model, test_case_id):
        logging.debug(f'Computing {test_case_id}')
        recommendations = list(model.recommend(test_case_id).keys())
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
    """Script for calculating similar books recommendations of a given model.

    Args:
        model_filepath: filepath to the model used for recommendations.
        test_cases_filepath: filepath containing book ids for which similar
        books will be calculated.
        output_filepath: file where the result data frame should be saved.
    """
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
    main()  # pylint: disable=no-value-for-parameter
