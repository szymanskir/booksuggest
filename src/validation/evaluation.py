import click
import logging
import pandas as pd

from os import listdir
from os.path import basename, join
from typing import Dict, List
from statistics import mean

from .metrics import precision, recall

from ..utils.csv_utils import save_csv


def calculate_scores(
        dir_path: str,
        test_cases: Dict[int, List[int]]
) -> pd.DataFrame:
    """Evaluates the precision and accuracy_score for all models
    present in the input directory.

    Args:
        dir_path: directory in which calculated similar books predictions
        are saved for each model.
        test_cases: similar books ground truth.

    Returns:
        Data frame containg presicion and recall scores for all models.
    """

    def calculate_single_score(predictions: Dict[int, List[int]],
                               test_cases: Dict[int, List[int]]) -> Dict:
        precision_score = mean([
            precision(similar_book_ids, test_cases[book_id])
            for book_id, similar_book_ids in predictions.items()
        ])

        recall_score = mean([
            recall(similar_book_ids, test_cases[book_id])
            for book_id, similar_book_ids in predictions.items()
        ])

        return precision_score, recall_score

    prediction_files = listdir(dir_path)
    prediction_files = [filename for filename in prediction_files
                        if filename.endswith('.csv')]

    scores = list()
    for prediction in prediction_files:
        predicted_similar_books = read_similar_books(join(dir_path, prediction))
        score = calculate_single_score(predicted_similar_books, test_cases)
        scores.append({
            'model': basename(prediction),
            'precision': score[0],
            'recall': score[1]
        })

    return scores


def read_similar_books(similar_books_filepath: str) -> Dict[int, List[int]]:
    similar_books = pd.read_csv(similar_books_filepath)
    test_cases = similar_books.groupby('book_id').groups

    return test_cases


@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.argument('similar_books_input', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_directory: str, similar_books_input: str, output_filepath: str):
    logger = logging.getLogger(__name__)
    logger.info(f'Preparing test cases...')
    test_cases = read_similar_books(similar_books_input)

    logger.info(f'Evaluating scores for predictions from {input_directory}...')
    scores = calculate_scores(input_directory, test_cases)

    logger.info(f'Saving results to {output_filepath}...')
    save_csv(scores, output_filepath, ['model', 'precision', 'recall'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
