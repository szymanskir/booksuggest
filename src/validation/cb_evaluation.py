import click
import logging
import pandas as pd

from glob import glob
from os.path import basename, join
from typing import Dict, List, Union
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
        dir_path: Directory in which calculated similar books predictions
        are saved for each model.
        test_cases: Similar books ground truth.

    Returns:
        Data frame containg presicion and recall scores for all models.
    """
    prediction_files = glob(join(dir_path, '*.csv'))

    scores = [calculate_single_score(prediction_file, test_cases)
              for prediction_file in prediction_files]

    return scores


def calculate_single_score(
        prediction_file: str,
        test_cases: Dict[int, List[int]]
) -> Dict[str, Union[str, float, int]]:
    """Calculates precision, accuracy scores and the amount of correct hits
    for a single prediction file.

    Args:
        prediction_file: Path to a file containing similar books predictions.
        test_cases: Similar books ground truth.
    """
    predictions = read_similar_books(prediction_file)
    precision_score = mean([
        precision(similar_book_ids, test_cases[book_id])
        for book_id, similar_book_ids in predictions.items()
    ])

    recall_score = mean([
        recall(similar_book_ids, test_cases[book_id])
        for book_id, similar_book_ids in predictions.items()
    ])

    correct_hits_score = sum([
        len(set(similar_book_ids) & set(test_cases[book_id]))
        for book_id, similar_book_ids in predictions.items()
    ])

    return {
        'model': basename(prediction_file),
        'precision': round(precision_score, 3),
        'recall': round(recall_score, 3),
        'correct_hits': correct_hits_score
    }


def read_similar_books(similar_books_filepath: str) -> Dict[int, List[int]]:
    """Converts similar books data from data frame form to dictionary form.

        Example:
            If books 2, 3 are similar to 1 then the result dictionary will have a record in the following form: 1: [2, 3].

        Args:
            similar_books_filepath: Path to the file containing similar books data.

        Returns:
            Dictionary in which keys are book_ids and values are lists of books that are similar to the book_id key.
    """
    similar_books = pd.read_csv(similar_books_filepath)
    return {
        book_id: list(table['similar_book_id'])
        for book_id, table in similar_books.groupby('book_id')
    }


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
    save_csv(scores, output_filepath,
             ['model', 'precision', 'recall', 'correct_hits'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
