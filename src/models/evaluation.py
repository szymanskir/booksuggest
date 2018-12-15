import click
import logging
import pandas as pd

from os import listdir
from os.path import basename, join
from typing import Dict, List

from ..utils.serialization import read_object
from ..utils.csv_utils import save_csv


def evaluate_content_based_models(
        dir_path: str,
        test_cases: Dict[int, List[int]]
) -> pd.DataFrame:
    """Evaluates the precision and accuracy_score for all models
    present in the input directory
    """

    def evaluate_single_model(model_path: str,
                              test_cases: Dict[int, List[int]]) -> Dict:
        model_name = basename(model_path)
        logging.info(f'Evaluating {model_name}')
        model = read_object(model_path)
        score = model.score(test_cases)
        return {
            'model': model_name,
            'precision': score[0],
            'recall': score[1]
        }

    models = listdir(dir_path)
    scores = [evaluate_single_model(join(dir_path, model), test_cases)
              for model in models]
    return scores


def prepare_test_case(similar_books_filepath: str) -> Dict[int, List[int]]:
    similar_books = pd.read_csv(similar_books_filepath).head(1)
    test_cases = similar_books.groupby('book_id').groups

    return test_cases


@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.argument('similar_books_input', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_directory: str, similar_books_input: str, output_filepath: str):
    logger = logging.getLogger(__name__)
    logger.info(f'Preparing test cases...')
    test_cases = prepare_test_case(similar_books_input)

    logger.info(f'Evaluating scores for models from {input_directory}...')
    scores = evaluate_content_based_models(input_directory, test_cases)

    logger.info(f'Saving results to {output_filepath}...')
    save_csv(scores, output_filepath, ['model', 'precision', 'recall'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
