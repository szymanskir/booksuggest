import logging
from os import listdir
from os.path import join
import click
import pandas as pd
from typing import Tuple

from .metrics import precision_thresholded, recall_thresholded


def evaluate_on_predictions(
        predictions_df: pd.DataFrame,
        test_df: pd.DataFrame,
        threshold: float,
        n: int
) -> Tuple[float, float]:
    """Calculates the precision and recall of predictions using to_read data.

    Args:
        predictions_df (pd.DataFrame): Data frame with predictions.
        test_df (pd.DataFrame): Data frame containg testing data.
            Should contain `['user_id', 'book_id']` columns.
        threshold (float): Treshold for rating to be valid recommendation.

    Returns:
        Tuple[float, float]: Average `(precision, recall)` for all users.
    """
    def evaluate(group):
        to_read_ids = group['book_id'].values
        predictions = predictions_grouped_df.get_group(group.name).head(n)
        pred_tuples = [(x.book_id, x.est)
                       for x in predictions.itertuples()]
        return (precision_thresholded(pred_tuples, to_read_ids, threshold),
                recall_thresholded(pred_tuples, to_read_ids, threshold))

    predictions_grouped_df = predictions_df.groupby('user_id')[
        'book_id', 'est']
    metrics_series = test_df.groupby('user_id').apply(evaluate)
    df = pd.DataFrame(metrics_series.values.tolist(),
                      index=metrics_series.index)
    return tuple(df.mean().values)


@click.command()
@click.argument('predictions_dir', type=click.Path(exists=True))
@click.argument('to_read_filepath', type=click.Path(exists=True))
@click.argument('testset_filepath', type=click.Path(exists=True))
@click.option('--threshold', default=4.0,
              help='Treshold for rating to be valid recommendation.')
@click.option('--n-min', default=10,
              help='Lower bound of predictions number generated loop.')
@click.option('--n-max', default=10,
              help='Upper bound of predictions number generated loop.')
@click.argument('output_filepath', type=click.Path())
def main(predictions_dir: str, to_read_filepath: str, testset_filepath: str,
         threshold: float, n_min: int, n_max: int, output_filepath: str):
    """Evaluates precision and recall metrics of predictions on given testsets.

    Args:
        predictions_dir (str): Directory with predictions files.
        to_read_filepath (str): Path to a file with to_read data.
        testset_filepath (str): Path to a file with testset data.
        threshold (float): Threshold for considering specific
            recommendation a good one.
        n_min (int): Lower bound of predictions number generated loop.
        n_max (int): Upper bound of predictions number generated loop.
        output_filepath (str): Output filepath.
    """
    logger = logging.getLogger(__name__)

    predictions_files = listdir(predictions_dir)
    predictions_files = [filename for filename in predictions_files
                         if filename.endswith('.csv')]
    to_read_df = pd.read_csv(to_read_filepath)
    testset_df = pd.read_csv(testset_filepath)
    logger.info('Evaluating predictions from %s...', predictions_dir)
    results = list()
    for prediction_file in predictions_files:
        prediction_df = pd.read_csv(join(predictions_dir, prediction_file))
        for n in range(n_min, n_max + 1):
            p_to_read, r_to_read = evaluate_on_predictions(
                prediction_df, to_read_df, threshold, n)
            p_testset, r_testset = evaluate_on_predictions(
                prediction_df, testset_df, threshold, n)
            results.append((prediction_file, n, p_to_read, p_testset,
                            r_to_read, r_testset))

    logger.info('Saving results to %s...', output_filepath)
    labels = ['model', 'n', 'precision-to_read', 'precision-testset',
              'recall-to_read', 'recall-testset']
    results_df = pd.DataFrame.from_records(results, columns=labels)
    with open(output_filepath, 'a') as f:
        results_df.to_csv(f, header=f.tell() == 0, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
