import logging
from os import listdir
from os.path import join
import click
import pandas as pd
from typing import Tuple

from .metrics import precision_thresholded, recall_thresholded


def evaluate_to_read(
        predictions_df: pd.DataFrame,
        to_read_df: pd.DataFrame,
        threshold: float
) -> Tuple[float, float]:
    """Calculates the precision and recall of predictions using to_read data.

    Args:
        predictions_df (pd.DataFrame): Data frame with predictions.
        to_read_df (pd.DataFrame): Data frame containg to_read data.
        threshold (float): Treshold for rating to be valid recommendation.

    Returns:
        Tuple[float, float]: Average `(precision, recall)` for all users.
    """
    def evaluate(group):
        to_read_ids = group['book_id'].values
        predictions = predictions_df[predictions_df['user_id'] ==
                                     group.name][['book_id', 'est']]
        pred_tuples = [(x.book_id, x.est)
                       for x in predictions.itertuples()]

        return (precision_thresholded(pred_tuples, to_read_ids, threshold),
                recall_thresholded(pred_tuples, to_read_ids, threshold))

    metrics_series = to_read_df.groupby('user_id').apply(evaluate)
    df = pd.DataFrame(metrics_series.values.tolist(),
                      index=metrics_series.index)
    return (df[df.columns[0]].mean(), df[df.columns[1]].mean())


@click.command()
@click.argument('predictions_dir', type=click.Path(exists=True))
@click.argument('to_read_filepath', type=click.Path(exists=True))
@click.option('--threshold', default=4.0,
              help='Treshold for rating to be valid recommendation.')
@click.argument('output_filepath', type=click.Path())
def main(predictions_dir: str, to_read_filepath: str,
         threshold: float, output_filepath: str):
    logger = logging.getLogger(__name__)

    predictions_files = listdir(predictions_dir)
    predictions_files = [filename for filename in predictions_files
                         if filename.endswith('.csv')]
    to_read_df = pd.read_csv(to_read_filepath)

    logger.info('Evaluating predictions from %s...', predictions_dir)
    results = list()
    for prediction_file in predictions_files:
        prediction_df = pd.read_csv(join(predictions_dir, prediction_file))
        precision, recall = evaluate_to_read(
            prediction_df, to_read_df, threshold)
        results.append((prediction_file, precision, recall))

    logger.info('Saving results to %s...', output_filepath)
    labels = ['model', 'precision', 'recall']
    results_df = pd.DataFrame.from_records(results, columns=labels)
    results_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
