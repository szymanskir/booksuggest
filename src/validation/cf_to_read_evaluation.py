import logging
from os import listdir
from os.path import join
import click
import pandas as pd

from .metrics import precision


def test_to_read(
        predictions_df: pd.DataFrame,
        to_read_df: pd.DataFrame
) -> float:
    """Calculates the precision of predictions using to_read data.

    Args:
        predictions_df (pd.DataFrame): Data frame with predictions.
        to_read_df (pd.DataFrame): Data frame containg to_read data.

    Returns:
        float: Average precision for all users.
    """
    def evaluate(group):
        to_read_ids = group['book_id'].values
        recommended_ids = predictions_df[predictions_df['user_id']
                                         == group.name]['book_id'].values
        return precision(recommended_ids, to_read_ids)

    return to_read_df.groupby('user_id').apply(evaluate).mean()


@click.command()
@click.argument('predictions_dir', type=click.Path(exists=True))
@click.argument('to_read_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(predictions_dir: str, to_read_filepath: str, output_filepath: str):
    logger = logging.getLogger(__name__)

    predictions_files = listdir(predictions_dir)
    predictions_files = [filename for filename in predictions_files
                         if filename.endswith('.csv')]
    to_read_df = pd.read_csv(to_read_filepath)

    logger.info('Evaluating predictions from %s...', predictions_dir)
    results = list()
    for prediction_file in predictions_files:
        prediction_df = pd.read_csv(join(predictions_dir, prediction_file))
        precision = evaluate_to_read(prediction_df, to_read_df)
        results.append((prediction_file, precision))

    logger.info('Saving results to %s...', output_filepath)
    labels = ['model', 'precision']
    results_df = pd.DataFrame.from_records(results, columns=labels)
    results_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
