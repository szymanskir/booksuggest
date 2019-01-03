import click
import logging
import pandas as pd

import concurrent.futures as cf
from itertools import chain, islice, repeat
from typing import Any, Iterable, List, Tuple

from .cf_recommend_models import ICfRecommendationModel
from ..utils.serialization import read_object

logger = logging.getLogger(__name__)


def predict_model(model: ICfRecommendationModel,
                  recommendation_count: int,
                  chunks_count: int,
                  batch_size: int = 100000
                  ) -> pd.DataFrame:
    """Calculates top recommendations for every user in the trainset.

    Calculations are done in batches to avoid huge memory consumption.

    Args:
        model (ICfRecommendationModel): Already trained model.
        recommendation_count (int): Specifies how many recommendations to save.
        chunks_count (int): Number of chunks.
        batch_size (int, optional): Defaults to 100000. Size of single batch.

    Returns:
        pd.DataFrame: Data frame with predictions.
    """
    main_df = pd.DataFrame(columns=['user_id', 'book_id', 'est'])
    users_chunked = _chunk_users(list(model.users), chunks_count)
    args = (users_chunked, repeat(model), repeat(recommendation_count),
            repeat(batch_size))
    with cf.ProcessPoolExecutor(max_workers=chunks_count) as executor:
        for df in executor.map(_process_chunk, *args):
            main_df = main_df.append(df)
    return main_df.sort_values('user_id')


def _chunk_users(users, chunks_count):
    return [users[start::chunks_count] for start in range(chunks_count)]


def _process_chunk(users, model, recommendation_count, batch_size):
    batches = _batch(model.generate_antitest_set(users), batch_size)
    batch_counter = 0
    main_df = pd.DataFrame(columns=['user_id', 'book_id', 'est'])
    for batch in batches:
        logger.debug('Batch: %s', batch_counter)
        batch_counter += 1
        df = _predict_batch(model, list(batch), recommendation_count)
        main_df = main_df.append(df)
    return main_df


def _predict_batch(
        model: ICfRecommendationModel,
        cases_batch: List[Tuple[int, int, float]],
        recommendation_count: int
) -> pd.DataFrame:
    predictions = model.test(cases_batch)
    labels = ['user_id', 'book_id', '2', 'est', '4']
    pred_df = pd.DataFrame.from_records(
        predictions, exclude=['2', '4'], columns=labels)
    pred_df = pred_df.groupby('user_id')['book_id', 'est'].apply(
        lambda x: x.nlargest(recommendation_count, columns=['est']))
    return pred_df.reset_index(level=0)


def _batch(iterable: Iterable[Any], batch_size: int) -> Iterable[Any]:
    iterable = iter(iterable)
    while True:
        group = islice(iterable, batch_size)
        try:
            item = next(group)
        except StopIteration:
            return
        yield chain((item,), group)


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--n', default=10,
              help='How many recommendations should be returned by the model')
@click.option('--chunks-count', type=int, help='Numbers of chunks')
def main(model_filepath: str, output_filepath: str, n: int, chunks_count: int):
    """Calculates and saves predictions for the given model.

    Args:
        model_filepath (str): Path to a file containg model.
        output_filepath (str): Output filepath.
        n (int): Number of recommendations to return.
        chunks_count (int): In how many chunks split the users set
            during parallel processing.
    """
    logger.info('Loading model...')
    model = read_object(model_filepath)

    logger.info('Calculating predictions...')
    predictions = predict_model(model, n, chunks_count)

    logger.info('Appending results to %s...', output_filepath)
    with open(output_filepath, 'a') as f:
        predictions.to_csv(f, header=f.tell() == 0, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
