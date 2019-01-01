import itertools
import logging
from typing import List, Tuple, Iterable, Any
import click
import pandas as pd

from .cf_recommend_models import ICfRecommendationModel
from ..utils.serialization import read_object

logger = logging.getLogger(__name__)


def predict_model(model: ICfRecommendationModel,
                  recommendation_count: int,
                  chunk_count: int,
                  chunk: int,
                  batch_size: int = 100000
                  ) -> pd.DataFrame:
    """Calculates top recommendations for every user in the trainset.

    Calculations are done in batches to avoid huge memory consumption.

    Args:
        model (ICfRecommendationModel): Already trained model.
        recommendation_count (int): Specifies how many recommendations to save.
        chunk_count (int): Number of chunks.
        chunk (int): Index of calculated chunk.
        batch_size (int, optional): Defaults to 100000. Size of single batch.

    Returns:
        pd.DataFrame: Data frame with predictions.
    """
    batch_counter = 1
    main_df = pd.DataFrame(columns=['user_id', 'book_id', 'est'])
    users = _get_users_chunk(list(model.users), chunk_count, chunk)
    batches = _batch(model.generate_antitest_set(users), batch_size)
    for batch in batches:
        logger.debug('Batch: %s', batch_counter)
        batch_counter += 1
        df = _predict_batch(model, list(batch), recommendation_count)
        main_df = main_df.append(df)

    return main_df


def _get_users_chunk(users, chunk_count, chunk):
    users_chunked = [users[start::chunk_count] for start in range(chunk_count)]
    return users_chunked[chunk]


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
        group = itertools.islice(iterable, batch_size)
        try:
            item = next(group)
        except StopIteration:
            return
        yield itertools.chain((item,), group)


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--n', default=10,
              help='How many recommendations should be returned by the model')
@click.option('--chunk-count', type=int, help='Numbers of chunks')
@click.option('--chunk', type=int,
              help='Number of chunk to calculate(starting from 0)')
def main(model_filepath: str, output_filepath: str, n: int, chunk_count: int, chunk: int):
    logger.info('Loading model...')
    model = read_object(model_filepath)

    logger.info('Calculating predictions...')
    predictions = predict_model(model, n, chunk_count, chunk)

    logger.info('Appending results to %s...', output_filepath)
    with open(output_filepath, 'a') as f:
        predictions.to_csv(f, header=f.tell() == 0, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
