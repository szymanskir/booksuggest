import click
import logging
import pandas as pd
from typing import List, Tuple, Iterable, Any

from .cf_recommend_models import ICfRecommendationModel
from ..utils.serialization import read_object
import itertools

logger = logging.getLogger(__name__)


def predict_model(model: ICfRecommendationModel, recommendation_count: int, batch_size: int = 100000) -> pd.DataFrame:
    """Calculates top recommendations for every user in the trainset.

    Calculations are done in batches to avoid huge memory consumption.

    Args:
        model (ICfRecommendationModel): Already trained model.
        recommendation_count (int): Specifies how many recommendations to save.
        batch_size (int, optional): Defaults to 100000. Size of single batch.


    Returns:
        pd.DataFrame: Data frame with predictions.
    """
    batch_counter = 1
    main_df = pd.DataFrame(columns=['user_id', 'book_id', 'est'])
    for batchiter in _batch(model.generate_antitest_set(), batch_size):
        logger.debug(f"Batch: {batch_counter}")
        batch_counter += 1
        df = _predict_batch(model, list(batchiter), recommendation_count)
        main_df = main_df.append(df)

    return main_df


def _predict_batch(model: ICfRecommendationModel, cases_batch: List[Tuple[int, int, float]], recommendation_count: int) -> pd.DataFrame:
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
              help='How many recommendations are returned by the model')
def main(model_filepath: str, output_filepath: str, n: int):
    logger.info('Loading model...')
    model = read_object(model_filepath)

    logger.info('Calculating predictions...')
    predictions = predict_model(model, n)

    logger.info(f'Saving results to {output_filepath}...')
    predictions.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
