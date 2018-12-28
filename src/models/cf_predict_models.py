import click
import logging
import pandas as pd
from typing import List, Tuple, Iterable, Any

from .cf_recommend_models import ICfRecommendationModel
from ..utils.serialization import read_object
import itertools

logger = logging.getLogger(__name__)


def predict_model(model: ICfRecommendationModel) -> pd.DataFrame:
    """Calculates top recommendations for every user in the trainset.

    Args:
        model (ICfRecommendationModel): Already trained model.

    Returns:
        pd.DataFrame: Data frame with predictions.
    """
    x = 1
    main_df = pd.DataFrame(columns=['user_id', 'book_id', 'est'])
    for batchiter in _batch(model.generate_antitest_set(), 100000):
        logger.debug(f"Batch: {x}")
        x += 1
        df = _predict_batch(model, list(batchiter))
        main_df = main_df.append(df)

    return main_df


def _predict_batch(model: ICfRecommendationModel, cases_batch: List[Tuple[int, int, float]]) -> pd.DataFrame:
    predictions = model.test(cases_batch)
    labels = ['user_id', 'book_id', '2', 'est', '4']
    pred_df = pd.DataFrame.from_records(
        predictions, exclude=['2', '4'], columns=labels)
    pred_df = pred_df.groupby('user_id')['book_id', 'est'].apply(
        lambda x: x.nlargest(model.recommendation_count, columns=['est']))
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
def main(model_filepath: str, output_filepath: str):
    logger.info('Loading model...')
    model = read_object(model_filepath)

    logger.info('Calculating predictions...')
    predictions = predict_model(model)

    logger.info(f'Saving results to {output_filepath}...')
    predictions.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
