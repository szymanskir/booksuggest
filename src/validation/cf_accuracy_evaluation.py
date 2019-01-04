import logging
from os import listdir
from os.path import join

import click
import pandas as pd
from surprise import Reader, Dataset, accuracy

from ..utils.serialization import read_object
from ..models.cf_recommend_models import ICfRecommendationModel


def test_accuracy(
        model: ICfRecommendationModel,
        testset_filepath: str
) -> float:
    """Calculates RMSE value for the given model and testset

    Args:
        model (ICfRecommendationModel): Model to test
        testset_filepath (str): Path to a file containing testset

    Returns:
        float: Value of the Root Mean Squared Error metric
    """
    ratings_df = pd.read_csv(testset_filepath)
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(
        ratings_df[['user_id', 'book_id', 'rating']], reader)
    testset = dataset.build_full_trainset().build_testset()
    est_ratings = model.test(testset)
    return (accuracy.rmse(est_ratings, verbose=False),
            accuracy.mae(est_ratings, verbose=False),
            accuracy.fcp(est_ratings, verbose=False))


@click.command()
@click.argument('models_dir', type=click.Path(exists=True))
@click.argument('testset_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(models_dir: str, testset_filepath: str, output_filepath: str):
    """Evaluates RMSE, MAE and FCP metrics of models on the given testset.

    Args:
        models_dir (str): Directory with models to test.
        testset_filepath (str): Path to a file with test data.
        output_filepath (str): Output filepath.
    """
    logger = logging.getLogger(__name__)

    models_files = listdir(models_dir)
    models_files = [filename for filename in models_files
                    if filename.endswith('.pkl')]

    logger.info('Evaluating models from %s...', models_dir)
    results = list()
    for model_file in models_files:
        model = read_object(join(models_dir, model_file))
        result = [model_file] + list(test_accuracy(model, testset_filepath))
        results.append(result)

    labels = ['model', 'rmse', 'mae', 'fcp']
    results_df = pd.DataFrame.from_records(results, columns=labels)
    logger.info('Saving results to %s...', output_filepath)
    results_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
