import click
import logging
import pandas as pd

from os import listdir
from os.path import basename, join
from ..utils.serialization import read_object
from ..models.cf_recommend_models import ICfRecommendationModel
from .metrics import precision

from surprise import Reader, Dataset, accuracy


def test_accuracy(model: ICfRecommendationModel, testset_filepath: str) -> float:
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
    return accuracy.rmse(est_ratings, verbose=False)


def test_to_read(model: ICfRecommendationModel, to_read_filepath: str) -> float:
    """Caulcates the precision of the given model using to_read data

    Args:
        model (ICfRecommendationModel): Model to test
        to_read_filepath (str): Path to a file containg to_read data

    Returns:
        float: Average precision for all users
    """
    to_read_df = pd.read_csv(to_read_filepath)

    def evaluate(group):
        to_read_ids = group['book_id'].values
        recommended_ids = model.recommend(group.name).keys()
        return precision(recommended_ids, to_read_ids)

    return to_read_df.groupby('user_id').apply(evaluate).mean()


@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.argument('testset_filepath', type=click.Path(exists=True))
@click.argument('to_read_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_directory: str, testset_filepath: str, to_read_filepath: str, output_filepath: str):
    logger = logging.getLogger(__name__)

    models_files = listdir(input_directory)
    models_files = [filename for filename in models_files
                    if filename.endswith('.pkl')]

    logger.info(f'Evaluating models from {input_directory}...')
    results = list()
    for model_file in models_files:
        model = read_object(join(input_directory, model_file))
        results.append((model_file, test_accuracy(model, testset_filepath),
                        test_to_read(model, to_read_filepath)))

    labels = ['model', 'rmse', 'to_read']
    results_df = pd.DataFrame.from_records(results, columns=labels)
    logger.info(f'Saving results to {output_filepath}...')
    results_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
