import pandas as pd
import click
import logging

from typing import Any, Dict, Tuple

from surprise import AlgoBase, KNNBaseline, SVD
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise.model_selection.split import KFold
from sklearn.utils.random import sample_without_replacement

logger = logging.getLogger(__name__)


def knn_grid_search(dataset: Dataset,
                    random_state: int) -> Tuple[AlgoBase, pd.DataFrame]:
    """Performs a grid searcg procedure for KNN model.

    Args:
        dataset (Dataset): Dataset to run tests on.
        random_state (int): Value for random seed.

    Returns:
        Tuple[AlgoBase, pd.DataFrame]: `(model_constructor, best_parameters)`
    """
    params = {
        'bsl_options': {
            'method': ['als'],
            'reg_i': [10],
            'reg_u': [15],
            'n_epochs': [10]
        },
        'k': [30],
        'sim_options': {
            'name': ['pearson_baseline'],
            'min_support': [1],
            'user_based': [False],
            'shrinkage': [100]
        },
        'verbose': [False]
    }
    algo = KNNBaseline
    return (algo, _perform_grid_search(algo, params, dataset, random_state))


def svd_grid_search(dataset: Dataset,
                    random_state: int) -> Tuple[AlgoBase, pd.DataFrame]:
    """Performs a grid searcg procedure for SVD model.

    Args:
        dataset (Dataset): Dataset to run tests on.
        random_state (int): Value for random seed.

    Returns:
        Tuple[AlgoBase, pd.DataFrame]: `(model_constructor, best_parameters)`
    """
    params = {
        'n_factors': [100],
        'biased': [True],
        'init_mean': [0.1],
        'init_std_dev': [0.05],
        'n_epochs': [25],
        'lr_all': [0.005],
        'reg_all': [0.02],
        'random_state': [random_state]
    }
    algo = SVD
    return (algo, _perform_grid_search(algo, params, dataset, random_state))


def _perform_grid_search(algo_class: AlgoBase, param_grid: Dict[str, Any],
                         dataset: Dataset, random_state: int) -> pd.DataFrame:
    gs = GridSearchCV(
        algo_class,
        param_grid,
        measures=['rmse', 'mae', 'fcp'],
        cv=KFold(5, random_state=random_state),
        n_jobs=2,
        joblib_verbose=100,
        pre_dispatch=2)
    gs.fit(dataset)
    return pd.DataFrame.from_dict(gs.cv_results).sort_values('rank_test_rmse')


def _minify_dataset(ratings_df: pd.DataFrame,
                    random_state: int) -> pd.DataFrame:
    users_count = len(ratings_df['user_id'].unique())
    samples = 200 if 200 < users_count else users_count / 2
    users_subset = set(
        sample_without_replacement(
            users_count, samples, random_state=random_state))
    return ratings_df[ratings_df['user_id'].isin(users_subset)]


@click.command()
@click.argument('ratings_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--model', type=click.Choice(['knn', 'svd']))
@click.option('--random-state', type=int, default=None)
@click.option('--use-subset', is_flag=True)
def main(ratings_filepath: str, output_filepath: str, model: str,
         random_state: int, use_subset: bool):
    """Searchs over model parameters values to find best combination.

    Args:
        ratings_filepath (str): Path to a file with ratings data,
        output_filepath (str): Output filepath.
        model (str): Model type shortname. Values: `['knn', 'svd']`
        random_state (int): Value for random seed.
        use_subset (bool): Whether to use a subset of 200 random users.

    Raises:
        KeyError: When `model` is out of the specified range.
    """
    ratings_df = pd.read_csv(ratings_filepath)

    if use_subset:
        ratings_df = _minify_dataset(ratings_df, random_state)

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(
        ratings_df[['user_id', 'book_id', 'rating']], reader)

    logger.info('Searching parameters values for %s model...', model)
    model_func = {
        'knn': lambda x: knn_grid_search(x, random_state=random_state),
        'svd': lambda x: svd_grid_search(x, random_state=random_state)
    }

    try:
        algo, parameters_df = model_func[model](dataset)
    except KeyError:
        raise KeyError("Model shortname not predefined!")

    logger.info('Saving parameters values to %s...', output_filepath)
    parameters_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
