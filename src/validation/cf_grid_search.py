import pandas as pd
import click
import logging

from typing import Any, Callable, Dict, Tuple

from surprise import AlgoBase, KNNBaseline, SVD
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise.model_selection.validation import cross_validate
from surprise.model_selection.split import KFold
import numpy as np
from sklearn.utils.random import sample_without_replacement

logger = logging.getLogger(__name__)


def knn_grid_search(dataset: Dataset) -> Tuple[AlgoBase, pd.DataFrame]:
    """Performs a grid searcg procedure for KNN model.

    Args:
        dataset (Dataset): Dataset to run tests on.

    Returns:
        Tuple[AlgoBase, pd.DataFrame]: `(model_constructor, best_parameters)`
    """
    params = {'bsl_options': {'method': ['als'],
                              'reg_i': [10, 5, 15],
                              'reg_u': [15, 10, 20],
                              'n_epochs': [10, 15]},
              'k': [20, 40, 60],
              'sim_options': {'name': ['msd', 'cosine', 'pearson_baseline'],
                              'min_support': [1, 5],
                              'user_based': [True, False],
                              'shrinkage': [100, 10, 200]},
              'verbose': [False]}
    algo = KNNBaseline
    return (algo, _perform_grid_search(algo, params, dataset))


def svd_grid_search(dataset: Dataset, random_state: int
                    ) -> Tuple[AlgoBase, pd.DataFrame]:
    """Performs a grid searcg procedure for SVD model.

    Args:
        dataset (Dataset): Dataset to run tests on.

    Returns:
        Tuple[AlgoBase, pd.DataFrame]: `(model_constructor, best_parameters)`
    """
    params = {'n_factors': [50, 100, 200, 500],
              'biased': [False, True],
              'init_std_dev': [0.1, 0.05, 0.2],
              'n_epochs': [20, 25, 30],
              'lr_all': [0.002, 0.001, 0.010, 0.050],
              'reg_all': [0.01, 0.1, 0.4],
              'random_state': [random_state]}
    algo = SVD
    return (algo, _perform_grid_search(algo, params, dataset))


def _perform_grid_search(algo_class: AlgoBase, param_grid: Dict[str, Any],
                         dataset: Dataset) -> pd.DataFrame:
    gs = GridSearchCV(algo_class, param_grid, measures=['rmse', 'mae', 'fcp'],
                      cv=5, n_jobs=4, joblib_verbose=10)
    gs.fit(dataset)
    return pd.DataFrame.from_dict(gs.cv_results).sort_values('rank_test_rmse')


def test_best_parameters(parameters_df: pd.DataFrame, full_dataset: Dataset,
                         algo_func: Callable[..., AlgoBase], random_state: int
                         ) -> pd.DataFrame:
    """Performs a 5-fold cross validation for best 5 parameters combinations.

    Args:
        parameters_df (pd.DataFrame): Dataframe with parameters values.
        full_dataset (Dataset): Dataset to perform cross validation.
        algo_func (Callable[..., AlgoBase]): Model constructor.

    Returns:
        pd.DataFrame: Metrics for the given parameters combinations.
    """
    best_performers = parameters_df.sort_values('rank_test_rmse').head(5)
    results = list()
    for row in best_performers.itertuples():
        logger.info("[%s]: %s",  row.rank_test_rmse, row.params)
        cv_iter = KFold(n_splits=5, random_state=random_state)
        result = cross_validate(algo_func(**row.params), full_dataset,
                                measures=['rmse', 'mae', 'fcp'], cv=cv_iter,
                                verbose=True)
        results.append((row.rank_test_rmse,
                        np.mean(result['test_rmse']),
                        np.mean(result['test_mae']),
                        np.mean(result['test_fcp']),
                        row.params))

    labels = ['search_rank', 'test_rmse', 'test_mae', 'test_fcp', 'params']
    df = pd.DataFrame.from_records(
        results, columns=labels).sort_values('test_rmse')
    return df


@click.command()
@click.argument('ratings_filepath', type=click.Path(exists=True))
@click.argument('params_output_filepath', type=click.Path())
@click.argument('metrics_output_filepath', type=click.Path())
@click.option('--model', type=click.Choice(['knn', 'svd']))
@click.option('--random-state', type=int, default=None)
def main(ratings_filepath: str, params_output_filepath: str,
         metrics_output_filepath: str, model: str, random_state: int):
    """Searchs over model parameters values to find best combination.

    Args:
        ratings_filepath (str): Path to a file with ratings data,
        output_dir (str): Output directory.
        model (str): Model type shortname. Values: `['knn', 'svd']`

    Raises:
        ValueError: When `model` is out of the specified range.
    """
    ratings_df = pd.read_csv(ratings_filepath)
    users_count = len(ratings_df['user_id'].unique())
    samples = 200 if 200 < users_count else users_count / 2
    users_subset = set(sample_without_replacement(
        users_count, samples, random_state=random_state))
    ratings_minified_df = ratings_df[ratings_df['user_id'].isin(users_subset)]
    reader = Reader(rating_scale=(1, 5))
    search_dataset = Dataset.load_from_df(
        ratings_minified_df[['user_id', 'book_id', 'rating']], reader)

    logger.info('Searching parameters values for %s model...', model)
    model_func = {
        'knn': lambda x: knn_grid_search(x),
        'svd': lambda x: svd_grid_search(x, random_state=random_state)}

    try:
        algo, parameters_df = model_func[model](search_dataset)
    except KeyError:
        raise KeyError("Model shortname not predefined!")

    logger.info('Saving parameters values to %s...', params_output_filepath)
    parameters_df.to_csv(params_output_filepath, index=False)

    logger.info('Evaluating best parameters on full dataset...')
    full_dataset = Dataset.load_from_df(
        ratings_df[['user_id', 'book_id', 'rating']], reader)
    full_dataset_metrics_df = test_best_parameters(
        parameters_df, full_dataset, algo, random_state)

    logger.info('Saving metrics for best parameters on full dataset to %s...',
                metrics_output_filepath)
    full_dataset_metrics_df.to_csv(metrics_output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
