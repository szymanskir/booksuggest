import pandas as pd
import click
import logging

from surprise import KNNBaseline, SVD
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise.model_selection.validation import cross_validate
from surprise.model_selection.split import KFold
from os.path import join


def svd_grid_search(dataset):
    params = {'n_epochs': [20, 25],
              'lr_all': [0.002, 0.010],
              'reg_all': [0.01, 0.4]}
    algo = SVD
    return (algo, _perform_grid_search(algo, params, dataset))


def knn_grid_search(dataset):
    params = {'bsl_options': {'method': ['als', 'sgd'],
                              'reg': [1, 2]},
              'k': [20, 40, 60],
              'sim_options': {'name': ['msd', 'cosine', 'pearson_baseline'],
                              'min_support': [1, 5],
                              'user_based': [False]},
              'verbose': [False]}
    algo = KNNBaseline
    return (algo, _perform_grid_search(algo, params, dataset))


def _perform_grid_search(algo_class, param_grid, dataset):
    gs = GridSearchCV(algo_class, param_grid, measures=['rmse', 'mae'], cv=5)
    gs.fit(dataset)
    return pd.DataFrame.from_dict(gs.cv_results).sort_values('rank_test_rmse')


def validate_best(results_df, full_dataset, algo_func):
    best_five = results_df.sort_values('rank_test_rmse').head(3)
    for row in best_five.itertuples():
        print(f"{row.rank_test_rmse}: {row.params}")
        cv_iter = KFold(n_splits=5, random_state=44)
        cross_validate(algo_func(**row.params), full_dataset,
                       cv=cv_iter, verbose=True)


@click.command()
@click.argument('ratings_filepath', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--model', type=click.Choice(['knn', 'svd']))
def main(ratings_filepath: str, output_dir: str, model: str):
    logger = logging.getLogger(__name__)

    ratings_df = pd.read_csv(ratings_filepath)
    ratings_minified_df = ratings_df[ratings_df['user_id'] <= 100]
    reader = Reader(rating_scale=(1, 5))
    search_dataset = Dataset.load_from_df(
        ratings_minified_df[['user_id', 'book_id', 'rating']], reader)

    logger.info('Performing grid search for parameters...')
    if model == 'knn':
        gs_func = knn_grid_search
    elif model == 'svd':
        gs_func = svd_grid_search
    else:
        raise ValueError
    algo, results_df = gs_func(search_dataset)

    output_filepath = join(output_dir, f"{model}-grid_search-results.csv")
    logger.info('Saving results to %s...', output_filepath)
    results_df.to_csv(output_filepath)

    logger.info('Evaluating best parameters on full dataset...')
    full_dataset = Dataset.load_from_df(
        ratings_df[['user_id', 'book_id', 'rating']], reader)
    validate_best(results_df, full_dataset, algo)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
