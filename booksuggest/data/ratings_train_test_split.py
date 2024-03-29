import logging

import click
import pandas as pd

from sklearn.model_selection import train_test_split


@click.command()
@click.argument('ratings_filepath', type=click.Path(exists=True))
@click.argument('trainset_filepath', type=click.Path())
@click.argument('testset_filepath', type=click.Path())
def main(ratings_filepath: str, trainset_filepath: str, testset_filepath: str):
    """Splits the data about ratings into traning and test datasets.

    Split function is randomized, but with predefined seed to
    allow for reproducible results.

    Args:
        ratings_filepath (str): Ratings data frame filepath.
        trainset_filepath (str): Output filepath for training dataset.
        testset_filepath (str): Output filepath for test dataset.
    """

    logging.info('Splitting ratings data into training and test sets...')
    ratings_df = pd.read_csv(ratings_filepath, index_col=[
        'user_id', 'book_id'])
    train_df, test_df = train_test_split(
        ratings_df, test_size=0.1, random_state=44)

    train_df = train_df.sort_index()
    test_df = test_df.sort_index()

    logging.info(
        'Saving sets to: %s, %s...',
        trainset_filepath,
        testset_filepath
    )
    train_df.to_csv(trainset_filepath)
    test_df.to_csv(testset_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
