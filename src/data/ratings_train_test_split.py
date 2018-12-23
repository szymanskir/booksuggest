import click
import logging

from sklearn.model_selection import train_test_split
import pandas as pd

logger = logging.getLogger(__name__)


@click.command()
@click.argument('ratings_filepath', type=click.Path(exists=True))
@click.argument('trainset_filepath', type=click.Path())
@click.argument('testset_filepath', type=click.Path())
def main(ratings_filepath: str, trainset_filepath: str, testset_filepath: str):
    """Splits the data about ratings into traning and test datasets.

    Split function is randomized, but with predefined seed to allow for reproducible results.

    Args:
        ratings_filepath (str): Ratings data frame filepath.
        trainset_filepath (str): Output filepath for training dataset.
        testset_filepath (str): Output filepath for test dataset.
    """

    logger.info('Splitting ratings data into training and test sets...')
    ratings_df = pd.read_csv(ratings_filepath, index_col=[
                             'user_id', 'book_id'])
    train, test = train_test_split(ratings_df, test_size=0.1, random_state=44)

    logger.info(f'Saving sets to: {trainset_filepath}, {testset_filepath}...')
    train.to_csv(trainset_filepath)
    test.to_csv(testset_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
