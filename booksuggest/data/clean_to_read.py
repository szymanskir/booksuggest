import logging

import click
import pandas as pd


@click.command()
@click.argument('to_read_filepath', type=click.Path(exists=True))
@click.argument('ratings_trainset_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(to_read_filepath: str, ratings_trainset_filepath: str,
         output_filepath: str):
    logger = logging.getLogger(__name__)
    logger.info('Cleaning already rated books from to_read...')

    to_read_df = pd.read_csv(to_read_filepath)
    ratings_df = pd.read_csv(ratings_trainset_filepath)

    merged_df = to_read_df.merge(
        ratings_df, on=['user_id', 'book_id'], how='left')
    merged_df = merged_df[merged_df['rating'].isnull()]
    merged_df = merged_df.drop(
        'rating', axis=1).sort_values(['user_id', 'book_id'])

    logger.info('Saving results to %s...', output_filepath)
    merged_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
