import click
import logging
import numpy as np
import pandas as pd

from typing import List


class InvalidTagFeaturesData(Exception):
    pass


def build_all_tag_features(
        book_tags: pd.DataFrame,
        tags: pd.DataFrame
) -> List[List[float]]:
    book_tags_grouped = book_tags.groupby(by='book_id')
    tag_features = book_tags_grouped.apply(
        lambda single_book_tags: pd.Series(
            build_tag_features(single_book_tags, tags)
        )
    )

    return tag_features.reset_index().set_index('book_id')


def build_tag_features(
        book_tags: pd.DataFrame,
        tags: pd.DataFrame
) -> List[float]:
    """Builds tags based features for a single book.

        Args:
            book_tags: data frame containing book_id and tag_id columns
            tags: data frame containing tag_id and tag_names columns
    """
    logging.debug('Building single feature...')
    if check_book_tags_ans_tags_compatibility(book_tags, tags):
        raise InvalidTagFeaturesData

    tags_data = tags.merge(book_tags, on='tag_id', how='left')
    tags_counts = tags_data['count'].fillna(0)
    feature_vector = tags_counts/tags_counts.sum()
    return feature_vector.tolist()


def check_book_tags_ans_tags_compatibility(
        book_tags: pd.DataFrame,
        tags: pd.DataFrame
) -> bool:
    """Checks if the book tags and tags data are compatible.

        Args:
            book_tags: data frame containg information about tags assigned
                to books
            tags: data frame containg information about tags

        Returns:
            bool: True if data frames are compatible
    """
    if not validate_book_tags_data(book_tags):
        return False

    if not validate_tags_data(tags):
        return False

    return set(book_tags['tag_id']) < set(tags['tag_id'])


def validate_book_tags_data(book_tags: pd.DataFrame) -> bool:
    """Checks if the book tags data frame contains valid data.

        Args:
            book_tags: data frame to check

        Returns:
            bool: True if the data frame is a valid book_tags data frame
    """
    required_columns = {'book_id', 'tag_id'}

    if not required_columns < set(book_tags.columns):
        return False

    book_ids = book_tags['book_id'].unique()
    concerns_single_book = len(book_ids) == 1
    is_book_relevant = not np.isnan(book_ids[0])

    return all([
        concerns_single_book,
        is_book_relevant
    ])


def validate_tags_data(tags: pd.DataFrame) -> bool:
    """Checks if the tags data frame contains valid data.

        Args:
            tags: data frame to check

        Returns:
            bool: True if the data frame is a valid book_tags data frame
    """
    required_columns = {'tag_id', 'tag_names'}
    return required_columns < set(tags.columns)


@click.command()
@click.argument('book_tags_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(book_tags_filepath, output_filepath):
    book_tags = pd.read_csv(book_tags_filepath)
    tags = pd.DataFrame({'tag_id': list(book_tags['tag_id'].unique())})

    tag_features_df = build_all_tag_features(
        book_tags, tags
    )

    tag_features_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
