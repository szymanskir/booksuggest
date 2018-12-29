import click
import logging
import pandas as pd

from typing import List

logger = logging.getLogger(__name__)


def switch_to_book_id(book_df: pd.DataFrame, book_tags_df: pd.DataFrame) -> pd.DataFrame:
    """Changes ids from ``goodreads_book_id`` to abstract ``book_id``, thus aggregating tags for all books editions.

    Args:
        book_df (pd.DataFrame): Books data frame.
        book_tags_df (pd.DataFrame): Book tags data frame.

    Returns:
        pd.DataFrame: Books data frame with switched ids.
    """"""
    """
    book_ids_df = book_df[['book_id', 'goodreads_book_id']]
    book_tags_fixed_ids_df = book_tags_df.merge(
        book_ids_df, on='goodreads_book_id')
    return book_tags_fixed_ids_df[['tag_id', 'book_id', 'count']]


def filter_non_genres_tags(tags_df: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
    """Filters out the non-genre tags.

    Args:
        tags_df (pd.DataFrame): Tags data frame.
        genres (List[str]): List of genres.

    Returns:
        pd.DataFrame: Filtered tags data frame.
    """
    tags_filtered_df = tags_df[tags_df.tag_name.isin(genres)]
    return tags_filtered_df[['tag_id', 'tag_name']]


def join_tag_names(book_tags_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
    """Replaces ``tag_id`` column with ``tag_name`` data.

    Args:
        book_tags_df (pd.DataFrame): Book tags data frame.
        tags_df (pd.DataFrame): Tags data frame.

    Returns:
        pd.DataFrame: Book tags data frame with tag names and tag ids.
    """
    book_tags_joined_df = book_tags_df.merge(tags_df, on='tag_id')
    return book_tags_joined_df[['book_id', 'tag_name', 'count']]


@click.command()
@click.argument('book_filepath', type=click.Path(exists=True))
@click.argument('book_tags_filepath', type=click.Path(exists=True))
@click.argument('tags_filepath', type=click.Path(exists=True))
@click.argument('genres_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(book_filepath: str, book_tags_filepath: str, tags_filepath: str, genres_filepath: str, output_filepath: str):
    """Cleans up the book tags data.

    Aggregates tags for all editions for specific book by switching to ``book_id``, filters out tags which do not indicate book genre and replaces tags ids with names.

    Args:
        book_filepath (str): Book data frame filepath.
        book_tags_filepath (str): Book tags data frame filepath.
        tags_filepath (str): Tags data frame filepath.
        genres_filepath (str): Path to a file containing valid genre names.
        output_filepath (str): Output filepath.
    """
    book_df = pd.read_csv(book_filepath)
    book_tags_df = pd.read_csv(book_tags_filepath)
    tags_df = pd.read_csv(tags_filepath)
    with open(genres_filepath) as file:
        goodreads_genres = [line.rstrip('\n') for line in file]

    book_tags_fixed_ids_df = switch_to_book_id(book_df, book_tags_df)
    tags_filtered_df = filter_non_genres_tags(tags_df, goodreads_genres)
    book_tags_joined_df = join_tag_names(
        book_tags_fixed_ids_df, tags_filtered_df)
    book_tags_joined_df = book_tags_joined_df.sort_values(
        ['book_id', 'count'], ascending=[True, False])
    book_tags_joined_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
