import click
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


def unify_book_ids(book_df, book_tags_df):
    """Change ids from book edition id to abstract book_id aggregating tags for all books editions.
    """
    book_ids_df = book_df[['book_id', 'goodreads_book_id']]
    book_tags_fixed_ids_df = book_tags_df.merge(
        book_ids_df, on='goodreads_book_id')
    return book_tags_fixed_ids_df[['tag_id', 'book_id', 'count']]


def filter_non_genres_tags(tags_df, genres):
    tags_filtered_df = tags_df[tags_df.tag_name.isin(genres)]
    return tags_filtered_df[['tag_id', 'tag_name']]


def join_tag_names(book_tags_df, tags_df):
    book_tags_joined_df = book_tags_df.merge(tags_df, on='tag_id')
    return book_tags_joined_df[['book_id', 'tag_name', 'count']]


@click.command()
@click.argument('book_filepath', type=click.Path(exists=True))
@click.argument('book_tags_filepath', type=click.Path(exists=True))
@click.argument('tags_filepath', type=click.Path(exists=True))
@click.argument('genres_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(book_filepath, book_tags_filepath, tags_filepath, genres_filepath, output_filepath):
    book_df = pd.read_csv(book_filepath)
    book_tags_df = pd.read_csv(book_tags_filepath)
    tags_df = pd.read_csv(tags_filepath)
    with open(genres_filepath) as file:
        goodreads_genres = [line.rstrip('\n') for line in file]

    book_tags_fixed_ids_df = unify_book_ids(book_df, book_tags_df)
    tags_filtered_df = filter_non_genres_tags(tags_df, goodreads_genres)
    book_tags_joined_df = join_tag_names(
        book_tags_fixed_ids_df, tags_filtered_df)
    book_tags_joined_df = book_tags_joined_df.sort_values(
        ['book_id', 'count'], ascending=[True, False])
    book_tags_joined_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
