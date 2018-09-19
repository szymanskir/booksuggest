# -*- coding: utf-8 -*-
import click
import logging
import os
import pandas

logger = logging.getLogger(__name__)


def unify_book(book_df, book_info_df, output_data_dir):
    """
    Append additional data columns to book.csv
    """

    book_df = book_df.drop(columns=['isbn13'])
    merged_df = book_df.merge(book_info_df, on='work_id')

    merged_df.to_csv(os.path.join(output_data_dir, 'book.csv'), index=False)


def unify_book_tags(book_df, book_tags_df, output_data_dir):
    """
    Change ids from book edition id to abstract book_id.
    Thus, aggregates tags for all books editions.
    """
    book_ids_df = book_df[['book_id', 'goodreads_book_id']]
    merged_df = book_tags_df.merge(book_ids_df, on='goodreads_book_id')
    merged_df = merged_df.drop(columns=['goodreads_book_id'])

    merged_df.to_csv(os.path.join(
        output_data_dir, 'book_tags.csv'), index=False)


def unify_similar_books(book_df, similar_books_df, output_data_dir):
    """
    Change ids from work id to abstract book_id
    """
    book_ids_df = book_df[['book_id', 'work_id']]
    book_ids_df = book_ids_df.rename(
        columns={'book_id': 'tmp_book_id', 'work_id': 'tmp_work_id'})

    merged_book_df = similar_books_df.merge(
        book_ids_df, left_on='work_id', right_on='tmp_work_id')
    merged_book_df = merged_book_df[['tmp_book_id', 'similar_book_work_id']]
    merged_book_df = merged_book_df.rename(columns={'tmp_book_id': 'book_id'})

    merged_similar_book_df = merged_book_df.merge(
        book_ids_df, left_on='similar_book_work_id', right_on='tmp_work_id')
    merged_similar_book_df = merged_similar_book_df[['book_id', 'tmp_book_id']]
    merged_similar_book_df = merged_similar_book_df.rename(
        columns={'tmp_book_id': 'similar_book_id'})

    merged_similar_book_df.to_csv(
        os.path.join(output_data_dir, 'similar_books.csv'), index=False)


@click.command()
@click.argument('raw_data_dir', type=click.Path(exists=True))
@click.argument('interim_data_dir', type=click.Path(exists=True))
@click.argument('output_data_dir', type=click.Path())
def main(raw_data_dir, interim_data_dir, output_data_dir):
    """
    Unifies identifiers used throughout data files.
    All books references are now using book_id.

    :param raw_data_dir: raw data directory
    :type raw_data_dir: string
    :param interim_data_dir: intermediate data directory
    :type interim_data_dir: string
    :param output_data_dir: output data directory
    :type output_data_dir: string
    """

    logger.info(f"Unifying ids in data files...")

    book_df = pandas.read_csv(os.path.join(raw_data_dir, 'book.csv'))
    book_info_df = pandas.read_csv(
        os.path.join(interim_data_dir, 'book_info.csv'))
    book_tags_df = pandas.read_csv(os.path.join(raw_data_dir, 'book_tags.csv'))
    similar_books_df = pandas.read_csv(
        os.path.join(interim_data_dir, 'similar_books.csv'))

    unify_book(book_df, book_info_df, output_data_dir)
    unify_book_tags(book_df, book_tags_df, output_data_dir)
    unify_similar_books(book_df, similar_books_df, output_data_dir)

    logger.info(f"Completed unifying ids")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
