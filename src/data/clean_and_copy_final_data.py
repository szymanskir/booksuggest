# -*- coding: utf-8 -*-
import click
import logging
import os
import pandas
from shutil import copyfile

logger = logging.getLogger(__name__)


def clean_ratings_and_to_read(ratings_df, to_read_df, output_data_dir):
    merged_df = ratings_df.merge(to_read_df, on=['user_id', 'book_id'],
                                 how='outer', indicator=True)

    ratings_clean_df = merged_df[merged_df['_merge'] == 'left_only']
    ratings_clean_df = ratings_clean_df[['user_id', 'book_id', 'rating']]
    ratings_clean_df.to_csv(os.path.join(
        output_data_dir, 'ratings-clean.csv'), index=False)

    to_read_clean_df = merged_df[merged_df['_merge'] == 'right_only']
    to_read_clean_df = to_read_clean_df[['user_id', 'book_id']]
    to_read_clean_df.to_csv(os.path.join(
        output_data_dir, 'to_read-clean.csv'), index=False)


def clean_tags_and_book_tags(tags_df, book_tags_df, genres, output_data_dir):
    tags_clean_df = tags_df[tags_df.tag_name.isin(genres)]
    tags_clean_df.to_csv(os.path.join(
        output_data_dir, 'tags-clean.csv'), index=False)

    book_tags_clean_df = book_tags_df.merge(tags_clean_df, on='tag_id')
    book_tags_clean_df = book_tags_clean_df[['book_id', 'tag_id']]
    book_tags_clean_df.to_csv(os.path.join(
        output_data_dir, 'book_tags-clean.csv'), index=False)


@click.command()
@click.argument('raw_data_dir', type=click.Path(exists=True))
@click.argument('external_data_dir', type=click.Path(exists=True))
@click.argument('interim_data_dir', type=click.Path(exists=True))
@click.argument('output_data_dir', type=click.Path())
def main(raw_data_dir, external_data_dir, interim_data_dir, output_data_dir):
    logger.info(f"Cleaning data...")

    ratings_df = pandas.read_csv(os.path.join(raw_data_dir, 'ratings.csv'))
    to_read_df = pandas.read_csv(os.path.join(raw_data_dir, 'to_read.csv'))
    clean_ratings_and_to_read(ratings_df, to_read_df, interim_data_dir)

    tags_df = pandas.read_csv(os.path.join(raw_data_dir, 'tags.csv'))
    book_tags_df = pandas.read_csv(os.path.join(
        interim_data_dir, 'book_tags-unified_ids.csv'))
    with open(os.path.join(external_data_dir, 'genres.txt')) as file:
        goodreads_genres = [line.rstrip('\n') for line in file]
    clean_tags_and_book_tags(tags_df, book_tags_df,
                             goodreads_genres, interim_data_dir)

    logger.info(f"Data cleaned")

    logger.info(f"Moving final data...")

    copyfile(os.path.join(interim_data_dir, 'ratings-clean.csv'),
             os.path.join(output_data_dir, 'ratings.csv'))
    copyfile(os.path.join(interim_data_dir, 'to_read-clean.csv'),
             os.path.join(output_data_dir, 'to_read.csv'))
    copyfile(os.path.join(interim_data_dir, 'tags-clean.csv'),
             os.path.join(output_data_dir, 'tags.csv'))
    copyfile(os.path.join(interim_data_dir, 'book_tags-clean.csv'),
             os.path.join(output_data_dir, 'book_tags.csv'))
    copyfile(os.path.join(interim_data_dir, 'similar_books-unified_ids.csv'),
             os.path.join(output_data_dir, 'similar_books.csv'))
    copyfile(os.path.join(interim_data_dir, 'book-unified_ids.csv'),
             os.path.join(output_data_dir, 'book.csv'))
    copyfile(os.path.join(interim_data_dir, 'ratings-clean.csv'),
             os.path.join(output_data_dir, 'ratings.csv'))

    logger.info(f"Data moved")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
