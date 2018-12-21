import pandas as pd
import click
import logging

from xml_parser import extract_all_book_xml_roots

from typing import List, Tuple
from lxml import etree

logger = logging.getLogger(__name__)


def extract_similar_books(xmls_dir: str) -> List[etree.Element]:
    similar_books_rows = list()
    for book in extract_all_book_xml_roots(xmls_dir):
        similar_books_rows.extend(_extract_similar_books_(book))

    return similar_books_rows


def _extract_similar_books_(book: etree.Element) -> List[Tuple[int, int]]:
    similar_books_rows = list()
    book_work_id = book.find("work").findtext("id")
    similar_books = book.find("similar_books")
    if similar_books is not None:
        for similar_book in similar_books.findall("book"):
            similar_book_id = similar_book.find("work").findtext("id")
            similar_books_rows.append(
                (int(book_work_id), int(similar_book_id)))
    return similar_books_rows


def process_similar_books(similar_books_rows: List[Tuple[int, int]]) -> pd.DataFrame:
    similar_book_labels = ['work_id', 'similar_book_work_id']
    similar_book_df = pd.DataFrame.from_records(
        similar_books_rows, columns=similar_book_labels)
    return similar_book_df


def switch_to_book_id(similar_books_df: pd.DataFrame, book_df: pd.DataFrame) -> pd.DataFrame:
    """Change ids from work id to abstract book_id
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

    return merged_similar_book_df


@click.command()
@click.argument('books_xml_dir', type=click.Path(exists=True))
@click.argument('books_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(books_xml_dir: str, books_filepath: str, output_filepath: str):
    similar_books_rows = extract_similar_books(books_xml_dir)
    similar_books_df = process_similar_books(similar_books_rows)

    book_df = pd.read_csv(books_filepath)
    similar_books_switched_ids_df = switch_to_book_id(
        similar_books_df, book_df)

    similar_books_switched_ids_df.to_csv(output_filepath, index=False)
    logger.info(f"Created: {output_filepath}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
