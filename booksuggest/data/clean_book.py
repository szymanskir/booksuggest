import logging

from typing import List, Tuple

import click
import pandas as pd

from lxml import etree

from .xml_parser import extract_all_book_xml_roots


def extract_book_extra_info(xmls_dir: str) -> List[Tuple[int, str, str]]:
    """Extracts extra information about books from .xml files in given directory.

    Args:
        xmls_dir (str): Directory with .xml files.

    Returns:
        List[Tuple[int, str, str]]:
            List of ``(work_id, isbn13, description)`` book data.
    """
    return [_extract_book_info(book)
            for book in extract_all_book_xml_roots(xmls_dir)]


def _extract_book_info(book: etree.Element) -> Tuple[int, str, str]:
    book_work_id = book.find("work").findtext("id")
    book_info = (int(book_work_id), book.findtext(
        "isbn13"), book.findtext("description"))
    return book_info


def process_book_extra_info(
        book_extra_info_rows: List[Tuple[int, str, str]]
) -> pd.DataFrame:
    """Processes books extra data and transforms it to a data frame.

    Apart from joining data it also removes HTML tags from
    the ``description`` column.

    Args:
        book_extra_info_rows (List[Tuple[int, str, str]]):
            List of ``(work_id, isbn13, description)`` book data.

    Returns:
        pd.DataFrame: Data frame with columns corresponding to input tuple.
    """
    book_info_labels = ['work_id', 'isbn13', 'description']
    book_extra_info_df = pd.DataFrame.from_records(
        book_extra_info_rows, columns=book_info_labels)

    html_tag_regexp = '<[^<]+?>'
    multiple_whitespaces_regexp = r"\s\s+"
    book_extra_info_df.description = book_extra_info_df.description.astype(
        str).str.replace(html_tag_regexp, ' ')
    book_extra_info_df.description = book_extra_info_df.description.astype(
        str).str.replace(multiple_whitespaces_regexp, ' ')

    return book_extra_info_df


def merge_book_data(
        book_df: pd.DataFrame,
        book_extra_info_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges standard book data with new extra info.

    Args:
        book_df (pd.DataFrame): Book data frame.
        book_extra_info_df (pd.DataFrame): Book extra info data frame.

    Returns:
        pd.DataFrame: Book data frame with additional ``description`` column.
    """
    book_df = book_df.drop(columns=['isbn13'])
    book_merged_data_df = book_df.merge(book_extra_info_df, on='work_id')
    return book_merged_data_df


@click.command()
@click.argument('book_filepath', type=click.Path(exists=True))
@click.argument('books_xml_dir', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(book_filepath: str, books_xml_dir: str, output_filepath: str):
    """Extracts additional data about books from .xml files
       and joins it with previous data.

    Args:
        book_filepath (str): Book data frame.
        books_xml_dir (str): Directory with books .xml files.
        output_filepath (str): Output filepath.
    """
    book_extra_info_rows = extract_book_extra_info(books_xml_dir)
    book_extra_info_df = process_book_extra_info(book_extra_info_rows)

    book_df = pd.read_csv(book_filepath)
    book_merged_data_df = merge_book_data(book_df, book_extra_info_df)

    book_merged_data_df.to_csv(output_filepath, index=False)
    logging.info('Created: %s', output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
