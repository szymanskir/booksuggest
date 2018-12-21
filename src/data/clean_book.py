import click
import logging
import pandas as pd

from xml_parser import extract_all_book_xml_roots

logger = logging.getLogger(__name__)


def extract_book_extra_info(xmls_dir):
    book_extra_info_rows = list()
    for book in extract_all_book_xml_roots(xmls_dir):
        book_extra_info_rows.append(_extract_book_info_(book))

    return book_extra_info_rows


def _extract_book_info_(book):
    book_work_id = book.find("work").find("id").text
    book_info = [int(book_work_id), book.find(
        "isbn13").text, book.find("description").text]
    return book_info


def process_book_extra_info(book_extra_info_rows):
    book_info_labels = ['work_id', 'isbn13', 'description']
    book_extra_info_df = pd.DataFrame.from_records(
        book_extra_info_rows, columns=book_info_labels)

    # clean html tags from description column
    html_tag_regexp = '<[^<]+?>'
    multiple_whitespaces_regexp = r"\s\s+"
    book_extra_info_df.description = book_extra_info_df.description.astype(
        str).str.replace(html_tag_regexp, ' ')
    book_extra_info_df.description = book_extra_info_df.description.astype(
        str).str.replace(multiple_whitespaces_regexp, ' ')

    return book_extra_info_df


def merge_book_data(book_df, book_extra_info_df):
    book_df = book_df.drop(columns=['isbn13'])
    book_merged_data_df = book_df.merge(book_extra_info_df, on='work_id')
    return book_merged_data_df


@click.command()
@click.argument('book_filepath', type=click.Path(exists=True))
@click.argument('books_xml_dir', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(book_filepath, books_xml_dir, output_filepath):
    book_extra_info_rows = extract_book_extra_info(books_xml_dir)
    book_extra_info_df = process_book_extra_info(book_extra_info_rows)

    book_df = pd.read_csv(book_filepath)
    book_merged_data_df = merge_book_data(book_df, book_extra_info_df)

    book_merged_data_df.to_csv(output_filepath, index=False)
    logger.info(f"Created: {output_filepath}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()