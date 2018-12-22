# -*- coding: utf-8 -*-
import click
import logging
import os
import zipfile
import shutil
import pandas
from glob import glob
from lxml import etree

logger = logging.getLogger(__name__)


def unzip_archive(xml_archive_path):
    """
    Extracts the archive given by xml_archive_path
    and returns the filepath to extracted files.
    :param xml_archive_path: path to the archive
    :type xml_archive_path: string
    :return: filepath to the extracted files
    :rtype: string
    """
    logger.info(f"Extracting {xml_archive_path}...")
    output_file = os.path.join(os.path.dirname(xml_archive_path), "xmls")
    zip_reference = zipfile.ZipFile(xml_archive_path, 'r')
    zip_reference.extractall(output_file)
    zip_reference.close()

    logger.info(f"Extracted files into {output_file}")
    return output_file


def process(xmls_dir, output_dir):
    logger.info(f"Processing xml files ...")

    book_info_rows = list()
    similar_books_rows = list()
    for filename in glob(os.path.join(xmls_dir, '**', '*.xml')):
        logger.debug(f"Processing file {filename} ...")
        book = _extract_book_element_(filename)
        book_info_rows.append(_extract_book_info_(book))
        similar_books_rows.extend(_extract_similar_books_(book))

    book_info_labels = ['work_id', 'isbn13', 'description']
    book_info_df = pandas.DataFrame.from_records(
        book_info_rows, columns=book_info_labels)

    # clean html tags from description column
    html_tag_regexp = '<[^<]+?>'
    multiple_whitespaces_regexp = r"\s\s+"
    book_info_df.description = book_info_df.description.astype(
        str).str.replace(html_tag_regexp, ' ')
    book_info_df.description = book_info_df.description.astype(
        str).str.replace(multiple_whitespaces_regexp, ' ')

    book_info_file = os.path.join(output_dir, 'book-additional_info.csv')
    book_info_df.to_csv(book_info_file, index=False)

    similar_book_labels = ['work_id', 'similar_book_work_id']
    similar_book_df = pandas.DataFrame.from_records(
        similar_books_rows, columns=similar_book_labels)

    similar_book_file = os.path.join(output_dir, 'similar_books.csv')
    similar_book_df.to_csv(similar_book_file, index=False)

    logger.info(f"Created: {book_info_file}, {similar_book_file}")


def _extract_book_element_(filename):
    with open(filename, 'r') as data_file:
        data = data_file.read()

    data_bytes = bytes(bytearray(data, encoding='utf-8'))
    root = etree.fromstring(data_bytes)
    return root.find("book")


def _extract_book_info_(book):
    book_work_id = book.find("work").find("id").text
    book_info = [book_work_id, book.find(
        "isbn13").text, book.find("description").text]
    return book_info


def _extract_similar_books_(book):
    similar_books_rows = list()

    book_work_id = book.find("work").find("id").text
    similar_books = book.find("similar_books")
    if similar_books is not None:
        for similar_book in similar_books.findall("book"):
            similar_book_id = similar_book.find("work").find("id").text
            similar_books_rows.append([book_work_id, similar_book_id])
    return similar_books_rows


@click.command()
@click.argument('xml_archive_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(xml_archive_path, output_dir):
    """
    Extract xml files from archive xml_archive_path, process them and
    saves output cvs in output_dir path.
    :param xml_archive_path: path to archive with xmls
    :param output_dirpath: path where the data will be saved
    """
    xml_directory = unzip_archive(xml_archive_path)

    process(xml_directory, output_dir)

    shutil.rmtree(xml_directory)
    logger.info(f"Cleaned up temporary files")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()