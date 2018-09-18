# -*- coding: utf-8 -*-
import click
import logging
import os
import zipfile
import shutil
import pandas
from glob import glob
from lxml import objectify

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

    logger.info(f"Extracted files into {output_file}...")
    return(output_file)


def process(xmls_dir, output_dir):
    logger.info(f"Processing xml files ...")

    book_info_rows = list()
    similar_books_rows = list()
    for filename in glob(os.path.join(xmls_dir, '**', '*.xml')):
        with open(filename, 'r') as data_file:
            logger.debug(f"Processing file {filename} ...")

            data = data_file.read()
            data_bytes = bytes(bytearray(data, encoding='utf-8'))
            root = objectify.fromstring(data_bytes)

            # parse simple columns
            book_info = [root.book.work.id, root.book.isbn13,
                         root.book.description]
            book_info_rows.append(book_info)

            # parse similar_books
            if hasattr(root.book, 'similar_books'):
                for similar_book in root.book.similar_books.iterchildren():
                    similar_books_rows.append(
                        [root.book.work.id, similar_book.id])

    book_info_labels = ['work_id', 'isbn13', 'description']
    book_info_df = pandas.DataFrame.from_records(
        book_info_rows, columns=book_info_labels)

    book_info_file = os.path.join(output_dir, 'book_info.csv')
    book_info_df.to_csv(book_info_file, index=False)

    similar_book_labels = ['book_goodreads_id', 'similar_book_goodreads_id']
    similar_book_df = pandas.DataFrame.from_records(
        similar_books_rows, columns=similar_book_labels)

    similar_book_file = os.path.join(output_dir, 'similar_books.csv')
    similar_book_df.to_csv(similar_book_file, index=False)

    logger.info(f"Created: {book_info_file}, {similar_book_file}")


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
