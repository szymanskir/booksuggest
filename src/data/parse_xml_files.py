# -*- coding: utf-8 -*-
import click
import logging
import os
import zipfile

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

@click.command()
@click.argument('xml_archive_path', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def main(xml_archive_path, output_file):
    """
    Extract xml files from archive xml_archive_path and
    saves it in output_filepath.
    :param xml_archive_path: upath to archive with xmls
    :param output_filepath: path where the data will be saved
    """
    xml_directory = unzip_archive(xml_archive_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()



data_dir = "/home/rzepinskip/Documents/Books/books_xml"

for filename in os.listdir(data_dir):
    with open(os.path.join(data_dir, filename), 'r') as data_file:
        data = data_file.read()
        data_bytes = bytes(bytearray(data, encoding='utf-8'))
        root = objectify.fromstring(data_bytes)

        # parse simple columns
        print(root.book.work.id)
        # print(root.book.work.best_book_id)
        # print(root.book.book.id)
        # print(root.book.isbn)
        # print(root.book.isbn13)
        # print(root.book.title)
        # print(root.book.description)
        # print(root.book.image_url)
        # print(root.book.small_image_url)

        # parse authors
        # for author in root.book.authors.iterchildren():
        #     print(author.id)
        #     print(author.name)

        # parse ratings dist?
        # rating_dist_pairs = root.book.work.rating_dist.text.split('|')
        # ratings = dict()

        # for rating_pair in rating_dist_pairs:
        #     rating_value = rating_pair.split(':')[0]
        #     rating_count = rating_pair.split(':')[1]
        #     ratings[rating_value] = rating_count

        # print(ratings)

        # parse shelves?
        # for shelf in root.book.popular_shelves.iterchildren():
        #     print(shelf.get("name"))
        #     print(shelf.get("count"))

        # parse similar_books?
        # for book in root.book.similar_books.iterchildren():
        #     print(book.id)
        #     print(book.title)
