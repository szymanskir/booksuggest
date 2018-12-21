import logging
import os

from glob import glob
from lxml import etree

logger = logging.getLogger(__name__)


def extract_all_book_xml_roots(xmls_dir):
    logger.info(f"Processing xml files in {xmls_dir} ...")
    book_roots = list()
    for filename in glob(os.path.join(xmls_dir, '*.xml')):
        logger.debug(f"Processing file {filename} ...")
        book_roots.append(_extract_book_element_(filename))

    return book_roots


def _extract_book_element_(filename):
    with open(filename, 'r') as data_file:
        data = data_file.read()

    data_bytes = bytes(bytearray(data, encoding='utf-8'))
    root = etree.fromstring(data_bytes)
    return root.find("book")
