import logging
import os

from typing import List
from glob import glob
from lxml import etree
import concurrent.futures as cf

logger = logging.getLogger(__name__)


def extract_all_book_xml_roots(xmls_dir: str) -> List[etree.Element]:
    logger.info(f"Processing xml files in {xmls_dir} ...")

    with cf.ThreadPoolExecutor() as executor:
        xml_files = glob(os.path.join(xmls_dir, '*.xml'))
        for xml_file, book_root in zip(
                xml_files, executor.map(_extract_book_element, xml_files)):
            yield book_root


def _extract_book_element(filename: str) -> etree.Element:
    with open(filename, 'r') as data_file:
        data = data_file.read()

    data_bytes = bytes(bytearray(data, encoding='utf-8'))
    root = etree.fromstring(data_bytes)
    return root.find("book")
