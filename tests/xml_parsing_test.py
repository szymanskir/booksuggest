import pytest
import re
import os

import src.data.xml_parser
import src.data.clean_book
import src.data.prepare_similar_books

current_path = os.path.dirname(os.path.realpath(__file__))
simple_file_path = os.path.join(current_path, "data/test-simple.xml")
full_file_path = os.path.join(current_path, "data/test-full.xml")


@pytest.mark.parametrize("file_path", [
    simple_file_path,
    full_file_path
])
def test_book_element_parsed(file_path):
    book = src.data.xml_parser._extract_book_element_(file_path)
    assert book is not None


@pytest.mark.parametrize("file_path, expected", [
    (simple_file_path, [(41335427, 1003876), (41335427, 946088)]),
    (full_file_path,
     [(908211, 858297), (908211, 40711), (908211, 3171254), (908211, 2777504), (908211, 1174485), (908211, 3590796), (908211, 1500323), (908211, 924161), (908211, 873021), (908211, 1247570), (908211, 924711), (908211, 348798), (908211, 3634673), (908211, 816647), (908211, 1218966), (908211, 953721), (908211, 820134)])
])
def test_similar_books_parsed(file_path, expected):
    book = src.data.xml_parser._extract_book_element_(file_path)
    similar_books = src.data.prepare_similar_books._extract_similar_books_(
        book)
    assert similar_books == expected


@pytest.mark.parametrize("file_path, expected", [
    (simple_file_path, (41335427, "9780439785969",
                         "The war against Voldemort is not going well")),
    (full_file_path, (908211, "9780441788385",
                      "<b>NAME: Valentine Michael Smith<br />ANCESTRY: Human<br />ORIGIN: Mars</b><br /><br />Valentine Michael Smith is a human being raised on Mars, newly returned to Earth. Among his people for the first time, he struggles to understand the social mores and prejudices of human nature that are so alien to him, while teaching them his own fundamental beliefs in grokking, watersharing, and love."))
])
def test_book_info_parsed(file_path, expected):
    book = src.data.xml_parser._extract_book_element_(file_path)
    book_info = src.data.clean_book._extract_book_info_(book)
    assert book_info == expected


@pytest.mark.parametrize("description, expected", [
    ("<b>NAME: Valentine<br />ANCESTRY: Human<br />ORIGIN: Mars</b><br /><br />",
     " NAME: Valentine ANCESTRY: Human ORIGIN: Mars "),
    ("<div>a</div><br />", " a ")
])
def test_book_description_html_tags_stripping(description, expected):
    html_tag_regexp = r"<[^<]+?>"
    multiple_whitespaces_regexp = r"\s\s+"
    clean_description = re.sub(html_tag_regexp, ' ', description)
    clean_description = re.sub(
        multiple_whitespaces_regexp, ' ', clean_description)
    assert clean_description == expected
