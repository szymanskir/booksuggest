import pytest
import re
import os
import src.data.xml_parser
import src.data.clean_book

current_path = os.path.dirname(os.path.realpath(__file__))
simple_file_path = os.path.join(current_path, "data/test-simple.xml")
full_file_path = os.path.join(current_path, "data/test-full.xml")


@pytest.mark.parametrize("filename", [
    simple_file_path,
    full_file_path,
])
def test_book_element_parsed(filename):
    book = src.data.xml_parser._extract_book_element_(filename)
    assert book is not None


@pytest.mark.parametrize("filename, expected", [
    (simple_file_path, (41335427, "9780439785969",
                         "The war against Voldemort is not going well")),
    (full_file_path, (908211, "9780441788385",
                      "<b>NAME: Valentine Michael Smith<br />ANCESTRY: Human<br />ORIGIN: Mars</b><br /><br />Valentine Michael Smith is a human being raised on Mars, newly returned to Earth. Among his people for the first time, he struggles to understand the social mores and prejudices of human nature that are so alien to him, while teaching them his own fundamental beliefs in grokking, watersharing, and love.")),
])
def test_book_info_parsed(filename, expected):
    book = src.data.xml_parser._extract_book_element_(filename)
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
