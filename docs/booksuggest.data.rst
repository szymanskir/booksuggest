Data preparation package
========================

download\_dataset script
-----------------------------------------

.. automodule:: booksuggest.data.download_dataset
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(download_url: str, output_filepath: str)

extract\_xml\_files script
-------------------------------------------

.. automodule:: booksuggest.data.extract_xml_files
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(xml_archive_path: str, output_dir: str)

xml\_parser module
-----------------------------------

.. automodule:: booksuggest.data.xml_parser
    :members:
    :undoc-members:
    :show-inheritance:

clean\_book script
-----------------------------------

.. automodule:: booksuggest.data.clean_book
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(book_filepath: str, books_xml_dir: str, output_filepath: str)

clean\_book\_tags script
-----------------------------------------

.. automodule:: booksuggest.data.clean_book_tags
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(book_filepath: str, book_tags_filepath: str, tags_filepath: str, genres_filepath: str, output_filepath: str)

prepare\_description script
--------------------------------------------

.. automodule:: booksuggest.data.prepare_description
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(input_filepath: str, output_filepath: str, remove_nouns: bool)

prepare\_similar\_books script
-----------------------------------------------

.. automodule:: booksuggest.data.prepare_similar_books
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(books_xml_dir: str, books_filepath: str, output_filepath: str)

ratings\_train\_test\_split script
---------------------------------------------------

.. automodule:: booksuggest.data.ratings_train_test_split
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(ratings_filepath: str, trainset_filepath: str, testset_filepath: str)