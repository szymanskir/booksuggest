Data preparation package
========================

download\_dataset script
-----------------------------------------

.. automodule:: booksuggest.data.download_dataset
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(download_url, output_filepath)

extract\_xml\_files script
-------------------------------------------

.. automodule:: booksuggest.data.extract_xml_files
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(xml_archive_path, output_dir)

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

    .. autofunction:: main(book_filepath, books_xml_dir, output_filepath)

clean\_book\_tags script
-----------------------------------------

.. automodule:: booksuggest.data.clean_book_tags
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(book_filepath, book_tags_filepath, tags_filepath, genres_filepath, output_filepath)

prepare\_description script
--------------------------------------------

.. automodule:: booksuggest.data.prepare_description
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(input_filepath, output_filepath, remove_nouns)

prepare\_similar\_books script
-----------------------------------------------

.. automodule:: booksuggest.data.prepare_similar_books
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(books_xml_dir, books_filepath, output_filepath)

ratings\_train\_test\_split script
---------------------------------------------------

.. automodule:: booksuggest.data.ratings_train_test_split
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(ratings_filepath, trainset_filepath, testset_filepath)