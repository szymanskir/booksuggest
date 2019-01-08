Models evaluation package
==============================

cb\_predict\_models script
-------------------------------------------------

.. automodule:: booksuggest.evaluation.cb_predict_models
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(model_filepath, test_cases_filepath, output_filepath)

cb\_evaluation script
--------------------------------------------

.. automodule:: booksuggest.evaluation.cb_evaluation
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(input_directory, similar_books_input, output_filepath)


cf\_predict\_models script
-------------------------------------------------

.. automodule:: booksuggest.evaluation.cf_predict_models
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(model_filepath, output_filepath, n, chunks_count)

cf\_accuracy\_evaluation script
------------------------------------------------------

.. automodule:: booksuggest.evaluation.cf_accuracy_evaluation
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(models_dir, testset_filepath, output_filepath)

cf\_effectiveness\_evaluation script
-----------------------------------------------------------

.. automodule:: booksuggest.evaluation.cf_effectiveness_evaluation
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(predictions_dir, to_read_filepath, testset_filepath, threshold, output_filepath)

metrics module
-------------------------------------

.. automodule:: booksuggest.evaluation.metrics
    :members:
    :undoc-members:
    :show-inheritance:
