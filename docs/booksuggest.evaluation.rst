Models evaluation package
==============================

cb\_predict\_models script
-------------------------------------------------

.. automodule:: booksuggest.evaluation.cb_predict_models
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(model_filepath: str, test_cases_filepath: str, output_filepath: str)

cb\_evaluation script
--------------------------------------------

.. automodule:: booksuggest.evaluation.cb_evaluation
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(input_directory: str, similar_books_input: str, output_filepath: str)


cf\_predict\_models script
-------------------------------------------------

.. automodule:: booksuggest.evaluation.cf_predict_models
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(model_filepath: str, output_filepath: str, n: int, chunks_count: int)

cf\_accuracy\_evaluation script
------------------------------------------------------

.. automodule:: booksuggest.evaluation.cf_accuracy_evaluation
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(models_dir: str, testset_filepath: str, output_filepath: str)

cf\_effectiveness\_evaluation script
-----------------------------------------------------------

.. automodule:: booksuggest.evaluation.cf_effectiveness_evaluation
    :members:
    :undoc-members:
    :show-inheritance:

    .. autofunction:: main(predictions_dir: str, to_read_filepath: str, testset_filepath: str, threshold: float, output_filepath: str)

metrics module
-------------------------------------

.. automodule:: booksuggest.evaluation.metrics
    :members:
    :undoc-members:
    :show-inheritance:
