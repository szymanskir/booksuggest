import logging
import click

from typing import List

import pandas as pd
import numpy as np

from booksuggest.models.cb_recommend_models import ICbRecommendationModel

from booksuggest.utils.serialization import read_object
from booksuggest.utils.csv_utils import save_csv


def get_all_vectors(model):
    def get_vector(test_case_id):
        return model.content_analyzer.get_feature_vector(test_case_id)

    return [get_vector(test_case_id) for test_case_id in range(1, 10001)]


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(model_filepath: str, output_filepath: str):
    logger = logging.getLogger(__name__)
    logger.info("Loading model...")
    model = read_object(model_filepath)

    logger.info("Extracting feature vectors...")
    feature_vectors = get_all_vectors(model)
    feature_vectors_flattened = [x.ravel() for x in feature_vectors]
    pd.DataFrame(feature_vectors_flattened).to_csv(
        output_filepath, sep="\t", header=False, index=False
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
