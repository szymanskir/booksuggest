import logging
import click

from typing import List

import pandas as pd
import numpy as np

from booksuggest.models.cb_recommend_models import ICbRecommendationModel

from booksuggest.utils.serialization import read_object
from booksuggest.utils.csv_utils import save_csv


def get_all_vectors(model):
    return [
        (book_id, model.content_analyzer.get_feature_vector(book_id))
        for book_id in range(1, 10001)
    ]


def get_metadata(book, book_tags):

    book_tags["title"] = book_tags.book_id.map(book.set_index("book_id")["title"])
    book_tags_top = (
        book_tags.sort_values(["book_id", "count"], ascending=False)
        .groupby("book_id")
        .head(3)
    )
    book_tags_top_listed = book_tags_top.groupby(["book_id", "title"])[
        "tag_name"
    ].apply(list)
    book_tags_top_listed.reset_index()
    df2 = pd.DataFrame(book_tags_top_listed)
    df2[["label1", "label2", "label3"]] = pd.DataFrame(
        df2.tag_name.values.tolist(), index=df2.index
    )
    df2 = df2.reset_index()
    book_tags_top_listed_split = df2[["book_id", "title", "label1", "label2", "label3"]]
    return book_tags_top_listed_split


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("book_filepath", type=click.Path(exists=True))
@click.argument("book_tags_filepath", type=click.Path(exists=True))
@click.argument("features_output_filepath", type=click.Path())
@click.argument("labels_output_filepath", type=click.Path())
def main(
    model_filepath: str,
    book_filepath: str,
    book_tags_filepath: str,
    features_output_filepath: str,
    labels_output_filepath: str,
):
    logger = logging.getLogger(__name__)
    logger.info("Loading model...")
    model = read_object(model_filepath)

    logger.info("Extracting feature vectors...")
    feature_vectors = [
        (book_id, feature_vector)
        for book_id, feature_vector in get_all_vectors(model)
        if feature_vector is not None
    ]
    feature_vectors_flattened = [x.ravel() for _, x in feature_vectors]
    pd.DataFrame(feature_vectors_flattened).to_csv(
        features_output_filepath, sep="\t", header=False, index=False
    )

    logger.info("Saving book metadata...")
    valid_book_ids = {book_id for book_id, _ in feature_vectors}
    book = pd.read_csv(book_filepath, usecols=["book_id", "title"])
    book_tags = pd.read_csv(book_tags_filepath)
    book_metadata = get_metadata(book, book_tags)
    book_metadata = book_metadata[book_metadata["book_id"].isin(valid_book_ids)]
    book_metadata.to_csv(labels_output_filepath, sep="\t", header=True, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
