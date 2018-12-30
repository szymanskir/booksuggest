import click
import logging
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .cb_recommend_models import ContentBasedRecommendationModel
from .content_analyzer import build_content_analyzer
from ..utils.serialization import save_object


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--ngrams', default=1,
              help='Length of n-grams to be considered')
@click.option('--n', default=1,
              help='How many recommendations are returned by the model')
@click.option('--tf_idf/--count', default=True)
@click.option('--tag_features', default=None)
def main(
        input_filepath: str,
        output_filepath: str,
        n: int,
        ngrams: int,
        tf_idf: bool,
        tag_features
):
    logger = logging.getLogger(__name__)

    if tf_idf:
        logger.info('Training tf-idf model...')
        text_feature_extractor = TfidfVectorizer(ngram_range=(1, ngrams))
    else:
        logger.info('Training count model...')
        text_feature_extractor = CountVectorizer(ngram_range=(1, ngrams))

    if tag_features:
        tag_features = pd.read_csv(tag_features, index_col='book_id')

    book_data = pd.read_csv(input_filepath, index_col='book_id')
    content_analyzer = build_content_analyzer(
        book_data=book_data[~book_data['description'].isna()],
        text_feature_extractor=text_feature_extractor,
        tag_features=tag_features
    )

    cb_model = ContentBasedRecommendationModel(
        content_analyzer, n
    )
    cb_model.train()

    logger.info(f'Saving model to {output_filepath}...')
    save_object(cb_model, output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
