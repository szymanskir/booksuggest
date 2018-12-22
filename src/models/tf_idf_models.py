import click
import logging

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .recommendation_models import ContentBasedRecommendationModel
from ..utils.serialization import save_object


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--ngrams', default=1,
              help='Length of n-grams to be considered')
@click.option('--n', default=1,
              help='How many recommendations are returned by the model')
@click.option('--tf_idf/--count', default=True)
def main(
        input_filepath: str,
        output_filepath: str,
        n: int,
        ngrams: int,
        tf_idf: bool
):
    logger = logging.getLogger(__name__)

    if tf_idf:
        logger.info('Training tf-idf model...')
        content_analyzer = TfidfVectorizer(ngram_range=(1, ngrams))
    else:
        logger.info('Training count model...')
        content_analyzer = CountVectorizer(ngram_range=(1, ngrams))

    cb_model = ContentBasedRecommendationModel(
        input_filepath, n, content_analyzer
    )
    cb_model.train()

    logger.info(f'Saving model to {output_filepath}...')
    save_object(cb_model, output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
