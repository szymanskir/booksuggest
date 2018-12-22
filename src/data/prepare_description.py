import click
import logging
import nltk
import numpy as np
import pandas as pd

from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from ..utils.logging import setup_root_logger


def clean_single_description(
        description: str,
        remove_proper_nouns: bool
) -> str:
    """Prepares the description for the tf-idf method.

    Args:
        description: string containg a book description.
        remove_proper_nouns: if True proper nouns will be removed
        from the description

    Returns:
        description with removed punctuation, stopwords and stemmed,
        lemmatized vocabulary.
    """
    logging.debug('Cleaning description...')
    if detect(description) != 'en':
        return np.nan

    word_list = description.split()

    if remove_proper_nouns:
        logging.debug('Removing proper nouns...')
        tagged_words = nltk.tag.pos_tag(word_list)
        word_list = [word for word, tag in tagged_words
                     if tag not in {'NNP', 'NNPS'}]

    word_list = [word.lower() for word in word_list
                 if word.isalpha() and word not in stopwords.words('english')]

    stemmer = SnowballStemmer('english')
    word_list = [stemmer.stem(word) for word in word_list]

    lemmatizer = WordNetLemmatizer()
    word_list = [lemmatizer.lemmatize(word) for word in word_list]

    return " ".join(word_list)


def clean_descriptions(
        input_filepath: str,
        remove_proper_nouns: bool
) -> pd.DataFrame:
    """Cleans all descriptions in the data from the input
    file.

    Args:
        input_filepath: filepath to the data containing a 'description'
        column that will be cleaned.

    Returns:
        the original data frame but with the 'description' column cleaned.
    """
    data = pd.read_csv(input_filepath, index_col='book_id')
    descriptions = data['description'].dropna()
    data['description'] = descriptions.apply(
        lambda x: clean_single_description(x, remove_proper_nouns)
    )

    return data.dropna()


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--remove_nouns', is_flag=True)
def main(input_filepath: str, output_filepath: str, remove_nouns: bool):
    """Cleans book descriptions.

    Args:
        input_filepath: input file to clean descriptions in.
        output_filepath: filepath where the results should be saved.
    """
    logging.info('Downloading nltk resources...')
    nltk.download(['stopwords', 'wordnet', 'averaged_perceptron_tagger'])
    logging.info('Cleaning descriptions...')
    cleaned_descriptions = clean_descriptions(input_filepath, remove_nouns)

    logging.info(f'Saving results to {output_filepath}...')
    cleaned_descriptions.to_csv(output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
