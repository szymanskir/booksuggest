"""Description cleaning functions.

This module contains functions used for cleaning book descriptions.
The main script is responsible for preparing descriptions that will
be later used for feature extraction.
"""

import logging
import click
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd

from langdetect import detect


def clean_single_description(
        description: str,
        remove_proper_nouns: bool
) -> str:
    """Prepares the description for the tf-idf method.

    Args:
        description (str): Book description.
        remove_proper_nouns (bool): Whether to remove proper nouns from text.

    Returns:
        str: Description with removed punctuation, stopwords and stemmed,
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
        input_filepath (str):
            Filepath to the data containing a ``description`` column that
            will be cleaned.
        remove_proper_nouns (bool): Whether to remove proper nouns from text.

    Returns:
        pd.DataFrame:
            The original data frame but with the ``description``
            column cleaned.
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
    """Cleans books descriptions.

    Args:
        input_filepath (str): Input file to clean descriptions in.
        output_filepath (str): Filepath where the results should be saved.
        remove_nouns (bool): Whether to remove proper nouns from text.
    """
    logging.info('Cleaning descriptions...')
    cleaned_descriptions = clean_descriptions(input_filepath, remove_nouns)

    logging.info('Saving results to %s...', output_filepath)
    cleaned_descriptions.to_csv(output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
