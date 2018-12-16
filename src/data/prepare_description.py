import click
import logging
import numpy as np
import pandas as pd

from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


def clean_single_description(description: str) -> str:
    """Prepares the description for the tf-idf method.

        Removes punctuation, stopwords and performs
        stemming and lemmatization.
    """
    if detect(description) != 'en':
        return np.nan

    word_list = description.split()
    word_list = [word.lower() for word in word_list
                 if word.isalpha() and word not in stopwords.words('english')]

    stemmer = SnowballStemmer('english')
    word_list = [stemmer.stem(word) for word in word_list]

    lemmatizer = WordNetLemmatizer()
    word_list = [lemmatizer.lemmatize(word) for word in word_list]

    return " ".join(word_list)


def clean_descriptions(input_filepath: str) -> pd.DataFrame:
    data = pd.read_csv(input_filepath, index_col='book_id')
    descriptions = data['description'].dropna()
    data['description'] = descriptions.apply(clean_single_description)

    return data.dropna()


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """Cleans book descriptions
    """
    logging.info('Downloading nltk resources...')
    import nltk
    nltk.download(['stopwords', 'wordnet'])
    logging.info('Cleaning descriptions...')
    cleaned_descriptions = clean_descriptions(input_filepath)

    logging.info(f'Saving results to {output_filepath}...')
    cleaned_descriptions.to_csv(output_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
