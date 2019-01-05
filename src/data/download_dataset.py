import logging

import click
import requests


@click.command()
@click.argument('download_url')
@click.argument('output_filepath', type=click.Path())
def main(download_url: str, output_filepath: str):
    """Downloads data from the ``download_url`` and saves
       it in the ``output_filepath``.

    Args:
        download_url (str): Url from which the data should be downloaded.
        output_filepath (str): Filepath to which the results should be saved.
    """
    logger = logging.getLogger(__name__)

    logger.info('Downloading data from %s...', download_url)
    response = requests.get(download_url)

    logger.info('Saving data in %s...', output_filepath)
    with open(output_filepath, 'wb') as output_file:
        output_file.write(response.content)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
