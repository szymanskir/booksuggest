# -*- coding: utf-8 -*-
import click
import logging
import requests


@click.command()
@click.argument('download_url')
@click.argument('output_filepath', type=click.Path())
def main(download_url, output_filepath):
    """
    Downloads data from the download_url and saves it in the output_filepath.
    :param download_url: url for downloading data
    :param output_filepath: path where the data will be saved
    """
    logger = logging.getLogger(__name__)

    logger.info(f'Downloading data from {download_url}...')
    response = requests.get(download_url)

    logger.info(f'Saving data in {output_filepath}...')
    with open(output_filepath, 'wb') as output_file:
        output_file.write(response.content)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
