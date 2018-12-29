import click
import logging
import pandas as pd


@click.command()
@click.argument('dataframe_filepath', type=click.Path(exists=True))
@click.option('--n', default=100, help='Number of rows to be left')
def main(dataframe_filepath: str, n: int):
    df = pd.read_csv(dataframe_filepath)
    df = df.head(n)
    df.to_csv(dataframe_filepath, index=False)
    logging.info(f'TEST_RUN: Minified {dataframe_filepath}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
