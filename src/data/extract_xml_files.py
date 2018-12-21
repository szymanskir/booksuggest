import click
import logging
import os
import zipfile

logger = logging.getLogger(__name__)


@click.command()
@click.argument('xml_archive_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(xml_archive_path, output_dir):
    """Extract xml files from archive xml_archive_path to output_dir
    """
    logger.info(f"Extracting {xml_archive_path}...")
    output_dir = os.path.join(os.path.dirname(xml_archive_path), "xmls")
    zip_reference = zipfile.ZipFile(xml_archive_path, 'r')
    zip_reference.extractall(output_dir)
    zip_reference.close()

    logger.info(f"Extracted files into {output_dir}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()