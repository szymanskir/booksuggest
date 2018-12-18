import csv

from typing import Dict, List


def save_csv(data: List[Dict], filepath: str, fieldnames: List[str]):
    """Saves data stored in a list of dictionaries to the given filepath.

    Args:
        data: data to save.
        fieldnames: header of the csv file.
    """
    with open(filepath, 'w') as save_file:
        csv_writer = csv.DictWriter(save_file, fieldnames)
        csv_writer.writeheader()
        for row in data:
            csv_writer.writerow(row)
