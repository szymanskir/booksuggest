import csv

from typing import Dict, List


def save_csv(data: List[Dict], filepath: str, fieldnames: List[str]):
    """Saves data stored in a list of dictionaries to the given filepath.
    """
    with open(filepath, 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames)
        csv_writer.writeheader()
        for row in data:
            csv_writer.writerow(row)
