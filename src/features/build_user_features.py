from typing import Callable
import pandas as pd

from scipy.sparse import coo_matrix


class InvalidUserDataError(Exception):
    """Exception used for informing that the passed user data
    is invalid.
    """


def validate_user_data(func: Callable):
    def user_data_validation(user_data: pd.DataFrame):
        """Checks if the passed data frame contains
        the ``(user_id, book_id, rating)`` columns
        """
        required_columns = {'user_id', 'book_id', 'rating'}
        missing_columns = required_columns - set(user_data.columns)
        if missing_columns:
            raise InvalidUserDataError(f'{missing_columns} are missing.')

        return func(user_data)

    return user_data_validation


@validate_user_data
def build_interaction_matrix(
        user_data: pd.DataFrame
) -> coo_matrix:
    """Builds the interaction matrix based on the received
    user data.
    """

    user_data['rating'][user_data['rating'] > 0] = 1
    interaction_matrix = coo_matrix(
        user_data.pivot(
            index='user_id',
            columns='book_id',
            values='rating'
        ).fillna(0)
    )

    return interaction_matrix
