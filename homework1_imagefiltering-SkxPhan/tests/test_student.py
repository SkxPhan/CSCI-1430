import pytest
import numpy as np
from src.student import my_imfilter


@pytest.mark.parametrize(
    "matrix, kernel, expected_matrix",
    [
        (
            np.identity(10, dtype=np.float32),
            np.array(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ),
            np.identity(10, dtype=np.float32),
        ),
        (
            np.identity(10, dtype=np.float32),
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ),
            np.identity(10, dtype=np.float32),
        ),
        (
            np.identity(10, dtype=np.int8),
            np.array(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ),
            np.identity(10, dtype=np.int8),
        ),
        (
            np.identity(10, dtype=np.uint8),
            np.array(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ),
            np.identity(10, dtype=np.uint8),
        ),
        (
            np.ones((5, 5, 3)),
            np.array(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ),
            np.ones((5, 5, 3)),
        ),
        (
            np.ones((5, 5, 3)),
            np.array(
                [[0, 1, 0]],
            ),
            np.ones((5, 5, 3)),
        ),
    ],
)
def test_my_imfilter(matrix, kernel, expected_matrix):
    filtered_matrix = my_imfilter(matrix, kernel)
    assert np.array_equal(filtered_matrix, expected_matrix)
    assert filtered_matrix.dtype == expected_matrix.dtype
