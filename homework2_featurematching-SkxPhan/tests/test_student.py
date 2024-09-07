import pytest
from src.student import get_feature_points
from skimage import io


@pytest.fixture
def example_image():
    image_file = "./data/Tests/four_corners.png"
    image = io.imread(image_file)
    return image[:, :, :3]


def test_get_feature_points(example_image):
    xs, ys = get_feature_points(example_image, 10)
    assert xs.size == 4 and ys.size == 4
