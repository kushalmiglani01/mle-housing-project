from housingmodel.custom_logger import *
from housingmodel.score import main

logger = configure_logger()


def test_main(data_path):
    test_path = os.path.join(data_path, "test_data_test.csv")
    model_path = os.path.join(data_path, "grid_search_model.pkl")
    score = main(test_path, model_path)
    print(score)
    assert score == (6686886416.609811, 81773.38452461052)
