from housingmodel.custom_logger import *
from housingmodel.ingest_data import main

logger = configure_logger()


def test_main(data_path):
    main(data_path)
    assert os.path.exists(os.path.join(data_path, "housing.csv"))
    assert os.path.exists(os.path.join(data_path, "train.csv"))
    assert os.path.exists(os.path.join(data_path, "test.csv"))
