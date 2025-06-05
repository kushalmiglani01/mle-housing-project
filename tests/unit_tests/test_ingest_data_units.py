import os
from unittest.mock import patch

import pandas as pd
from housingmodel.default_args import HOUSING_URL
from housingmodel.ingest_data import fetch_housing_data, load_housing_data


def test_load_housing_data(data_path):
    dataframe = load_housing_data(data_path)
    assert isinstance(dataframe, pd.DataFrame)


@patch("housingmodel.ingest_data.urllib.request.urlretrieve")
def test_fetch_housing_data(mock_urlretrieve, data_path):
    # Call the function being tested
    fetch_housing_data(data_path)

    # Check that urlretrieve was called with the expected arguments
    mock_urlretrieve.assert_called_once_with(
        HOUSING_URL, os.path.join(data_path, "housing.tgz")
    )
