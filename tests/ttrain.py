from src.housingmodel.train import main


def test_main():
    score = main()
    assert "process completed!"
