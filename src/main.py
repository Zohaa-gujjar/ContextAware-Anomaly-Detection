def main():
    print("Context-Aware Anomaly Detection Project")


if __name__ == "__main__":
    main()

"""
This script is used to test the Dataset class and its methods.
from pathlib import Path

from src.dataset.dataset import Dataset


def main():

    dataset = Dataset.from_directory(
        Path(
            r"C:\Users\Zoi\Downloads\shanghaitech_EDA\dataset\SHANGHAI_TRAIN"
        )
    )

    print(dataset)
    print()

    print(dataset.info())
    print()

    print(dataset.get_sequence("12_0143"))
    print()

    print(dataset.get_sequence("12_0143").info())


if __name__ == "__main__":
    main()
    """