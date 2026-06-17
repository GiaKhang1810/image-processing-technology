from argparse import ArgumentParser
from os.path import join
from sklearn.model_selection import train_test_split
from pandas import read_csv

from model import Model
from preprocess import Preprocess

DATASET_DIR = "dataset"
IMAGE_DIR = join(DATASET_DIR, "images")
LABEL_PATH = join(DATASET_DIR, "labels.csv")

OUTPUT_DIR = "output"
PROCESSED_DIR = join(OUTPUT_DIR, "processed")

def main(model: str) -> None:
    try:
        preprocess = Preprocess(processed_dir=PROCESSED_DIR)
        model = Model(model)

        df = read_csv(LABEL_PATH)
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
    except Exception as error:
        pass

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="easyocr",
        choices=["easyocr", "tesseract", "paddle"],
        help="Model selection"
    )

    args = parser.parse_args()
    main(args.model)