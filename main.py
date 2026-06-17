from argparse import ArgumentParser
from os.path import join
from sklearn.model_selection import train_test_split
from pandas import read_csv, DataFrame
from traceback import print_exception

from metrics import cer, accuracy
from model import Model
from preprocess import Preprocess

DATASET_DIR = "dataset"
IMAGE_DIR = join(DATASET_DIR, "images")
LABEL_PATH = join(DATASET_DIR, "labels.csv")

OUTPUT_DIR = "output"
PROCESSED_DIR = join(OUTPUT_DIR, "processed")

def evaluate(df: DataFrame, model: Model, process: Preprocess, threshold: int) -> DataFrame:
    rows = []

    for _, row in df.iterrows():
        filename = row["filename"]
        expected_text = row["text"]

        image = process.modify(
            join(IMAGE_DIR, filename),
            f"processed_{threshold}_{filename}"
        )

        predicted_text = model.read(image)

        cer_score = cer(expected_text, predicted_text)
        accuracy_score = accuracy(cer_score)

        rows.append({
            "filename": filename,
            "threshold": threshold,
            "expected": expected_text,
            "predicted": predicted_text,
            "cer": cer_score,
            "accuracy": accuracy_score
        })

    return DataFrame(rows)

def find_best_threshold(train_df, model: Model, process: Preprocess) -> tuple[int, float]:
    thresholds = [100, 120, 140, 150, 160, 180, 200]
    best_threshold: int = None
    best_score: float = -1

    for threshold in thresholds:
        result_df = evaluate(train_df, model, process, threshold)
        mean_accuracy = result_df["accuracy"].mean()

        print(f"Threshold {threshold}: accuracy = {mean_accuracy:.4f}")

        if mean_accuracy > best_score:
            best_score = mean_accuracy
            best_threshold = threshold

    return best_threshold, best_score

def main(modelname: str) -> None:
    try:
        preprocess = Preprocess(processed_dir=PROCESSED_DIR)
        model = Model(modelname)

        df = read_csv(LABEL_PATH)
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        print("Finding best preprocessing threshold on train data...")

        best_threshold, train_score = find_best_threshold(train_df, model, preprocess)

        print(f"\nBest threshold: {best_threshold}")
        print(f"Train accuracy: {train_score:.4f}")
        print("\nEvaluating on test data...")

        RESULT_PATH = join(OUTPUT_DIR, f"{modelname}.csv")
        test_result_df = evaluate(test_df, model, preprocess, best_threshold)
        test_result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

        mean_test_accuracy = test_result_df["accuracy"].mean()
        mean_test_cer = test_result_df["cer"].mean()

        print("\n===== FINAL RESULT =====")
        print(f"Best threshold: {best_threshold}")
        print(f"Test CER: {mean_test_cer:.4f}")
        print(f"Test Accuracy: {mean_test_accuracy:.4f}")
        print(f"Saved result to: {RESULT_PATH}")
    except Exception as error:
        with open("error.log", "w") as f:
            print_exception(
                type(error), 
                error, 
                error.__traceback__, 
                file=f
            )

        print("Error have been saved to the error.log file")


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