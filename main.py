from argparse import ArgumentParser
from os import makedirs
from os.path import join
from sklearn.model_selection import train_test_split
from pandas import read_csv, DataFrame
from traceback import print_exception
from datetime import datetime
from csv import QUOTE_ALL
from json import load

from metrics import cer, accuracy
from model import Model
from preprocess import Preprocess

DATASET_DIR = "dataset"
IMAGE_DIR = join(DATASET_DIR, "images")
LABEL_PATH = join(DATASET_DIR, "labels.csv")

OUTPUT_DIR = "output"
PROCESSED_DIR = join(OUTPUT_DIR, "processed")
ERRORLOG_DIR = "errorlog"


def clean_text_for_csv(text: str) -> str:
    text = str(text)

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    text = text.replace("\t", " ")

    text = " ".join(text.split())

    return text


def save_csv(df: DataFrame, path: str) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=QUOTE_ALL,
        lineterminator="\n",
    )


def evaluate(
    df: DataFrame, mode: str, model: Model, process: Preprocess, threshold: int
) -> DataFrame:
    rows = []

    for _, row in df.iterrows():
        filename = row["filename"]
        expected_text = row["text"]

        image = process.modify(
            path=join(IMAGE_DIR, filename),
            mode=mode,
            name=f"processed_{model.model}_{mode}_{threshold}_{filename}",
        )

        predicted_text = model.read(image)

        cer_score = cer(expected_text, predicted_text)
        accuracy_score = accuracy(cer_score)

        expected_text_csv = clean_text_for_csv(expected_text)
        predicted_text_csv = clean_text_for_csv(predicted_text)

        rows.append(
            {
                "filename": filename,
                "model": model.model,
                "mode": mode,
                "threshold": threshold,
                "expected": expected_text_csv,
                "predicted": predicted_text_csv,
                "cer": cer_score,
                "accuracy": accuracy_score,
            }
        )

    return DataFrame(rows)


def find_best_threshold(
    train_df: DataFrame, model: Model, process: Preprocess
) -> tuple[dict[str, str | int], float]:
    with open("options.json", "r", encoding="utf-8") as file:
        options = load(file)

    best_option = options[0]
    best_score = -1.0

    for option in options:
        result_df = evaluate(
            df=train_df,
            model=model,
            process=process,
            mode=option["mode"],
            threshold=option["threshold"],
        )

        mean_accuracy = result_df["accuracy"].mean()

        print(
            f"Mode {option['mode']}, "
            f"threshold {option['threshold']}: "
            f"accuracy = {mean_accuracy:.4f}"
        )

        if mean_accuracy > best_score:
            best_score = mean_accuracy
            best_option = option

    return best_option, best_score


def save_summary(
    modelname: str,
    best_option: dict[str, str | int],
    train_score: float,
    mean_test_cer: float,
    mean_test_accuracy: float,
) -> str:
    summary_path = join(OUTPUT_DIR, f"{modelname}_summary.csv")

    summary_df = DataFrame(
        [
            {
                "model": modelname,
                "best_mode": best_option["mode"],
                "best_threshold": best_option["threshold"],
                "train_accuracy": train_score,
                "test_cer": mean_test_cer,
                "test_accuracy": mean_test_accuracy,
            }
        ]
    )

    save_csv(summary_df, summary_path)

    return summary_path


def main(modelname: str) -> None:
    try:
        makedirs(OUTPUT_DIR, exist_ok=True)
        makedirs(PROCESSED_DIR, exist_ok=True)
        makedirs(ERRORLOG_DIR, exist_ok=True)

        preprocess = Preprocess(processed_dir=PROCESSED_DIR)
        model = Model(model=modelname)

        df = read_csv(LABEL_PATH)

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=True
        )

        print("===== START TRAINING =====")
        print(f"Using OCR: {modelname}")
        print("Finding best preprocessing threshold on train data...\n")

        best_option, train_score = find_best_threshold(
            train_df=train_df, model=model, process=preprocess
        )

        print(f"\nBest mode: {best_option['mode']}")
        print(f"Best threshold: {best_option['threshold']}")
        print(f"Train accuracy: {train_score:.4f}")
        print("\nEvaluating on test data...")

        result_path = join(OUTPUT_DIR, f"{modelname}.csv")

        test_result_df = evaluate(
            df=test_df,
            model=model,
            process=preprocess,
            mode=best_option["mode"],
            threshold=best_option["threshold"],
        )

        save_csv(test_result_df, result_path)

        mean_test_accuracy = test_result_df["accuracy"].mean()
        mean_test_cer = test_result_df["cer"].mean()

        summary_path = save_summary(
            modelname=modelname,
            best_option=best_option,
            train_score=train_score,
            mean_test_cer=mean_test_cer,
            mean_test_accuracy=mean_test_accuracy,
        )

        print("\n===== FINAL RESULT =====")
        print(f"Best mode: {best_option['mode']}")
        print(f"Best threshold: {best_option['threshold']}")
        print(f"Test CER: {mean_test_cer:.4f}")
        print(f"Test Accuracy: {mean_test_accuracy:.4f}")
        print(f"Saved detail result to: {result_path}")
        print(f"Saved summary result to: {summary_path}")

    except Exception as error:
        makedirs(ERRORLOG_DIR, exist_ok=True)

        errorlog = join(
            ERRORLOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )

        with open(errorlog, "w", encoding="utf-8") as f:
            print_exception(type(error), error, error.__traceback__, file=f)

        print(f"Saved error to {errorlog}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="easyocr",
        choices=["easyocr", "tesseract"],
        help="Model selection",
    )

    args = parser.parse_args()

    main(args.model)