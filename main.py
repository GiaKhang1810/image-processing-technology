from csv import QUOTE_ALL
from argparse import ArgumentParser
from pathlib import Path
from traceback import print_exception
from datetime import datetime
from time import perf_counter
from typing import TypedDict

from sklearn.model_selection import train_test_split
from pandas import read_csv, DataFrame

from metrics import metricser
from model import Model
from preprocess import Preprocesser
from utils import clean_text_for_csv, read_options, ModesDict


class BestOption(TypedDict):
    mode: str
    threshold: int


def save_csv(df: DataFrame, path: str) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=QUOTE_ALL,
        lineterminator="\n",
    )


def save_summary(
    modelname: str,
    best_option: BestOption,
    train_score: float,
    mean_test_cer: float,
    mean_test_accuracy: float,
    output_dir: Path,
) -> str:
    summary_path = output_dir / f"{modelname}_summary.csv"

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


class PipeLine:
    def __init__(self, model: Model, preprocesser: Preprocesser, dataset: Path) -> None:
        self.__model = model
        self.__preprocesser = preprocesser
        self.__dataset = dataset

    def evaluate(self, df: DataFrame, mode: str, threshold: int) -> DataFrame:
        rows = []

        total_preprocess_time = 0.0
        total_ocr_time = 0.0
        total_eval_time = perf_counter()

        for _, row in df.iterrows():
            filename: str = row["filename"]
            expected_text: str = row["text"]

            start_preprocess = perf_counter()

            image = self.__preprocesser.modify(
                image_path=self.__dataset / filename,
                mode=mode,
                threshold=threshold,
                name=f"{self.__model.modelname}_{mode}_{threshold}_{filename}",
            )

            preprocess_time = perf_counter() - start_preprocess
            start_ocr = perf_counter()

            predicted_text = self.__model.read(image)

            ocr_time = perf_counter() - start_ocr

            accuracy_score, cer_score = metricser(expected_text, predicted_text)

            total_time = preprocess_time + ocr_time
            total_preprocess_time += preprocess_time
            total_ocr_time += ocr_time

            rows.append(
                {
                    "filename": filename,
                    "model": self.__model.modelname,
                    "mode": mode,
                    "threshold": threshold,
                    "expected": clean_text_for_csv(expected_text),
                    "predicted": clean_text_for_csv(predicted_text),
                    "cer": cer_score,
                    "accuracy": accuracy_score,
                    "preprocess_time": preprocess_time,
                    "ocr_time": ocr_time,
                    "total_time": total_time,
                }
            )

        result_df = DataFrame(rows)
        total_eval_time = perf_counter() - total_eval_time

        print(
            f"Evaluate mode={mode}, threshold={threshold} | "
            f"preprocess={total_preprocess_time:.2f}s | "
            f"ocr={total_ocr_time:.2f}s | "
            f"total={total_eval_time:.2f}s"
        )

        return result_df

    def findthreshold(
        self, df: DataFrame, options: ModesDict
    ) -> tuple[BestOption, float]:
        best_option: BestOption | None = None
        best_score = -1.0

        def test_option(mode: str, threshold: int) -> None:
            nonlocal best_option, best_score

            result_df = self.evaluate(
                df=df,
                mode=mode,
                threshold=threshold,
            )

            mean_accuracy = float(result_df["accuracy"].mean())
            print(f"Mode {mode}, threshold {threshold}: accuracy = {mean_accuracy:.4f}")

            if mean_accuracy > best_score:
                best_score = mean_accuracy
                best_option = {
                    "mode": mode,
                    "threshold": threshold,
                }

        if options["raw"]["enable"]:
            test_option(
                mode="raw",
                threshold=options["raw"]["threshold"],
            )

        if options["gray"]["enable"]:
            test_option(
                mode="gray",
                threshold=options["gray"]["threshold"],
            )

        if options["contrast"]["enable"]:
            test_option(
                mode="contrast",
                threshold=options["contrast"]["threshold"],
            )

        if options["binary"]["enable"]:
            for threshold in options["binary"]["thresholds"]:
                test_option(
                    mode="binary",
                    threshold=threshold,
                )

        if options["background"]["enable"]:
            for threshold in options["background"]["thresholds"]:
                test_option(
                    mode="background",
                    threshold=threshold,
                )

        if best_option is None:
            raise Exception("No preprocessing option is enabled.")

        return best_option, best_score


def main(modelname: str) -> None:
    options = read_options()

    DATASET_DIR = Path(options["dataset"]["dataset_dir"])
    IMAGE_DIR = DATASET_DIR / options["dataset"]["image_dir"]
    LABEL_PATH = DATASET_DIR / options["dataset"]["label_path"]

    OUTPUT_DIR = Path(options["output"]["output_dir"])
    PROCESSED_DIR = OUTPUT_DIR / options["output"]["processed_dir"]
    ERRORLOG_DIR = OUTPUT_DIR / options["output"]["errorlog_dir"]

    try:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ERRORLOG_DIR.mkdir(parents=True, exist_ok=True)

        if not LABEL_PATH.is_file():
            raise Exception(f"{LABEL_PATH} not found")

        model = Model(modelname)

        prepresser = Preprocesser(
            processed_dir=PROCESSED_DIR,
            resize_scale=options["preprocesser"]["resize_scale"],
            median_filter_size=options["preprocesser"]["median_filter_size"],
            contrast_factor=options["preprocesser"]["contrast_factor"],
            background_blur_radius=options["preprocesser"]["background_blur_radius"],
            auto_rotate=options["preprocesser"]["auto_rotate"],
            save_processed=options["preprocesser"]["save_processed"],
        )

        pipeline = PipeLine(
            model=model,
            preprocesser=prepresser,
            dataset=IMAGE_DIR,
        )

        df = read_csv(LABEL_PATH)
        train_df, test_df = train_test_split(
            df,
            test_size=options["dataset"]["test_size"],
            random_state=options["dataset"]["random_state"],
            shuffle=options["dataset"]["shuffle"],
        )

        print("===== START TRAINING =====")
        print(f"Using OCR: {modelname}")
        print("Finding best preprocessing threshold on train data...\n")

        best_option, train_score = pipeline.findthreshold(
            df, options=options["preprocesser"]["modes"]
        )

        print(f"\nBest mode: {best_option['mode']}")
        print(f"Best threshold: {best_option['threshold']}")
        print(f"Train accuracy: {train_score:.4f}")
        print("\nEvaluating on test data...")

        result_path = OUTPUT_DIR / f"{modelname}.csv"

        test_result_df = pipeline.evaluate(
            df=test_df,
            mode=best_option["mode"],
            threshold=best_option["threshold"],
        )

        mean_test_accuracy = test_result_df["accuracy"].mean()
        mean_test_cer = test_result_df["cer"].mean()

        summary_path = save_summary(
            modelname=modelname,
            best_option=best_option,
            train_score=train_score,
            mean_test_cer=mean_test_cer,
            mean_test_accuracy=mean_test_accuracy,
            output_dir=OUTPUT_DIR,
        )

        print("\n===== FINAL RESULT =====")
        print(f"Best mode: {best_option['mode']}")
        print(f"Best threshold: {best_option['threshold']}")
        print(f"Test CER: {mean_test_cer:.4f}")
        print(f"Test Accuracy: {mean_test_accuracy:.4f}")
        print(f"Saved detail result to: {result_path}")
        print(f"Saved summary result to: {summary_path}")
    except Exception as error:
        ERRORLOG_DIR.mkdir(parents=True, exist_ok=True)
        errorfile = ERRORLOG_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        with open(errorfile, "w", encoding="utf-8") as content:
            print_exception(type(error), error, error.__traceback__, file=content)

        print(f"Saved error to {errorfile}")


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
