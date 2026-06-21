from csv import QUOTE_ALL
from argparse import ArgumentParser
from pathlib import Path
from threading import Thread, Lock
from traceback import print_exception
from datetime import datetime
from time import perf_counter
from typing import TypedDict

from sklearn.model_selection import train_test_split
from pandas import read_csv, DataFrame

from metrics import metricser
from model import Model
from preprocess import Preprocesser
from utils import clean_text_for_csv, read_options, get_limited_workers, ModesDict


class BestOption(TypedDict):
    mode: str
    threshold: int


# Ghi DataFrame ra CSV theo định dạng thống nhất cho toàn bộ project.
def save_csv(df: DataFrame, path: str | Path) -> None:
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=QUOTE_ALL,
        lineterminator="\n",
    )


# Lưu kết quả tổng hợp gồm cấu hình tốt nhất và điểm đánh giá cuối cùng.
def save_summary(
    modelname: str,
    best_option: BestOption,
    train_score: float,
    mean_test_cer: float,
    mean_test_accuracy: float,
    output_dir: Path,
) -> Path:
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
    # Quản lý toàn bộ quá trình tiền xử lý ảnh, OCR và đánh giá kết quả.
    def __init__(
        self,
        model: Model,
        preprocesser: Preprocesser,
        dataset: Path,
        quantitythreads: int,
    ) -> None:
        self.__model = model
        self.__preprocesser = preprocesser
        self.__dataset = dataset
        self.__quantitythreads = quantitythreads

    # Tiền xử lý một ảnh.
    def __preprocess_one(
        self,
        filename: str,
        expected_text: str,
        mode: str,
        threshold: int,
    ) -> dict:
        start_preprocess = perf_counter()

        image = self.__preprocesser.modify(
            image_path=self.__dataset / filename,
            mode=mode,
            threshold=threshold,
            name=filename,
        )

        preprocess_time = perf_counter() - start_preprocess

        return {
            "filename": filename,
            "expected_text": expected_text,
            "image": image,
            "preprocess_time": preprocess_time,
        }

    # Chạy preprocess song song cho một batch ảnh bằng threading.
    def __preprocess_batch(
        self, batch: list[dict], mode: str, threshold: int
    ) -> tuple[list[dict], float]:
        results: list[dict] = []
        errors: list[BaseException] = []
        lock = Lock()
        batch_start = perf_counter()

        def worker(record: dict) -> None:
            try:
                item = self.__preprocess_one(
                    filename=record["filename"],
                    expected_text=record["expected_text"],
                    mode=mode,
                    threshold=threshold,
                )
                with lock:
                    results.append(item)

            except BaseException as error:
                with lock:
                    errors.append(error)

        threads = [Thread(target=worker, args=(record,)) for record in batch]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        batch_wall_time = perf_counter() - batch_start

        if errors:
            raise errors[0]

        return results, batch_wall_time

    # Đánh giá một tập dữ liệu với một cấu hình tiền xử lý cụ thể.
    def evaluate(
        self, df: DataFrame, mode: str, threshold: int
    ) -> tuple[DataFrame, float, float, float]:
        rows = []

        total_preprocess_time = 0.0
        total_ocr_time = 0.0
        total_eval_time = perf_counter()

        records = [
            {
                "filename": str(row["filename"]),
                "expected_text": str(row["text"]),
            }
            for _, row in df.iterrows()
        ]

        for start in range(0, len(records), self.__quantitythreads):
            batch = records[start : start + self.__quantitythreads]

            # Preprocess tối đa theo 40% số luồng của cpu
            preprocessed_items, batch_preprocess_time = self.__preprocess_batch(
                batch=batch,
                mode=mode,
                threshold=threshold,
            )

            total_preprocess_time += batch_preprocess_time

            # OCR vẫn chạy tuần tự để tránh tranh GPU hoặc tạo nhiều model OCR.
            for item in preprocessed_items:
                filename = item["filename"]
                expected_text = item["expected_text"]
                image = item["image"]
                preprocess_time = item["preprocess_time"]

                start_ocr = perf_counter()

                predicted_text = self.__model.read(image)

                ocr_time = perf_counter() - start_ocr

                accuracy_score, cer_score = metricser(
                    expected_text,
                    predicted_text,
                )

                total_time = preprocess_time + ocr_time
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

        return result_df, total_preprocess_time, total_ocr_time, total_eval_time

    # Thử các mode và threshold trong options để tìm cấu hình có accuracy cao nhất.
    def findthreshold(
        self, df: DataFrame, options: ModesDict
    ) -> tuple[BestOption, float, float]:
        best_option: BestOption | None = None
        best_score = -1.0
        total_time_train = 0.0

        # Chạy đánh giá cho một cấu hình và cập nhật kết quả tốt nhất nếu cần.
        def test_option(mode: str, threshold: int) -> None:
            nonlocal best_option, best_score, total_time_train

            result_df, total_preprocess_time, total_ocr_time, total_eval_time = (
                self.evaluate(
                    df=df,
                    mode=mode,
                    threshold=threshold,
                )
            )

            total_time_train += total_eval_time

            print(
                f"Evaluate mode={mode}, threshold={threshold} | "
                f"preprocess={total_preprocess_time:.2f}s | "
                f"ocr={total_ocr_time:.2f}s | "
                f"total={total_eval_time:.2f}s"
            )

            mean_accuracy = float(result_df["accuracy"].mean())
            print(f"Mode {mode}, threshold {threshold}: accuracy = {mean_accuracy:.4f}")

            if mean_accuracy > best_score:
                best_score = mean_accuracy
                best_option = {
                    "mode": mode,
                    "threshold": threshold,
                }

        # Kiểm tra các mode không cần nhiều threshold.
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

        # Kiểm tra các mode có nhiều threshold.
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

        return best_option, best_score, total_time_train


# Chạy toàn bộ pipeline: đọc cấu hình, chia dữ liệu, train chọn cấu hình và test model.
def main(modelname: str) -> None:
    print("===== START TRAINING =====")
    print(f"Using OCR: {modelname}")

    options = read_options()

    # Chuẩn bị các đường dẫn chính từ file cấu hình.
    DATASET_DIR = Path(options["dataset"]["dataset_dir"])
    IMAGE_DIR = DATASET_DIR / options["dataset"]["image_dir"]
    LABEL_PATH = DATASET_DIR / options["dataset"]["label_path"]

    OUTPUT_DIR = Path(options["output"]["output_dir"])
    PROCESSED_DIR = OUTPUT_DIR / options["output"]["processed_dir"]
    ERRORLOG_DIR = OUTPUT_DIR / options["output"]["errorlog_dir"]

    try:
        # Đảm bảo các thư mục cần thiết đã tồn tại trước khi chạy.
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ERRORLOG_DIR.mkdir(parents=True, exist_ok=True)

        if not LABEL_PATH.is_file():
            raise Exception(f"{LABEL_PATH} not found")

        # Khởi tạo model OCR, bộ tiền xử lý và pipeline chính.
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

        # Kiểm số luồng được phép dùng để chạy, tối đa 40% số luồng CPU
        quantitythreads = 1
        if options["main"]["multithreading"]:
            quantitythreads = get_limited_workers()

        print(f"Using {quantitythreads} thread(s)")

        pipeline = PipeLine(
            model=model,
            preprocesser=prepresser,
            dataset=IMAGE_DIR,
            quantitythreads=quantitythreads,
        )

        # Đọc nhãn và chia dữ liệu thành train/test.
        df = read_csv(LABEL_PATH)

        train_df, test_df = train_test_split(
            df,
            test_size=options["dataset"]["test_size"],
            random_state=options["dataset"]["random_state"],
            shuffle=options["dataset"]["shuffle"],
        )

        print("Finding best preprocessing threshold on train data...\n")

        # Tìm cấu hình tiền xử lý tốt nhất trên tập train.
        best_option, train_score, total_time_train = pipeline.findthreshold(
            df=train_df,
            options=options["preprocesser"]["modes"],
        )

        print(f"\nBest mode: {best_option['mode']}")
        print(f"Best threshold: {best_option['threshold']}")
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Total train time: {total_time_train:.2f}s")
        print("\nEvaluating on test data...")

        result_path = OUTPUT_DIR / f"{modelname}.csv"

        # Đánh giá lại trên tập test bằng cấu hình tốt nhất.
        test_result_df, total_preprocess_time, total_ocr_time, total_eval_time = (
            pipeline.evaluate(
                df=test_df,
                mode=best_option["mode"],
                threshold=best_option["threshold"],
            )
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
            output_dir=OUTPUT_DIR,
        )

        # In kết quả cuối cùng và vị trí các file output.
        print("\n===== FINAL RESULT =====")
        print(f"Best mode: {best_option['mode']}")
        print(f"Best threshold: {best_option['threshold']}")
        print(f"Test CER: {mean_test_cer:.4f}")
        print(f"Test Accuracy: {mean_test_accuracy:.4f}")
        print(f"Preprocess: {total_preprocess_time:.2f}s")
        print(f"Ocr: {total_ocr_time:.2f}s")
        print(f"Total: {total_eval_time:.2f}s")
        print(f"Saved detail result to: {result_path}")
        print(f"Saved summary result to: {summary_path}")

    except Exception as error:
        # Ghi traceback đầy đủ vào file log để dễ kiểm tra lỗi sau khi chạy.
        ERRORLOG_DIR.mkdir(parents=True, exist_ok=True)
        errorfile = ERRORLOG_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        with open(errorfile, "w", encoding="utf-8") as content:
            print_exception(type(error), error, error.__traceback__, file=content)

        print(f"Saved error to {errorfile}")


# Đọc tham số dòng lệnh và chạy chương trình.
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
