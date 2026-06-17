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

# Khai báo các biến mặc định
DATASET_DIR = "dataset"
IMAGE_DIR = join(DATASET_DIR, "images")
LABEL_PATH = join(DATASET_DIR, "labels.csv")
OUTPUT_DIR = "output"
PROCESSED_DIR = join(OUTPUT_DIR, "processed")
ERRORLOG_DIR = "errorlog"


# Hàm này dùng để làm sạch text trước khi ghi vào CSV.
def clean_text_for_csv(text: str) -> str:
    # Ép dữ liệu về kiểu string để tránh lỗi nếu dữ liệu là None hoặc NaN.
    text = str(text)

    # Thay ký tự xuống dòng kiểu Windows hoặc Linux bằng khoảng trắng.
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Thay ký tự tab bằng khoảng trắng.
    text = text.replace("\t", " ")

    # Xóa khoảng trắng dư và gom text về một dòng.
    text = " ".join(text.split())

    # Trả về text đã được làm sạch.
    return text


# Hàm này dùng để lưu DataFrame ra file CSV.
def save_csv(df: DataFrame, path: str) -> None:
    # Lưu CSV với UTF-8 BOM để Excel đọc tiếng Việt tốt hơn.
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=QUOTE_ALL,
        lineterminator="\n",
    )


# Hàm này dùng để đánh giá một tập dữ liệu bằng một model và một kiểu xử lý ảnh.
def evaluate(
    df: DataFrame, mode: str, model: Model, process: Preprocess, threshold: int
) -> DataFrame:
    rows = []

    # Duyệt từng dòng trong DataFrame labels.
    for _, row in df.iterrows():
        # Lấy thông tin
        filename = row["filename"]
        expected_text = row["text"]

        # Xử lý ảnh bằng class Preprocess.
        image = process.modify(
            path=join(IMAGE_DIR, filename),
            mode=mode,
            name=f"processed_{model.model}_{mode}_{threshold}_{filename}",
            threshold=threshold,
        )

        # Đưa ảnh đã xử lý vào model OCR để đọc text.
        predicted_text = model.read(image)

        # Tính CER giữa text đúng và text OCR dự đoán.
        cer_score = cer(expected_text, predicted_text)

        # Tính accuracy dựa trên CER.
        accuracy_score = accuracy(cer_score)

        # Làm sạch texttrước khi ghi CSV.
        expected_text_csv = clean_text_for_csv(expected_text)
        predicted_text_csv = clean_text_for_csv(predicted_text)

        # Thêm kết quả của ảnh hiện tại vào danh sách rows.
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

    # Chuyển danh sách rows thành DataFrame và trả về.
    return DataFrame(rows)


# Hàm này dùng để tìm cấu hình xử lý ảnh tốt nhất trên tập train.
def find_best_threshold(
    train_df: DataFrame, model: Model, process: Preprocess
) -> tuple[dict[str, str | int], float]:
    # Mở file options.json để đọc danh sách các cấu hình xử lý ảnh cần thử.
    with open("options.json", "r", encoding="utf-8") as file:
        # Đọc nội dung JSON thành list/dict Python.
        options = load(file)

    # Gán cấu hình tốt nhất ban đầu là cấu hình đầu tiên trong options.json.
    best_option = options[0]
    best_score = -1.0

    # Duyệt từng cấu hình xử lý ảnh trong options.json.
    for option in options:
        # Đánh giá tập train với cấu hình hiện tại.
        result_df = evaluate(
            df=train_df,
            model=model,
            process=process,
            mode=option["mode"],
            threshold=option["threshold"],
        )

        # Tính accuracy trung bình của cấu hình hiện tại.
        mean_accuracy = result_df["accuracy"].mean()

        print(
            f"Mode {option['mode']}, "
            f"threshold {option['threshold']}: "
            f"accuracy = {mean_accuracy:.4f}"
        )

        # Nếu cấu hình hiện tại tốt hơn cấu hình tốt nhất trước đó thì cập nhật.
        if mean_accuracy > best_score:
            best_score = mean_accuracy
            best_option = option

    return best_option, best_score


# Hàm này dùng để lưu file tổng kết kết quả cuối cùng.
def save_summary(
    modelname: str,
    best_option: dict[str, str | int],
    train_score: float,
    mean_test_cer: float,
    mean_test_accuracy: float,
) -> str:
    # Tạo đường dẫn file summary theo tên model.
    summary_path = join(OUTPUT_DIR, f"{modelname}_summary.csv")

    # Tạo DataFrame chỉ gồm một dòng tổng kết.
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

    # Lưu DataFrame tổng kết ra CSV.
    save_csv(summary_df, summary_path)

    return summary_path


# Hàm chính của chương trình.
def main(modelname: str) -> None:
    # Bọc toàn bộ chương trình bằng try để nếu lỗi thì ghi log.
    try:
        # Tạo thư mục nếu chưa tồn tại.
        makedirs(OUTPUT_DIR, exist_ok=True)
        makedirs(PROCESSED_DIR, exist_ok=True)
        makedirs(ERRORLOG_DIR, exist_ok=True)

        # Khởi tạo các object
        preprocess = Preprocess(processed_dir=PROCESSED_DIR)
        model = Model(model=modelname)

        # Đọc file labels.csv thành DataFrame.
        df = read_csv(LABEL_PATH)

        # Chia dataset thành 80% train và 20% test.
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=True
        )

        print("===== START TRAINING =====")
        print(f"Using OCR: {modelname}")
        print("Finding best preprocessing threshold on train data...\n")

        # Tìm cấu hình xử lý ảnh tốt nhất trên tập train.
        best_option, train_score = find_best_threshold(
            train_df=train_df, model=model, process=preprocess
        )

        print(f"\nBest mode: {best_option['mode']}")
        print(f"Best threshold: {best_option['threshold']}")
        print(f"Train accuracy: {train_score:.4f}")
        print("\nEvaluating on test data...")

        # Tạo đường dẫn file kết quả chi tiết.
        result_path = join(OUTPUT_DIR, f"{modelname}.csv")

        # Đánh giá tập test bằng cấu hình tốt nhất tìm được từ tập train.
        test_result_df = evaluate(
            df=test_df,
            model=model,
            process=preprocess,
            mode=best_option["mode"],
            threshold=best_option["threshold"],
        )

        # Lưu kết quả chi tiết ra CSV.
        save_csv(test_result_df, result_path)

        # Tính accuracy trung bình trên tập test.
        mean_test_accuracy = test_result_df["accuracy"].mean()

        # Tính CER trung bình trên tập test.
        mean_test_cer = test_result_df["cer"].mean()

        # Lưu file summary tổng kết kết quả.
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

    # Bắt tất cả lỗi xảy ra trong quá trình chạy chương trình.
    except Exception as error:
        # Tạo thư mục errorlog nếu chưa tồn tại.
        makedirs(ERRORLOG_DIR, exist_ok=True)

        # Tạo tên file log theo thời gian hiện tại.
        errorlog = join(
            ERRORLOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )

        # Mở file log để ghi lỗi.
        with open(errorlog, "w", encoding="utf-8") as f:
            # Ghi đầy đủ loại lỗi, nội dung lỗi và traceback vào file log.
            print_exception(type(error), error, error.__traceback__, file=f)

        print(f"Saved error to {errorlog}")


# Kiểm tra file này có đang được chạy trực tiếp hay không.
if __name__ == "__main__":
    # Tạo object parser để đọc tham số dòng lệnh.
    parser = ArgumentParser()

    # Thêm tham số --model để người dùng chọn model OCR.
    parser.add_argument(
        "--model",
        type=str,
        default="easyocr",
        choices=["easyocr", "tesseract"],
        help="Model selection",
    )

    # Đọc tham số từ terminal.
    args = parser.parse_args()

    main(args.model)


"""
File này là chương trình chính của project OCR.
Chương trình đọc dataset từ labels.csv và ảnh từ thư mục dataset/images.
Sau đó chương trình chia dữ liệu thành 80% train và 20% test.
Tập train được dùng để tìm cấu hình tiền xử lý ảnh tốt nhất từ options.json.
Tập test được dùng để đánh giá kết quả cuối cùng của model OCR.
Kết quả chi tiết được lưu vào output/easyocr.csv hoặc output/tesseract.csv.
Kết quả tổng kết được lưu vào output/easyocr_summary.csv hoặc output/tesseract_summary.csv.
Nếu chương trình bị lỗi, nội dung lỗi sẽ được ghi vào thư mục errorlog.
"""
