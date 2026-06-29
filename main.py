from datetime import datetime
from pathlib import Path
from traceback import print_exception

from source.model import Model
# from source.pipeline import Pipeline
from source.prep import Prep
from source.util import get_options, get_thread_count


def main() -> None:
    options = get_options()

    OP_ERRORLOG_P = Path(options["main"]["errorlog"])

    try:
        print("=" * 10, "CONFIG OCR", "=" * 10)

        model = Model(modelOptions=options["main"]["model"])
        print("-> Model:", model.modelname)
        print("-> Graphics:", model.devicename or "unavailable")

        usethread = 1
        multithread = options["main"]["multithread"]

        if multithread["enable"]:
            usethread = get_thread_count(use_threads=multithread["usethreads"])

        print("-> Use thread(s):", usethread)

        prep = Prep(options=options["preprocessor"])
        # pipeline = Pipeline(model=model, prep=prep)

        prep.modify(IMAGE_P=Path("D:/LapTrinhTaiGia/python/012012410016/optical-character-recognition-processing-pipeline/dataset/tune/images/image_49.png"), PROCESSED_P=".", FILE_NAME="image_49.png")

    except Exception as error:
        print("Something went wrong during processing, error:", error)

        if not OP_ERRORLOG_P.exists():
            OP_ERRORLOG_P.mkdir(parents=True, exist_ok=True)

        ERRORFILE_P = (
            OP_ERRORLOG_P / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )

        with open(ERRORFILE_P, "w", encoding="utf-8") as writer:
            print_exception(
                type(error), value=error, tb=error.__traceback__, file=writer
            )


if __name__ == "__main__":
    main()
