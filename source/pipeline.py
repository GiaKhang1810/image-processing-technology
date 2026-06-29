from pathlib import Path

from .model import Model
from .prep import Prep


class Pipeline:
    def __init__(
        self,
        model: Model,
        prep: Prep,
        usethreads: int,
        PROCESSING_P: Path | None = None,
        PROCESSED_P: Path | None = None,
    ) -> None:
        self.__model = model
        self.__prep = prep
        self.__usethreads = usethreads
        self.N_PROCESSING_P = PROCESSING_P
        self.__PROCESSED_P = PROCESSED_P

    @property
    def PROCESSING_P(self) -> Path | None:
        return self.__PROCESSED_P

    @PROCESSING_P.setter
    def PROCESSING_P(self, N_PROCESSING_P: Path | None) -> None:
        if N_PROCESSING_P is not None and not N_PROCESSING_P.exists():
            raise FileNotFoundError(f"{N_PROCESSING_P} not found")

        self.N_PROCESSING_P = N_PROCESSING_P

    @property
    def PROCESSED_P(self) -> Path | None:
        return self.__PROCESSED_P

    @PROCESSED_P.setter
    def processed(self, N_PROCESSED_P: Path | None) -> None:
        if N_PROCESSED_P is not None and not N_PROCESSED_P.exists():
            N_PROCESSED_P.mkdir(parents=True, exist_ok=True)

        self.__PROCESSED_P = N_PROCESSED_P


