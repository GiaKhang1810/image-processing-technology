from typing import TypedDict


class DatasetOptions(TypedDict):
    tune: str
    evaluate: str


class OutputOptions(TypedDict):
    tune: str
    evaluate: str


class StorageOptions(TypedDict):
    dataset: DatasetOptions
    output: OutputOptions
    processed: bool


class RotationOptions(TypedDict):
    enable: bool
    step: float
    minConfidence: float
    maxangle: float
    minangle: float


class MethodSharpnessOptions(TypedDict):
    unsharp: bool
    kernel: bool


class SharpnessOptions(TypedDict):
    enable: bool
    method: MethodSharpnessOptions
    sigma: float
    amount: float
    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


class EnhancerOptions(TypedDict):
    enable: bool
    resizeScale: int
    contrastfactor: float
    medianfilterSize: int


class MethodRemoverOptions(TypedDict):
    division: bool
    gaussian: bool
    median: bool
    morphology: bool


class RemoverOptions(TypedDict):
    enable: bool
    method: MethodRemoverOptions
    blurRadius: int
    kernelsize: int
    iterations: int


class MethodBinarizationOptions(TypedDict):
    fixed: bool
    adaptive: bool
    otsu: bool


class BinarizationOptions(TypedDict):
    enable: bool
    method: MethodBinarizationOptions
    threshold: float
    blockSize: int
    constant: float


class PreprocessorOptions(TypedDict):
    storage: StorageOptions
    rotation: RotationOptions
    sharpness: SharpnessOptions
    enhancer: EnhancerOptions
    remover: RemoverOptions
    binarization: BinarizationOptions


class ModelOptions(TypedDict):
    easyocr: bool
    tesseract: bool


class MultithreadOptions(TypedDict):
    enable: bool
    usethreads: int


class MainOptions(TypedDict):
    errorlog: str
    bestoptions: str
    model: ModelOptions
    multithread: MultithreadOptions


class Options(TypedDict):
    preprocessor: PreprocessorOptions
    main: MainOptions
