from typing import Optional, List, Dict, Literal

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):
    # 一般设置
    TORCH_DEVICE: Optional[str] = None  # 注意：MPS设备不适用于文本检测，将默认为CPU
    IMAGE_DPI: int = 96  # 从 pdf 获取图像时的 DPI
    EXTRACT_IMAGES: bool = True  # 从 pdf 中提取图像并保存
    PAGINATE_OUTPUT: bool = False  # 分页输出 markdown

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        """
        确定要使用的 torch 设备模型。
        """
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    INFERENCE_RAM: int = 40  # GPU 的 VRAM（以 GB 为单位）
    VRAM_PER_TASK: float = 4.5  # 每个任务分配的 VRAM（以 GB 为单位）
    DEFAULT_LANG: str = "English"  # 默认语言，应该是 TESSERACT_LANGUAGES 中的键

    SUPPORTED_FILETYPES: Dict = {
        "application/pdf": "pdf",
    }

    # 文本提取
    PDFTEXT_CPU_WORKERS: int = 4  # 用于 pdf 文本提取的 CPU 工作线程数

    # 文本行检测
    DETECTOR_BATCH_SIZE: Optional[int] = None  # 默认为 CPU 的 6，否则为 12
    SURYA_DETECTOR_DPI: int = 96
    DETECTOR_POSTPROCESSING_CPU_WORKERS: int = 4

    # OCR
    INVALID_CHARS: List[str] = [chr(0xfffd), "�"]
    OCR_ENGINE: Optional[Literal["surya", "ocrmypdf"]] = "surya"  # 使用的 OCR 引擎，默认为 CPU 的 "ocrmypdf"，GPU 的 "surya"
    OCR_ALL_PAGES: bool = False  # 即使可以提取文本，也运行 OCR 所有页面

    ## Surya
    SURYA_OCR_DPI: int = 96
    RECOGNITION_BATCH_SIZE: Optional[int] = None  # 默认为 cuda 的 64，否则为 32

    ## Tesseract
    OCR_PARALLEL_WORKERS: int = 2  # 用于 OCR 的 CPU 工作线程数
    TESSERACT_TIMEOUT: int = 20  # 放弃 OCR 的时间
    TESSDATA_PREFIX: str = ""

    # Texify 模型
    TEXIFY_MODEL_MAX: int = 384  # Texify 的最大推理长度
    TEXIFY_TOKEN_BUFFER: int = 256  # Texify 的令牌缓冲区大小
    TEXIFY_DPI: int = 96  # 渲染图像时的 DPI
    TEXIFY_BATCH_SIZE: Optional[int] = None  # 默认为 cuda 的 6，否则为 12
    TEXIFY_MODEL_NAME: str = "vikp/texify"

    # 布局模型
    SURYA_LAYOUT_DPI: int = 96
    BAD_SPAN_TYPES: List[str] = ["Caption", "Footnote", "Page-footer", "Page-header", "Picture"]
    LAYOUT_MODEL_CHECKPOINT: str = "vikp/surya_layout3"
    BBOX_INTERSECTION_THRESH: float = 0.7  # 布局和 pdf 边框需要重叠多少才能被认为是相同的
    LAYOUT_BATCH_SIZE: Optional[int] = None  # 默认为 cuda 的 12，否则为 6

    # 排序模型
    SURYA_ORDER_DPI: int = 96
    ORDER_BATCH_SIZE: Optional[int] = None  # 默认为 cuda 的 12，否则为 6
    ORDER_MAX_BBOXES: int = 255

    # 最终编辑模型
    EDITOR_BATCH_SIZE: Optional[int] = None  # 默认为 cuda 的 6，否则为 12
    EDITOR_MAX_LENGTH: int = 1024
    EDITOR_MODEL_NAME: str = "vikp/pdf_postprocessor_t5"
    ENABLE_EDITOR_MODEL: bool = False  # 编辑器模型可能会产生误报
    EDITOR_CUTOFF_THRESH: float = 0.9  # 忽略概率低于此值的预测

    # 调试
    DEBUG: bool = False  # 启用调试日志
    DEBUG_DATA_FOLDER: Optional[str] = None
    DEBUG_LEVEL: int = 0  # 0 到 2，2 表示记录所有内容

    @computed_field
    @property
    def CUDA(self) -> bool:
        """
        确定 torch 设备模型是否为 cuda。
        """
        return "cuda" in self.TORCH_DEVICE_MODEL

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        """
        确定要使用的模型数据类型。
        """
        if self.TORCH_DEVICE_MODEL == "cuda":
            return torch.bfloat16
        else:
            return torch.float32

    @computed_field
    @property
    def TEXIFY_DTYPE(self) -> torch.dtype:
        """
        确定要使用的 texify 数据类型。
        """
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"

settings = Settings()
