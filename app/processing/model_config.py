import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_path: str = ""
    context_size: int = 4096
    gpu_layers: int = 0
    max_tokens: int = 256
    seed: int = 42
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0
    stop: tuple[str, ...] = ("<|im_end|>", "<|endoftext|>")

    def __post_init__(self) -> None:
        if type(self.context_size) is not int or self.context_size < 1024:
            raise ValueError("MODEL_CONTEXT_SIZE must be an integer of at least 1024.")
        if type(self.gpu_layers) is not int or self.gpu_layers < -1:
            raise ValueError("MODEL_GPU_LAYERS must be an integer of -1 or greater.")
        if (
            type(self.max_tokens) is not int
            or not 0 < self.max_tokens < self.context_size
        ):
            raise ValueError(
                "max_tokens must be positive and smaller than context_size."
            )

    @classmethod
    def from_env(cls) -> "ModelConfig":
        try:
            return cls(
                model_path=os.environ.get("MODEL_PATH", "").strip(),
                context_size=int(os.environ.get("MODEL_CONTEXT_SIZE", "4096")),
                gpu_layers=int(os.environ.get("MODEL_GPU_LAYERS", "0")),
            )
        except ValueError:
            raise ValueError(
                "Invalid model configuration; check context size and GPU layers."
            ) from None
