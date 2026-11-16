import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WebpageConfig:
    timeout_seconds: float = 30
    max_redirects: int = 5
    max_bytes: int = 5_000_000
    retries: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 30:
            raise ValueError(
                "PAGE_FETCH_TIMEOUT_SECONDS must be greater than 0 and at most 30."
            )
        for name, value, minimum, maximum in (
            ("MAX_REDIRECTS", self.max_redirects, 0, 5),
            ("MAX_BYTES", self.max_bytes, 1, 5_000_000),
            ("RETRIES", self.retries, 0, 2),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(
                    f"PAGE_FETCH_{name} must be an integer from {minimum} to {maximum}."
                )

    @classmethod
    def from_env(cls) -> "WebpageConfig":
        try:
            return cls(
                float(os.environ.get("PAGE_FETCH_TIMEOUT_SECONDS", "30")),
                int(os.environ.get("PAGE_FETCH_MAX_REDIRECTS", "5")),
                int(os.environ.get("PAGE_FETCH_MAX_BYTES", "5000000")),
                int(os.environ.get("PAGE_FETCH_RETRIES", "2")),
            )
        except (ValueError, OverflowError):
            raise ValueError(
                "Invalid PAGE_FETCH configuration; check timeout, redirects, bytes and retries."
            ) from None
