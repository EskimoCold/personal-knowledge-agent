class PageError(Exception):
    """Page failure safe to display without URL or response details."""


class RetryablePageError(PageError):
    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
