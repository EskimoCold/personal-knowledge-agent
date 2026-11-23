class ProcessingError(Exception):
    """Expected processing failure with a message safe to display to the owner."""


class InferenceBusy(Exception):
    """The local model cannot accept another search."""


class SearchUnavailable(Exception):
    """Search inference failed without exposing request or provider details."""
