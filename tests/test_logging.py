import io
import json
import logging
import os
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest
import structlog

from app.logging import configure_logging


class TestLogging:
    @pytest.fixture(autouse=True)
    def restore_logging(self):
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        config = structlog.get_config()
        httpx_level = logging.getLogger("httpx").level
        try:
            yield
        finally:
            for handler in root.handlers:
                if handler not in handlers:
                    handler.close()
            root.handlers = handlers
            root.setLevel(level)
            logging.getLogger("httpx").setLevel(httpx_level)
            structlog.configure(**config)

    def test_json_formats_structured_and_standard_logs(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"LOG_FORMAT": "json"}), redirect_stdout(output):
            configure_logging()
            with structlog.contextvars.bound_contextvars(note_id=7):
                structlog.get_logger("test").info(
                    "generation_started", prompt_tokens=100
                )
                logging.getLogger("legacy").warning("Something failed")
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        assert rows[0]["event"] == "generation_started"
        assert rows[0]["prompt_tokens"] == 100
        assert rows[0]["note_id"] == 7
        assert rows[1]["level"] == "warning"
        assert rows[1]["note_id"] == 7
        assert "timestamp" in rows[0]
        assert "_record" not in rows[0]

    def test_console_is_default(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(output):
            configure_logging()
            structlog.get_logger("test").info("model_loaded", gpu_layers=-1)
        assert "model_loaded" in output.getvalue()
        assert "gpu_layers=-1" in output.getvalue()
        assert "\x1b[" not in output.getvalue()

    def test_invalid_format_fails_explicitly(self):
        with (
            patch.dict(os.environ, {"LOG_FORMAT": "xml"}),
            pytest.raises(ValueError, match="LOG_FORMAT"),
        ):
            configure_logging()
