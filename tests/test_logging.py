import io
import json
import logging
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import structlog

from app.logging import configure_logging


class LoggingTests(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        self.handlers, self.level = root.handlers[:], root.level
        self.config = structlog.get_config()
        self.httpx_level = logging.getLogger("httpx").level

    def tearDown(self):
        root = logging.getLogger()
        for handler in root.handlers:
            if handler not in self.handlers:
                handler.close()
        root.handlers = self.handlers
        root.setLevel(self.level)
        logging.getLogger("httpx").setLevel(self.httpx_level)
        structlog.configure(**self.config)

    def test_json_formats_structured_and_standard_logs(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"LOG_FORMAT": "json"}), redirect_stdout(output):
            configure_logging()
            with structlog.contextvars.bound_contextvars(note_id=7):
                structlog.get_logger("test").info("generation_started", prompt_tokens=100)
                logging.getLogger("legacy").warning("Something failed")
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(rows[0]["event"], "generation_started")
        self.assertEqual(rows[0]["prompt_tokens"], 100)
        self.assertEqual(rows[0]["note_id"], 7)
        self.assertEqual(rows[1]["level"], "warning")
        self.assertEqual(rows[1]["note_id"], 7)
        self.assertIn("timestamp", rows[0])
        self.assertNotIn("_record", rows[0])

    def test_console_is_default(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(output):
            configure_logging()
            structlog.get_logger("test").info("model_loaded", gpu_layers=-1)
        self.assertIn("model_loaded", output.getvalue())
        self.assertIn("gpu_layers=-1", output.getvalue())
        self.assertNotIn("\x1b[", output.getvalue())

    def test_invalid_format_fails_explicitly(self):
        with patch.dict(os.environ, {"LOG_FORMAT": "xml"}):
            with self.assertRaisesRegex(ValueError, "LOG_FORMAT"):
                configure_logging()
