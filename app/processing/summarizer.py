import json
import time
from contextlib import contextmanager
from pathlib import Path

import structlog
import yaml
from jinja2.sandbox import ImmutableSandboxedEnvironment

from app.processing.exceptions import ProcessingError
from app.processing.model_config import ModelConfig

ROOT = Path(__file__).resolve().parent
logger = structlog.get_logger(__name__)


class Summarizer:
    def __init__(self, model, grammars, config: ModelConfig):
        self.model = model
        self.grammars = grammars
        self.config = config
        self.prompt = json.loads((ROOT / "prompt.json").read_text())
        self.tags = yaml.safe_load((ROOT / "tags.yaml").read_text())["tags"]
        self.template = ImmutableSandboxedEnvironment().from_string(
            model.metadata["tokenizer.chat_template"]
        )

    def _tokens(self, text: str, intermediate: bool = False, short_note: bool = False):
        content = "<source>\n" + text.replace("<|", "< |") + "\n</source>\n\n"
        if intermediate:
            content += self.prompt["map_task"]
        else:
            content += (
                "<tag_options>\n"
                + "\n".join(f"{tag}: {meaning}" for tag, meaning in self.tags.items())
                + "\n</tag_options>\n\n"
            )
            rule = self.prompt["note_rule" if short_note else "article_rule"]
            content += self.prompt["final_task"].format(sentence_rule=rule)
        rendered = self.template.render(
            messages=[
                {"role": "system", "content": self.prompt["system"]},
                {"role": "user", "content": content},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.model.tokenize(rendered.encode(), add_bos=False, special=True)

    def _fits(self, text: str, intermediate: bool = False) -> bool:
        return (
            len(self._tokens(text, intermediate)) + self.config.max_tokens
            <= self.config.context_size
        )

    def _generate(self, text: str, intermediate: bool = False, short_note: bool = False):
        tokens = self._tokens(text, intermediate, short_note)
        log = logger.bind(phase="intermediate" if intermediate else "final")
        if len(tokens) + self.config.max_tokens > self.config.context_size:
            log.error(
                "model_context_exceeded",
                prompt_tokens=len(tokens),
                max_tokens=self.config.max_tokens,
                context_size=self.config.context_size,
            )
            raise ProcessingError(
                "The combined summary exceeds the model context. Use a larger context or a shorter source."
            )
        for attempt in range(2):
            started = time.perf_counter()
            log.info(
                "model_generation_started",
                attempt=attempt + 1,
                prompt_tokens=len(tokens),
                max_tokens=self.config.max_tokens,
                seed=self.config.seed + attempt,
            )
            try:
                response = self.model.create_completion(
                    prompt=tokens,
                    grammar=self.grammars[intermediate],
                    max_tokens=self.config.max_tokens,
                    seed=self.config.seed + attempt,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    min_p=self.config.min_p,
                    stop=list(self.config.stop),
                )
            except Exception as exc:  # noqa: BLE001 - keep native error details out of saved notes.
                log.error(
                    "model_generation_failed",
                    attempt=attempt + 1,
                    error_type=type(exc).__name__,
                    duration_seconds=round(time.perf_counter() - started, 3),
                )
                raise ProcessingError("Local model generation failed.") from None
            finish_reason = None
            try:
                choice = response["choices"][0]
                finish_reason = choice["finish_reason"]
                result = json.loads(choice["text"])
                if finish_reason != "stop":
                    raise ValueError("generation_incomplete")
                keys = {"summary"} if intermediate else {"summary", "tags"}
                if not isinstance(result, dict) or set(result) != keys:
                    raise ValueError("invalid_fields")
                summary = result["summary"]
                if not isinstance(summary, str) or not summary.strip() or "\0" in summary:
                    raise ValueError("invalid_summary")
                summary.encode("utf-8")
                if not intermediate:
                    tags = result["tags"]
                    if (
                        not isinstance(tags, list)
                        or len(tags) > 3
                        or any(not isinstance(tag, str) or tag not in self.tags for tag in tags)
                        or len(set(tags)) != len(tags)
                    ):
                        raise ValueError("invalid_tags")
                usage = response.get("usage") or {}
                log.info(
                    "model_generation_completed",
                    attempt=attempt + 1,
                    duration_seconds=round(time.perf_counter() - started, 3),
                    finish_reason=finish_reason,
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                )
                return result
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                reason = str(exc) if type(exc) is ValueError else type(exc).__name__
                log.warning(
                    "model_response_invalid",
                    attempt=attempt + 1,
                    reason=reason,
                    finish_reason=finish_reason,
                    retrying=attempt == 0,
                )
                if attempt:
                    raise ProcessingError(
                        "Local model returned an invalid or incomplete response after two attempts."
                    ) from None

    def _chunks(self, text: str) -> list[str]:
        if not self._fits("", True):
            raise ProcessingError(
                "The prompt exceeds the model context. Increase MODEL_CONTEXT_SIZE."
            )
        parts, pending = [], ""
        for paragraph in text.splitlines(keepends=True):
            while paragraph:
                if self._fits(pending + paragraph, True):
                    pending += paragraph
                    break
                if pending:
                    parts.append(pending)
                    pending = ""
                    continue
                low, high = 0, len(paragraph)
                while low < high:
                    middle = (low + high + 1) // 2
                    if self._fits(paragraph[:middle], True):
                        low = middle
                    else:
                        high = middle - 1
                if low == 0:
                    raise ProcessingError("The model context cannot fit source text.")
                parts.append(paragraph[:low])
                paragraph = paragraph[low:]
        if pending:
            parts.append(pending)
        return parts

    def summarize(self, text: str) -> tuple[str, list[str]]:
        if not text.strip():
            raise ProcessingError("No source text to summarize.")
        short_note = len(text.split()) < 45
        started = time.perf_counter()
        prompt_tokens = len(self._tokens(text, short_note=short_note))
        logger.info(
            "summarization_started",
            source_characters=len(text),
            prompt_tokens=prompt_tokens,
        )
        if prompt_tokens + self.config.max_tokens <= self.config.context_size:
            result = self._generate(text, short_note=short_note)
        else:
            logger.info("source_chunking_started")
            parts = self._chunks(text)
            logger.info("source_chunked", chunks=len(parts))
            sections = []
            for index, part in enumerate(parts, 1):
                with structlog.contextvars.bound_contextvars(chunk=index, chunks=len(parts)):
                    sections.append(self._generate(part, True)["summary"])
            combined = "\n\n".join(f"Section {i + 1}: {s}" for i, s in enumerate(sections))
            result = self._generate(combined)
        logger.info(
            "summarization_completed",
            duration_seconds=round(time.perf_counter() - started, 3),
            tag_count=len(result["tags"]),
        )
        return result["summary"], result["tags"]


@contextmanager
def open_summarizer(config: ModelConfig | None = None):
    config = config if config is not None else ModelConfig.from_env()
    if not config.model_path or not Path(config.model_path).is_file():
        logger.error(
            "model_configuration_failed",
            reason="model_file_not_found",
            model_path=config.model_path,
        )
        raise ValueError("Set MODEL_PATH to a local Qwen3-4B-Instruct-2507 GGUF file.")

    from llama_cpp import Llama, LlamaGrammar

    tags = yaml.safe_load((ROOT / "tags.yaml").read_text())["tags"]
    properties = {
        "summary": {"type": "string", "minLength": 1},
        "tags": {
            "type": "array",
            "items": {"type": "string", "enum": list(tags)},
            "maxItems": 3,
            "uniqueItems": True,
        },
    }
    grammars = {
        intermediate: LlamaGrammar.from_json_schema(
            json.dumps(
                {
                    "type": "object",
                    "properties": {"summary": properties["summary"]}
                    if intermediate
                    else properties,
                    "required": ["summary"] if intermediate else ["summary", "tags"],
                    "additionalProperties": False,
                }
            ),
            verbose=False,
        )
        for intermediate in (False, True)
    }
    started = time.perf_counter()
    logger.info(
        "model_loading",
        model_path=config.model_path,
        context_size=config.context_size,
        gpu_layers=config.gpu_layers,
        max_tokens=config.max_tokens,
        seed=config.seed,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
    )
    try:
        model = Llama(
            model_path=config.model_path,
            n_ctx=config.context_size,
            n_gpu_layers=config.gpu_layers,
            n_threads=4,
            n_threads_batch=4,
            n_batch=256,
            seed=config.seed,
            verbose=False,
        )
    except Exception as exc:
        logger.error("model_load_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    logger.info(
        "model_loaded",
        duration_seconds=round(time.perf_counter() - started, 3),
        gpu_layers=config.gpu_layers,
    )
    try:
        yield Summarizer(model, grammars, config)
    finally:
        model.close()
        logger.info("model_unloaded")
