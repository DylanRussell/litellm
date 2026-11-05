"""OpenTelemetry GenAI TelemetryHandler integration for ``OpenTelemetryV2``.

Delegates LLM call invocation lifecycle (span creation, context suppression,
and attribute/metric enrichment) to ``opentelemetry.util.genai.handler.TelemetryHandler``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from opentelemetry._logs import NoOpLoggerProvider
from opentelemetry.context import Context, get_value
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.trace import Span, Tracer, TracerProvider
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    CLIENT_EMBEDDING_CONTEXT_KEY,
    CLIENT_FETCH_RESPONSE_CONTEXT_KEY,
    CLIENT_INFERENCE_CONTEXT_KEY,
    CLIENT_RETRIEVAL_CONTEXT_KEY,
    ContextData,
    EmbeddingData,
    EmbeddingInvocation,
    FetchResponseData,
    FetchResponseInvocation,
    InferenceData,
    InferenceInvocation,
    RetrievalData,
    RetrievalInvocation,
)
from opentelemetry.util.genai.types import ContentCapturingMode, Error as GenAIError
from opentelemetry.util.types import AttributeValue

from litellm._version import version as litellm_version
from litellm.integrations.otel.emitter import attribute_budget, carried_keys, stamp_error
from litellm.integrations.otel.mappers.langfuse import LANGFUSE_REQUEST_METADATA_ATTRIBUTES
from litellm.integrations.otel.mappers.openinference import fit_indexed_messages
from litellm.integrations.otel.model.metadata import LLMCallEvent
from litellm.integrations.otel.model.payloads import LLMCallSpanData
from litellm.integrations.otel.model.semconv import Error, GenAI, GenAIOperation
from litellm.integrations.otel.model.spans import llm_call_span_name
from litellm.integrations.otel.plumbing.events import GenAIEventRecorder
from litellm.integrations.otel.plumbing.providers import LITELLM_TRACER_NAME

__all__ = [
    "CLIENT_EMBEDDING_CONTEXT_KEY",
    "CLIENT_FETCH_RESPONSE_CONTEXT_KEY",
    "CLIENT_INFERENCE_CONTEXT_KEY",
    "CLIENT_RETRIEVAL_CONTEXT_KEY",
    "ContextData",
    "EmbeddingData",
    "FetchResponseData",
    "GenAIInvocation",
    "InferenceData",
    "RetrievalData",
    "TelemetryHandler",
    "build_telemetry_handler",
    "finish_genai_invocation",
    "has_active_genai_context",
    "is_suppressed_invocation",
    "start_genai_invocation",
]

_INFERENCE_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        GenAIOperation.CHAT.value,
        GenAIOperation.TEXT_COMPLETION.value,
        GenAIOperation.GENERATE_CONTENT.value,
    }
)

_FETCH_RESPONSE_CALL_TYPES: Final[frozenset[str]] = frozenset(
    {
        "get_responses",
        "aget_responses",
    }
)

_CONTENT_ATTRIBUTE_KEYS: Final[frozenset[str]] = frozenset(
    {
        GenAI.INPUT_MESSAGES,
        GenAI.OUTPUT_MESSAGES,
        "langfuse.observation.input",
        "langfuse.observation.output",
        "input.value",
        "output.value",
        "gen_ai.prompt",
        "gen_ai.completion",
    }
)

_CONTENT_ATTRIBUTE_PREFIXES: Final[tuple[str, ...]] = (
    "llm.input_messages.",
    "llm.output_messages.",
    "gen_ai.prompt.",
    "gen_ai.completion.",
)


def _is_content_attribute(key: str) -> bool:
    return key in _CONTENT_ATTRIBUTE_KEYS or key.startswith(_CONTENT_ATTRIBUTE_PREFIXES)


def build_telemetry_handler(
    tracer_provider: TracerProvider | None = None,
    *,
    capture_content: bool = False,
) -> TelemetryHandler:
    """Create a ``TelemetryHandler`` configured with LiteLLM's tracer provider and scope.

    Uses no-op meter/logger providers so ``GenAIMetricRecorder`` and
    ``GenAIEventRecorder`` remain the single owners of LiteLLM's metric ceiling
    and ``gen_ai.client.operation.exception`` log events when unsuppressed, while
    ``Suppressed*Invocation`` still publishes ``metric_attributes`` to outer contexts.
    """
    handler: Final = TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=NoOpMeterProvider(),
        logger_provider=NoOpLoggerProvider(),
        instrumentation_scope_name=LITELLM_TRACER_NAME,
        instrumentation_scope_version=litellm_version,
    )
    if capture_content:
        handler._content_capturing_mode = ContentCapturingMode.SPAN_ONLY
        handler._capture_content = True
    return handler


def is_suppressed_invocation(invocation: GenAIInvocation) -> bool:
    """Return True when ``TelemetryHandler`` created a suppressed invocation due to an outer context."""
    return not invocation._start_span


def has_active_genai_context(
    operation: GenAIOperation | str | None,
    *,
    call_type: str | None = None,
    context: Context | None = None,
) -> bool:
    """Return True when an outer ``<Op>Data`` is attached in ``opentelemetry.context``."""
    op_str: Final = operation.value if isinstance(operation, GenAIOperation) else (operation or "")
    if (call_type and call_type.lower() in _FETCH_RESPONSE_CALL_TYPES) or op_str == "fetch_response":
        spec: tuple[str, type[ContextData]] | None = (CLIENT_FETCH_RESPONSE_CONTEXT_KEY, FetchResponseData)
    elif op_str in _INFERENCE_OPERATIONS:
        spec = (CLIENT_INFERENCE_CONTEXT_KEY, InferenceData)
    elif op_str == GenAIOperation.EMBEDDINGS.value:
        spec = (CLIENT_EMBEDDING_CONTEXT_KEY, EmbeddingData)
    elif op_str == GenAIOperation.RETRIEVAL.value:
        spec = (CLIENT_RETRIEVAL_CONTEXT_KEY, RetrievalData)
    else:
        spec = None
    if spec is None:
        return False
    context_key, data_cls = spec
    for candidate_ctx in (context, None):
        if isinstance(get_value(context_key, context=candidate_ctx), data_cls):
            return True
        if candidate_ctx is None:
            break
    return False


class _TimedTracer(Tracer):
    """Thin proxy around a ``Tracer`` that injects ``start_time`` into ``start_span``."""

    def __init__(self, wrapped: Tracer, start_time_ns: int | None) -> None:
        self._wrapped = wrapped
        self._start_time_ns = start_time_ns

    def start_span(self, name: str, context: Context | None = None, kind: object = None, attributes: object = None, links: object = None, start_time: int | None = None, record_exception: bool = True, set_status_on_exception: bool = True) -> Span:  # type: ignore[override]
        return self._wrapped.start_span(
            name,
            context=context,
            kind=kind,  # type: ignore[arg-type]
            attributes=attributes,  # type: ignore[arg-type]
            links=links,  # type: ignore[arg-type]
            start_time=start_time if start_time is not None else self._start_time_ns,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )

    def start_as_current_span(self, *args: object, **kwargs: object) -> object:  # type: ignore[override]
        return self._wrapped.start_as_current_span(*args, **kwargs)  # type: ignore[arg-type]


def start_genai_invocation(
    handler: TelemetryHandler,
    call: LLMCallEvent,
    kwargs: Mapping[str, object],
    *,
    context: Context | None = None,
    tracer: Tracer | None = None,
    start_time_ns: int | None = None,
    capture_content: bool = False,
) -> GenAIInvocation:
    """Start a ``GenAIInvocation`` via ``TelemetryHandler`` for ``call``.

    If an outer GenAI context key is active in ``context``, ``TelemetryHandler``
    automatically returns a ``Suppressed*Invocation`` (``_start_span=False``).
    Otherwise it starts a ``CLIENT`` span and returns the live invocation.
    """
    prev_tracer: Final = handler._tracer
    prev_mode: Final = handler._content_capturing_mode
    prev_capture: Final = handler._capture_content
    effective_tracer: Tracer = tracer if tracer is not None else prev_tracer
    if start_time_ns is not None:
        effective_tracer = _TimedTracer(effective_tracer, start_time_ns)
    handler._tracer = effective_tracer
    handler._content_capturing_mode = (
        ContentCapturingMode.SPAN_ONLY if capture_content else ContentCapturingMode.NO_CONTENT
    )
    handler._capture_content = capture_content
    try:
        call_type: Final = str(kwargs.get("call_type") or "")
        model: Final = str(kwargs.get("model") or "") or None
        litellm_params: Final = kwargs.get("litellm_params")
        provider: Final = (
            str(litellm_params.get("custom_llm_provider") or "")
            if isinstance(litellm_params, Mapping)
            else ""
        )
        op: Final = call.operation

        if call_type.lower() in _FETCH_RESPONSE_CALL_TYPES or op.value == "fetch_response":
            return handler.fetch_response(
                provider,
                response_id="",
                context=context,
                _attach_to_context=False,
            )
        if op is GenAIOperation.EMBEDDINGS:
            return handler.embedding(
                provider,
                request_model=model,
                context=context,
                _attach_to_context=False,
            )
        if op is GenAIOperation.RETRIEVAL:
            return handler.retrieval(
                provider=provider or None,
                request_model=model,
                context=context,
                _attach_to_context=False,
            )
        return handler.inference(
            provider,
            request_model=model,
            operation_name=op.value,
            conversation_id=call.session_id,
            context=context,
            _attach_to_context=False,
        )
    finally:
        handler._tracer = prev_tracer
        handler._content_capturing_mode = prev_mode
        handler._capture_content = prev_capture


def _populate_typed_invocation_fields(
    invocation: GenAIInvocation,
    data: LLMCallSpanData,
) -> None:
    """Populate typed fields on ``invocation`` from ``LLMCallSpanData``."""
    server_address: Final = data.server.address if data.server is not None else None
    server_port: Final = data.server.port if data.server is not None else None

    if isinstance(invocation, InferenceInvocation):
        invocation.data.provider_name = data.provider or None
        invocation.data.request_model = data.request_model or None
        invocation.data.operation_name = data.operation.value
        invocation.data.server_address = server_address
        invocation.data.server_port = server_port
        invocation.conversation_id = data.session_id
        invocation.response_model_name = data.response_model
        invocation.response_id = data.response_id
        invocation.finish_reasons = list(data.finish_reasons) if data.finish_reasons else None
        invocation.input_tokens = data.usage.input_tokens
        invocation.output_tokens = data.usage.output_tokens
        invocation.thinking_tokens = data.usage.reasoning_tokens
        invocation.cache_read_input_tokens = data.usage.cache_read_input_tokens
        invocation.cache_write_input_tokens = data.usage.cache_creation_input_tokens
        invocation.temperature = data.request_params.temperature
        invocation.top_p = data.request_params.top_p
        invocation.top_k = data.request_params.top_k
        invocation.max_tokens = data.request_params.max_tokens
        invocation.frequency_penalty = data.request_params.frequency_penalty
        invocation.presence_penalty = data.request_params.presence_penalty
        invocation.stop_sequences = (
            list(data.request_params.stop_sequences) if data.request_params.stop_sequences else None
        )
        invocation.seed = data.request_params.seed
        invocation.request_stream = data.is_streaming
        invocation.ttfc_seconds = data.time_to_first_chunk_seconds
        invocation.output_type = data.output_type.value if data.output_type is not None else None
    elif isinstance(invocation, EmbeddingInvocation):
        invocation._data.provider_name = data.provider or None
        invocation._data.request_model = data.request_model or None
        invocation._data.server_address = server_address
        invocation._data.server_port = server_port
        invocation.response_model_name = data.response_model
        invocation.input_tokens = data.usage.input_tokens
        if data.embedding_output is not None:
            invocation.dimension_count = data.embedding_output.dimensions
    elif isinstance(invocation, RetrievalInvocation):
        invocation._data.provider_name = data.provider or None
        invocation._data.request_model = data.request_model or None
        invocation._data.server_address = server_address
        invocation._data.server_port = server_port
        if data.request_params.top_k is not None:
            invocation.top_k = float(data.request_params.top_k)
    elif isinstance(invocation, FetchResponseInvocation):
        invocation._data.provider_name = data.provider or None
        invocation._data.server_address = server_address
        invocation._data.server_port = server_port
        if data.response_id is not None:
            invocation._data.response_id = data.response_id
        invocation.response_model_name = data.response_model
        invocation.finish_reasons = list(data.finish_reasons) if data.finish_reasons else None


class _TimedEndSpan(Span):
    """Thin proxy around a ``Span`` that injects ``end_time`` into ``end()``."""

    def __init__(self, wrapped: Span, end_time_ns: int | None) -> None:
        self._wrapped = wrapped
        self._end_time_ns = end_time_ns

    def end(self, end_time: int | None = None) -> None:
        self._wrapped.end(end_time=end_time if end_time is not None else self._end_time_ns)

    def get_span_context(self):  # type: ignore[no-untyped-def]
        return self._wrapped.get_span_context()

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        self._wrapped.set_attributes(attributes)

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self._wrapped.set_attribute(key, value)

    def add_event(self, name: str, attributes: object = None, timestamp: int | None = None) -> None:  # type: ignore[override]
        self._wrapped.add_event(name, attributes=attributes, timestamp=timestamp)  # type: ignore[arg-type]

    def add_link(self, context: object, attributes: object = None) -> None:  # type: ignore[override]
        self._wrapped.add_link(context, attributes=attributes)  # type: ignore[arg-type]

    def update_name(self, name: str) -> None:
        self._wrapped.update_name(name)

    def is_recording(self) -> bool:
        return self._wrapped.is_recording()

    def set_status(self, status: object, description: str | None = None) -> None:  # type: ignore[override]
        self._wrapped.set_status(status, description=description)  # type: ignore[arg-type]

    def record_exception(self, exception: BaseException, attributes: object = None, timestamp: int | None = None, escaped: bool = False) -> None:  # type: ignore[override]
        self._wrapped.record_exception(exception, attributes=attributes, timestamp=timestamp, escaped=escaped)  # type: ignore[arg-type]


def finish_genai_invocation(
    invocation: GenAIInvocation,
    data: LLMCallSpanData,
    span_attrs: Mapping[str, AttributeValue],
    for_langfuse: Mapping[str, AttributeValue] = {},
    metric_attrs: Mapping[str, str | bool | int | float] | None = None,
    *,
    end_time_ns: int | None = None,
    event_recorder: GenAIEventRecorder | None = None,
) -> Span | None:
    """Populate ``invocation`` and finalize it via ``stop()`` or ``fail()``."""
    _populate_typed_invocation_fields(invocation, data)

    allow_content: Final = invocation.should_capture_content
    suppressed: Final = is_suppressed_invocation(invocation)

    if suppressed:
        for key, val in span_attrs.items():
            if not allow_content and _is_content_attribute(key):
                continue
            if val is not None and isinstance(val, (str, bool, int, float, list, tuple)):
                invocation.attributes[key] = val
        if data.error is not None:
            if data.error.error_type:
                invocation.attributes[Error.TYPE] = data.error.error_type
            if data.error.message:
                invocation.attributes[Error.MESSAGE] = data.error.message
        if metric_attrs:
            for key, val in metric_attrs.items():
                if val is not None and isinstance(val, (str, bool, int, float)):
                    invocation.metric_attributes[key] = val
        if data.error is not None:
            err_type: Final = data.error.error_type or "error"
            err_msg: Final = data.error.message or err_type
            invocation.fail(GenAIError(message=err_msg, type=err_type))
        else:
            invocation.stop()
        return None

    # Unsuppressed path: update span name, apply attribute budget fitting, and finalize via TelemetryHandler
    span: Final = invocation.span
    span.update_name(llm_call_span_name(data))

    # Clear start_attributes keys from invocation._start_attributes so _apply_finish can overwrite
    # provisional pre_call values (e.g. empty provider or model_group vs model)
    invocation._start_attributes.clear()

    overwrites: Final = carried_keys(span) & span_attrs.keys()
    reserved: Final = 5 if data.error else 0
    fitted: Final = fit_indexed_messages(
        {key: value for key, value in span_attrs.items() if key not in overwrites},
        attribute_budget(span, reserved),
        LANGFUSE_REQUEST_METADATA_ATTRIBUTES.union(for_langfuse),
    )
    for key, val in span_attrs.items():
        if key in overwrites or key in fitted:
            invocation.attributes[key] = val

    if end_time_ns is not None:
        invocation.span = _TimedEndSpan(span, end_time_ns)

    if data.error is not None:
        err_type_u: Final = data.error.error_type or "error"
        err_msg_u: Final = data.error.message or err_type_u
        stamped: Final = stamp_error(span, data.error)
        if stamped is not None and event_recorder is not None:
            error_type, message = stamped
            event_recorder.record_operation_exception(
                span_context=span.get_span_context(),
                error_type=error_type,
                message=message,
                stack_trace=data.error.stack_trace,
                timestamp_ns=end_time_ns,
            )
        invocation.fail(GenAIError(message=err_msg_u, type=err_type_u))
    else:
        invocation.stop()
    return span
