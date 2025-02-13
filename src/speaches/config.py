from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SAMPLES_PER_SECOND = 16000
BYTES_PER_SAMPLE = 2
BYTES_PER_SECOND = SAMPLES_PER_SECOND * BYTES_PER_SAMPLE
# 2 BYTES = 16 BITS = 1 SAMPLE
# 1 SECOND OF AUDIO = 32000 BYTES = 16000 SAMPLES


type Device = Literal["cpu", "cuda", "auto"]

# https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md#quantize-on-model-conversion
type Quantization = Literal[
    "int8", "int8_float16", "int8_bfloat16", "int8_float32", "int16", "float16", "bfloat16", "float32", "default"
]


class WhisperConfig(BaseModel):
    """See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599."""

    model: str = Field(default="Systran/faster-whisper-small")
    """
    Default HuggingFace model to use for transcription. Note, the model must support being ran using CTranslate2.
    This model will be used if no model is specified in the request.

    Models created by authors of `faster-whisper` can be found at https://huggingface.co/Systran
    You can find other supported models at https://huggingface.co/models?p=2&sort=trending&search=ctranslate2 and https://huggingface.co/models?sort=trending&search=ct2
    """
    inference_device: Device = "auto"
    device_index: int | list[int] = 0
    compute_type: Quantization = "default"  # TODO: should this even be a configuration option?
    cpu_threads: int = 0
    num_workers: int = 1
    ttl: int = Field(default=300, ge=-1)
    """
    Time in seconds until the model is unloaded if it is not being used.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """
    use_batched_mode: bool = False
    """
    Whether to use batch mode(introduced in 1.1.0 `faster-whisper` release) for inference. This will likely become the default in the future and the configuration option will be removed.
    """


# TODO: document `alias` behaviour within the docstring
class Config(BaseSettings):
    """Configuration for the application. Values can be set via environment variables.

    Pydantic will automatically handle mapping uppercased environment variables to the corresponding fields.
    To populate nested, the environment should be prefixed with the nested field name and an underscore. For example,
    the environment variable `LOG_LEVEL` will be mapped to `log_level`, `WHISPER__MODEL`(note the double underscore) to `whisper.model`, to set quantization to int8, use `WHISPER__COMPUTE_TYPE=int8`, etc.
    """

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    api_key: str | None = None
    """
    If set, the API key will be required for all requests.
    """
    log_level: str = "debug"
    """
    Logging level. One of: 'debug', 'info', 'warning', 'error', 'critical'.
    """
    host: str = Field(alias="UVICORN_HOST", default="0.0.0.0")
    port: int = Field(alias="UVICORN_PORT", default=8000)
    allow_origins: list[str] | None = None
    """
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#parsing-environment-variable-values
    Usage:
        `export ALLOW_ORIGINS='["http://localhost:3000", "http://localhost:3001"]'`
        `export ALLOW_ORIGINS='["*"]'`
    """

    enable_ui: bool = True
    """
    Whether to enable the Gradio UI. You may want to disable this if you want to minimize the dependencies and slightly improve the startup time.
    """

    whisper: WhisperConfig = WhisperConfig()

    loopback_host_url: str | None = None
    """
    If set this is the URL that the gradio app will use to connect to the API server hosting speaches.
    If not set the gradio app will use the url that the user connects to the gradio app on.
    """

    # TODO: document the below configuration options
    chat_completion_base_url: str = "https://api.openai.com/v1"
    chat_completion_api_key: str | None = None
    chat_completion_model: str | None = None

    speech_base_url: str | None = None
    speech_api_key: str | None = None
    speech_model: str = "hexgrad/Kokoro-82M"
    speech_extra_body: dict = {"sample_rate": 24000}  # TODO: should this be set by default?

    # TODO: mention that used by both Realtime API and audio chat
    transcription_base_url: str | None = None
    transcription_api_key: str | None = None
