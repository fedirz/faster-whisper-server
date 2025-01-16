import enum

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SAMPLES_PER_SECOND = 16000
BYTES_PER_SAMPLE = 2
BYTES_PER_SECOND = SAMPLES_PER_SECOND * BYTES_PER_SAMPLE
# 2 BYTES = 16 BITS = 1 SAMPLE
# 1 SECOND OF AUDIO = 32000 BYTES = 16000 SAMPLES


# https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-response_format
class ResponseFormat(enum.StrEnum):
    TEXT = "text"
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    SRT = "srt"
    VTT = "vtt"


class Device(enum.StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


# https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md
class Quantization(enum.StrEnum):
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    INT8_BFLOAT16 = "int8_bfloat16"
    INT8_FLOAT32 = "int8_float32"
    INT16 = "int16"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    DEFAULT = "default"


# TODO: this needs to be rethought
class Language(enum.StrEnum):
    AF = "af"
    AM = "am"
    AR = "ar"
    AS = "as"
    AZ = "az"
    BA = "ba"
    BE = "be"
    BG = "bg"
    BN = "bn"
    BO = "bo"
    BR = "br"
    BS = "bs"
    CA = "ca"
    CS = "cs"
    CY = "cy"
    DA = "da"
    DE = "de"
    EL = "el"
    EN = "en"
    ES = "es"
    ET = "et"
    EU = "eu"
    FA = "fa"
    FI = "fi"
    FO = "fo"
    FR = "fr"
    GL = "gl"
    GU = "gu"
    HA = "ha"
    HAW = "haw"
    HE = "he"
    HI = "hi"
    HR = "hr"
    HT = "ht"
    HU = "hu"
    HY = "hy"
    ID = "id"
    IS = "is"
    IT = "it"
    JA = "ja"
    JW = "jw"
    KA = "ka"
    KK = "kk"
    KM = "km"
    KN = "kn"
    KO = "ko"
    LA = "la"
    LB = "lb"
    LN = "ln"
    LO = "lo"
    LT = "lt"
    LV = "lv"
    MG = "mg"
    MI = "mi"
    MK = "mk"
    ML = "ml"
    MN = "mn"
    MR = "mr"
    MS = "ms"
    MT = "mt"
    MY = "my"
    NE = "ne"
    NL = "nl"
    NN = "nn"
    NO = "no"
    OC = "oc"
    PA = "pa"
    PL = "pl"
    PS = "ps"
    PT = "pt"
    RO = "ro"
    RU = "ru"
    SA = "sa"
    SD = "sd"
    SI = "si"
    SK = "sk"
    SL = "sl"
    SN = "sn"
    SO = "so"
    SQ = "sq"
    SR = "sr"
    SU = "su"
    SV = "sv"
    SW = "sw"
    TA = "ta"
    TE = "te"
    TG = "tg"
    TH = "th"
    TK = "tk"
    TL = "tl"
    TR = "tr"
    TT = "tt"
    UK = "uk"
    UR = "ur"
    UZ = "uz"
    VI = "vi"
    YI = "yi"
    YO = "yo"
    YUE = "yue"
    ZH = "zh"


class Task(enum.StrEnum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class WhisperConfig(BaseModel):
    """See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599."""

    model: str = Field(default="Systran/faster-whisper-small")
    """
    Default HuggingFace model to use for transcription. Note, the model must support being ran using CTranslate2.
    This model will be used if no model is specified in the request.

    Models created by authors of `faster-whisper` can be found at https://huggingface.co/Systran
    You can find other supported models at https://huggingface.co/models?p=2&sort=trending&search=ctranslate2 and https://huggingface.co/models?sort=trending&search=ct2
    """
    inference_device: Device = Field(default=Device.AUTO)
    device_index: int | list[int] = 0
    compute_type: Quantization = Field(default=Quantization.DEFAULT)
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
    """  # noqa: E501


# TODO: document `alias` behaviour within the docstring
class Config(BaseSettings):
    """Configuration for the application. Values can be set via environment variables.

    Pydantic will automatically handle mapping uppercased environment variables to the corresponding fields.
    To populate nested, the environment should be prefixed with the nested field name and an underscore. For example,
    the environment variable `LOG_LEVEL` will be mapped to `log_level`, `WHISPER__MODEL`(note the double underscore) to `whisper.model`, to set quantization to int8, use `WHISPER__COMPUTE_TYPE=int8`, etc.
    """  # noqa: E501

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
    """  # noqa: E501

    default_language: Language | None = None
    """
    Default language to use for transcription. If not set, the language will be detected automatically.
    It is recommended to set this as it will improve the performance.
    """
    default_response_format: ResponseFormat = ResponseFormat.JSON
    whisper: WhisperConfig = WhisperConfig()
    max_no_data_seconds: float = 1.0
    """
    Max duration to wait for the next audio chunk before transcription is finilized and connection is closed.
    Used only for live transcription (WS /v1/audio/transcriptions).
    """
    min_duration: float = 1.0
    """
    Minimum duration of an audio chunk that will be transcribed.
    Used only for live transcription (WS /v1/audio/transcriptions).
    """
    word_timestamp_error_margin: float = 0.2
    """
    Used only for live transcription (WS /v1/audio/transcriptions).
    """
    max_inactivity_seconds: float = 2.5
    """
    Max allowed audio duration without any speech being detected before transcription is finilized and connection is closed.
    Used only for live transcription (WS /v1/audio/transcriptions).
    """  # noqa: E501
    inactivity_window_seconds: float = 5.0
    """
    Controls how many latest seconds of audio are being passed through VAD. Should be greater than `max_inactivity_seconds`.
    Used only for live transcription (WS /v1/audio/transcriptions).
    """  # noqa: E501

    # NOTE: options below are not used yet and should be ignored. Added as a placeholder for future features I'm currently working on.  # noqa: E501

    chat_completion_base_url: str = "https://api.openai.com/v1"
    chat_completion_api_key: str | None = None

    speech_base_url: str | None = None
    speech_api_key: str | None = None
    speech_model: str = "piper"
    speech_extra_body: dict = {"sample_rate": 24000}

    transcription_base_url: str | None = None
    transcription_api_key: str | None = None

    loopback_host_url: str | None = None
    """
    If set this is the URL that the gradio app will use to connect to the API server hosting speaches.
    If not set the gradio app will use the url that the user connects to the gradio app on.
    """
