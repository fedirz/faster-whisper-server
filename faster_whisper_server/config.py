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
    # NOTE: While inspecting outputs of these formats with `curl`, I noticed there's one or two "\n" inserted at the end of the response.

    # VTT = "vtt" # TODO
    # 1
    # 00:00:00,000 --> 00:00:09,220
    # In his video on Large Language Models or LLMs, OpenAI co-founder and YouTuber Andrej Karpathy
    #
    # 2
    # 00:00:09,220 --> 00:00:12,280
    # likened LLMs to operating systems.
    #
    # 3
    # 00:00:12,280 --> 00:00:13,280
    # Karpathy said,
    #
    # SRT = "srt" # TODO
    # WEBVTT
    #
    # 00:00:00.000 --> 00:00:09.220
    # In his video on Large Language Models or LLMs, OpenAI co-founder and YouTuber Andrej Karpathy
    #
    # 00:00:09.220 --> 00:00:12.280
    # likened LLMs to operating systems.
    #
    # 00:00:12.280 --> 00:00:13.280
    # Karpathy said,
    #
    # 00:00:13.280 --> 00:00:19.799
    # I see a lot of equivalence between this new LLM OS and operating systems of today.


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
    model: str = Field(default="Systran/faster-whisper-medium.en")
    """
    Huggingface model to use for transcription. Note, the model must support being ran using CTranslate2.
    Models created by authors of `faster-whisper` can be found at https://huggingface.co/Systran
    You can find other supported models at https://huggingface.co/models?p=2&sort=trending&search=ctranslate2 and https://huggingface.co/models?sort=trending&search=ct2
    """
    inference_device: Device = Field(default=Device.AUTO)
    compute_type: Quantization = Field(default=Quantization.DEFAULT)


class Config(BaseSettings):
    """
    Configuration for the application. Values can be set via environment variables.
    Pydantic will automatically handle mapping uppercased environment variables to the corresponding fields.
    To populate nested, the environment should be prefixed with the nested field name and an underscore. For example,
    the environment variable `LOG_LEVEL` will be mapped to `log_level`, `WHISPER_MODEL` to `whisper.model`, etc.
    """

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    log_level: str = "info"
    default_language: Language | None = None
    default_response_format: ResponseFormat = ResponseFormat.JSON
    whisper: WhisperConfig = WhisperConfig()
    max_models: int = 1
    max_no_data_seconds: float = 1.0
    """
    Max duration to for the next audio chunk before transcription is finilized and connection is closed.
    """
    min_duration: float = 1.0
    word_timestamp_error_margin: float = 0.2
    max_inactivity_seconds: float = 5.0
    """
    Max allowed audio duration without any speech being detected before transcription is finilized and connection is closed.
    """
    inactivity_window_seconds: float = 10.0
    """
    Controls how many latest seconds of audio are being passed through VAD.
    Should be greater than `max_inactivity_seconds`
    """


config = Config()
