<!-- https://mkdocstrings.github.io/python/usage/configuration/general/ -->
::: speaches.config.Config
    options:
        show_bases: true
        show_if_no_docstring: true
        show_labels: false
        separate_signature: true
        show_signature_annotations: true
        signature_crossrefs: true
        summary: false
        source: true
        members_order: source
        filters:
            - "!model_config"
            - "!chat_completion_*"
            - "!speech_*"
            - "!transcription_*"

::: speaches.config.WhisperConfig

<!-- TODO: nested model `whisper`  -->
<!-- TODO: Insert new lines for multi-line docstrings  -->
