import asyncio
from contextvars import Context
import logging
import random
import string

logger = logging.getLogger(__name__)


def generate_id_suffix() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=21))  # noqa: S311


def generate_event_id() -> str:
    return "event_" + generate_id_suffix()


def generate_conversation_id() -> str:
    return "conv_" + generate_id_suffix()


def generate_item_id() -> str:
    return "item_" + generate_id_suffix()


def generate_response_id() -> str:
    return "resp_" + generate_id_suffix()


def generate_session_id() -> str:
    return "sess_" + generate_id_suffix()


def generate_call_id() -> str:
    return "call_" + generate_id_suffix()


def task_done_callback(task: asyncio.Task, *, context: Context | None = None) -> None:  # noqa: ARG001
    try:
        task.result()
    except asyncio.CancelledError:
        logger.info(f"Task {task.get_name()} cancelled")
    except BaseException:  # TODO: should this be `Exception` instead?
        logger.exception(f"Task {task.get_name()} failed")
