import random
import string


def generate_id_suffix() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=21))  # noqa: S311


def generate_event_id() -> str:
    return "event_" + generate_id_suffix()


def generate_item_id() -> str:
    return "item_" + generate_id_suffix()


def generate_response_id() -> str:
    return "resp_" + generate_id_suffix()


def generate_session_id() -> str:
    return "sess_" + generate_id_suffix()


def generate_call_id() -> str:
    return "call_" + generate_id_suffix()
