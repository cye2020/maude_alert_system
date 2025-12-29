import uuid


def uuid5_from_str(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value)) if value is not None else None
