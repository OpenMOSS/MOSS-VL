"""Validation and settlement of authoritative, ordered backend counters."""

FIELDS = (
    "vision_tokens",
    "text_input_tokens",
    "text_output_tokens",
    "text_tokens",
    "total_tokens",
)


def empty_usage():
    return dict.fromkeys(FIELDS, 0)


def validate_usage(value):
    if not isinstance(value, dict):
        raise ValueError("missing authoritative usage")
    if any(type(value.get(key)) is not int or value[key] < 0 for key in FIELDS):
        raise ValueError("usage fields must be non-negative integers")
    result = {key: value[key] for key in FIELDS}
    if (
        result["text_tokens"]
        != result["text_input_tokens"] + result["text_output_tokens"]
    ):
        raise ValueError("text usage does not reconcile")
    if result["total_tokens"] != result["vision_tokens"] + result["text_tokens"]:
        raise ValueError("total usage does not reconcile")
    return result


class UsageLedger:
    def __init__(self):
        self.latest = empty_usage()
        self.settled = empty_usage()
        self.watermark = -1

    def record(self, snapshot, watermark):
        value = validate_usage(snapshot)
        if type(watermark) is not int or watermark < self.watermark:
            raise ValueError("accounting watermark regressed")
        if any(value[key] < self.latest[key] for key in FIELDS):
            raise ValueError("cumulative usage regressed")
        self.latest = value
        self.watermark = watermark

    def settle(self):
        delta = {key: self.latest[key] - self.settled[key] for key in FIELDS}
        self.settled = dict(self.latest)
        return {**delta, "cumulative": dict(self.latest)}
