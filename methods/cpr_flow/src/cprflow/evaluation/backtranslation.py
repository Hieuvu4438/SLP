"""Audited plugin boundary. No invented recognizer or substitute captioner."""


def backtranslation_status(checkpoint=None, adapter=None):
    if not checkpoint or not adapter:
        return {
            "metric_status": "unavailable",
            "reason": "No audited official adapter/checkpoint for canonical296",
        }
    raise ValueError(
        "Official adapter must be registered and validated on real-fit/AE sanity suite before use"
    )
