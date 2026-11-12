"""Normalize failure text so that two occurrences of the same bug look identical.

Removes what changes between occurrences but says nothing about the bug: memory addresses,
source line numbers, PIDs/TIDs, timestamps, install-path hashes, build ids and numeric
message parameters.
"""

from __future__ import annotations

import re

from logcat_triage.models import FailureEvent

MAX_SIGNATURE_FRAMES = 12

_TIMESTAMP = re.compile(r"\b(?:\d{4}-)?\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b")
_PID = re.compile(r"\b(?P<key>pid|tid|PID|TID|uid)(?P<sep>[:=]?\s*)\d+\b")
_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
_PC = re.compile(r"\bpc\s+[0-9a-fA-F]{6,}\b")
_LONG_HEX = re.compile(r"\b[0-9a-fA-F]{12,}\b")
_JAVA_LINE = re.compile(r"\((?P<file>[^():]+?):\d+\)")
_NATIVE_OFFSET = re.compile(r"\+\d+\)")
_BUILD_ID = re.compile(r"\s*\(BuildId: [0-9a-fA-F]+\)")
_APP_PATH = re.compile(r"/data/app/(?:~~[^/]+/)?(?P<pkg>[\w.]+)-[^/]+/")
_FRAME_INDEX = re.compile(r"^#\d+\s+")
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_URL = re.compile(r"\bhttps?://\S+")
_NUMBER = re.compile(r"(?<![\w$])\d+(?:\.\d+)?(?:ms|s|[KMG]i?B|%)?(?![\w$])")
_OBJ_HASH = re.compile(r"(?<=[{@])[0-9a-fA-F]{5,8}\b")
_SHORT_HEX = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{6,8}\b")
_QUOTED_VALUE = re.compile(r"'[^']{0,80}'")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize free text (messages, reasons, context lines)."""
    out = _TIMESTAMP.sub("<ts>", text)
    out = _UUID.sub("<uuid>", out)
    out = _URL.sub("<url>", out)
    out = _PID.sub(lambda m: f"{m['key']}{m['sep']}<pid>", out)
    out = _HEX.sub("<addr>", out)
    out = _LONG_HEX.sub("<hex>", out)
    out = _OBJ_HASH.sub("<hex>", out)
    out = _SHORT_HEX.sub("<hex>", out)
    out = _NUMBER.sub("<n>", out)
    return _SPACES.sub(" ", out).strip()


def normalize_message(text: str) -> str:
    """Normalize an exception message; quoted parameter values are masked too."""
    masked = _QUOTED_VALUE.sub(
        lambda m: m.group(0) if "(" in m.group(0) or "." in m.group(0) else "'<v>'", text
    )
    return normalize_text(masked)


def normalize_frame(frame: str) -> str:
    """Normalize one Java (``at a.b.C.m(C.java:12)``) or native (``#00 pc ...``) frame."""
    out = frame.strip().removeprefix("at ").strip()
    out = _BUILD_ID.sub("", out)
    out = _APP_PATH.sub(lambda m: f"/data/app/{m['pkg']}/", out)
    out = _FRAME_INDEX.sub("", out)
    out = _PC.sub("pc", out)
    out = _JAVA_LINE.sub(lambda m: f"({m['file']})", out)
    out = _NATIVE_OFFSET.sub(")", out)
    out = _HEX.sub("<addr>", out)
    return _SPACES.sub(" ", out).strip()


def build_signature(event: FailureEvent, max_frames: int = MAX_SIGNATURE_FRAMES) -> str:
    """Build the normalized text that represents ``event`` for clustering."""
    parts: list[str] = [event.kind.value]
    if event.exception_type:
        parts.append(event.exception_type)
    if event.package:
        parts.append(event.package)
    if event.message:
        parts.append(normalize_message(event.message))
    for cause in event.caused_by:
        parts.append(f"caused by {normalize_message(cause)}")
    parts.extend(normalize_frame(f) for f in event.frames[:max_frames])
    return "\n".join(p for p in parts if p)


def normalize_events(events: list[FailureEvent]) -> list[FailureEvent]:
    """Fill ``signature`` on every event in place and return the list for chaining."""
    for event in events:
        event.signature = build_signature(event)
    return events
