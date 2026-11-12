"""Parse raw logcat text into :class:`LogLine` objects.

Supported formats: ``threadtime`` (the ``adb logcat`` default), ``threadtime`` with ``-v year``
and ``brief``. Anything else (``--------- beginning of crash`` markers, wrapped lines) is kept
as an unparsed line so line numbers stay aligned with the source file.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from logcat_triage.models import LogLine

_THREADTIME = re.compile(
    r"^(?P<ts>(?:\d{4}-)?\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3,6})\s+"
    r"(?P<pid>\d+)\s+(?P<tid>\d+)\s+(?P<level>[VDIWEFA])\s+"
    r"(?P<tag>[^:]*?)\s*: ?(?P<msg>.*)$"
)
_BRIEF = re.compile(r"^(?P<level>[VDIWEFA])/(?P<tag>[^(]*?)\(\s*(?P<pid>\d+)\): ?(?P<msg>.*)$")


def parse_line(raw: str, number: int) -> LogLine:
    """Parse a single logcat line; never raises."""
    text = raw.rstrip("\r\n")
    match = _THREADTIME.match(text)
    if match:
        return LogLine(
            number=number,
            timestamp=match["ts"],
            pid=int(match["pid"]),
            tid=int(match["tid"]),
            level=match["level"],
            tag=match["tag"].strip(),
            message=match["msg"],
            raw=text,
        )
    match = _BRIEF.match(text)
    if match:
        return LogLine(
            number=number,
            pid=int(match["pid"]),
            level=match["level"],
            tag=match["tag"].strip(),
            message=match["msg"],
            raw=text,
        )
    return LogLine(number=number, message=text, raw=text)


def parse_lines(lines: Iterable[str]) -> Iterator[LogLine]:
    """Parse an iterable of raw lines, numbering them from 1."""
    for number, raw in enumerate(lines, start=1):
        yield parse_line(raw, number)


def read_log(path: Path) -> list[LogLine]:
    """Read and parse a logcat file (UTF-8, undecodable bytes replaced)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return list(parse_lines(text.splitlines()))
