"""Extract crashes, ANRs, native crashes and logged exceptions from parsed logcat lines.

Real logcat interleaves output from many processes, so multi-line blocks are tracked per
``(pid, tag)`` key instead of assuming contiguity. A block closes when its key emits a line
that does not continue it, or when it has been silent for ``_STALE_AFTER`` lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from logcat_triage.models import FailureEvent, FailureKind, LogLine

_STALE_AFTER = 60
_MAX_FRAMES = 64
_CONTEXT_LINES = 8
_CONTEXT_LOOKBACK = 400

EXCEPTION_HEADER = re.compile(
    r"^(?P<type>(?:[a-zA-Z_$][\w$]*\.)+[A-Z][\w$]*(?:Exception|Error|Throwable)[\w$]*)"
    r"(?::\s?(?P<msg>.*))?$"
)
_FRAME = re.compile(r"^\s*at\s+\S")
_MORE = re.compile(r"^\s*\.\.\.\s+\d+\s+more")
_CAUSED_BY = re.compile(r"^\s*Caused by:\s*(?P<rest>.*)$")
_SUPPRESSED = re.compile(r"^\s*Suppressed:\s*")
_PROCESS = re.compile(r"^Process:\s*(?P<pkg>[\w.:]+),\s*PID:\s*(?P<pid>\d+)")
_FATAL = re.compile(r"^FATAL EXCEPTION:\s*(?P<thread>.*)$")
_ANR = re.compile(r"^ANR in (?P<pkg>[\w.:]+)(?:\s+\((?P<component>[^)]*)\))?")
_ANR_BODY = re.compile(
    r"^(?:PID:|Reason:|Parent:|ErrorId:|Frozen:|Load:|Subject:|CPU usage|Process:|"
    r"Package is|Dropbox|\s|\d+(?:\.\d+)?%|-?\d)"
)
_ANR_PID = re.compile(r"^PID:\s*(?P<pid>\d+)")
_ANR_REASON = re.compile(r"^(?:Reason|Subject):\s*(?P<reason>.*)$")
_FATAL_SIGNAL = re.compile(
    r"Fatal signal (?P<num>\d+) \((?P<sig>SIG\w+)\)(?P<rest>.*?)"
    r"(?:,\s*pid (?P<pid>\d+) \((?P<name>[^)]*)\))?$"
)
_TID_NAME = re.compile(r"in tid \d+ \((?P<thread>[^)]*)\)")
_TOMBSTONE_START = re.compile(r"^\*\*\* \*\*\* \*\*\*")
_TOMBSTONE_PID = re.compile(
    r"pid: (?P<pid>\d+), tid: \d+, name: (?P<thread>.*?)\s+>>> (?P<pkg>\S+) <<<"
)
_TOMBSTONE_SIGNAL = re.compile(r"^signal \d+ \((?P<sig>SIG\w+)\)(?P<rest>.*)$")
_TOMBSTONE_ABORT = re.compile(r"^Abort message: '(?P<msg>.*)'")
_NATIVE_FRAME = re.compile(r"^\s*#\d+\s+pc\s+\S+")
_START_PROC = re.compile(r"Start proc (?P<pid>\d+):(?P<pkg>[\w.]+)(?:/|\s)")

_FRAMEWORK_PREFIXES = (
    "android.",
    "androidx.",
    "com.android.",
    "dalvik.",
    "java.",
    "javax.",
    "jdk.",
    "kotlin.",
    "kotlinx.",
    "libcore.",
    "sun.",
    "okhttp3.",
    "okio.",
    "com.google.",
)

FrameKey = tuple[int | None, str | None]


@dataclass
class _Block:
    kind: FailureKind
    start: LogLine
    last_line: int
    thread: str | None = None
    package: str | None = None
    pid: int | None = None
    exception_type: str | None = None
    message: str = ""
    frames: list[str] = field(default_factory=list)
    caused_by: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    hint: str | None = None

    def add_raw(self, line: LogLine) -> None:
        self.raw.append(line.raw)
        self.last_line = line.number


class Extractor:
    """Single-pass state machine over a list of parsed lines."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.pid_to_package: dict[int, str] = {}
        self.open_blocks: dict[FrameKey, _Block] = {}
        self.native_by_pid: dict[int, list[_Block]] = {}
        self.finished: list[_Block] = []
        self.last_message: dict[FrameKey, str] = {}

    # ---- public -------------------------------------------------------------------------
    def run(self, lines: list[LogLine]) -> list[FailureEvent]:
        for line in lines:
            self._expire(line.number)
            self._learn_package(line)
            self._feed(line)
            if line.tag is not None:
                self.last_message[(line.pid, line.tag)] = line.message
        for key in list(self.open_blocks):
            self._close(key)
        events = [self._to_event(block) for block in self.finished if self._is_valid(block)]
        events.sort(key=lambda e: e.start_line)
        _attach_context(events, lines)
        return events

    # ---- bookkeeping ----------------------------------------------------------------------
    def _expire(self, number: int) -> None:
        for key, block in list(self.open_blocks.items()):
            if number - block.last_line > _STALE_AFTER:
                self._close(key)

    def _learn_package(self, line: LogLine) -> None:
        match = _START_PROC.search(line.message)
        if match:
            self.pid_to_package[int(match["pid"])] = match["pkg"]

    def _close(self, key: FrameKey) -> None:
        block = self.open_blocks.pop(key, None)
        if block is None:
            return
        if block.kind is FailureKind.NATIVE_CRASH and block.start.tag == "DEBUG":
            target = self._native_target(block.pid)
            if target is not None:
                target.frames.extend(block.frames)
                target.raw.extend(block.raw)
                target.message = target.message or block.message
                target.package = target.package or block.package
                target.thread = target.thread or block.thread
                return
        self.finished.append(block)

    def _native_target(self, pid: int | None) -> _Block | None:
        if pid is None:
            return None
        for candidate in reversed(self.native_by_pid.get(pid, [])):
            if not candidate.frames:
                return candidate
        return None

    # ---- dispatch -------------------------------------------------------------------------
    def _feed(self, line: LogLine) -> None:
        if line.tag is None:
            return
        key: FrameKey = (line.pid, line.tag)
        block = self.open_blocks.get(key)
        if block is not None and self._continue(block, line):
            block.add_raw(line)
            return
        if block is not None:
            self._close(key)
        self._maybe_open(key, line)

    def _maybe_open(self, key: FrameKey, line: LogLine) -> None:
        msg = line.message
        if line.tag == "AndroidRuntime" and (fatal := _FATAL.match(msg)):
            self._open(key, FailureKind.CRASH, line, thread=fatal["thread"].strip() or None)
            return
        if anr := _ANR.match(msg):
            block = self._open(key, FailureKind.ANR, line, package=anr["pkg"])
            block.exception_type = "ANR"
            block.hint = anr["component"]
            return
        if sig := _FATAL_SIGNAL.search(msg):
            pid = int(sig["pid"]) if sig["pid"] else line.pid
            thread = _TID_NAME.search(msg)
            block = _Block(
                kind=FailureKind.NATIVE_CRASH,
                start=line,
                last_line=line.number,
                pid=pid,
                package=sig["name"] or self.pid_to_package.get(pid or -1),
                thread=thread["thread"] if thread else None,
                exception_type=sig["sig"],
                message=f"{sig['sig']}{sig['rest']}".strip().rstrip(","),
                raw=[line.raw],
            )
            self.finished.append(block)
            if pid is not None:
                self.native_by_pid.setdefault(pid, []).append(block)
            return
        if line.tag == "DEBUG" and _TOMBSTONE_START.match(msg):
            self._open(key, FailureKind.NATIVE_CRASH, line)
            return
        if line.tag != "AndroidRuntime" and (header := EXCEPTION_HEADER.match(msg.strip())):
            block = self._open(key, FailureKind.LOGGED_EXCEPTION, line)
            block.exception_type = header["type"]
            block.message = (header["msg"] or "").strip()
            block.hint = self.last_message.get(key)

    def _open(
        self,
        key: FrameKey,
        kind: FailureKind,
        line: LogLine,
        *,
        thread: str | None = None,
        package: str | None = None,
    ) -> _Block:
        block = _Block(
            kind=kind,
            start=line,
            last_line=line.number,
            thread=thread,
            package=package,
            pid=line.pid,
            raw=[line.raw],
        )
        self.open_blocks[key] = block
        return block

    # ---- continuation rules ---------------------------------------------------------------
    def _continue(self, block: _Block, line: LogLine) -> bool:
        if block.kind is FailureKind.ANR:
            return self._continue_anr(block, line)
        if block.kind is FailureKind.NATIVE_CRASH:
            return self._continue_tombstone(block, line)
        return self._continue_java(block, line)

    def _continue_java(self, block: _Block, line: LogLine) -> bool:
        msg = line.message
        if _FATAL.match(msg):
            return False
        if block.kind is FailureKind.CRASH and (proc := _PROCESS.match(msg)):
            block.package = proc["pkg"]
            block.pid = int(proc["pid"])
            return True
        if _FRAME.match(msg):
            if len(block.frames) < _MAX_FRAMES:
                block.frames.append(msg.strip())
            return True
        if _MORE.match(msg) or _SUPPRESSED.match(msg):
            return True
        if cause := _CAUSED_BY.match(msg):
            block.caused_by.append(cause["rest"].strip())
            return True
        if block.exception_type is None and (header := EXCEPTION_HEADER.match(msg.strip())):
            block.exception_type = header["type"]
            block.message = (header["msg"] or "").strip()
            return True
        # Multi-line exception messages appear before the first frame.
        if block.exception_type is not None and not block.frames and not block.caused_by:
            block.message = f"{block.message} {msg.strip()}".strip()
            return True
        return False

    def _continue_anr(self, block: _Block, line: LogLine) -> bool:
        msg = line.message
        if _ANR.match(msg) or not _ANR_BODY.match(msg) or len(block.raw) >= 40:
            return False
        if pid := _ANR_PID.match(msg):
            block.pid = int(pid["pid"])
        elif (reason := _ANR_REASON.match(msg)) and not block.message:
            block.message = reason["reason"].strip()
        return True

    def _continue_tombstone(self, block: _Block, line: LogLine) -> bool:
        msg = line.message
        if line.tag != "DEBUG" or _TOMBSTONE_START.match(msg):
            return False
        if proc := _TOMBSTONE_PID.search(msg):
            block.pid = int(proc["pid"])
            block.package = proc["pkg"]
            block.thread = proc["thread"].strip()
        elif sig := _TOMBSTONE_SIGNAL.match(msg.strip()):
            block.exception_type = sig["sig"]
            block.message = block.message or f"{sig['sig']}{sig['rest']}".strip()
        elif abort := _TOMBSTONE_ABORT.match(msg.strip()):
            block.message = f"Abort message: {abort['msg']}"
        elif _NATIVE_FRAME.match(msg) and len(block.frames) < _MAX_FRAMES:
            block.frames.append(msg.strip())
        return True

    # ---- output ---------------------------------------------------------------------------
    @staticmethod
    def _is_valid(block: _Block) -> bool:
        if block.kind is FailureKind.LOGGED_EXCEPTION:
            return bool(block.frames)
        if block.kind is FailureKind.CRASH:
            return block.exception_type is not None
        return True

    def _to_event(self, block: _Block) -> FailureEvent:
        package = block.package
        if package is None and block.pid is not None:
            package = self.pid_to_package.get(block.pid)
        if package is None:
            package = guess_package(block.frames)
        message = block.message
        if block.kind is FailureKind.LOGGED_EXCEPTION and block.hint:
            message = f"{message} [logged after: {block.hint.strip()}]".strip()
        if block.kind is FailureKind.ANR and block.hint:
            message = f"{message} [component: {block.hint}]".strip()
        return FailureEvent(
            id=f"{self.source}:{block.start.number}",
            kind=block.kind,
            source=self.source,
            start_line=block.start.number,
            timestamp=block.start.timestamp,
            pid=block.pid,
            package=package.split(":")[0] if package else None,
            thread=block.thread,
            exception_type=block.exception_type,
            message=message,
            frames=block.frames,
            caused_by=block.caused_by,
            raw=block.raw,
        )


def guess_package(frames: list[str]) -> str | None:
    """Return the first three segments of the first non-framework Java frame."""
    for frame in frames:
        target = frame.removeprefix("at ").strip()
        if not target or target.startswith(_FRAMEWORK_PREFIXES) or target.startswith("#"):
            continue
        parts = target.split("(")[0].split(".")
        if len(parts) >= 3:
            return ".".join(parts[:3])
    return None


def _attach_context(events: list[FailureEvent], lines: list[LogLine]) -> None:
    """Attach the last few related lines before each failure; used to infer repro steps."""
    for event in events:
        context: list[str] = []
        index = event.start_line - 2  # line numbers are 1-based
        floor = max(0, event.start_line - 1 - _CONTEXT_LOOKBACK)
        while index >= floor and len(context) < _CONTEXT_LINES:
            line = lines[index]
            index -= 1
            if line.tag is None or line.tag in ("AndroidRuntime", "DEBUG"):
                continue
            related = event.pid is not None and line.pid == event.pid
            mentions = event.package is not None and event.package in line.message
            if (related or mentions) and not _FRAME.match(line.message):
                context.append(f"{line.level}/{line.tag}: {line.message}")
        event.context = list(reversed(context))


def extract_events(lines: list[LogLine], source: str) -> list[FailureEvent]:
    """Extract every failure event from ``lines``; ``source`` labels the event ids."""
    return Extractor(source).run(lines)
