"""Minimal, allowlisted ADB access.

Only two read-only commands are allowed: ``adb devices`` and ``adb logcat -d`` with a small
set of flags. Commands are passed to ``subprocess.run`` as an argument list with
``shell=False``; nothing user-supplied is ever interpreted by a shell. Tests and the demo use
:class:`FakeTransport`, so no device is needed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

LOGCAT_BUFFERS = frozenset({"main", "system", "crash", "events", "radio", "default", "all"})
MAX_TAIL = 200_000
_SERIAL = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")


class AdbError(RuntimeError):
    """Raised for disallowed commands, missing adb, timeouts and non-zero exits."""


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one adb invocation."""

    returncode: int
    stdout: str
    stderr: str = ""


class Transport(Protocol):
    """Runs an already-validated adb argument list."""

    def run(self, args: Sequence[str], timeout: float) -> CommandResult: ...


def validate_args(args: Sequence[str]) -> list[str]:
    """Return ``args`` as a list if allowlisted, else raise :class:`AdbError`."""
    items = list(args)
    if not items:
        raise AdbError("empty adb command")
    position = 0
    if items[0] == "-s":
        if len(items) < 2 or not _SERIAL.match(items[1]):
            raise AdbError("invalid device serial")
        position = 2
    command, rest = items[position], items[position + 1 :]
    if command == "devices":
        if rest not in ([], ["-l"]):
            raise AdbError(f"flags not allowed for 'devices': {rest}")
        return items
    if command != "logcat":
        raise AdbError(f"adb command not allowed: {command!r}")
    if "-d" not in rest:
        raise AdbError("logcat must be called with -d (dump and exit)")
    index = 0
    while index < len(rest):
        flag = rest[index]
        value = rest[index + 1] if index + 1 < len(rest) else ""
        if flag == "-d":
            index += 1
            continue
        if flag == "-t" and value.isdigit():
            if not 0 < int(value) <= MAX_TAIL:
                raise AdbError(f"-t must be between 1 and {MAX_TAIL}")
        elif not (
            (flag == "-v" and value == "threadtime") or (flag == "-b" and value in LOGCAT_BUFFERS)
        ):
            raise AdbError(f"logcat argument not allowed: {flag!r}")
        index += 2
    return items


class SubprocessTransport:
    """Runs the real ``adb`` binary with ``shell=False``."""

    def __init__(self, adb_path: str | None = None) -> None:
        resolved = adb_path or shutil.which("adb")
        if resolved is None:
            raise AdbError("adb not found on PATH; install Android platform-tools")
        self.adb_path = resolved

    def run(self, args: Sequence[str], timeout: float) -> CommandResult:
        try:
            completed = subprocess.run(
                [self.adb_path, *validate_args(args)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"adb timed out after {timeout:.0f}s") from exc
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass
class FakeTransport:
    """In-memory transport: answers ``devices`` and ``logcat`` from canned text."""

    logcat_text: str = ""
    devices: Sequence[str] = ("emulator-5554",)
    calls: list[list[str]] = field(default_factory=list)
    handler: Callable[[list[str]], CommandResult] | None = None

    @classmethod
    def from_file(cls, path: Path) -> FakeTransport:
        return cls(logcat_text=path.read_text(encoding="utf-8", errors="replace"))

    def run(self, args: Sequence[str], timeout: float) -> CommandResult:
        items = validate_args(args)
        self.calls.append(items)
        if self.handler is not None:
            return self.handler(items)
        command = items[2] if items[0] == "-s" else items[0]
        if command == "devices":
            body = "".join(f"{serial}\tdevice\n" for serial in self.devices)
            return CommandResult(0, f"List of devices attached\n{body}\n")
        return CommandResult(0, self.logcat_text)


class AdbClient:
    """High-level, read-only operations used by the CLI."""

    def __init__(self, transport: Transport, serial: str | None = None, timeout: float = 60.0):
        if serial is not None and not _SERIAL.match(serial):
            raise AdbError("invalid device serial")
        self.transport = transport
        self.serial = serial
        self.timeout = timeout

    def _run(self, args: list[str]) -> str:
        full = (["-s", self.serial] if self.serial else []) + args
        result = self.transport.run(validate_args(full), self.timeout)
        if result.returncode != 0:
            raise AdbError(f"adb {' '.join(args)} failed: {result.stderr.strip()[:300]}")
        return result.stdout

    def list_devices(self) -> list[str]:
        """Serials of devices in the ``device`` state."""
        serials = []
        for line in self._run(["devices"]).splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                serials.append(parts[0])
        return serials

    def dump_logcat(
        self, buffers: Sequence[str] = ("main", "system", "crash"), tail: int | None = None
    ) -> str:
        """``adb logcat -d -v threadtime -b ...``: dump the current buffers and exit."""
        args = ["logcat", "-d", "-v", "threadtime"]
        for buffer in buffers:
            args += ["-b", buffer]
        if tail is not None:
            args += ["-t", str(tail)]
        return self._run(args)
