"""Mutable state used while writing one synthetic logcat file."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from logcat_triage.synth import noise

APPS = ("org.example.notes", "org.example.player", "org.example.weather")
MAIN_ACTIVITY = {
    "org.example.notes": ".NotesListActivity",
    "org.example.player": ".PlaybackActivity",
    "org.example.weather": ".ForecastActivity",
}


@dataclass
class Label:
    """Ground truth for one injected failure."""

    line: int
    family: str
    kind: str
    package: str


@dataclass
class SynthContext:
    """Writes threadtime-formatted lines and tracks processes, time and labels."""

    rng: random.Random
    clock: datetime
    noise_rate: float = 0.12
    lines: list[str] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    pids: dict[str, int] = field(default_factory=dict)
    next_pid: int = 0
    system_pid: int = 0
    uids: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.system_pid = self.rng.randint(600, 1600)
        self.next_pid = self.rng.randint(2500, 6000)
        self.uids = {app: 10100 + i * 7 + self.rng.randint(0, 5) for i, app in enumerate(APPS)}

    # ---- low-level emit -------------------------------------------------------------------
    def tick(self, low_ms: int = 1, high_ms: int = 40) -> None:
        self.clock += timedelta(milliseconds=self.rng.randint(low_ms, high_ms))

    def emit(self, pid: int, tid: int, level: str, tag: str, msg: str) -> int:
        """Append one line and return its 1-based line number."""
        self.tick()
        stamp = self.clock.strftime("%m-%d %H:%M:%S.") + f"{self.clock.microsecond // 1000:03d}"
        self.lines.append(f"{stamp} {pid:5d} {tid:5d} {level} {tag:<8}: {msg}")
        return len(self.lines)

    def emit_block(self, pid: int, tid: int, level: str, tag: str, messages: list[str]) -> int:
        """Emit a multi-line block, interleaving unrelated noise like a real device does."""
        first = 0
        for index, msg in enumerate(messages):
            number = self.emit(pid, tid, level, tag, msg)
            if index == 0:
                first = number
            elif self.rng.random() < self.noise_rate / 2:
                self.noise_line(exclude_pid=pid)
        return first

    def noise_line(self, exclude_pid: int | None = None) -> None:
        pid, tid, level, tag, msg = noise.random_line(self, exclude_pid)
        self.emit(pid, tid, level, tag, msg)

    def noise(self, count: int) -> None:
        for _ in range(count):
            self.noise_line()

    # ---- process model --------------------------------------------------------------------
    def app_pid(self, app: str) -> int:
        """Return the app's pid, starting the process (with a log line) if needed."""
        if app not in self.pids:
            self.next_pid += self.rng.randint(3, 400)
            pid = self.next_pid
            self.pids[app] = pid
            activity = MAIN_ACTIVITY[app]
            self.emit(
                self.system_pid,
                self.system_pid + self.rng.randint(20, 90),
                "I",
                "ActivityManager",
                f"Start proc {pid}:{app}/u0a{self.uids[app] - 10000} for "
                f"pre-top-activity {{{app}/{app}{activity}}}",
            )
            self.emit(pid, pid, "I", "Zygote", "Process: zygote64 forked child")
        return self.pids[app]

    def start_activity(self, app: str, activity: str) -> None:
        self.app_pid(app)
        self.emit(
            self.system_pid,
            self.system_pid + self.rng.randint(20, 90),
            "I",
            "ActivityTaskManager",
            f"START u0 {{flg=0x10000000 cmp={app}/{activity}}} from uid 2000",
        )
        self.tick(80, 400)
        self.emit(
            self.system_pid,
            self.system_pid + self.rng.randint(20, 90),
            "I",
            "ActivityTaskManager",
            f"Displayed {app}/{activity}: +{self.rng.randint(180, 1400)}ms",
        )

    def kill(self, app: str) -> None:
        """Simulate the process dying after a fatal crash."""
        pid = self.pids.pop(app, None)
        if pid is None:
            return
        self.emit(pid, pid, "I", "Process", f"Sending signal. PID: {pid} SIG: 9")
        self.emit(
            self.system_pid,
            self.system_pid + self.rng.randint(20, 90),
            "I",
            "ActivityManager",
            f"Process {app} (pid {pid}) has died: fg  TOP ",
        )

    def label(self, line: int, family: str, kind: str, package: str) -> None:
        self.labels.append(Label(line=line, family=family, kind=kind, package=package))
