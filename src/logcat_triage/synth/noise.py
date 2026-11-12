"""Background noise lines: the kind of chatter that surrounds every real failure."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logcat_triage.synth.context import SynthContext

NoiseLine = tuple[int, int, str, str, str]

_SYSTEM_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    (
        "I",
        "ActivityManager",
        "Background concurrent copying GC freed {n}({kb}KB) AllocSpace objects",
    ),
    ("D", "ConnectivityService", "requestNetwork for uid/pid:{uid}/{pid} activeRequest: null"),
    ("I", "WifiService", "getWifiEnabledState uid={uid}"),
    ("D", "BatteryService", "Processing new values: level={pct}, temperature={temp}"),
    ("W", "ActivityManager", "Slow operation: {ms}ms so far, now at startProcess: done"),
    ("I", "chatty", "uid=1000(system) Binder:{pid}_{n} identical {n} lines"),
    ("D", "PowerManagerService", 'acquireWakeLockInternal: lock={hex}, flags=0x1, tag="*alarm*"'),
    ("I", "InputDispatcher", "Focus entering window (channel {hex})"),
    ("V", "AlarmManager", "Triggering alarm #{n}: 3 when={ms} type=ELAPSED_WAKEUP"),
    ("W", "PackageManager", "Not granting permission android.permission.QUERY_ALL_PACKAGES"),
    ("D", "BluetoothAdapter", "isLeEnabled(): ON"),
    ("I", "NetworkScheduler", "Scheduling job uid={uid} for network change"),
)
_APP_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("I", "Choreographer", "Skipped {n} frames!  The application may be doing too much work"),
    ("D", "OpenGLRenderer", "Davey! duration={ms}ms; Flags=0, FrameTimelineVsyncId={n}"),
    ("I", "art", "Background young concurrent copying GC freed {n}({kb}KB) AllocSpace objects"),
    ("D", "OkHttp", "<-- 200 {url} ({ms}ms, {kb}-byte body)"),
    ("D", "ProfileInstaller", "Installing profile for {pkg}"),
    ("W", "System", "A resource failed to call close."),
    ("D", "SyncWorker", "retrying after IOException, attempt {small}"),
    ("I", "WM-WorkerWrapper", "Worker result SUCCESS for Work [ id={uuid} ]"),
    ("W", "System.err", "java.io.FileNotFoundException: /cache/tmp-{n}.json (No such file)"),
)
_URLS = (
    "https://api.example.org/v1/sync",
    "https://cdn.example.net/assets/manifest.json",
    "https://tiles.example.com/radar/7/42/63.png",
)


def _fill(template: str, ctx: SynthContext, pid: int, pkg: str) -> str:
    rng = ctx.rng
    return template.format(
        n=rng.randint(2, 9999),
        kb=rng.randint(100, 40000),
        uid=rng.randint(10050, 10400),
        pid=pid,
        pct=rng.randint(5, 100),
        temp=rng.randint(240, 410),
        ms=rng.randint(16, 3500),
        hex=f"{rng.getrandbits(28):07x}",
        url=rng.choice(_URLS),
        pkg=pkg,
        small=rng.randint(1, 5),
        uuid=f"{rng.getrandbits(32):08x}-{rng.getrandbits(16):04x}-4{rng.getrandbits(12):03x}-"
        f"a{rng.getrandbits(12):03x}-{rng.getrandbits(48):012x}",
    )


def random_line(ctx: SynthContext, exclude_pid: int | None = None) -> NoiseLine:
    """Return ``(pid, tid, level, tag, message)`` for one noise line."""
    rng = ctx.rng
    running = [(pkg, pid) for pkg, pid in ctx.pids.items() if pid != exclude_pid]
    if running and rng.random() < 0.45:
        pkg, pid = rng.choice(running)
        level, tag, template = rng.choice(_APP_TEMPLATES)
        tid = pid if rng.random() < 0.6 else pid + rng.randint(1, 60)
        return pid, tid, level, tag, _fill(template, ctx, pid, pkg)
    level, tag, template = rng.choice(_SYSTEM_TEMPLATES)
    pid = ctx.system_pid
    return pid, pid + rng.randint(1, 120), level, tag, _fill(template, ctx, pid, "android")
