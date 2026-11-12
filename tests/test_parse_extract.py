from __future__ import annotations

from pathlib import Path

from logcat_triage.extract import extract_events, guess_package
from logcat_triage.models import FailureKind
from logcat_triage.parser import parse_line, read_log


def test_parse_threadtime_line() -> None:
    line = parse_line("03-01 10:00:00.300  4321  4322 E AndroidRuntime: FATAL EXCEPTION: main\n", 7)
    assert line.number == 7
    assert line.timestamp == "03-01 10:00:00.300"
    assert (line.pid, line.tid, line.level, line.tag) == (4321, 4322, "E", "AndroidRuntime")
    assert line.message == "FATAL EXCEPTION: main"


def test_parse_threadtime_with_year_and_padded_tag() -> None:
    line = parse_line("2026-03-01 10:00:06.000  6000  6020 F libc    : Fatal signal 11", 1)
    assert line.tag == "libc"
    assert line.message == "Fatal signal 11"


def test_parse_brief_line() -> None:
    line = parse_line("E/NotesRepository( 4400): Failed to save note id=7", 3)
    assert (line.level, line.tag, line.pid, line.tid) == ("E", "NotesRepository", 4400, None)
    assert line.message == "Failed to save note id=7"


def test_unparseable_line_is_kept() -> None:
    line = parse_line("--------- beginning of crash", 1)
    assert line.tag is None
    assert line.message == "--------- beginning of crash"


def test_extracts_every_kind(mixed_lines) -> None:
    events = extract_events(mixed_lines, "mixed.logcat")
    by_line = {e.start_line: e for e in events}
    assert sorted(by_line) == [5, 14, 22, 33, 38]
    kinds = {line: e.kind for line, e in by_line.items()}
    assert kinds == {
        5: FailureKind.CRASH,
        14: FailureKind.ANR,
        22: FailureKind.NATIVE_CRASH,
        33: FailureKind.LOGGED_EXCEPTION,
        38: FailureKind.NATIVE_CRASH,
    }


def test_crash_block_survives_interleaved_noise(mixed_lines) -> None:
    crash = next(e for e in extract_events(mixed_lines, "m") if e.kind is FailureKind.CRASH)
    assert crash.package == "org.example.notes"
    assert crash.pid == 4321
    assert crash.thread == "main"
    assert crash.exception_type == "java.lang.NullPointerException"
    assert "getTitle()" in crash.message
    assert len(crash.frames) == 4
    assert crash.id == "m:5"
    assert any("openNote id=42" in line for line in crash.context)


def test_anr_fields(mixed_lines) -> None:
    anr = next(e for e in extract_events(mixed_lines, "m") if e.kind is FailureKind.ANR)
    assert anr.package == "org.example.player"
    assert anr.pid == 5555
    assert anr.exception_type == "ANR"
    assert anr.message.startswith("Input dispatching timed out")
    assert "component: org.example.player/.PlaybackActivity" in anr.message


def test_native_crash_merges_tombstone(mixed_lines) -> None:
    natives = [e for e in extract_events(mixed_lines, "m") if e.kind is FailureKind.NATIVE_CRASH]
    merged = next(e for e in natives if e.start_line == 22)
    assert merged.exception_type == "SIGSEGV"
    assert merged.package == "org.example.player"
    assert merged.thread == "DecoderThread"
    assert len(merged.frames) == 2
    assert "libplayercore.so" in merged.frames[0]
    assert any("MediaCodecWrapper" in line for line in merged.context)


def test_standalone_tombstone(mixed_lines) -> None:
    natives = [e for e in extract_events(mixed_lines, "m") if e.kind is FailureKind.NATIVE_CRASH]
    lone = next(e for e in natives if e.start_line == 38)
    assert lone.package == "org.example.weather"
    assert lone.exception_type == "SIGABRT"
    assert lone.thread == "RenderThread"
    assert lone.frames
    assert "abort" in lone.frames[0]


def test_logged_exception_with_cause_and_hint(mixed_lines) -> None:
    logged = next(
        e for e in extract_events(mixed_lines, "m") if e.kind is FailureKind.LOGGED_EXCEPTION
    )
    assert logged.exception_type == "java.lang.RuntimeException"
    assert "logged after: Failed to save note id=7" in logged.message
    assert logged.caused_by
    assert "SQLiteConstraintException" in logged.caused_by[0]
    assert len(logged.frames) == 2
    assert logged.package == "org.example.notes"  # guessed from frames


def test_header_without_frames_is_not_an_event(mixed_lines) -> None:
    events = extract_events(mixed_lines, "m")
    assert all(e.exception_type != "java.io.FileNotFoundException" for e in events)


def test_stale_block_is_closed() -> None:
    from logcat_triage.parser import parse_lines

    lines = [
        "01-01 00:00:00.000   100   100 E Repo: java.lang.IllegalArgumentException: bad",
        "01-01 00:00:00.000   100   100 E Repo:     at org.example.a.B.c(B.kt:1)",
        *[f"01-01 00:00:00.000   200   200 I Other: filler {i}" for i in range(70)],
        "01-01 00:00:00.000   100   100 E Repo:     at org.example.a.B.d(B.kt:2)",
    ]
    events = extract_events(list(parse_lines(lines)), "s")
    assert len(events) == 1
    assert len(events[0].frames) == 1


def test_multiline_exception_message() -> None:
    from logcat_triage.parser import parse_lines

    lines = [
        "01-01 00:00:00.000   100   100 E AndroidRuntime: FATAL EXCEPTION: worker-1",
        "01-01 00:00:00.000   100   100 E AndroidRuntime: Process: org.example.notes, PID: 100",
        "01-01 00:00:00.000   100   100 E AndroidRuntime: java.lang.IllegalStateException: first",
        "01-01 00:00:00.000   100   100 E AndroidRuntime: second line of message",
        "01-01 00:00:00.000   100   100 E AndroidRuntime:     at org.example.notes.X.y(X.kt:3)",
        "01-01 00:00:00.000   100   100 E AndroidRuntime: FATAL EXCEPTION: main",
        "01-01 00:00:00.000   100   100 E AndroidRuntime: Process: org.example.notes, PID: 100",
        "01-01 00:00:00.000   100   100 E AndroidRuntime: java.lang.RuntimeException",
    ]
    events = extract_events(list(parse_lines(lines)), "s")
    assert [e.start_line for e in events] == [1, 6]
    assert events[0].message == "first second line of message"
    assert events[0].thread == "worker-1"


def test_crash_without_exception_line_is_dropped() -> None:
    from logcat_triage.parser import parse_lines

    lines = ["01-01 00:00:00.000   100   100 E AndroidRuntime: FATAL EXCEPTION: main"]
    assert extract_events(list(parse_lines(lines)), "s") == []


def test_guess_package_skips_framework_frames() -> None:
    frames = [
        "at android.os.Handler.dispatchMessage(Handler.java:106)",
        "at org.example.player.ui.Foo.bar(Foo.kt:1)",
    ]
    assert guess_package(frames) == "org.example.player"
    assert guess_package(["at java.lang.Thread.run(Thread.java:1)"]) is None


def test_read_log(tmp_path: Path) -> None:
    path = tmp_path / "x.logcat"
    path.write_bytes(b"01-01 00:00:00.000   1   1 I Tag: ok\n\xff\xfe broken\n")
    lines = read_log(path)
    assert len(lines) == 2
    assert lines[0].tag == "Tag"
