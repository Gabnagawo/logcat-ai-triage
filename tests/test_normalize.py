from __future__ import annotations

from logcat_triage.extract import extract_events
from logcat_triage.models import FailureEvent, FailureKind
from logcat_triage.normalize import (
    build_signature,
    normalize_events,
    normalize_frame,
    normalize_message,
    normalize_text,
)


def test_java_frame_drops_line_numbers() -> None:
    assert normalize_frame("\tat org.example.notes.A.b(A.kt:142)") == "org.example.notes.A.b(A.kt)"
    assert normalize_frame("at java.lang.reflect.Method.invoke(Native Method)") == (
        "java.lang.reflect.Method.invoke(Native Method)"
    )


def test_native_frame_drops_pc_offsets_paths_and_build_ids() -> None:
    a = normalize_frame(
        "#00 pc 000000000004a2b4  /data/app/~~Ab12==/org.example.player-Cd34==/lib/arm64/"
        "libplayercore.so (decode_frame+132) (BuildId: 0123456789abcdef)"
    )
    b = normalize_frame(
        "#00 pc 00000000000ffff0  /data/app/~~Zz99==/org.example.player-Qq11==/lib/arm64/"
        "libplayercore.so (decode_frame+400) (BuildId: fedcba9876543210)"
    )
    assert a == b == "pc /data/app/org.example.player/lib/arm64/libplayercore.so (decode_frame)"


def test_text_strips_pids_timestamps_addresses_and_numbers() -> None:
    text = "03-01 10:00:00.300 pid 4321 tid: 77 fault addr 0x7f00 took 5001ms size=8MB at 0xdead"
    assert normalize_text(text) == (
        "<ts> pid <pid> tid: <pid> fault addr <addr> took <n> size=<n> at <addr>"
    )


def test_text_masks_uuids_urls_and_object_hashes() -> None:
    text = (
        "Fragment F{eb2c79d} (7644d38c-0da3-4196-a788-0993c775b3dd) GET https://x.example/a?b=1 "
        "Object@1a2b3c window 52dd34d"
    )
    assert normalize_text(text) == (
        "Fragment F{<hex>} (<uuid>) GET <url> Object@<hex> window <hex>"
    )


def test_message_masks_quoted_values_but_keeps_signatures() -> None:
    msg = "Invalid name 'bob' for 'java.lang.String a.B.c()'"
    assert normalize_message(msg) == "Invalid name '<v>' for 'java.lang.String a.B.c()'"


def test_signature_identical_for_two_occurrences(mixed_lines) -> None:
    event = extract_events(mixed_lines, "m")[0]
    twin = event.model_copy(
        update={
            "pid": 999,
            "frames": [f.replace(":142", ":150").replace(":61", ":66") for f in event.frames],
        }
    )
    assert build_signature(event) == build_signature(twin)


def test_signature_includes_cause_and_kind() -> None:
    event = FailureEvent(
        id="x:1",
        kind=FailureKind.LOGGED_EXCEPTION,
        source="x",
        start_line=1,
        exception_type="java.lang.RuntimeException",
        message="batch 12 failed",
        caused_by=["android.database.sqlite.SQLiteException: code 1"],
        frames=["at org.example.a.B.c(B.kt:9)"],
    )
    sig = build_signature(event)
    assert sig.splitlines()[0] == "logged_exception"
    assert "caused by android.database.sqlite.SQLiteException: code <n>" in sig
    assert "batch <n> failed" in sig
    assert normalize_events([event])[0].signature == sig
