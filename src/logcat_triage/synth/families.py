"""Ten bugs from seven failure families (NPE, IllegalStateException, OOM, ANR, SQLite,
native SIGSEGV, NetworkOnMainThread) in three fictional apps.

NPE, IllegalStateException and ANR each have two unrelated bugs, so clustering has to
separate failures that share an exception type, an app and most framework frames. Each
ground-truth family is one underlying bug. Every occurrence varies the details a real device varies:
line numbers (different app/OS builds), PIDs/TIDs, thread names, message parameters, call
paths into the faulty method, memory addresses and how much framework stack is printed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logcat_triage.synth.context import SynthContext

NOTES = "org.example.notes"
PLAYER = "org.example.player"
WEATHER = "org.example.weather"


def _ln(ctx: SynthContext, low: int, high: int) -> int:
    return ctx.rng.randint(low, high)


def _looper_tail(ctx: SynthContext) -> list[str]:
    """Main-thread framework frames; the printed depth varies like real traces do."""
    frames = [
        f"\tat android.os.Handler.handleCallback(Handler.java:{ctx.rng.choice([942, 958, 959])})",
        f"\tat android.os.Handler.dispatchMessage(Handler.java:{ctx.rng.choice([99, 100, 106])})",
        f"\tat android.os.Looper.loopOnce(Looper.java:{ctx.rng.choice([201, 205, 232])})",
        f"\tat android.os.Looper.loop(Looper.java:{ctx.rng.choice([288, 294, 317])})",
        f"\tat android.app.ActivityThread.main(ActivityThread.java:{_ln(ctx, 7800, 8900)})",
        "\tat java.lang.reflect.Method.invoke(Native Method)",
        "\tat com.android.internal.os.RuntimeInit$MethodAndArgsCaller.run"
        f"(RuntimeInit.java:{ctx.rng.choice([548, 569, 574])})",
        f"\tat com.android.internal.os.ZygoteInit.main(ZygoteInit.java:{_ln(ctx, 920, 1010)})",
    ]
    return frames[: ctx.rng.randint(4, len(frames))]


def _fatal(ctx: SynthContext, app: str, thread: str, body: list[str], family: str) -> None:
    pid = ctx.app_pid(app)
    tid = pid if thread == "main" else pid + ctx.rng.randint(5, 80)
    messages = [f"FATAL EXCEPTION: {thread}", f"Process: {app}, PID: {pid}", *body]
    if thread == "main" and ctx.rng.random() < 0.5:
        ctx.emit(pid, tid, "D", "AndroidRuntime", "Shutting down VM")
    line = ctx.emit_block(pid, tid, "E", "AndroidRuntime", messages)
    ctx.label(line, family, "crash", app)
    ctx.kill(app)


# ---- 1. NullPointerException in the notes editor -------------------------------------------
def npe_note_editor(ctx: SynthContext) -> None:
    rng = ctx.rng
    pid = ctx.app_pid(NOTES)
    note_id = rng.randint(1, 5000)
    ctx.start_activity(NOTES, ".editor.NoteEditorActivity")
    ctx.emit(pid, pid, "D", "NoteEditor", f"openNote id={note_id} fromWidget={rng.random() < 0.3}")
    ctx.emit(
        pid, pid, "W", "NotesRepository", f"note {note_id} not found in cache, deleted remotely?"
    )
    entry = rng.choice(
        [
            [
                f"\tat org.example.notes.editor.NoteEditorFragment.onViewCreated(NoteEditorFragment.kt:{_ln(ctx, 60, 75)})",
                f"\tat androidx.fragment.app.Fragment.performViewCreated(Fragment.java:{_ln(ctx, 3080, 3130)})",
            ],
            [
                f"\tat org.example.notes.editor.NoteEditorFragment.onNoteLoaded(NoteEditorFragment.kt:{_ln(ctx, 110, 130)})",
                f"\tat org.example.notes.editor.NoteEditorViewModel$load$1.invokeSuspend(NoteEditorViewModel.kt:{_ln(ctx, 40, 52)})",
                f"\tat kotlin.coroutines.jvm.internal.BaseContinuationImpl.resumeWith(ContinuationImpl.kt:{_ln(ctx, 33, 34)})",
            ],
        ]
    )
    body = [
        "java.lang.NullPointerException: Attempt to invoke virtual method "
        "'java.lang.String org.example.notes.model.Note.getTitle()' on a null object reference",
        f"\tat org.example.notes.editor.NoteEditorFragment.bindNote(NoteEditorFragment.kt:{_ln(ctx, 140, 158)})",
        *entry,
        *_looper_tail(ctx),
    ]
    _fatal(ctx, NOTES, "main", body, "npe_note_editor")


# ---- 2. IllegalStateException: fragment used after detach ------------------------------------
def ise_player_fragment(ctx: SynthContext) -> None:
    rng = ctx.rng
    pid = ctx.app_pid(PLAYER)
    ctx.start_activity(PLAYER, ".PlaybackActivity")
    ctx.emit(pid, pid, "D", "QueueObserver", f"queue changed size={rng.randint(1, 80)}")
    ctx.emit(pid, pid, "D", "PlaybackActivity", "onPause isFinishing=true")
    frag = f"{rng.getrandbits(28):07x}"
    uuid = f"{rng.getrandbits(32):08x}-{rng.getrandbits(16):04x}-4{rng.getrandbits(12):03x}-b{rng.getrandbits(12):03x}-{rng.getrandbits(48):012x}"
    body = [
        f"java.lang.IllegalStateException: Fragment PlaybackControlsFragment{{{frag}}} ({uuid}) "
        "not attached to a context.",
        f"\tat androidx.fragment.app.Fragment.requireContext(Fragment.java:{_ln(ctx, 966, 995)})",
        f"\tat org.example.player.ui.PlaybackControlsFragment.updateQueueBadge(PlaybackControlsFragment.kt:{_ln(ctx, 201, 219)})",
        f"\tat org.example.player.ui.PlaybackControlsFragment.access$updateQueueBadge(PlaybackControlsFragment.kt:{_ln(ctx, 40, 44)})",
        f"\tat org.example.player.queue.QueueObserver$onChanged$1.run(QueueObserver.kt:{_ln(ctx, 30, 38)})",
        *_looper_tail(ctx),
    ]
    _fatal(ctx, PLAYER, "main", body, "ise_player_fragment")


# ---- 3. OutOfMemoryError decoding radar tiles -----------------------------------------------
def oom_weather_bitmap(ctx: SynthContext) -> None:
    rng = ctx.rng
    pid = ctx.app_pid(WEATHER)
    ctx.start_activity(WEATHER, ".radar.RadarActivity")
    zoom = rng.randint(6, 12)
    ctx.emit(pid, pid, "D", "RadarTileLoader", f"prefetch zoom={zoom} tiles={rng.randint(40, 300)}")
    for _ in range(rng.randint(1, 3)):
        ctx.emit(
            pid, pid, "I", "art", f"Clamp target GC heap from {rng.randint(260, 300)}MB to 256MB"
        )
    alloc = rng.choice([4194316, 8388620, 16777228, 33554444, 52428816])
    free = rng.randint(1_000_000, alloc - 1)
    body = [
        f"java.lang.OutOfMemoryError: Failed to allocate a {alloc} byte allocation with {free} "
        f"free bytes and {rng.randint(1, 30)}MB until OOM, target footprint 268435456, "
        "growth limit 268435456",
        "\tat android.graphics.BitmapFactory.nativeDecodeStream(Native Method)",
        f"\tat android.graphics.BitmapFactory.decodeStreamInternal(BitmapFactory.java:{rng.choice([761, 790])})",
        f"\tat android.graphics.BitmapFactory.decodeStream(BitmapFactory.java:{rng.choice([736, 765])})",
        f"\tat org.example.weather.radar.RadarTileLoader.decodeTile(RadarTileLoader.kt:{_ln(ctx, 88, 97)})",
        f"\tat org.example.weather.radar.RadarTileLoader.loadTile(RadarTileLoader.kt:{_ln(ctx, 60, 70)})",
        f"\tat org.example.weather.radar.RadarTileLoader$prefetch$1.run(RadarTileLoader.kt:{_ln(ctx, 41, 45)})",
        f"\tat java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:{rng.choice([1137, 1145, 1167])})",
        f"\tat java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:{rng.choice([637, 644])})",
        f"\tat java.lang.Thread.run(Thread.java:{rng.choice([923, 1012, 1119])})",
    ]
    _fatal(ctx, WEATHER, f"RadarLoader-{rng.randint(1, 8)}", body, "oom_weather_bitmap")


# ---- 4. ANR: main thread blocked during playback --------------------------------------------
def anr_player_main(ctx: SynthContext) -> None:
    rng = ctx.rng
    pid = ctx.app_pid(PLAYER)
    ctx.start_activity(PLAYER, ".PlaybackActivity")
    ctx.emit(
        pid, pid, "D", "PlaylistImporter", f"importing {rng.randint(200, 4000)} tracks on main"
    )
    ctx.emit(
        pid,
        pid,
        "I",
        "Choreographer",
        f"Skipped {rng.randint(120, 600)} frames!  The application may be doing too much work on its main thread.",
    )
    ctx.tick(4000, 6000)
    waited = rng.randint(5000, 5100)
    sys_tid = ctx.system_pid + rng.randint(20, 90)
    messages = [
        f"ANR in {PLAYER} ({PLAYER}/.PlaybackActivity)",
        f"PID: {pid}",
        f"Reason: Input dispatching timed out ({rng.getrandbits(28):07x} {PLAYER}/{PLAYER}.PlaybackActivity "
        f"(server) is not responding. Waited {waited}ms for MotionEvent)",
        f"Parent: {PLAYER}/.PlaybackActivity",
        f"ErrorId: {rng.getrandbits(32):08x}-{rng.getrandbits(16):04x}-4{rng.getrandbits(12):03x}-8{rng.getrandbits(12):03x}-{rng.getrandbits(48):012x}",
        "Frozen: false",
        f"Load: {rng.uniform(1, 9):.2f} / {rng.uniform(1, 9):.2f} / {rng.uniform(1, 9):.2f}",
        f"CPU usage from {rng.randint(0, 900)}ms to {rng.randint(9000, 12000)}ms ago:",
        f"  {rng.randint(60, 99)}% {pid}/{PLAYER}: {rng.randint(50, 90)}% user + {rng.randint(1, 20)}% kernel",
        f"  {rng.randint(5, 30)}% {ctx.system_pid}/system_server: {rng.randint(3, 20)}% user + {rng.randint(1, 9)}% kernel",
    ]
    line = ctx.emit_block(ctx.system_pid, sys_tid, "E", "ActivityManager", messages)
    ctx.label(line, "anr_player_main", "anr", PLAYER)
    if rng.random() < 0.6:
        ctx.kill(PLAYER)


# ---- 5. SQLiteConstraintException caught and logged ----------------------------------------
def sqlite_notes_logged(ctx: SynthContext) -> None:
    rng = ctx.rng
    pid = ctx.app_pid(NOTES)
    tid = pid + rng.randint(10, 90)
    note_id = rng.randint(1, 5000)
    ctx.emit(
        pid, pid, "D", "NoteEditor", f"onSaveClicked id={note_id} chars={rng.randint(1, 9000)}"
    )
    from_sync = rng.random() < 0.5
    caller = (
        f"\tat org.example.notes.sync.SyncWorker.pushPending(SyncWorker.kt:{_ln(ctx, 55, 66)})"
        if from_sync
        else f"\tat org.example.notes.editor.NoteEditorViewModel$save$1.invokeSuspend(NoteEditorViewModel.kt:{_ln(ctx, 70, 84)})"
    )
    sqlite_header = (
        "android.database.sqlite.SQLiteConstraintException: UNIQUE constraint failed: "
        f"notes.uuid (code {rng.choice([2067, 1555])} SQLITE_CONSTRAINT_UNIQUE)"
    )
    sqlite_frames = [
        "\tat android.database.sqlite.SQLiteConnection.nativeExecuteForLastInsertedRowId(Native Method)",
        f"\tat android.database.sqlite.SQLiteConnection.executeForLastInsertedRowId(SQLiteConnection.java:{rng.choice([938, 961])})",
        f"\tat android.database.sqlite.SQLiteSession.executeForLastInsertedRowId(SQLiteSession.java:{rng.choice([790, 818])})",
        f"\tat android.database.sqlite.SQLiteStatement.executeInsert(SQLiteStatement.java:{rng.choice([89, 90])})",
        f"\tat org.example.notes.data.NoteDao_Impl.insert(NoteDao_Impl.java:{_ln(ctx, 120, 160)})",
        f"\tat org.example.notes.data.NotesRepository.save(NotesRepository.kt:{_ln(ctx, 77, 95)})",
        caller,
    ]
    if from_sync and rng.random() < 0.5:
        # The sync path wraps the database error: same bug, different top-level exception.
        messages = [
            f"Failed to save note id={note_id}",
            f"java.lang.RuntimeException: sync push failed for batch {rng.randint(1, 900)}",
            f"\tat org.example.notes.sync.SyncWorker.pushPending(SyncWorker.kt:{_ln(ctx, 67, 72)})",
            f"\tat org.example.notes.sync.SyncWorker.doWork(SyncWorker.kt:{_ln(ctx, 30, 41)})",
            f"\tat androidx.work.Worker$2.run(Worker.java:{rng.choice([82, 86])})",
            f"Caused by: {sqlite_header}",
            *sqlite_frames[:-1],
            f"\t... {rng.randint(6, 14)} more",
        ]
    else:
        messages = [f"Failed to save note id={note_id}", sqlite_header, *sqlite_frames]
        if rng.random() < 0.5:
            messages.append(f"\t... {rng.randint(6, 14)} more")
    first = ctx.emit_block(pid, tid, "E", "NotesRepository", messages)
    ctx.label(first + 1, "sqlite_notes_logged", "logged_exception", NOTES)


# ---- 6. Native SIGSEGV in the decoder -------------------------------------------------------
def native_sigsegv_player(ctx: SynthContext) -> None:
    rng = ctx.rng
    pid = ctx.app_pid(PLAYER)
    ctx.emit(
        pid,
        pid,
        "I",
        "MediaCodecWrapper",
        f"configure codec=video/avc {rng.choice(['1920x1080', '1280x720', '3840x2160'])}",
    )
    tid = pid + rng.randint(20, 200)
    thread = rng.choice(["DecoderThread", "DecoderThread-2", "CodecLooper"])
    fault = rng.choice(["0x0", "0x8", "0x10", f"0x{rng.getrandbits(36):x}"])
    line = ctx.emit(
        pid,
        tid,
        "F",
        "libc",
        f"Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr {fault} in tid {tid} ({thread}), pid {pid} ({PLAYER})",
    )
    ctx.label(line, "native_sigsegv_player", "native_crash", PLAYER)
    if rng.random() < 0.85:
        dbg = pid + rng.randint(300, 900)
        app_dir = (
            f"/data/app/~~{rng.getrandbits(64):x}==/{PLAYER}-{rng.getrandbits(64):x}==/lib/arm64"
        )
        build = f"{rng.getrandbits(64):016x}"
        frames = [
            f"      #00 pc {rng.getrandbits(32):016x}  {app_dir}/libplayercore.so (decode_frame+{rng.randint(80, 400)}) (BuildId: {build})",
            f"      #01 pc {rng.getrandbits(32):016x}  {app_dir}/libplayercore.so (VideoDecoder::submit(Packet const&)+{rng.randint(40, 300)}) (BuildId: {build})",
            f"      #02 pc {rng.getrandbits(32):016x}  {app_dir}/libplayercore.so (decoder_loop+{rng.randint(100, 900)}) (BuildId: {build})",
            f"      #03 pc {rng.getrandbits(32):016x}  /apex/com.android.runtime/lib64/bionic/libc.so (__pthread_start(void*)+{rng.choice([196, 208])})",
            f"      #04 pc {rng.getrandbits(32):016x}  /apex/com.android.runtime/lib64/bionic/libc.so (__start_thread+{rng.choice([64, 68])})",
        ]
        messages = [
            "*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***",
            f"Build fingerprint: 'example/sdk_gphone64/emu64a:{rng.choice([13, 14, 15])}/AE3A.{rng.randint(240000, 249999)}/{rng.randint(1000000, 9999999)}:userdebug/dev-keys'",
            f"Revision: '{rng.randint(0, 2)}'",
            "ABI: 'arm64'",
            f"pid: {pid}, tid: {tid}, name: {thread}  >>> {PLAYER} <<<",
            f"signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr {fault}",
            f"    x0  {rng.getrandbits(64):016x}  x1  {rng.getrandbits(64):016x}",
            "backtrace:",
            *frames,
        ]
        ctx.emit_block(dbg, dbg, "F", "DEBUG", messages)
    ctx.kill(PLAYER)


# ---- 7. NetworkOnMainThreadException on refresh ---------------------------------------------
def network_main_weather(ctx: SynthContext) -> None:
    rng = ctx.rng
    pid = ctx.app_pid(WEATHER)
    ctx.start_activity(WEATHER, ".ForecastActivity")
    city = rng.choice(["Lisbon", "Toronto", "Manaus", "Berlin", "Austin", "Dublin"])
    ctx.emit(
        pid,
        pid,
        "D",
        "ForecastActivity",
        f"onRefreshClicked city={city} lastSync={rng.randint(1, 600)}s",
    )
    via_swipe = rng.random() < 0.5
    trigger = (
        [
            f"\tat org.example.weather.ui.ForecastActivity.onRefresh(ForecastActivity.kt:{_ln(ctx, 120, 131)})",
            f"\tat androidx.swiperefreshlayout.widget.SwipeRefreshLayout.finishSpinner(SwipeRefreshLayout.java:{_ln(ctx, 960, 970)})",
        ]
        if via_swipe
        else [
            f"\tat org.example.weather.ui.ForecastActivity.onRefreshClicked(ForecastActivity.kt:{_ln(ctx, 98, 110)})",
            f"\tat android.view.View.performClick(View.java:{rng.choice([7448, 7659, 7892])})",
            f"\tat android.view.View.performClickInternal(View.java:{rng.choice([7425, 7636, 7869])})",
        ]
    )
    body = [
        "android.os.NetworkOnMainThreadException",
        f"\tat android.os.StrictMode$AndroidBlockGuardPolicy.onNetwork(StrictMode.java:{rng.choice([1605, 1668, 1675])})",
        f"\tat java.net.Inet6AddressImpl.lookupHostByName(Inet6AddressImpl.java:{rng.choice([115, 117])})",
        f"\tat java.net.Inet6AddressImpl.lookupAllHostAddr(Inet6AddressImpl.java:{rng.choice([103, 105])})",
        f"\tat java.net.InetAddress.getAllByName(InetAddress.java:{rng.choice([1152, 1155])})",
        f"\tat okhttp3.Dns$Companion$DnsSystem.lookup(Dns.kt:{rng.choice([49, 50])})",
        f"\tat okhttp3.internal.connection.RouteSelector.resetNextInetSocketAddress(RouteSelector.kt:{rng.choice([169, 171])})",
        f"\tat okhttp3.internal.connection.RealCall.execute(RealCall.kt:{rng.choice([154, 160])})",
        f"\tat org.example.weather.forecast.ForecastClient.fetchNow(ForecastClient.kt:{_ln(ctx, 44, 58)})",
        *trigger,
        *_looper_tail(ctx),
    ]
    _fatal(ctx, WEATHER, "main", body, "network_main_weather")


# ---- 8. A second, unrelated NPE in the same notes app ----------------------------------------
def npe_notes_list(ctx: SynthContext) -> None:
    rng = ctx.rng
    pid = ctx.app_pid(NOTES)
    ctx.start_activity(NOTES, ".NotesListActivity")
    ctx.emit(
        pid,
        pid,
        "D",
        "NotesListAdapter",
        f"submitList size={rng.randint(0, 400)} filter={rng.choice(['all', 'pinned', 'archived'])}",
    )
    body = [
        "java.lang.NullPointerException: Attempt to invoke virtual method "
        "'long org.example.notes.model.Note.getUpdatedAt()' on a null object reference",
        f"\tat org.example.notes.list.NotesListAdapter.onBindViewHolder(NotesListAdapter.kt:{_ln(ctx, 48, 63)})",
        f"\tat org.example.notes.list.NotesListAdapter.onBindViewHolder(NotesListAdapter.kt:{_ln(ctx, 20, 24)})",
        f"\tat androidx.recyclerview.widget.RecyclerView$Adapter.onBindViewHolder(RecyclerView.java:{_ln(ctx, 7254, 7300)})",
        f"\tat androidx.recyclerview.widget.RecyclerView$Adapter.bindViewHolder(RecyclerView.java:{_ln(ctx, 7337, 7380)})",
        f"\tat androidx.recyclerview.widget.RecyclerView$Recycler.tryBindViewHolderByDeadline(RecyclerView.java:{_ln(ctx, 6194, 6240)})",
        *_looper_tail(ctx),
    ]
    _fatal(ctx, NOTES, "main", body, "npe_notes_list")


# ---- 9. A second, unrelated IllegalStateException in the player -----------------------------
def ise_player_released(ctx: SynthContext) -> None:
    rng = ctx.rng
    pid = ctx.app_pid(PLAYER)
    ctx.emit(
        pid, pid, "D", "PlaybackService", f"onStartCommand action=PLAY startId={rng.randint(1, 40)}"
    )
    ctx.emit(pid, pid, "D", "PlayerEngine", "release() called from onTaskRemoved")
    body = [
        "java.lang.IllegalStateException: Cannot call play() after release()",
        f"\tat org.example.player.engine.PlayerEngine.checkNotReleased(PlayerEngine.kt:{_ln(ctx, 310, 322)})",
        f"\tat org.example.player.engine.PlayerEngine.play(PlayerEngine.kt:{_ln(ctx, 120, 133)})",
        f"\tat org.example.player.service.PlaybackService.onStartCommand(PlaybackService.kt:{_ln(ctx, 77, 90)})",
        f"\tat android.app.ActivityThread.handleServiceArgs(ActivityThread.java:{_ln(ctx, 4890, 5020)})",
        *_looper_tail(ctx)[2:],
    ]
    _fatal(ctx, PLAYER, "main", body, "ise_player_released")


# ---- 10. A second ANR: a service timeout in the notes app -----------------------------------
def anr_notes_service(ctx: SynthContext) -> None:
    rng = ctx.rng
    pid = ctx.app_pid(NOTES)
    ctx.emit(
        pid, pid, "D", "SyncService", f"onStartCommand full resync accounts={rng.randint(1, 3)}"
    )
    ctx.tick(15000, 21000)
    sys_tid = ctx.system_pid + rng.randint(20, 90)
    error_id = (
        f"{rng.getrandbits(32):08x}-{rng.getrandbits(16):04x}-4{rng.getrandbits(12):03x}-"
        f"9{rng.getrandbits(12):03x}-{rng.getrandbits(48):012x}"
    )
    messages = [
        f"ANR in {NOTES}",
        f"PID: {pid}",
        f"Reason: executing service {NOTES}/.sync.SyncService",
        f"ErrorId: {error_id}",
        "Frozen: false",
        f"Load: {rng.uniform(1, 9):.2f} / {rng.uniform(1, 9):.2f} / {rng.uniform(1, 9):.2f}",
        f"CPU usage from {rng.randint(0, 900)}ms to {rng.randint(19000, 22000)}ms ago:",
        f"  {rng.randint(30, 90)}% {pid}/{NOTES}: {rng.randint(20, 80)}% user + {rng.randint(1, 20)}% kernel",
    ]
    line = ctx.emit_block(ctx.system_pid, sys_tid, "E", "ActivityManager", messages)
    ctx.label(line, "anr_notes_service", "anr", NOTES)
    ctx.kill(NOTES)


FAMILIES: dict[str, Callable[[SynthContext], None]] = {
    "npe_note_editor": npe_note_editor,
    "ise_player_fragment": ise_player_fragment,
    "oom_weather_bitmap": oom_weather_bitmap,
    "anr_player_main": anr_player_main,
    "sqlite_notes_logged": sqlite_notes_logged,
    "native_sigsegv_player": native_sigsegv_player,
    "network_main_weather": network_main_weather,
    "npe_notes_list": npe_notes_list,
    "ise_player_released": ise_player_released,
    "anr_notes_service": anr_notes_service,
}
