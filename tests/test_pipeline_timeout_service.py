from __future__ import annotations

import json
import multiprocessing.process
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineFunnelSnapshot,
    PipelineRunStatus,
    PipelineStage,
)
from services.pipeline_timeout_service import (
    GlobalPipelineTimeoutRunner,
    PipelineDiagnosticCheckpointStore,
)


def _track_parent_process_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, bool]]:
    events: list[tuple[str, bool]] = []
    process_type = multiprocessing.process.BaseProcess
    original_join = process_type.join
    original_close = process_type.close
    original_is_alive = process_type.is_alive

    def tracked_join(self: Any, *args: Any, **kwargs: Any) -> None:
        original_join(self, *args, **kwargs)
        events.append(("join", original_is_alive(self)))

    def tracked_close(self: Any) -> None:
        events.append(("close", original_is_alive(self)))
        original_close(self)

    monkeypatch.setattr(process_type, "join", tracked_join)
    monkeypatch.setattr(process_type, "close", tracked_close)
    return events


def _partial_diagnostic(stage: PipelineStage) -> PipelineDiagnosticResult:
    return PipelineDiagnosticResult(
        status=PipelineRunStatus.PARTIAL,
        source_file=Path("sample.py"),
        function_name="target",
        last_completed_stage=PipelineStage.PATH_DISCOVERY,
        stopped_stage=stage,
        total_duration_seconds=0.01,
        funnel=PipelineFunnelSnapshot(bounded_path_count=7),
    )


def blocking_checkpoint_worker(
    connection: Any,
    checkpoint_path: str,
    temporary_root: str,
    run_arguments: dict[str, Any],
) -> None:
    del connection, temporary_root
    stage = PipelineStage(run_arguments["stage"])
    PipelineDiagnosticCheckpointStore(Path(checkpoint_path)).write(
        _partial_diagnostic(stage)
    )
    time.sleep(10.0)


def blocking_worker_without_checkpoint(
    connection: Any,
    checkpoint_path: str,
    temporary_root: str,
    run_arguments: dict[str, Any],
) -> None:
    del connection, checkpoint_path, temporary_root, run_arguments
    time.sleep(10.0)


def successful_worker(
    connection: Any,
    checkpoint_path: str,
    temporary_root: str,
    run_arguments: dict[str, Any],
) -> None:
    del checkpoint_path, temporary_root, run_arguments
    connection.send(("result", "completed"))
    connection.close()


def large_result_worker(
    connection: Any,
    checkpoint_path: str,
    temporary_root: str,
    run_arguments: dict[str, Any],
) -> None:
    del checkpoint_path, temporary_root, run_arguments
    connection.send(("result", "x" * 2_000_000))
    connection.close()


def failing_worker(
    connection: Any,
    checkpoint_path: str,
    temporary_root: str,
    run_arguments: dict[str, Any],
) -> None:
    del checkpoint_path, temporary_root, run_arguments
    connection.send(("error", RuntimeError("worker failed")))
    connection.close()


def crashing_worker(
    connection: Any,
    checkpoint_path: str,
    temporary_root: str,
    run_arguments: dict[str, Any],
) -> None:
    del connection, checkpoint_path, temporary_root, run_arguments
    raise SystemExit(3)


def child_process_worker(
    connection: Any,
    checkpoint_path: str,
    temporary_root: str,
    run_arguments: dict[str, Any],
) -> None:
    del connection, temporary_root
    child = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(10)"),
    )
    Path(run_arguments["pid_file"]).write_text(
        str(child.pid), encoding="utf-8"
    )
    PipelineDiagnosticCheckpointStore(Path(checkpoint_path)).write(
        _partial_diagnostic(PipelineStage.TEST_EXECUTION)
    )
    time.sleep(10.0)


def coverage_subprocess_worker(
    connection: Any,
    checkpoint_path: str,
    temporary_root: str,
    run_arguments: dict[str, Any],
) -> None:
    del connection
    coverage_root = Path(temporary_root) / "coverage_subprocess"
    coverage_root.mkdir()
    source_file = coverage_root / "target_module.py"
    test_file = coverage_root / "test_target_module.py"
    source_file.write_text("def value() -> int:\n    return 7\n", encoding="utf-8")
    test_file.write_text(
        "import time\n"
        "from target_module import value\n\n"
        "def test_value() -> None:\n"
        "    assert value() == 7\n"
        "    time.sleep(10.0)\n",
        encoding="utf-8",
    )
    coverage_file = coverage_root / ".coverage"
    child = subprocess.Popen(
        (
            sys.executable,
            "-m",
            "coverage",
            "run",
            f"--data-file={coverage_file}",
            "-m",
            "pytest",
            "-q",
            test_file.name,
        ),
        cwd=coverage_root,
    )
    Path(run_arguments["pid_file"]).write_text(
        str(child.pid), encoding="utf-8"
    )
    PipelineDiagnosticCheckpointStore(Path(checkpoint_path)).write(
        _partial_diagnostic(PipelineStage.COVERAGE_MEASUREMENT)
    )
    time.sleep(10.0)


def _system_pipeline_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="rl-unit-test-pipeline-"))


def _remove_test_pipeline_root(root: Path) -> None:
    if not root.exists():
        return
    for candidate in sorted(root.rglob("*"), reverse=True):
        try:
            candidate.chmod(stat.S_IWRITE)
        except OSError:
            pass
    shutil.rmtree(root)


def test_checkpoint_round_trip_is_atomic_and_json_safe(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    store = PipelineDiagnosticCheckpointStore(path)
    diagnostic = _partial_diagnostic(PipelineStage.SCENARIO_GENERATION)

    store.write(diagnostic)

    assert store.read() == diagnostic
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "PARTIAL"
    assert not tuple(tmp_path.glob("*.tmp"))


def test_corrupt_checkpoint_is_ignored_safely(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text('{"status":', encoding="utf-8")

    assert PipelineDiagnosticCheckpointStore(path).read() is None


def test_checkpoint_does_not_contain_sensitive_values(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    PipelineDiagnosticCheckpointStore(path).write(
        _partial_diagnostic(PipelineStage.CONCRETE_VALIDATION)
    )

    text = path.read_text(encoding="utf-8")
    assert "keyword_arguments" not in text
    assert "expected_result" not in text
    assert "actual_result" not in text


@pytest.mark.parametrize(
    "stage",
    (
        PipelineStage.PATH_DISCOVERY,
        PipelineStage.SCENARIO_GENERATION,
        PipelineStage.CONCRETE_VALIDATION,
        PipelineStage.COVERAGE_MEASUREMENT,
        PipelineStage.RL_TRAINING,
    ),
)
def test_runner_times_out_and_preserves_latest_checkpoint(
    stage: PipelineStage,
) -> None:
    runner = GlobalPipelineTimeoutRunner(worker_target=blocking_checkpoint_worker)

    result = runner.run(
        run_arguments={"stage": stage.value},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=0.5,
    )

    assert result.status is PipelineRunStatus.TIMED_OUT
    assert result.stopped_stage is stage
    assert result.funnel.bounded_path_count == 7
    assert result.pipeline_timeout_seconds == pytest.approx(0.5)
    assert runner.last_worker_alive is False
    assert runner.last_temporary_root_exists is False


def test_runner_builds_minimum_timeout_without_checkpoint() -> None:
    runner = GlobalPipelineTimeoutRunner(
        worker_target=blocking_worker_without_checkpoint
    )

    result = runner.run(
        run_arguments={},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=0.2,
    )

    assert result.status is PipelineRunStatus.TIMED_OUT
    assert result.funnel.bounded_path_count is None
    assert result.line_coverage_percent is None


def test_runner_returns_normal_worker_result() -> None:
    runner = GlobalPipelineTimeoutRunner(worker_target=successful_worker)
    assert runner.run(
        run_arguments={},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=2.0,
    ) == "completed"


def test_normal_worker_process_handle_closes_once_after_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _track_parent_process_cleanup(monkeypatch)
    runner = GlobalPipelineTimeoutRunner(worker_target=successful_worker)

    assert runner.run(
        run_arguments={},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=2.0,
    ) == "completed"

    assert [event for event, _ in events].count("close") == 1
    close_index = next(
        index for index, event in enumerate(events) if event[0] == "close"
    )
    assert any(event[0] == "join" for event in events[:close_index])
    assert events[close_index] == ("close", False)


def test_pipe_endpoints_close_before_the_process_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeConnection:
        def __init__(self, message: tuple[str, str] | None = None) -> None:
            self.message = message
            self.closed = False

        def poll(self, timeout: float) -> bool:
            del timeout
            return self.message is not None

        def recv(self) -> tuple[str, str]:
            assert self.message is not None
            message = self.message
            self.message = None
            return message

        def close(self) -> None:
            self.closed = True
            events.append("pipe_close")

    class FakeProcess:
        pid = 12345

        def __init__(self) -> None:
            self.started = False
            self.alive = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            assert not self.closed
            return self.alive

        def join(self, timeout: float) -> None:
            del timeout
            assert self.started
            events.append("join")

        def close(self) -> None:
            assert self.started
            assert not self.alive
            assert not self.closed
            self.closed = True
            events.append("process_close")

    receive_connection = FakeConnection(("result", "completed"))
    send_connection = FakeConnection()
    process = FakeProcess()

    class FakeContext:
        @staticmethod
        def Pipe(duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            assert duplex is False
            return receive_connection, send_connection

        @staticmethod
        def Process(*args: Any, **kwargs: Any) -> FakeProcess:
            del args, kwargs
            return process

    monkeypatch.setattr(
        "services.pipeline_timeout_service.multiprocessing.get_context",
        lambda method: FakeContext() if method == "spawn" else None,
    )

    result = GlobalPipelineTimeoutRunner(worker_target=successful_worker).run(
        run_arguments={},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=2.0,
    )

    assert result == "completed"
    assert receive_connection.closed is True
    assert send_connection.closed is True
    assert process.closed is True
    assert events[-1] == "process_close"
    assert "join" in events[: events.index("process_close")]
    assert events.count("process_close") == 1


def test_timed_out_worker_process_handle_closes_once_after_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _track_parent_process_cleanup(monkeypatch)
    runner = GlobalPipelineTimeoutRunner(
        worker_target=blocking_worker_without_checkpoint
    )

    result = runner.run(
        run_arguments={},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=0.2,
    )

    assert result.status is PipelineRunStatus.TIMED_OUT
    assert [event for event, _ in events].count("close") == 1
    close_index = next(
        index for index, event in enumerate(events) if event[0] == "close"
    )
    assert any(event[0] == "join" for event in events[:close_index])
    assert events[close_index] == ("close", False)


def test_failed_worker_process_handle_closes_once_after_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _track_parent_process_cleanup(monkeypatch)
    runner = GlobalPipelineTimeoutRunner(worker_target=failing_worker)

    with pytest.raises(RuntimeError, match="worker failed"):
        runner.run(
            run_arguments={},
            source_file=Path("sample.py"),
            function_name="target",
            timeout_seconds=2.0,
        )

    assert [event for event, _ in events].count("close") == 1
    close_index = next(
        index for index, event in enumerate(events) if event[0] == "close"
    )
    assert any(event[0] == "join" for event in events[:close_index])
    assert events[close_index] == ("close", False)


def test_runner_drains_large_worker_result_without_false_timeout() -> None:
    runner = GlobalPipelineTimeoutRunner(worker_target=large_result_worker)
    result = runner.run(
        run_arguments={},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=3.0,
    )

    assert len(result) == 2_000_000


def test_runner_propagates_worker_failure_as_failure_not_timeout() -> None:
    runner = GlobalPipelineTimeoutRunner(worker_target=failing_worker)
    with pytest.raises(RuntimeError, match="worker failed"):
        runner.run(
            run_arguments={},
            source_file=Path("sample.py"),
            function_name="target",
            timeout_seconds=2.0,
        )


def test_worker_crash_is_failure_not_timeout() -> None:
    runner = GlobalPipelineTimeoutRunner(worker_target=crashing_worker)
    result = runner.run(
        run_arguments={},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=2.0,
    )

    assert result.status is PipelineRunStatus.FAILED
    assert result.error_category == "PIPELINE_WORKER_FAILURE"


@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree assertion")
def test_timeout_terminates_worker_child_process_tree(tmp_path: Path) -> None:
    import ctypes

    pid_file = tmp_path / "child.pid"
    runner = GlobalPipelineTimeoutRunner(worker_target=child_process_worker)
    result = runner.run(
        run_arguments={"pid_file": str(pid_file)},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=0.8,
    )

    child_pid = int(pid_file.read_text(encoding="utf-8"))
    process_handle = ctypes.windll.kernel32.OpenProcess(
        0x1000, False, child_pid
    )
    if process_handle:
        ctypes.windll.kernel32.CloseHandle(process_handle)

    assert result.status is PipelineRunStatus.TIMED_OUT
    assert process_handle == 0


def test_timeout_runner_can_be_reused_after_timeout() -> None:
    runner = GlobalPipelineTimeoutRunner(worker_target=blocking_checkpoint_worker)
    first = runner.run(
        run_arguments={"stage": PipelineStage.PATH_DISCOVERY.value},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=0.3,
    )
    runner.worker_target = successful_worker
    second = runner.run(
        run_arguments={},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=2.0,
    )

    assert first.status is PipelineRunStatus.TIMED_OUT
    assert second == "completed"


def test_cleanup_retries_permission_error_then_succeeds() -> None:
    root = _system_pipeline_root()
    attempts: list[Path] = []
    sleeps: list[float] = []

    def flaky_rmtree(path: Path, **kwargs: Any) -> None:
        attempts.append(Path(path))
        if len(attempts) == 1:
            raise PermissionError("private locked path")
        shutil.rmtree(path, **kwargs)

    runner = GlobalPipelineTimeoutRunner(
        cleanup_attempts=3,
        cleanup_backoff_seconds=0.01,
        cleanup_sleeper=sleeps.append,
        cleanup_rmtree=flaky_rmtree,
    )
    try:
        runner._cleanup_temporary_root(root)
    finally:
        _remove_test_pipeline_root(root)

    assert attempts == [root.resolve(), root.resolve()]
    assert sleeps == [pytest.approx(0.01)]
    assert not root.exists()


def test_cleanup_retries_delayed_release_with_increasing_backoff() -> None:
    root = _system_pipeline_root()
    attempts = 0
    sleeps: list[float] = []

    def delayed_rmtree(path: Path, **kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("delayed release")
        shutil.rmtree(path, **kwargs)

    runner = GlobalPipelineTimeoutRunner(
        cleanup_attempts=4,
        cleanup_backoff_seconds=0.02,
        cleanup_sleeper=sleeps.append,
        cleanup_rmtree=delayed_rmtree,
    )
    try:
        runner._cleanup_temporary_root(root)
    finally:
        _remove_test_pipeline_root(root)

    assert attempts == 3
    assert sleeps == [pytest.approx(0.02), pytest.approx(0.04)]
    assert not root.exists()


def test_cleanup_makes_read_only_coverage_file_writable_inside_owned_root() -> None:
    root = _system_pipeline_root()
    coverage_file = root / ".coverage"
    coverage_file.write_bytes(b"coverage")
    coverage_file.chmod(stat.S_IREAD)

    try:
        GlobalPipelineTimeoutRunner()._cleanup_temporary_root(root)
    finally:
        _remove_test_pipeline_root(root)

    assert not root.exists()


def test_persistent_cleanup_failure_is_safe_domain_result() -> None:
    created: list[Path] = []

    def failing_rmtree(path: Path, **kwargs: Any) -> None:
        del kwargs
        created.append(Path(path))
        raise PermissionError("private OS path and message")

    runner = GlobalPipelineTimeoutRunner(
        worker_target=successful_worker,
        cleanup_attempts=2,
        cleanup_backoff_seconds=0.0,
        cleanup_sleeper=lambda _: None,
        cleanup_rmtree=failing_rmtree,
    )
    try:
        result = runner.run(
            run_arguments={},
            source_file=Path("sample.py"),
            function_name="target",
            timeout_seconds=2.0,
        )
        payload = json.dumps(result.to_dict())
    finally:
        for root in set(created):
            _remove_test_pipeline_root(root)

    assert result.status is PipelineRunStatus.FAILED
    assert result.error_category == "PIPELINE_CLEANUP_FAILED"
    assert result.exception_type == "PipelineCleanupError"
    assert "private OS path" not in payload
    assert "PermissionError" not in payload
    assert runner.last_worker_alive is False
    assert runner.last_temporary_root_exists is True


def test_cleanup_failure_preserves_timeout_checkpoint_snapshot() -> None:
    created: list[Path] = []

    def failing_rmtree(path: Path, **kwargs: Any) -> None:
        del kwargs
        created.append(Path(path))
        raise OSError("private cleanup failure")

    runner = GlobalPipelineTimeoutRunner(
        worker_target=blocking_checkpoint_worker,
        cleanup_attempts=2,
        cleanup_backoff_seconds=0.0,
        cleanup_sleeper=lambda _: None,
        cleanup_rmtree=failing_rmtree,
    )
    try:
        result = runner.run(
            run_arguments={"stage": PipelineStage.COVERAGE_MEASUREMENT.value},
            source_file=Path("sample.py"),
            function_name="target",
            timeout_seconds=0.3,
        )
    finally:
        for root in set(created):
            _remove_test_pipeline_root(root)

    assert result.status is PipelineRunStatus.FAILED
    assert result.error_category == "PIPELINE_CLEANUP_FAILED"
    assert result.last_completed_stage is PipelineStage.PATH_DISCOVERY
    assert result.stopped_stage is PipelineStage.COVERAGE_MEASUREMENT
    assert result.funnel.bounded_path_count == 7
    assert result.pipeline_timeout_seconds == pytest.approx(0.3)


def test_pipe_process_handle_and_filesystem_cleanup_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeConnection:
        def __init__(
            self,
            name: str,
            message: tuple[str, str] | None = None,
        ) -> None:
            self.name = name
            self.message = message

        def poll(self, timeout: float) -> bool:
            del timeout
            return self.message is not None

        def recv(self) -> tuple[str, str]:
            assert self.message is not None
            message = self.message
            self.message = None
            return message

        def close(self) -> None:
            events.append(f"{self.name}_close")

    class FakeProcess:
        pid = 12345

        def start(self) -> None:
            events.append("start")

        @staticmethod
        def is_alive() -> bool:
            return False

        @staticmethod
        def join(timeout: float) -> None:
            del timeout
            events.append("join")

        @staticmethod
        def close() -> None:
            events.append("process_close")

    receive_connection = FakeConnection("receive", ("result", "completed"))
    send_connection = FakeConnection("send")

    class FakeContext:
        @staticmethod
        def Pipe(duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            assert duplex is False
            return receive_connection, send_connection

        @staticmethod
        def Process(*args: Any, **kwargs: Any) -> FakeProcess:
            del args, kwargs
            return FakeProcess()

    monkeypatch.setattr(
        "services.pipeline_timeout_service.multiprocessing.get_context",
        lambda method: FakeContext() if method == "spawn" else None,
    )
    runner = GlobalPipelineTimeoutRunner(worker_target=successful_worker)
    original_cleanup = runner._cleanup_temporary_root

    def tracked_cleanup(root: Path) -> None:
        events.append("filesystem_cleanup")
        original_cleanup(root)

    monkeypatch.setattr(runner, "_cleanup_temporary_root", tracked_cleanup)

    assert runner.run(
        run_arguments={},
        source_file=Path("sample.py"),
        function_name="target",
        timeout_seconds=2.0,
    ) == "completed"
    assert events.index("send_close") < events.index("join")
    assert events.index("join") < events.index("receive_close")
    assert events.index("receive_close") < events.index("process_close")
    assert events.index("process_close") < events.index("filesystem_cleanup")


def test_live_worker_never_closes_handle_or_deletes_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    created_root = _system_pipeline_root()

    class FakeConnection:
        @staticmethod
        def poll(timeout: float) -> bool:
            del timeout
            return False

        @staticmethod
        def recv() -> tuple[str, str]:
            raise AssertionError("recv çağrılmamalı")

        @staticmethod
        def close() -> None:
            events.append("pipe_close")

    class FakeProcess:
        pid = 12345

        @staticmethod
        def start() -> None:
            events.append("start")

        @staticmethod
        def is_alive() -> bool:
            return True

        @staticmethod
        def join(timeout: float) -> None:
            del timeout
            events.append("join")

        @staticmethod
        def close() -> None:
            events.append("process_close")

        @staticmethod
        def kill() -> None:
            events.append("kill")

    class FakeContext:
        @staticmethod
        def Pipe(duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            assert duplex is False
            return FakeConnection(), FakeConnection()

        @staticmethod
        def Process(*args: Any, **kwargs: Any) -> FakeProcess:
            del args, kwargs
            return FakeProcess()

    monkeypatch.setattr(
        "services.pipeline_timeout_service.tempfile.mkdtemp",
        lambda prefix: str(created_root),
    )
    monkeypatch.setattr(
        "services.pipeline_timeout_service.multiprocessing.get_context",
        lambda method: FakeContext() if method == "spawn" else None,
    )
    runner = GlobalPipelineTimeoutRunner(worker_target=successful_worker)
    monkeypatch.setattr(runner, "_terminate_process_tree", lambda process: None)
    monkeypatch.setattr(
        runner,
        "_cleanup_temporary_root",
        lambda root: events.append("filesystem_cleanup"),
    )

    try:
        with pytest.raises(RuntimeError, match="cleanup sonrasında çalışıyor"):
            runner.run(
                run_arguments={},
                source_file=Path("sample.py"),
                function_name="target",
                timeout_seconds=0.01,
            )
    finally:
        _remove_test_pipeline_root(created_root)

    assert "process_close" not in events
    assert "filesystem_cleanup" not in events


@pytest.mark.parametrize(
    "root_factory",
    (
        lambda: Path(tempfile.mkdtemp(prefix="foreign-prefix-")),
        lambda: _system_pipeline_root() / "nested",
    ),
)
def test_cleanup_rejects_unowned_or_nested_root(root_factory: Any) -> None:
    root = root_factory()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "user-data.txt"
    marker.write_text("preserve", encoding="utf-8")

    try:
        with pytest.raises(RuntimeError, match="güvenli cleanup kapsamı"):
            GlobalPipelineTimeoutRunner()._cleanup_temporary_root(root)
        assert marker.read_text(encoding="utf-8") == "preserve"
    finally:
        top = root
        while top.parent.name.startswith("rl-unit-test-pipeline-"):
            top = top.parent
        _remove_test_pipeline_root(top)


def test_cleanup_touches_only_the_current_pipeline_root() -> None:
    current = _system_pipeline_root()
    pre_existing = _system_pipeline_root()
    marker = pre_existing / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    try:
        GlobalPipelineTimeoutRunner()._cleanup_temporary_root(current)
        assert not current.exists()
        assert marker.read_text(encoding="utf-8") == "preserve"
    finally:
        _remove_test_pipeline_root(current)
        _remove_test_pipeline_root(pre_existing)


@pytest.mark.parametrize(
    "exception_type",
    (AssertionError, TypeError, RuntimeError),
)
def test_unexpected_cleanup_exception_still_propagates(
    exception_type: type[Exception],
) -> None:
    created: list[Path] = []

    def unexpected_rmtree(path: Path, **kwargs: Any) -> None:
        del kwargs
        created.append(Path(path))
        raise exception_type("programming bug")

    runner = GlobalPipelineTimeoutRunner(
        worker_target=successful_worker,
        cleanup_rmtree=unexpected_rmtree,
    )
    try:
        with pytest.raises(exception_type, match="programming bug"):
            runner.run(
                run_arguments={},
                source_file=Path("sample.py"),
                function_name="target",
                timeout_seconds=2.0,
            )
    finally:
        for root in set(created):
            _remove_test_pipeline_root(root)


@pytest.mark.skipif(os.name != "nt", reason="Windows coverage process-tree acceptance")
def test_real_coverage_timeout_leaves_no_new_pipeline_temp_and_next_run_succeeds(
    tmp_path: Path,
) -> None:
    temp_root = Path(tempfile.gettempdir())
    pre_existing = _system_pipeline_root()
    marker = pre_existing / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    before = frozenset(temp_root.glob("rl-unit-test-pipeline-*"))
    pid_file = tmp_path / "coverage.pid"
    runner = GlobalPipelineTimeoutRunner(worker_target=coverage_subprocess_worker)

    try:
        first = runner.run(
            run_arguments={"pid_file": str(pid_file)},
            source_file=Path("sample.py"),
            function_name="target",
            timeout_seconds=1.0,
        )
        runner.worker_target = successful_worker
        second = runner.run(
            run_arguments={},
            source_file=Path("sample.py"),
            function_name="target",
            timeout_seconds=2.0,
        )
        after = frozenset(temp_root.glob("rl-unit-test-pipeline-*"))

        import ctypes

        coverage_pid = int(pid_file.read_text(encoding="utf-8"))
        process_handle = ctypes.windll.kernel32.OpenProcess(
            0x1000, False, coverage_pid
        )
        if process_handle:
            ctypes.windll.kernel32.CloseHandle(process_handle)

        assert first.status is PipelineRunStatus.TIMED_OUT
        assert first.stopped_stage is PipelineStage.COVERAGE_MEASUREMENT
        assert second == "completed"
        assert after == before
        assert marker.read_text(encoding="utf-8") == "preserve"
        assert process_handle == 0
        assert runner.last_worker_alive is False
        assert runner.last_temporary_root_exists is False
    finally:
        _remove_test_pipeline_root(pre_existing)
