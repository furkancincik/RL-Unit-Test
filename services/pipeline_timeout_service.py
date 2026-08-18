from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable

from models.pipeline_diagnostic_result import (
    PipelineDiagnosticResult,
    PipelineFunnelSnapshot,
    PipelineRunStatus,
)


class PipelineWorkerFailure(RuntimeError):
    """Global timeout worker'ı sonuç üretemeden durduğunda oluşur."""


class PipelineDiagnosticCheckpointStore:
    """Diagnostic JSON checkpoint'ini atomik olarak yazar ve okur."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("Checkpoint yolu bir Path olmalıdır.")
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def write(self, diagnostic: PipelineDiagnosticResult) -> None:
        if not isinstance(diagnostic, PipelineDiagnosticResult):
            raise TypeError(
                "diagnostic bir PipelineDiagnosticResult olmalıdır."
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_name(
            f".{self._path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary_path.write_text(
                json.dumps(
                    diagnostic.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary_path, self._path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def read(self) -> PipelineDiagnosticResult | None:
        if not self._path.is_file():
            return None
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
            return PipelineDiagnosticResult.from_dict(value)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None


WorkerTarget = Callable[
    [Connection, str, str, dict[str, Any]],
    None,
]


def _worker_bootstrap(
    worker_target: WorkerTarget,
    connection: Connection,
    checkpoint_path: str,
    temporary_root: str,
    run_arguments: dict[str, Any],
) -> None:
    if os.name != "nt":
        os.setsid()
    os.environ["TMP"] = temporary_root
    os.environ["TEMP"] = temporary_root
    os.environ["TMPDIR"] = temporary_root
    worker_target(
        connection,
        checkpoint_path,
        temporary_root,
        run_arguments,
    )


def _real_rl_training_worker(
    connection: Connection,
    checkpoint_path: str,
    temporary_root: str,
    run_arguments: dict[str, Any],
) -> None:
    del temporary_root
    from services.real_rl_training_service import RealRLTrainingService

    service = RealRLTrainingService(
        diagnostic_checkpoint_path=Path(checkpoint_path)
    )
    try:
        result = service.run_with_diagnostics(**run_arguments)
        connection.send(("result", result))
    except Exception as error:
        try:
            connection.send(("error", error))
        except Exception:
            connection.send(("worker_failure", type(error).__name__))
    finally:
        connection.close()


class GlobalPipelineTimeoutRunner:
    """Pipeline'ı spawn worker'da çalıştırıp global deadline uygular."""

    def __init__(
        self,
        worker_target: WorkerTarget = _real_rl_training_worker,
    ) -> None:
        self.worker_target = worker_target
        self.last_worker_pid: int | None = None
        self.last_worker_alive = False
        self.last_temporary_root_exists = False

    def run(
        self,
        *,
        run_arguments: dict[str, Any],
        source_file: Path,
        function_name: str,
        timeout_seconds: float,
    ) -> Any:
        self._validate_timeout(timeout_seconds)
        started_at = time.monotonic()
        temporary_root = Path(
            tempfile.mkdtemp(prefix="rl-unit-test-pipeline-")
        )
        checkpoint_path = temporary_root / "diagnostic.json"
        checkpoint_store = PipelineDiagnosticCheckpointStore(checkpoint_path)
        self.last_temporary_root_exists = True

        context = multiprocessing.get_context("spawn")
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker_bootstrap,
            args=(
                self.worker_target,
                send_connection,
                str(checkpoint_path),
                str(temporary_root),
                run_arguments,
            ),
        )
        process_started = False
        process_stopped = False

        try:
            process.start()
            process_started = True
            self.last_worker_pid = process.pid
            send_connection.close()
            deadline = started_at + float(timeout_seconds)
            message: tuple[str, Any] | None = None

            while True:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0.0:
                    break
                try:
                    if receive_connection.poll(
                        min(0.05, remaining_seconds)
                    ):
                        message = receive_connection.recv()
                        break
                except (BrokenPipeError, EOFError, OSError):
                    break
                if not process.is_alive():
                    try:
                        if receive_connection.poll(0.05):
                            message = receive_connection.recv()
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                    break

            process.join(
                timeout=max(0.0, deadline - time.monotonic())
            )

            if process.is_alive():
                self._terminate_process_tree(process)
                elapsed = max(0.0, time.monotonic() - started_at)
                checkpoint = checkpoint_store.read()
                return self._create_timeout_result(
                    checkpoint=checkpoint,
                    source_file=source_file,
                    function_name=function_name,
                    elapsed_seconds=elapsed,
                    timeout_seconds=float(timeout_seconds),
                )

            if message is not None:
                message_type, payload = message
                if message_type == "result":
                    return payload
                if message_type == "error" and isinstance(
                    payload, BaseException
                ):
                    raise payload
                return self._create_failure_result(
                    checkpoint=checkpoint_store.read(),
                    source_file=source_file,
                    function_name=function_name,
                    elapsed_seconds=max(
                        0.0, time.monotonic() - started_at
                    ),
                    timeout_seconds=float(timeout_seconds),
                )

            return self._create_failure_result(
                checkpoint=checkpoint_store.read(),
                source_file=source_file,
                function_name=function_name,
                elapsed_seconds=max(0.0, time.monotonic() - started_at),
                timeout_seconds=float(timeout_seconds),
            )
        finally:
            try:
                if process_started:
                    if process.is_alive():
                        self._terminate_process_tree(process)
                    process.join(timeout=5.0)
                    self.last_worker_alive = process.is_alive()
                    process_stopped = not self.last_worker_alive
                    if self.last_worker_alive:
                        raise RuntimeError(
                            "Pipeline worker cleanup sonrasında çalışıyor."
                        )
                else:
                    self.last_worker_alive = False
            finally:
                try:
                    receive_connection.close()
                    send_connection.close()
                finally:
                    try:
                        if process_started and process_stopped:
                            process.close()
                    finally:
                        self._cleanup_temporary_root(temporary_root)
                        self.last_temporary_root_exists = (
                            temporary_root.exists()
                        )

    @staticmethod
    def _create_timeout_result(
        *,
        checkpoint: PipelineDiagnosticResult | None,
        source_file: Path,
        function_name: str,
        elapsed_seconds: float,
        timeout_seconds: float,
    ) -> PipelineDiagnosticResult:
        if checkpoint is None:
            return PipelineDiagnosticResult.timed_out(
                source_file=source_file,
                function_name=function_name,
                stopped_stage=None,
                last_completed_stage=None,
                total_duration_seconds=elapsed_seconds,
                funnel=PipelineFunnelSnapshot(),
                pipeline_timeout_seconds=timeout_seconds,
                message="Global pipeline süre sınırı aşıldı.",
            )

        return replace(
            checkpoint,
            status=PipelineRunStatus.TIMED_OUT,
            stopped_stage=(
                checkpoint.stopped_stage
                or checkpoint.last_completed_stage
            ),
            error_category="PIPELINE_TIMEOUT",
            error_message="Global pipeline süre sınırı aşıldı.",
            exception_type=None,
            total_duration_seconds=elapsed_seconds,
            pipeline_timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _create_failure_result(
        *,
        checkpoint: PipelineDiagnosticResult | None,
        source_file: Path,
        function_name: str,
        elapsed_seconds: float,
        timeout_seconds: float,
    ) -> PipelineDiagnosticResult:
        if checkpoint is None:
            return PipelineDiagnosticResult(
                status=PipelineRunStatus.FAILED,
                source_file=source_file,
                function_name=function_name,
                last_completed_stage=None,
                stopped_stage=None,
                error_category="PIPELINE_WORKER_FAILURE",
                error_message="Pipeline worker güvenli sonuç iletmeden durdu.",
                exception_type="PipelineWorkerFailure",
                total_duration_seconds=elapsed_seconds,
                funnel=PipelineFunnelSnapshot(),
                pipeline_timeout_seconds=timeout_seconds,
            )
        return replace(
            checkpoint,
            status=PipelineRunStatus.FAILED,
            stopped_stage=(
                checkpoint.stopped_stage
                or checkpoint.last_completed_stage
            ),
            error_category="PIPELINE_WORKER_FAILURE",
            error_message="Pipeline worker güvenli sonuç iletmeden durdu.",
            exception_type="PipelineWorkerFailure",
            total_duration_seconds=elapsed_seconds,
            pipeline_timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _terminate_process_tree(process: multiprocessing.Process) -> None:
        pid = process.pid
        if pid is None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ("taskkill", "/PID", str(pid), "/T", "/F"),
                    capture_output=True,
                    check=False,
                    timeout=10.0,
                )
            except subprocess.TimeoutExpired:
                pass
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)

    @staticmethod
    def _cleanup_temporary_root(temporary_root: Path) -> None:
        resolved_root = temporary_root.resolve()
        resolved_system_temp = Path(tempfile.gettempdir()).resolve()
        if (
            resolved_root.parent != resolved_system_temp
            or not resolved_root.name.startswith("rl-unit-test-pipeline-")
        ):
            raise RuntimeError(
                "Pipeline geçici klasörü güvenli cleanup kapsamı dışında."
            )
        shutil.rmtree(resolved_root, ignore_errors=True)

    @staticmethod
    def _validate_timeout(value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("pipeline_timeout_seconds sayısal olmalıdır.")
        if not 0.0 < float(value) < float("inf"):
            raise ValueError(
                "pipeline_timeout_seconds pozitif ve sonlu olmalıdır."
            )
