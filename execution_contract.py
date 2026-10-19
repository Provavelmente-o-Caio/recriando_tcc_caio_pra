import json
import os
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = 1


def _write_json(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    os.replace(temporary_path, path)


def write_execution_contract(
    path: str,
    job_id: str,
    operation: str,
    status: str,
    message: str = "",
    error_type: str | None = None,
) -> None:
    if status not in {"running", "success", "error", "cancelled"}:
        raise ValueError(f"Unsupported execution status: {status}")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "operation": operation,
        "status": status,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error_type:
        payload["error_type"] = error_type
    _write_json(path, payload)


def write_progress_contract(
    path: str,
    job_id: str,
    operation: str,
    percent: float,
    stage: str,
    message: str = "",
) -> None:
    if not 0 <= percent <= 100:
        raise ValueError("Progress percent must be between 0 and 100.")

    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "operation": operation,
            "status": "progress",
            "percent": percent,
            "stage": stage,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def emit_progress(
    path: str,
    job_id: str,
    operation: str,
    percent: float,
    stage: str,
    message: str = "",
) -> None:
    write_progress_contract(path, job_id, operation, percent, stage, message)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "job_id": job_id,
                "operation": operation,
                "status": "progress",
                "percent": percent,
                "stage": stage,
                "message": message,
            }
        ),
        flush=True,
    )
