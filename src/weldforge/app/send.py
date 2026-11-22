"""Getting the program out of the application and into the machine.

Three routes, because cells differ in how they are reached:

``folder``
    Write the files somewhere. A USB stick, a network share, the robot's own
    mapped drive. Always available, and the one that works when nothing else
    does.

``ftp``
    Upload straight to the controller. ABB, KUKA and FANUC controllers all run
    an FTP server for exactly this, and it is how most cells are loaded
    without walking a stick across the shop.

``cell``
    Push to the on-machine PC running the WeldForge cell software, over its
    REST API. This is the only route that carries the *whole plan* -- seams,
    procedures, positioner settings, the report -- rather than just a program
    file, so the cell can show the operator what it is about to do.

Every dispatch is recorded on the project: when, what, where, and whether it
worked. A program that reached a machine is a thing that needs a record.

Safety
------
Nothing here starts a weld. A controller receiving a file does not run it, and
the REST route uploads a plan rather than commanding execution. The gate that
matters is in the cell: the arc interlock ships off, and the execution action
refuses a plan the planner flagged unless the request is a dry run.

What this module does enforce is that a job the planner marked as having
problems cannot be sent without the caller saying so explicitly. A program
whose own report says the torch collides should not leave the desk by
accident.
"""

from __future__ import annotations

import ftplib
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .project import Project, SentRecord

logger = logging.getLogger(__name__)

__all__ = [
    "SendError",
    "SendTarget",
    "DispatchResult",
    "send_to_folder",
    "send_over_ftp",
    "send_to_cell",
    "dispatch",
]


class SendError(RuntimeError):
    """The dispatch failed, with a reason the operator can act on."""


@dataclass
class SendTarget:
    """Where a program is going."""

    method: str = "folder"  # folder | ftp | cell
    destination: str = ""  # path, host, or base URL
    #: FTP only.
    username: str = ""
    password: str = field(default="", repr=False)
    remote_dir: str = ""
    port: int = 0
    passive: bool = True
    timeout_s: float = 20.0

    def describe(self) -> str:
        if self.method == "folder":
            return self.destination or "(no folder chosen)"
        if self.method == "ftp":
            user = f"{self.username}@" if self.username else ""
            directory = f"/{self.remote_dir.strip('/')}" if self.remote_dir else ""
            return f"ftp://{user}{self.destination}{directory}"
        return self.destination or "(no cell address)"


@dataclass
class DispatchResult:
    ok: bool = True
    message: str = ""
    files: list[str] = field(default_factory=list)
    destination: str = ""

    def as_record(self, target_key: str, method: str) -> SentRecord:
        return SentRecord(
            when=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            target=target_key,
            destination=self.destination,
            method=method,
            files=list(self.files),
            ok=self.ok,
            message=self.message,
        )


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


def send_to_folder(files: dict[str, str], target: SendTarget) -> DispatchResult:
    """Write the program files into a folder."""
    if not target.destination:
        raise SendError("choose a folder first")
    folder = Path(target.destination)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SendError(f"cannot write to {folder}: {exc}") from exc

    written: list[str] = []
    for name, text in files.items():
        # Only the basename: a generated filename must never be able to write
        # outside the folder the operator chose.
        path = folder / Path(name).name
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise SendError(f"cannot write {path.name}: {exc}") from exc
        written.append(str(path))

    return DispatchResult(
        ok=True,
        message=f"{len(written)} file(s) written to {folder}",
        files=written,
        destination=str(folder),
    )


def send_over_ftp(files: dict[str, str], target: SendTarget) -> DispatchResult:
    """Upload the program files to a controller over FTP."""
    if not target.destination:
        raise SendError("enter the controller's address")

    port = target.port or 21
    written: list[str] = []
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(target.destination, port, timeout=target.timeout_s)
            ftp.login(target.username or "anonymous", target.password or "")
            ftp.set_pasv(target.passive)
            if target.remote_dir:
                for part in target.remote_dir.strip("/").split("/"):
                    if not part:
                        continue
                    try:
                        ftp.cwd(part)
                    except ftplib.error_perm:
                        # Make the directory rather than failing: a job folder
                        # per part is the normal way a cell is organised.
                        ftp.mkd(part)
                        ftp.cwd(part)
            for name, text in files.items():
                import io

                payload = io.BytesIO(text.encode("utf-8"))
                ftp.storbinary(f"STOR {Path(name).name}", payload)
                written.append(f"{target.describe()}/{Path(name).name}")
    except (ftplib.all_errors, OSError) as exc:
        raise SendError(
            f"FTP upload to {target.destination} failed: {exc}. Check the address, "
            "the credentials, and that the controller's FTP server is enabled."
        ) from exc

    return DispatchResult(
        ok=True,
        message=f"{len(written)} file(s) uploaded to {target.describe()}",
        files=written,
        destination=target.describe(),
    )


def send_to_cell(
    project: Project,
    files: dict[str, str],
    target: SendTarget,
    target_key: str,
) -> DispatchResult:
    """Push the plan and its program to the on-machine cell software.

    Uploads the project's CAD and plans it on the cell, so the machine holds
    the whole job -- seams, procedures, positioner settings, the report -- not
    just a program file. The operator at the machine then sees what it is
    about to do and can dry-run it there.
    """
    base = (target.destination or "").rstrip("/")
    if not base:
        raise SendError("enter the cell's address, for example http://192.168.1.50:8000")
    if project.part_bytes is None:
        raise SendError("the project has no CAD attached")

    def request(path: str, data: bytes | None, headers: dict[str, str], method: str):
        req = urllib.request.Request(base + path, data=data, method=method)
        for key, value in headers.items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=target.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise SendError(
                f"the cell rejected {method} {path}: HTTP {exc.code} "
                f"{exc.read().decode('utf-8', 'replace')[:200]}"
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise SendError(
                f"cannot reach the cell at {base}: {getattr(exc, 'reason', exc)}. "
                "Is the on-machine software running, and is the address right?"
            ) from exc
        return json.loads(body) if body else None

    # Check what is on the other end before sending anything.
    health = request("/health", None, {}, "GET")
    logger.info("cell at %s: %s", base, health)

    # Upload the CAD as multipart, the same way the CAD plugins do.
    import uuid

    boundary = f"----weldforge{uuid.uuid4().hex}"
    filename = project.part_filename or "part.step"
    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="name"\r\n\r\n'
            f"{project.display_name}\r\n".encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode(),
            project.part_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    part = request(
        "/parts",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        "POST",
    )

    plan_request = {
        "part_id": part["id"],
        "cell": project.cell_name,
        "name": project.display_name,
        "base_material": project.base_material,
        "leg_thickness_ratio": project.leg_thickness_ratio,
    }
    if project.wire_diameter_mm:
        plan_request["wire_diameter_mm"] = project.wire_diameter_mm
    started = request(
        "/jobs",
        json.dumps(plan_request).encode(),
        {"Content-Type": "application/json"},
        "POST",
    )
    job_id = started["job_id"]

    export = request(
        f"/jobs/{job_id}/export",
        json.dumps(
            {"target": target_key, "program_name": project.display_name[:8].upper() or "WELDJOB"}
        ).encode(),
        {"Content-Type": "application/json"},
        "POST",
    )

    names = list(export.get("files", files).keys())
    return DispatchResult(
        ok=True,
        message=(
            f"plan sent to the cell as job {job_id}; it generated "
            f"{len(names)} file(s). Dry-run it at the machine before welding."
        ),
        files=names,
        destination=base,
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def dispatch(
    project: Project,
    target_key: str,
    target: SendTarget,
    force: bool = False,
) -> DispatchResult:
    """Send the generated program, recording the dispatch on the project.

    Refuses a job whose own report found problems unless *force*. A program
    that says its torch collides should not leave the desk by accident; it
    should leave only because somebody decided it should.
    """
    files = project.programs.get(target_key)
    if not files:
        raise SendError(f"no {target_key} program has been generated yet. Generate it first.")

    report = project.last_report or {}
    if report and not report.get("ok", False) and not force:
        errors = report.get("errors") or []
        detail = f" First: {errors[0]}" if errors else ""
        raise SendError(
            "the planner reported problems with this job, so it will not be sent "
            f"by accident.{detail} Send it anyway only after a dry run."
        )

    if target.method == "folder":
        result = send_to_folder(files, target)
    elif target.method == "ftp":
        result = send_over_ftp(files, target)
    elif target.method == "cell":
        result = send_to_cell(project, files, target, target_key)
    else:
        raise SendError(f"unknown dispatch method {target.method!r}")

    project.sent.append(result.as_record(target_key, target.method))
    project.touch()
    logger.info("dispatched %s via %s: %s", target_key, target.method, result.message)
    return result
