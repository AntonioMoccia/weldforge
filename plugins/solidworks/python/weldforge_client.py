"""HTTP client for the WeldForge API.

Written against ``urllib`` from the standard library and nothing else. That is
deliberate: FreeCAD ships its own Python, and what is installed in it varies by
platform and by how the user got FreeCAD -- an AppImage, a Snap, Homebrew, the
Windows installer. ``requests`` is frequently absent, and asking a shop-floor
user to pip-install into FreeCAD's interpreter is how a plugin stops being
used.

Multipart upload is assembled by hand for the same reason.

The same file is shared by the SolidWorks Python exporter, so there is one
client to keep working rather than two.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

__all__ = ["WeldForgeError", "WeldForgeClient"]

DEFAULT_URL = os.environ.get("WELDFORGE_URL", "http://127.0.0.1:8000")


class WeldForgeError(RuntimeError):
    """An API call failed. Carries the server's own message where there is one."""

    def __init__(self, message: str, status: int | None = None, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class WeldForgeClient:
    """Thin wrapper over the REST API."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- plumbing ------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            query = urllib.parse.urlencode(
                {k: ("true" if v is True else "false" if v is False else v)
                 for k, v in params.items() if v is not None}
            )
            url = f"{url}?{query}"

        request = urllib.request.Request(url, data=body, method=method)
        for key, value in (headers or {}).items():
            request.add_header(key, value)

        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            detail = raw
            try:
                parsed = json.loads(raw)
                detail = parsed.get("detail", parsed)
            except Exception:
                pass
            raise WeldForgeError(
                f"{method} {path} failed with HTTP {exc.code}: {detail}",
                status=exc.code,
                detail=detail,
            ) from exc
        except urllib.error.URLError as exc:
            raise WeldForgeError(
                f"cannot reach the WeldForge server at {self.base_url} ({exc.reason}). "
                "Start it with: weldforge serve"
            ) from exc

        if not payload:
            return None
        text = payload.decode("utf-8", "replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params=params)

    def _post_json(self, path: str, body: dict[str, Any], timeout: float | None = None) -> Any:
        return self._request(
            "POST",
            path,
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

    # -- service -------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def reachable(self) -> bool:
        try:
            self.health()
            return True
        except WeldForgeError:
            return False

    def cells(self) -> list[dict[str, Any]]:
        return self._get("/cells")["cells"]

    def materials(self) -> list[dict[str, Any]]:
        return self._get("/materials")["materials"]

    def targets(self) -> list[dict[str, Any]]:
        return self._get("/targets")["targets"]

    def machines(self) -> list[dict[str, Any]]:
        return self._get("/machines")["machines"]

    # -- parts ---------------------------------------------------------------

    def upload_part(self, path: str, name: str = "") -> dict[str, Any]:
        """Upload a CAD file as ``multipart/form-data``."""
        filename = os.path.basename(path)
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(path, "rb") as handle:
            data = handle.read()

        boundary = f"----weldforge{uuid.uuid4().hex}"
        parts: list[bytes] = []

        def add_field(field: str, value: str) -> None:
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{field}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode()
            )

        if name:
            add_field("name", name)
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
        )
        parts.append(data)
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        body = b"".join(parts)

        return self._request(
            "POST",
            "/parts",
            body=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            timeout=max(self.timeout, 300.0),
        )

    def seams(self, part_id: str, with_geometry: bool = True, **params: Any) -> dict[str, Any]:
        return self._get(
            f"/parts/{part_id}/seams", with_geometry=with_geometry, **params
        )

    # -- jobs ----------------------------------------------------------------

    def plan(self, part_id: str, **options: Any) -> str:
        """Start a plan; returns the job id."""
        body = {"part_id": part_id, **options}
        return self._post_json("/jobs", body)["job_id"]

    def job(self, job_id: str, with_points: bool = False) -> dict[str, Any]:
        return self._get(f"/jobs/{job_id}", with_points=with_points)

    def wait_for_job(
        self,
        job_id: str,
        timeout: float = 900.0,
        poll_interval: float = 1.0,
        on_progress=None,
    ) -> dict[str, Any]:
        """Poll until the job finishes.

        Polling rather than a blocking call because planning a large weldment
        takes minutes and a CAD plugin must keep its UI responsive. The
        callback is where a progress dialog gets updated.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.job(job_id)
            if on_progress is not None:
                on_progress(status)
            if status.get("status") in ("done", "failed"):
                return status
            time.sleep(poll_interval)
        raise WeldForgeError(f"job {job_id} did not finish within {timeout:.0f} s")

    def export(self, job_id: str, target: str, program_name: str = "WELDJOB") -> dict[str, Any]:
        return self._post_json(
            f"/jobs/{job_id}/export",
            {"target": target, "program_name": program_name},
        )

    def setpoints(self, job_id: str, machine: str = "generic_analog") -> dict[str, Any]:
        return self._get(f"/jobs/{job_id}/setpoints", machine=machine)
