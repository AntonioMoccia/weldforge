"""REST API and command-line interface tests.

The API is the contract the CAD plugins code against, so these tests pin the
shapes of its responses as much as its behaviour: a field renamed here breaks
a FreeCAD workbench on someone's workstation.
"""

from __future__ import annotations

import json

import pytest
from conftest import requires_cad

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from weldforge.api import create_app  # noqa: E402
from weldforge.api.store import Store  # noqa: E402
from weldforge.cli import main  # noqa: E402


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(Store(tmp_path / "store")))


def upload(client: TestClient, path) -> dict:
    with open(path, "rb") as handle:
        response = client.post(
            "/parts",
            files={"file": (path.name, handle, "application/octet-stream")},
        )
    assert response.status_code == 200, response.text
    return response.json()


def plan(client: TestClient, part_id: str, **overrides) -> dict:
    """Start a plan and wait for it: TestClient runs background tasks inline."""
    body = {
        "part_id": part_id,
        "cell": "bench_cell",
        "placement_grid_steps": 3,
        "placement_yaw_steps": 4,
        "placement_orientations": 3,
        **overrides,
    }
    response = client.post("/jobs", json=body)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    return status.json()


# ---------------------------------------------------------------------------
# service and reference data
# ---------------------------------------------------------------------------


def test_health_reports_what_the_server_can_do(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert "brep_supported" in payload
    assert ".step" in payload["formats"]


@pytest.mark.parametrize(
    "endpoint,key",
    [
        ("/cells", "cells"),
        ("/robots", "robots"),
        ("/machines", "machines"),
        ("/procedures", "procedures"),
        ("/materials", "materials"),
        ("/targets", "targets"),
    ],
)
def test_reference_endpoints_are_populated(client, endpoint, key):
    """Plugins build their dialogs from these; empty lists mean a broken UI."""
    payload = client.get(endpoint).json()
    assert payload[key], f"{endpoint} returned nothing"


def test_cells_carry_what_a_dialog_needs(client):
    cells = client.get("/cells").json()["cells"]
    bench = next(c for c in cells if c["key"] == "bench_cell")
    assert bench["robot"]
    assert bench["table_z_mm"] > 0
    assert bench["clamps"] > 0


def test_targets_declare_their_units(client):
    targets = {t["key"]: t for t in client.get("/targets").json()["targets"]}
    assert targets["urscript"]["units"] == "m"
    assert targets["rapid"]["units"] == "mm"


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "WeldForge"
    assert "/jobs" in schema["paths"]


# ---------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------


@requires_cad
def test_upload_returns_measured_geometry(client, tee_assembly_step):
    payload = upload(client, tee_assembly_step)
    assert payload["n_solids"] == 2
    assert payload["n_faces"] == 12
    assert payload["geometry_kind"] == "brep"
    assert sorted(payload["thicknesses_mm"]) == [6.0, 8.0]
    assert len(payload["solids"]) == 2
    assert payload["mass_kg"] > 0


def test_upload_rejects_an_unknown_format(client, tmp_path):
    bad = tmp_path / "part.dwg"
    bad.write_bytes(b"not cad")
    with open(bad, "rb") as handle:
        response = client.post("/parts", files={"file": ("part.dwg", handle)})
    assert response.status_code == 415
    assert "unsupported" in response.text


def test_upload_rejects_an_empty_file(client, tmp_path):
    empty = tmp_path / "empty.step"
    empty.write_bytes(b"")
    with open(empty, "rb") as handle:
        response = client.post("/parts", files={"file": ("empty.step", handle)})
    assert response.status_code == 400


def test_upload_of_a_corrupt_step_is_reported_not_crashed(client, tmp_path):
    bad = tmp_path / "broken.step"
    bad.write_text("ISO-10303-21; this is not really a STEP file")
    with open(bad, "rb") as handle:
        response = client.post("/parts", files={"file": ("broken.step", handle)})
    assert response.status_code == 422
    assert "could not read" in response.text


def test_unknown_part_is_a_404_with_advice(client):
    response = client.get("/parts/part_nope")
    assert response.status_code == 404
    assert "POST /parts" in response.text


@requires_cad
def test_seam_endpoint_returns_classified_seams(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    payload = client.get(f"/parts/{part['id']}/seams").json()
    assert payload["n_seams"] == 2
    assert payload["total_length_mm"] == pytest.approx(400.0, abs=1.0)
    seam = payload["seams"][0]
    assert seam["joint_type"] == "tee"
    assert seam["groove_deg"] == pytest.approx(90.0, abs=0.5)
    assert seam["position"] == "PB"
    assert seam["points_mm"] is None  # geometry omitted unless asked for


@requires_cad
def test_seam_geometry_is_available_on_request(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    payload = client.get(f"/parts/{part['id']}/seams", params={"with_geometry": True}).json()
    seam = payload["seams"][0]
    assert seam["points_mm"] and len(seam["points_mm"][0]) == 3
    assert seam["bisectors"] and len(seam["bisectors"]) == len(seam["points_mm"])


@requires_cad
def test_seam_detection_is_cached(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    first = client.get(f"/parts/{part['id']}/seams").json()
    second = client.get(f"/parts/{part['id']}/seams").json()
    assert [s["id"] for s in first["seams"]] == [s["id"] for s in second["seams"]]


@requires_cad
def test_part_can_be_deleted(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    assert client.delete(f"/parts/{part['id']}").status_code == 200
    assert client.get(f"/parts/{part['id']}").status_code == 404


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


@requires_cad
def test_planning_a_job_end_to_end(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    status = plan(client, part["id"])
    assert status["status"] == "done", status.get("error")

    result = status["result"]
    assert len(result["seams"]) == 2
    assert len(result["procedures"]) == 2
    assert result["placement"]["feasible"]
    assert result["paths"]
    assert result["report"]["ok"]
    assert result["report"]["total_weld_length_mm"] == pytest.approx(400.0, abs=1.0)


@requires_cad
def test_job_options_are_honoured(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    status = plan(
        client,
        part["id"],
        base_material="316L",
        leg_thickness_ratio=1.0,
        do_paths=False,
        do_simulate=False,
    )
    result = status["result"]
    procedure = result["procedures"][0]
    assert procedure["base_material"] == "316L"
    assert procedure["passes"][0]["gas"].startswith("M12")
    # A full-strength fillet on a 6 mm member is a 6 mm leg, not 4.
    assert procedure["target_leg_mm"] == pytest.approx(6.0)
    assert result["paths"] == []


@requires_cad
def test_job_on_an_unknown_part_is_a_404(client):
    response = client.post("/jobs", json={"part_id": "part_nope"})
    assert response.status_code == 404


@requires_cad
def test_job_with_an_unknown_cell_is_a_404(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    response = client.post("/jobs", json={"part_id": part["id"], "cell": "mars_base"})
    assert response.status_code == 404


@requires_cad
def test_report_endpoint_gives_just_the_verdict(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    status = plan(client, part["id"])
    report = client.get(f"/jobs/{status['id']}/report").json()
    assert report["ok"] is True
    assert report["cycle_time_s"] > 0
    assert 0.0 < report["duty_cycle"] <= 1.0


@requires_cad
def test_setpoints_endpoint_translates_for_a_named_machine(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    status = plan(client, part["id"])
    payload = client.get(
        f"/jobs/{status['id']}/setpoints", params={"machine": "fronius_tps"}
    ).json()
    assert payload["seams"]
    entry = payload["seams"][0]
    assert entry["synergic"] is True
    setpoints = entry["setpoints"][0]
    assert setpoints["job_number"] is not None
    assert setpoints["analog_words"]


@requires_cad
def test_setpoints_for_an_unknown_machine_is_a_404(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    status = plan(client, part["id"])
    response = client.get(f"/jobs/{status['id']}/setpoints", params={"machine": "sparkomatic"})
    assert response.status_code == 404


@requires_cad
@pytest.mark.parametrize("target", ["rapid", "krl", "urscript", "fanuc", "ros2"])
def test_export_produces_a_downloadable_program(client, tee_assembly_step, target):
    part = upload(client, tee_assembly_step)
    status = plan(client, part["id"])
    response = client.post(
        f"/jobs/{status['id']}/export",
        json={"target": target, "program_name": "TEE"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["moves"] > 0
    assert payload["files"]

    filename = next(iter(payload["files"]))
    download = client.get(f"/jobs/{status['id']}/export/{filename}")
    assert download.status_code == 200
    assert "WeldForge" in download.text


@requires_cad
def test_export_before_the_job_finishes_is_a_409(tmp_path, tee_assembly_step):
    """Polling too early must get a clear 409, not an empty program."""
    store = Store(tmp_path / "store")
    local = TestClient(create_app(store))
    part = upload(local, tee_assembly_step)
    record = local.post(
        "/jobs", json={"part_id": part["id"], "do_placement": False, "do_paths": False}
    ).json()
    # TestClient runs background tasks inline, so push the record back to
    # "running" to reproduce a client that polled before the plan was done.
    store.update_record(record["job_id"], status="running", job=None)
    response = local.post(f"/jobs/{record['job_id']}/export", json={"target": "rapid"})
    assert response.status_code == 409
    assert "not finished" in response.text

    report = local.get(f"/jobs/{record['job_id']}/report")
    assert report.status_code == 409


@requires_cad
def test_export_to_an_unknown_target_is_a_404(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    status = plan(client, part["id"])
    response = client.post(f"/jobs/{status['id']}/export", json={"target": "skynet"})
    assert response.status_code == 404


@requires_cad
def test_downloading_an_ungenerated_file_is_a_404(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    status = plan(client, part["id"])
    assert client.get(f"/jobs/{status['id']}/export/nope.mod").status_code == 404


@requires_cad
def test_jobs_can_be_listed_and_deleted(client, tee_assembly_step):
    part = upload(client, tee_assembly_step)
    status = plan(client, part["id"])
    listing = client.get("/jobs").json()["jobs"]
    assert any(j["id"] == status["id"] for j in listing)
    assert client.delete(f"/jobs/{status['id']}").status_code == 200
    assert client.get(f"/jobs/{status['id']}").status_code == 404


def test_deleting_an_unknown_job_is_a_404(client):
    assert client.delete("/jobs/job_nope").status_code == 404


@requires_cad
def test_a_failing_plan_is_recorded_not_lost(client, tee_assembly_step, monkeypatch):
    """A background task that raises must leave a readable failure, not a hang."""
    part = upload(client, tee_assembly_step)

    import weldforge.api.app as app_module

    def explode(*args, **kwargs):
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(app_module, "plan_from_part", explode)
    status = plan(client, part["id"])
    assert status["status"] == "failed"
    assert "deliberate failure" in status["error"]


# ---------------------------------------------------------------------------
# upload path safety
# ---------------------------------------------------------------------------


def test_upload_filename_cannot_escape_the_workspace(tmp_path):
    """A client sending ``../../etc/passwd`` must not write outside the store."""
    store = Store(tmp_path / "store")
    path = store.save_upload("job1", "../../escaped.step", b"data")
    assert path.parent == store.workspace("job1")
    assert path.name == "escaped.step"
    assert not (tmp_path / "escaped.step").exists()


def test_store_evicts_the_oldest_entries(tmp_path):
    store = Store(tmp_path / "store", capacity=3)
    for i in range(5):
        store.create_record(f"job{i}")
    held = [r.id for r in store.list_records()]
    assert len(held) == 3
    assert "job0" not in held and "job4" in held


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "WeldForge" in out
    assert "export targets" in out


def test_cli_info_as_json(capsys):
    assert main(["--json", "info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["export_targets"]
    assert "rapid" in payload["export_targets"]


@requires_cad
def test_cli_seams(capsys, tee_assembly_step):
    assert main(["seams", str(tee_assembly_step)]) == 0
    out = capsys.readouterr().out
    assert "2 seams" in out
    assert "tee" in out


@requires_cad
def test_cli_seams_as_json(capsys, tee_assembly_step):
    assert main(["--json", "seams", str(tee_assembly_step)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["seams"]) == 2
    assert payload["seams"][0]["joint_type"] == "tee"


@requires_cad
def test_cli_wps(capsys, tee_assembly_step):
    assert main(["wps", str(tee_assembly_step), "--material", "S355"]) == 0
    out = capsys.readouterr().out
    assert "pass" in out and "kJ/mm" in out
    assert "arc time" in out


@requires_cad
def test_cli_plan_writes_programs(capsys, tee_assembly_step, tmp_path):
    code = main(
        [
            "plan",
            str(tee_assembly_step),
            "--cell",
            "bench_cell",
            "--grid",
            "3",
            "--yaw",
            "4",
            "-t",
            "rapid",
            "-t",
            "ros2",
            "-o",
            str(tmp_path),
            "--program-name",
            "CLITEST",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "VERDICT         : OK" in out
    assert (tmp_path / "CLITEST.mod").exists()
    assert (tmp_path / "CLITEST_trajectories.yaml").exists()
    assert "MODULE CLITEST" in (tmp_path / "CLITEST.mod").read_text()


@requires_cad
def test_cli_plan_exit_code_reflects_the_verdict(capsys, tee_assembly_step):
    """0 means safe to run, 2 means planned but do not run it.

    A shop script can therefore gate on the exit status instead of grepping
    the output.
    """
    code = main(["plan", str(tee_assembly_step), "--grid", "3", "--yaw", "4"])
    out = capsys.readouterr().out
    assert code in (0, 2)
    assert ("VERDICT         : OK" in out) == (code == 0)


def test_cli_missing_file_is_a_clean_error(capsys):
    assert main(["seams", "/nonexistent/part.step"]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["frobnicate"])
