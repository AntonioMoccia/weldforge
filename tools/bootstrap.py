"""One command: install, document, package, run.

Run it with nothing but a system Python:

.. code-block:: text

    weldforge.cmd              Windows   -- everything, then start the studio
    ./weldforge.sh             macOS / Linux

Stages, in order, each one skippable and each one runnable on its own:

============  =======================================================
``install``   create ``.venv`` and install WeldForge with every extra
``docs``      build the HTML documentation into ``docs/_build``
``exe``       build standalone executables into ``dist``
``run``       start WeldForge Studio
============  =======================================================

Design points worth stating.

**Standard library only.** This file runs *before* anything is installed, so
it may import nothing but the standard library. That is also why it is a
script under ``tools`` rather than a console entry point in the package.

**A stage that fails does not stop the others.** Packaging needs a PyInstaller
that supports the running Python, and that support lags new releases by
months. When it fails, the documentation and the application itself are still
worth having -- so every stage is reported and the summary at the end says
plainly what worked and what did not.

**Idempotent.** Installing is skipped when ``pyproject.toml`` and the Python
version have not changed since the last successful install, so the common case
-- running this to start the program -- takes a second rather than a minute.
``--force`` redoes it anyway.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
STAMP = VENV / ".weldforge-install.json"
DOCS_OUT = ROOT / "docs" / "_build"
DIST = ROOT / "dist"
BUILD = ROOT / "build" / "pyinstaller"

STAGES = ("install", "docs", "exe", "run")

# Every extra in pyproject, so one command leaves nothing to install later.
EXTRAS = ("cad", "api", "sim", "studio", "dev", "docs", "build")

# PyInstaller cannot see what these packages load at runtime -- OpenCASCADE's
# shared libraries, VTK's module tree, trimesh's optional backends -- so it
# has to be told to take all of them.
COLLECT_ALL = ("OCP", "vtkmodules", "trimesh", "weldforge")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text


def heading(text: str) -> None:
    print()
    print(_paint(f"=== {text} ", "1;36").ljust(72, "="))


def say(text: str) -> None:
    print(f"  {text}")


def warn(text: str) -> None:
    print(_paint(f"  ! {text}", "33"))


def fail(text: str) -> None:
    print(_paint(f"  x {text}", "31"))


# ---------------------------------------------------------------------------
# the virtual environment
# ---------------------------------------------------------------------------


def venv_python(root: Path = VENV) -> Path:
    """Path to the interpreter inside *root*."""
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def venv_script(name: str, root: Path = VENV) -> Path:
    folder = "Scripts" if sys.platform == "win32" else "bin"
    suffix = ".exe" if sys.platform == "win32" else ""
    return root / folder / f"{name}{suffix}"


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Run a command, echoing it first so a failure can be repeated by hand."""
    printable = " ".join(str(c) for c in cmd)
    say(_paint(f"$ {printable}", "2"))
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True, **kwargs)


def ensure_venv() -> Path:
    """Create ``.venv`` if it is missing, and return its interpreter."""
    python = venv_python()
    if python.exists():
        return python
    say(f"creating a virtual environment in {VENV}")
    # with_pip, because a venv without pip cannot install anything;
    # upgrade_deps, so the first install is not done by a year-old pip.
    venv.EnvBuilder(with_pip=True, upgrade_deps=True, clear=False).create(VENV)
    if not python.exists():  # pragma: no cover - a broken Python install
        raise RuntimeError(f"the virtual environment has no interpreter at {python}")
    return python


# ---------------------------------------------------------------------------
# stage: install
# ---------------------------------------------------------------------------


def _fingerprint() -> str:
    """Hash of everything that decides the installed set."""
    digest = hashlib.sha256()
    digest.update(platform.python_version().encode())
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        digest.update(pyproject.read_bytes())
    return digest.hexdigest()


def stage_install(force: bool) -> bool:
    heading("install")
    python = ensure_venv()
    current = _fingerprint()

    if not force and STAMP.exists():
        try:
            previous = json.loads(STAMP.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if previous.get("fingerprint") == current:
            say("already installed, and pyproject.toml has not changed -- skipping")
            say("(pass --force to reinstall)")
            _report(python)
            return True

    run([python, "-m", "pip", "install", "--upgrade", "pip", "wheel"])

    extras = ",".join(EXTRAS)
    ok = True
    try:
        run([python, "-m", "pip", "install", "-e", f".[{extras}]"])
    except subprocess.CalledProcessError:
        # One unavailable wheel must not cost us the whole installation: fall
        # back to the core, then add each extra on its own and say which ones
        # this Python cannot have.
        warn(f"installing .[{extras}] as one failed; retrying extra by extra")
        run([python, "-m", "pip", "install", "-e", "."])
        for extra in EXTRAS:
            try:
                run([python, "-m", "pip", "install", "-e", f".[{extra}]"])
            except subprocess.CalledProcessError:
                fail(f"extra {extra!r} is not installable on Python {platform.python_version()}")
                ok = False

    STAMP.write_text(
        json.dumps({"fingerprint": current, "when": time.strftime("%Y-%m-%d %H:%M:%S")}),
        encoding="utf-8",
    )
    _report(python)
    return ok


def _report(python: Path) -> None:
    """Print what the installation can actually do.

    pip exiting zero does not mean the CAD kernel arrived, and the kernel is
    the difference between reading a STEP file and not.
    """
    print()
    subprocess.run([str(python), "-m", "weldforge.cli", "doctor"], cwd=ROOT, check=False)


# ---------------------------------------------------------------------------
# stage: docs
# ---------------------------------------------------------------------------

MARKDOWN_PAGES = (
    ("README.md", "Overview"),
    ("QUICKSTART.md", "Quick start"),
    ("docs/manuale-operativo.md", "Manuale operativo"),
    ("docs/architecture.md", "Architecture"),
    ("docs/commercial.md", "Commercial"),
)


def stage_docs() -> bool:
    heading("docs")
    python = venv_python()
    if not python.exists():
        fail("no virtual environment yet -- run the install stage first")
        return False

    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    ok = True

    # -- API reference, straight from the docstrings ------------------------
    api_out = DOCS_OUT / "api"
    try:
        run([python, "-m", "pdoc", "--output-directory", api_out, "weldforge"])
        say(f"API reference -> {api_out / 'index.html'}")
    except subprocess.CalledProcessError:
        fail("pdoc failed; the API reference was not built")
        ok = False

    # -- CLI reference, generated from argparse so it cannot drift ----------
    cli_page = DOCS_OUT / "cli.html"
    try:
        cli_page.write_text(_render_cli_reference(python), encoding="utf-8")
        say(f"CLI reference -> {cli_page}")
    except Exception as exc:
        fail(f"could not build the CLI reference: {exc}")
        ok = False

    # -- the hand-written pages --------------------------------------------
    pages: list[tuple[str, str]] = []
    for relative, title in MARKDOWN_PAGES:
        source = ROOT / relative
        if not source.exists():
            continue
        target = DOCS_OUT / (Path(relative).stem.lower() + ".html")
        try:
            target.write_text(
                _render_markdown(python, source.read_text(encoding="utf-8"), title),
                encoding="utf-8",
            )
            pages.append((target.name, title))
            say(f"{relative} -> {target.name}")
        except Exception as exc:
            warn(f"{relative}: {exc}")

    index = DOCS_OUT / "index.html"
    index.write_text(_render_index(pages), encoding="utf-8")
    say(f"index -> {index}")
    return ok


def _render_cli_reference(python: Path) -> str:
    """``--help`` for the CLI and every subcommand, as one page."""
    code = """
import contextlib, io, json
from weldforge.cli import build_parser

parser = build_parser()
out = {}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    parser.print_help()
out["weldforge"] = buf.getvalue()
for action in parser._actions:
    for name, sub in (getattr(action, "choices", None) or {}).items():
        if not hasattr(sub, "print_help"):
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sub.print_help()
        out["weldforge " + name] = buf.getvalue()
print(json.dumps(out))
"""
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        # Explicit UTF-8, because the default is the console code page, and on
        # Windows that is cp1252: every en dash in a help string would fail.
        encoding="utf-8",
        errors="replace",
    )
    sections = json.loads(result.stdout)
    body = [
        f"<h2><code>{_escape(name)}</code></h2>\n<pre>{_escape(text)}</pre>"
        for name, text in sections.items()
    ]
    return _page("Command-line reference", "\n".join(body))


def _render_markdown(python: Path, text: str, title: str) -> str:
    """Markdown to HTML, falling back to a preformatted block.

    No Markdown engine is a dependency of WeldForge itself, and adding one for
    the documentation build alone is not worth it: unrendered but readable
    beats a failed build.
    """
    code = (
        "import sys, markdown\n"
        "sys.stdout.write(markdown.markdown(sys.stdin.read(),"
        " extensions=['fenced_code', 'tables', 'toc']))\n"
    )
    probe = subprocess.run(
        [str(python), "-c", code],
        input=text,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if probe.returncode == 0 and probe.stdout.strip():
        return _page(title, probe.stdout)
    return _page(title, f"<pre>{_escape(text)}</pre>")


def _render_index(pages: list[tuple[str, str]]) -> str:
    items = [
        '<li><a href="api/index.html">API reference</a>'
        " &mdash; every module, from the docstrings</li>",
        '<li><a href="cli.html">Command-line reference</a>'
        " &mdash; every command and flag</li>",
    ]
    items += [f'<li><a href="{href}">{_escape(title)}</a></li>' for href, title in pages]
    newline = "\n"
    body = (
        f"<ul>{newline}{newline.join(items)}{newline}</ul>"
        f'{newline}<p class="meta">Built {time.strftime("%Y-%m-%d %H:%M")}</p>'
    )
    return _page("WeldForge documentation", body)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(title)} - WeldForge</title>
<style>
 :root {{ color-scheme: light dark; --fg:#1b1f24; --bg:#fbfbfa; --muted:#5d6670;
          --line:#d8dde3; --code:#f2f4f7; --link:#0b5ed7; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --fg:#e6e9ee; --bg:#14171c; --muted:#9aa6b5;
            --line:#2b313a; --code:#1c222b; --link:#7fb0ff; }}
 }}
 body {{ margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.6 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif; }}
 main {{ max-width:54rem; margin:0 auto; padding:2.5rem 1rem 4rem; }}
 h1 {{ font-size:1.9rem; margin:0 0 1.5rem; }}
 h2 {{ margin-top:2.2rem; padding-bottom:.3rem; border-bottom:1px solid var(--line); }}
 a {{ color:var(--link); }}
 pre {{ background:var(--code); padding:1rem; border-radius:6px; overflow-x:auto; }}
 code {{ font:14px/1.5 ui-monospace,"Cascadia Code",Consolas,monospace; }}
 table {{ border-collapse:collapse; display:block; overflow-x:auto; }}
 th,td {{ border:1px solid var(--line); padding:.4rem .6rem; text-align:left; }}
 ul {{ padding-left:1.2rem; }}
 li {{ margin:.4rem 0; }}
 nav {{ margin-bottom:2rem; font-size:.9rem; }}
 .meta {{ color:var(--muted); font-size:.9rem; margin-top:3rem; }}
</style></head>
<body><main>
<nav><a href="index.html">WeldForge documentation</a></nav>
<h1>{_escape(title)}</h1>
{body}
</main></body></html>
"""


# ---------------------------------------------------------------------------
# stage: exe
# ---------------------------------------------------------------------------


def _clear_output(folder: Path) -> None:
    """Remove a previous bundle, working around Windows file locks.

    PyInstaller's own ``--clean`` gives up with "Accesso negato" when anything
    still holds one of the hundreds of DLLs in the last build -- an indexer, a
    virus scanner, or an executable that was run to check it. Renaming the
    folder out of the way succeeds where deleting it does not, because Windows
    allows a rename of a directory whose files are open.
    """
    import shutil
    import time as _time

    if not folder.exists():
        return
    try:
        shutil.rmtree(folder)
        return
    except OSError:
        pass
    stale = folder.with_name(f"{folder.name}.old-{int(_time.time())}")
    try:
        folder.rename(stale)
        warn(f"{folder.name} was locked; moved the old build to {stale.name}")
    except OSError as exc:
        warn(f"could not clear {folder}: {exc}")
        return
    # Best effort: the lock usually goes before the next build finishes.
    with contextlib.suppress(OSError):
        shutil.rmtree(stale)


def stage_exe() -> bool:
    heading("exe")
    python = venv_python()
    if not python.exists():
        fail("no virtual environment yet -- run the install stage first")
        return False

    probe = subprocess.run(
        [str(python), "-c", "import PyInstaller; print(PyInstaller.__version__)"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if probe.returncode != 0:
        fail("PyInstaller is not installed in the virtual environment")
        say(f"it may not support Python {platform.python_version()} yet")
        say("the studio still runs from source: weldforge.cmd run")
        return False
    say(f"PyInstaller {probe.stdout.strip()}")

    collect: list[str] = []
    for package in COLLECT_ALL:
        collect += ["--collect-all", package]

    targets = (
        # (name, module with main(), windowed)
        ("WeldForgeStudio", "weldforge.app", True),
        ("weldforge", "weldforge.cli", False),
    )
    ok = True
    for name, module, windowed in targets:
        BUILD.mkdir(parents=True, exist_ok=True)
        _clear_output(DIST / name)
        entry = BUILD / f"{name}_entry.py"
        entry.write_text(
            "import sys\n"
            f"from {module} import main\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    sys.exit(main())\n",
            encoding="utf-8",
        )
        try:
            run(
                [
                    python,
                    "-m",
                    "PyInstaller",
                    "--noconfirm",
                    "--clean",
                    "--name",
                    name,
                    "--distpath",
                    DIST,
                    "--workpath",
                    BUILD,
                    "--specpath",
                    BUILD,
                    # One folder, not one file: OpenCASCADE and VTK together
                    # are hundreds of megabytes, and a one-file build unpacks
                    # all of it to a temporary directory on every launch.
                    "--onedir",
                    "--windowed" if windowed else "--console",
                    # The cell, robot, machine and procedure YAML that is read
                    # at runtime. Without it a packaged build has no data.
                    "--add-data",
                    f"{ROOT / 'data'}{os.pathsep}data",
                    *collect,
                    entry,
                ]
            )
            say(f"{name} -> {DIST / name}")
        except subprocess.CalledProcessError:
            fail(f"packaging {name} failed")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# stage: run
# ---------------------------------------------------------------------------


def stage_run(extra: list[str]) -> bool:
    heading("run")
    python = venv_python()
    if not python.exists():
        fail("no virtual environment yet -- run the install stage first")
        return False
    say("starting WeldForge Studio -- close the window to come back here")
    result = subprocess.run([str(python), "-m", "weldforge.app", *extra], cwd=ROOT, check=False)
    if result.returncode != 0:
        fail(f"the studio exited with code {result.returncode}")
        say(f"the reason is in the log: {_log_hint(python)}")
        return False
    return True


def _log_hint(python: Path) -> str:
    probe = subprocess.run(
        [str(python), "-c", "from weldforge.diagnostics import log_file; print(log_file())"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return probe.stdout.strip() or "run: weldforge doctor <file.step>"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    launcher = "weldforge.cmd" if sys.platform == "win32" else "./weldforge.sh"
    return _parser(launcher)


def _parser(launcher: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=launcher,
        description=(
            "Set up and start WeldForge. With no arguments it installs whatever is "
            "missing, builds the documentation and the executables, then starts the studio."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            f"  {launcher}                  everything, then start the studio\n"
            f"  {launcher} run              just start it\n"
            f"  {launcher} install docs     install and build the docs, do not start\n"
            f"  {launcher} --force          reinstall from scratch\n"
            f"  {launcher} --skip exe       everything except packaging\n"
            f"  {launcher} run -- part.step open a file on start-up\n"
        ),
    )
    parser.add_argument(
        "stages",
        nargs="*",
        choices=[*STAGES, "all"],
        help=f"stages to run; default is all of: {', '.join(STAGES)}",
    )
    parser.add_argument("--force", action="store_true", help="reinstall even if nothing changed")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=STAGES,
        help="a stage to leave out; repeatable",
    )
    parser.add_argument(
        "--open-docs", action="store_true", help="open the documentation in a browser once built"
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="after --, arguments handed to the studio",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 10):
        fail(
            f"WeldForge needs Python 3.10 or newer; this is "
            f"{platform.python_version()} at {sys.executable}"
        )
        return 1

    # Line buffering, so our own output stays interleaved with the output of
    # the commands we launch even when all of it is redirected to a file --
    # which is how someone sends us a failed run.
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    wanted = args.stages or ["all"]
    chosen = list(STAGES) if "all" in wanted else [s for s in STAGES if s in wanted]
    chosen = [s for s in chosen if s not in args.skip]
    if not chosen:
        print("nothing to do")
        return 0

    print(_paint("WeldForge setup", "1"))
    say(f"project       : {ROOT}")
    say(f"system python : {platform.python_version()}  ({sys.executable})")
    say(f"stages        : {', '.join(chosen)}")

    results: dict[str, bool] = {}
    passthrough = [a for a in args.args if a != "--"]

    for stage in chosen:
        try:
            if stage == "install":
                results[stage] = stage_install(args.force)
            elif stage == "docs":
                results[stage] = stage_docs()
                if results[stage] and args.open_docs:
                    import webbrowser

                    webbrowser.open((DOCS_OUT / "index.html").as_uri())
            elif stage == "exe":
                results[stage] = stage_exe()
            elif stage == "run":
                results[stage] = stage_run(passthrough)
        except KeyboardInterrupt:
            print()
            warn("interrupted")
            results[stage] = False
            break
        except subprocess.CalledProcessError as exc:
            fail(f"{stage}: a command failed with code {exc.returncode}")
            results[stage] = False
        except Exception as exc:
            fail(f"{stage}: {type(exc).__name__}: {exc}")
            results[stage] = False

    heading("summary")
    for stage in chosen:
        state = results.get(stage)
        mark = "ok     " if state else "FAILED "
        say(f"{_paint(mark, '32' if state else '31')} {stage}")
    if results.get("docs"):
        say(f"documentation : {DOCS_OUT / 'index.html'}")
    if results.get("exe"):
        say(f"executables   : {DIST}")
    say(f"why did it fail : {venv_script('weldforge')} doctor <file.step>")

    return 0 if all(results.get(s) for s in chosen) else 1


if __name__ == "__main__":
    sys.exit(main())
