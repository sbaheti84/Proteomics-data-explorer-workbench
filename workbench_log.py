"""
Log file for the desktop (standalone) builds.

PyInstaller runs this file as a *runtime hook* before anything else starts, so
start-up crashes, import errors, Streamlit server errors and app errors all end
up in a log file instead of disappearing (a windowed app has no console).

Where the log goes:
  Windows: %LOCALAPPDATA%\\ProteomicsWorkbench\\logs\\
  macOS:   ~/Library/Logs/ProteomicsWorkbench/
  other:   ~/.proteomics_workbench/logs/
Each start writes workbench_YYYYMMDD_HHMMSS.log; the 10 newest are kept.
The path is stored in the WORKBENCH_LOG_FILE environment variable, and the app
shows it under Help -> Log file.

Only the standard library is used here on purpose.
"""
import os
import sys


def _log_dir() -> str:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "ProteomicsWorkbench", "logs")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Logs/ProteomicsWorkbench")
    return os.path.expanduser("~/.proteomics_workbench/logs")


def _usable(s) -> bool:
    try:
        return s is not None and not s.closed
    except Exception:
        return False


class _Tee:
    """Writes to the console and the log file. Other attributes come from the console."""

    def __init__(self, console, logfile):
        self._console = console
        self._logfile = logfile

    def write(self, text):
        for s in (self._logfile, self._console):
            try:
                s.write(text)
            except Exception:
                pass
        return len(text)

    def flush(self):
        for s in (self._logfile, self._console):
            try:
                s.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._console, name)


def setup() -> str:
    """Start logging to a file. Safe to call more than once. Returns the log path."""
    if os.environ.get("WORKBENCH_LOG_FILE") and getattr(sys, "_workbench_log_ready", False):
        return os.environ["WORKBENCH_LOG_FILE"]
    import datetime
    import faulthandler
    import glob
    import logging
    import platform
    import threading
    import traceback

    try:
        d = _log_dir()
        os.makedirs(d, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(d, f"workbench_{stamp}_{os.getpid()}.log")
        # Keep the 10 newest logs.
        old = sorted(glob.glob(os.path.join(d, "workbench_*.log")))
        for f in old[:-9]:
            try:
                os.remove(f)
            except OSError:
                pass
        stream = open(path, "a", encoding="utf-8", buffering=1)
    except Exception:
        return ""

    # A windowed app has no console: sys.stdout/sys.stderr are None, and any
    # print() or warning would crash silently. Send both to the log file.
    # When there IS a console (the Windows build), write to both, so errors are
    # still visible on screen and not only hidden in the log file.
    sys.stdout = _Tee(sys.stdout, stream) if _usable(sys.stdout) else stream
    sys.stderr = _Tee(sys.stderr, stream) if _usable(sys.stderr) else stream
    try:
        faulthandler.enable(file=stream)  # hard crashes (segfaults) too
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(stream)],
        force=True,
    )
    # Very chatty libraries (font subsetting during PDF export, image plugins).
    for noisy in ("fontTools", "matplotlib", "PIL", "urllib3", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("workbench")

    console_err = sys.__stderr__ if _usable(sys.__stderr__) else None

    def _excepthook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.critical("Uncaught error:\n%s", text)
        if console_err is not None:  # also show it on screen
            try:
                console_err.write(text + f"\nLog file: {path}\n")
                console_err.flush()
            except Exception:
                pass

    def _thread_hook(args):
        log.critical(
            "Uncaught error in thread %s:\n%s",
            getattr(args.thread, "name", "?"),
            "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
        )

    sys.excepthook = _excepthook
    threading.excepthook = _thread_hook

    os.environ["WORKBENCH_LOG_FILE"] = path
    sys._workbench_log_ready = True
    log.info("Proteomics Workbench starting")
    log.info("Python %s | %s | frozen=%s", platform.python_version(), platform.platform(),
             bool(getattr(sys, "frozen", False)))
    log.info("Executable: %s", sys.executable)
    log.info("Log file: %s", path)
    return path


def log_dir() -> str:
    return _log_dir()


# Runs automatically when PyInstaller loads this file as a runtime hook.
if getattr(sys, "frozen", False):
    setup()