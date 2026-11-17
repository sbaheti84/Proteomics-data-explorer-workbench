"""
Desktop entry point for the standalone (PyInstaller) build.

This replaces the wrapper that `streamlit-desktop-app build` used to generate.
That wrapper had three problems:
  1. It gave the Streamlit server only 10 seconds to start. The packaged app is
     large, so a first start (antivirus scanning every file) or a start from a
     network drive (Q:\\, \\\\server\\share) easily takes longer. The app then
     quit with "[PYI-49000] Failed to execute script ... unhandled exception".
  2. Nothing was shown while waiting, so a slow start looked like a crash.
  3. Errors never reached the screen.

This launcher:
  - opens the window at once with a "starting" page,
  - waits up to 5 minutes (set WORKBENCH_START_TIMEOUT to change it, in seconds),
  - notices at once if the server process dies, instead of waiting,
  - shows any start-up error in the window (or a message box) with the log path.

It is used only for the desktop build. Streamlit Cloud and Docker still run app.py.
"""
import multiprocessing
import os
import socket
import sys
import threading
import time
import traceback

APP_TITLE = "Proteomics Workbench"
try:
    START_TIMEOUT = max(10, int(os.environ.get("WORKBENCH_START_TIMEOUT", "300")))
except ValueError:
    START_TIMEOUT = 300

# Set WORKBENCH_DEBUG=1 to open the window with developer tools available, so a
# layout problem that only shows up in the packaged app can be inspected the way
# it would be in a browser. Off by default.
DEBUG_WINDOW = os.environ.get("WORKBENCH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

# Set WORKBENCH_GUI to force the rendering engine, for example "edgechromium" or
# "mshtml" on Windows and "cocoa" on macOS. Left empty, pywebview picks one, and
# which one it picks depends on what is installed on that computer. That is worth
# being able to pin down when one machine behaves differently from another.
FORCED_GUI = os.environ.get("WORKBENCH_GUI", "").strip() or None

# Same settings as .streamlit/config.toml. They are passed on the command line
# because Streamlit only reads config.toml from the folder the app is started
# from, and that folder is not the app folder for a packaged build.
STREAMLIT_OPTIONS = {
    "server.maxUploadSize": "1024",
    "server.enableStaticServing": "true",
    "theme.base": "light",
    "theme.primaryColor": "#004b87",
    "theme.backgroundColor": "#ffffff",
    "theme.secondaryBackgroundColor": "#f0f4f8",
    "theme.textColor": "#1e293b",
    "theme.font": "sans serif",
}


def _bundle_dir() -> str:
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def _log(msg: str) -> None:
    try:
        import logging
        logging.getLogger("workbench.launcher").info(msg)
    except Exception:
        pass


def _log_file() -> str:
    return os.environ.get("WORKBENCH_LOG_FILE", "")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_streamlit(script_path: str, port: int) -> None:
    """Runs in the child process: the Streamlit server."""
    from streamlit.web import cli as stcli

    args = ["streamlit", "run", script_path]
    args += [f"--{k}={v}" for k, v in STREAMLIT_OPTIONS.items()]
    args += [
        "--server.address=localhost",
        f"--server.port={port}",
        "--server.headless=true",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
    ]
    sys.argv = args
    stcli.main()


def _server_is_up(port: int) -> bool:
    # A plain socket check. No HTTP library, so office proxy settings
    # (HTTP_PROXY etc.) cannot get in the way.
    try:
        with socket.create_connection(("localhost", port), timeout=1):
            return True
    except OSError:
        return False


def _page(title: str, body: str, spinner: bool) -> str:
    import html as _html
    spin = (
        '<div class="spin"></div>'
        if spinner else ""
    )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body {{ font-family: Segoe UI, -apple-system, sans-serif; color:#1e293b; background:#fff;
       display:flex; align-items:center; justify-content:center; height:95vh; margin:0; }}
.box {{ max-width:640px; text-align:center; padding:24px; }}
h2 {{ color:#004b87; font-weight:600; }}
p {{ line-height:1.5; }}
pre {{ text-align:left; background:#f0f4f8; padding:12px; border-radius:6px;
       white-space:pre-wrap; word-break:break-word; font-size:12px; max-height:40vh; overflow:auto; }}
.spin {{ margin:24px auto; width:40px; height:40px; border:4px solid #f0f4f8;
         border-top-color:#004b87; border-radius:50%; animation:s 1s linear infinite; }}
@keyframes s {{ to {{ transform:rotate(360deg); }} }}
</style></head><body><div class="box"><h2>{_html.escape(title)}</h2>{spin}{body}</div></body></html>"""


def _loading_page() -> str:
    return _page(
        f"Starting {APP_TITLE}...",
        "<p>The first start can take a minute or more, longer if the app is on a "
        "network drive. Please keep this window open.</p>",
        spinner=True,
    )


def _error_page(message: str) -> str:
    import html as _html
    log = _log_file()
    log_html = (
        f"<p>Details are in the log file:</p><pre>{_html.escape(log)}</pre>" if log else ""
    )
    return _page(
        f"{APP_TITLE} could not start",
        f"<pre>{_html.escape(message)}</pre>{log_html}"
        "<p>Tip: if the app folder is on a network drive, copy the whole folder to "
        "your own computer (for example to Documents) and start it from there.</p>",
        spinner=False,
    )


def _message_box(text: str) -> None:
    """Last-resort error display, used when the window itself cannot open."""
    log = _log_file()
    if log:
        text += f"\n\nLog file:\n{log}"
    try:
        if sys.platform.startswith("win"):
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, f"{APP_TITLE} could not start", 0x10)
            return
        if sys.platform == "darwin":
            import subprocess

            def _q(s: str) -> str:  # AppleScript string literal
                return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

            subprocess.run(
                ["osascript", "-e",
                 f"display dialog {_q(text)} with title {_q(APP_TITLE + ' could not start')} "
                 'buttons {"OK"} default button "OK" with icon stop'],
                timeout=600, check=False,
            )
            return
    except Exception:
        pass
    try:
        print(text, file=sys.stderr)
    except Exception:
        pass


WEBVIEW2_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"

WEBVIEW2_MESSAGE = (
    "This app needs the Microsoft Edge WebView2 Runtime, and it is not installed "
    "on this computer.\n\n"
    "Without it the window falls back to the Internet Explorer engine, which "
    "cannot lay the page out correctly: the page stops scrolling part of the way "
    "down and the rest of the content cannot be reached.\n\n"
    "To fix it, run the installer included next to this app:\n\n"
    '    MicrosoftEdgeWebview2Setup.exe /silent /install\n\n'
    "or download it from " + WEBVIEW2_URL + "\n\n"
    "Then start the app again."
)


def _windows_renderer():
    """
    Which engine the Windows build will draw with: 'edgechromium' or 'mshtml'.

    pywebview decides this when its WinForms backend is imported, by looking in
    the registry for the WebView2 Runtime. If the runtime is missing it drops
    back to MSHTML, the Internet Explorer engine, and says so only in a log
    message that nothing here was reading. The window then opens and renders a
    modern page with a browser engine that cannot do it.

    Returns None when the renderer cannot be determined, which is not treated as
    a failure.
    """
    if not sys.platform.startswith("win"):
        return None

    try:
        import webview.platforms.winforms as winforms

        return getattr(winforms, "renderer", None)
    except Exception:
        return None


def _describe_engine() -> str:
    """Engine name for the log, specific enough to be worth reading."""
    try:
        import webview as _wv

        name = getattr(getattr(_wv, "guilib", None), "__name__", "unknown")
    except Exception:
        name = "unknown"

    renderer = _windows_renderer()
    return f"{name} ({renderer})" if renderer else name


def _wait_then_load(window, proc, port: int, closed: threading.Event) -> None:
    """Runs in a background thread once the window is open."""
    start = time.time()
    _log(f"Rendering engine: {_describe_engine()}" + (" (debug window)" if DEBUG_WINDOW else ""))

    try:
        while not closed.is_set():
            if _server_is_up(port):
                _log(f"Streamlit server is up after {time.time() - start:.1f} s")
                window.load_url(f"http://localhost:{port}")
                return
            if not proc.is_alive():
                msg = (
                    "The analysis server stopped while starting "
                    f"(exit code {proc.exitcode})."
                )
                _log(msg)
                window.load_html(_error_page(msg))
                return
            if time.time() - start > START_TIMEOUT:
                msg = (
                    f"The analysis server did not start within {START_TIMEOUT} seconds."
                )
                _log(msg)
                window.load_html(_error_page(msg))
                return
            time.sleep(0.25)
    except Exception:
        tb = traceback.format_exc()
        _log("Error while waiting for the server:\n" + tb)
        try:
            window.load_html(_error_page(tb))
        except Exception:
            pass


def main() -> int:
    script_path = os.path.join(_bundle_dir(), "app.py")
    port = _free_port()
    _log(f"Launcher: app={script_path} port={port} timeout={START_TIMEOUT}s")

    # Not a daemon process: daemon processes may not start their own worker
    # processes, which some analysis libraries (joblib / scikit-learn) need.
    proc = multiprocessing.Process(
        target=run_streamlit, args=(script_path, port), name="streamlit-server"
    )
    proc.start()

    try:
        import webview

        # Checked before the window opens. Opening it anyway would show a page
        # that looks almost right and scrolls only part of the way down, which is
        # far harder to work out than being told what is wrong.
        if _windows_renderer() == "mshtml" and FORCED_GUI != "mshtml":
            _log("WebView2 Runtime is missing; refusing to render with MSHTML.")
            _message_box(WEBVIEW2_MESSAGE)
            return 1

        window = webview.create_window(
            APP_TITLE, html=_loading_page(), width=1280, height=860, min_size=(900, 600)
        )
        closed = threading.Event()
        window.events.closed += lambda *a: closed.set()

        start_kwargs = {}
        if DEBUG_WINDOW:
            start_kwargs["debug"] = True
        if FORCED_GUI:
            start_kwargs["gui"] = FORCED_GUI
            _log(f"Rendering engine forced to {FORCED_GUI} by WORKBENCH_GUI")

        webview.start(_wait_then_load, (window, proc, port, closed), **start_kwargs)
        return 0
    except Exception:
        tb = traceback.format_exc()
        _log("Could not open the app window:\n" + tb)
        hint = (
            "On Windows this usually means the Microsoft Edge WebView2 Runtime "
            "is missing. Run MicrosoftEdgeWebview2Setup.exe /silent /install from "
            "the folder next to this app, or get it from " + WEBVIEW2_URL
            if sys.platform.startswith("win") else
            "Make sure you are running the build for your Mac's chip "
            "(Apple Silicon = arm64, Intel = x86_64)."
        )
        _message_box(
            "The app window could not open.\n\n" + hint + "\n\n"
            + tb.strip().splitlines()[-1]
        )
        return 1
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=10)


if __name__ == "__main__":
    # Must come first: in the packaged app the server process starts by running
    # this same file, and freeze_support() sends it to run_streamlit() instead.
    multiprocessing.freeze_support()
    sys.exit(main())