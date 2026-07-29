
import os
import sys
import tkinter as tk
from korpusuj.runtime_paths import configure_ml_cache_environment as _configure_ml_cache_environment_182n
_configure_ml_cache_environment_182n()


splash = None
progress_label = None # Nowa zmienna globalna
# ---- macOS fix: Tkinter + NSWindow main-thread patch ----

# ---------------------------------------------------------
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# NOWA FUNKCJA: pozwala innym plikom zmieniać tekst na splashu
def update_status(text):
    global progress_label, splash
    if progress_label and splash:
        progress_label.config(text=text)
        splash.update() # Wymusza odświeżenie okna podczas importu

def show_boot_splash():
    global splash, progress_label
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.configure(bg="#1f2328")

    width = 460
    height = 320
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))
    splash.geometry(f"{width}x{height}+{x}+{y}")

    frame = tk.Frame(splash, bg="#1f2328", bd=0)
    frame.pack(fill="both", expand=True)

    try:
        from PIL import Image, ImageTk
        logo_path = resource_path("temp/logo.png")
        img = Image.open(logo_path)
        img.thumbnail((250, 120), Image.LANCZOS)
        logo_img = ImageTk.PhotoImage(img)
        splash.logo_img = logo_img
        logo_label = tk.Label(frame, image=logo_img, bg="#1f2328", bd=0)
        logo_label.pack(pady=(20, 0))
        title_pady = (10, 5)
    except Exception:
        title_pady = (40, 10)

    tk.Label(frame, text="Korpusuj", font=("Verdana", 22, "bold"), fg="white", bg="#1f2328").pack(pady=title_pady)
    tk.Label(frame, text="Uruchamianie aplikacji, proszę czekać...", font=("Verdana", 11), fg="#cfd8dc", bg="#1f2328").pack(pady=(0, 20))

    # Zapisujemy referencję do labela, żeby móc go zmieniać
    progress_label = tk.Label(frame, text="Inicjalizacja...", font=("Verdana", 10), fg="#9fb3c8", bg="#1f2328")
    progress_label.pack()

    splash.update_idletasks()
    splash.update()

def close_splash():
    global splash
    if splash:
        try:
            splash.destroy()
        except Exception:
            pass
        splash = None

def main():
    # NOWOŚĆ: Jeśli uruchamiamy proces poboczny (webview/fiszki), POMIŃ splash screen
    if "--run-webview" in sys.argv or "--run-fiszki" in sys.argv or "--run-semantic-trainer" in sys.argv or "--run-semantic-report" in sys.argv:
        import engine
        return  # engine.py i tak zrobi sys.exit(0) w swoim routerze, więc tu kończymy

    # Standardowe zachowanie dla głównej aplikacji
    if sys.platform == "darwin":
        import engine
        if hasattr(engine, "main"):
            engine.main()
        return

    show_boot_splash()
    import engine
    close_splash()
    if hasattr(engine, "main"):
        engine.main()




# KORPUSUJ_PATCH_137C_DIAGNOSTIC_LOGGING_FLAGS_AND_CONFIG_GUI
# Enables diagnostic/verbose logs from GUI startup using CLI-like flags and flat config keys:
#   python Korpusuj.py --verbose
#   python Korpusuj.py --diagnostics-logs
# Config defaults:
#   logging_verbose: false
#   logging_diagnostics_logs: false
# Priority: explicit CLI flag > existing env var > config default.
def _install_gui_logging_flags_137c():
    try:
        import os as _os_137c
        import sys as _sys_137c
        import json as _json_137c
        from pathlib import Path as _Path_137c
    except Exception:
        return

    if globals().get("_korpusuj_137c_gui_logging_flags_installed", False):
        return

    TRUTHY = {"1", "true", "yes", "tak", "on", "debug", "verbose"}

    def _truthy_137c(value):
        try:
            if value is True:
                return True
            if isinstance(value, str):
                return value.strip().lower() in TRUTHY
            return bool(value)
        except Exception:
            return False

    def _load_config_137c():
        candidates = []
        try:
            candidates.append(_Path_137c(__file__).resolve().parent / "config.json")
        except Exception:
            pass
        candidates.append(_Path_137c("config.json"))
        for p in candidates:
            try:
                if p.exists():
                    data = _json_137c.loads(p.read_text(encoding="utf-8", errors="replace"))
                    return data if isinstance(data, dict) else {}
            except Exception:
                pass
        return {}

    def _has_env_137c(*names):
        try:
            return any(name in _os_137c.environ for name in names)
        except Exception:
            return False

    def _apply_137c(argv=None, *, strip_flags=False):
        argv = list(_sys_137c.argv[1:] if argv is None else argv)
        cfg = _load_config_137c()
        verbose_flag = "--verbose" in argv
        diag_flag = "--diagnostics-logs" in argv

        # Config defaults only when no related env var already exists.
        if (not _has_env_137c("KORPUSUJ_VERBOSE_LOGS", "KORPUSUJ_VERBOSE")) and _truthy_137c(cfg.get("logging_verbose", False)):
            _os_137c.environ["KORPUSUJ_VERBOSE_LOGS"] = "1"
        if (not _has_env_137c("KORPUSUJ_137_DIAGNOSTIC_LOGS")) and _truthy_137c(cfg.get("logging_diagnostics_logs", False)):
            _os_137c.environ["KORPUSUJ_137_DIAGNOSTIC_LOGS"] = "1"

        # Explicit flags win over config/env-off states for this process.
        if verbose_flag:
            _os_137c.environ["KORPUSUJ_VERBOSE_LOGS"] = "1"
        if diag_flag:
            _os_137c.environ["KORPUSUJ_137_DIAGNOSTIC_LOGS"] = "1"
            # diagnostics imply verbose compatibility for older helpers
            _os_137c.environ.setdefault("KORPUSUJ_VERBOSE_LOGS", "1")

        if strip_flags:
            try:
                _sys_137c.argv[:] = [_sys_137c.argv[0]] + [a for a in _sys_137c.argv[1:] if a not in {"--verbose", "--diagnostics-logs"}]
            except Exception:
                pass

    orig_main = globals().get("main")
    if callable(orig_main) and not getattr(orig_main, "_korpusuj_137c_logging_flags_wrapped", False):
        def main_with_logging_flags_137c(*args, **kwargs):
            _apply_137c(strip_flags=True)
            return orig_main(*args, **kwargs)
        main_with_logging_flags_137c._korpusuj_137c_logging_flags_wrapped = True
        globals()["main"] = main_with_logging_flags_137c
    else:
        _apply_137c(strip_flags=True)

    globals()["_korpusuj_137c_gui_logging_flags_installed"] = True

try:
    _install_gui_logging_flags_137c()
except Exception:
    pass
# END KORPUSUJ_PATCH_137C_DIAGNOSTIC_LOGGING_FLAGS_AND_CONFIG_GUI
if __name__ == "__main__":
    main()