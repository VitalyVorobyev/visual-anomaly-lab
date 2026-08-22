// Hide the console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod sidecar;

use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_dialog::DialogExt;

use crate::sidecar::{Sidecar, StartupError};

struct SidecarState(Mutex<Option<Sidecar>>);

/// Where the backend source lives.
///
/// Development only: `ANOMALY_LAB_REPO_ROOT`, else the checkout this binary was built
/// from. A packaged build (M7) ships its own interpreter and sets
/// `ANOMALY_LAB_SIDECAR_CMD` instead, so it never consults this path.
fn repo_root() -> PathBuf {
    if let Ok(explicit) = std::env::var("ANOMALY_LAB_REPO_ROOT") {
        return PathBuf::from(explicit);
    }
    // <repo>/frontend/src-tauri -> <repo>
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|frontend| frontend.parent())
        .map(PathBuf::from)
        .expect("the crate should live at <repo>/frontend/src-tauri")
}

/// Choose a directory with the OS picker.
///
/// The import flow needs a path the *backend* can open, and a browser cannot produce one:
/// a file input yields a `File` and a bare name, deliberately. So the shell offers this
/// and the browser offers a text field (handbook frontend.md).
///
/// `async` on purpose — the blocking picker must not run on the main thread, and an async
/// command is dispatched to a worker.
#[tauri::command]
async fn pick_directory(app: tauri::AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .blocking_pick_folder()
        .and_then(|chosen| chosen.into_path().ok())
        .map(|path| path.to_string_lossy().into_owned())
}

/// Show a path in the OS file manager.
///
/// The second capability handbook frontend.md anticipated, and it exists for a concrete complaint: a
/// run spends eleven minutes producing a 31 MB checkpoint and the app never says where it
/// went. A browser has no equivalent and gets the path as selectable text instead.
///
/// **The path is checked against the data directory before it is handed to the OS.** It
/// arrives from the backend today, but a command that reveals any path the webview names
/// is a command that reveals any path at all, and the check costs one comparison.
#[tauri::command]
fn reveal_path(path: String) -> Result<(), String> {
    let target = std::fs::canonicalize(&path).map_err(|error| error.to_string())?;
    let root = data_root();
    let allowed = std::fs::canonicalize(&root).unwrap_or(root);
    if !target.starts_with(&allowed) {
        return Err(format!(
            "{} is outside the data directory",
            target.display()
        ));
    }

    // `open -R` selects the item in Finder rather than opening it, which is right for both
    // a directory and a checkpoint nothing should double-click.
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("open")
        .arg("-R")
        .arg(&target)
        .spawn();
    #[cfg(target_os = "windows")]
    let result = std::process::Command::new("explorer")
        .arg("/select,")
        .arg(&target)
        .spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let result = std::process::Command::new("xdg-open")
        .arg(target.parent().unwrap_or(&target))
        .spawn();

    result.map(|_| ()).map_err(|error| error.to_string())
}

/// The one directory `reveal_path` will open, matching the backend's `Settings.data_dir`.
fn data_root() -> PathBuf {
    if let Ok(explicit) = std::env::var("ANOMALY_LAB_DATA_DIR") {
        return PathBuf::from(explicit);
    }
    repo_root().join("data")
}

/// The globals a working shell injects.
///
/// Everything the shell offers is *injected*, never imported: that is what keeps
/// `frontend/src/` free of any Tauri dependency, so the same bundle runs in a plain browser
/// (§2, handbook frontend.md). The UI feature-detects `pickDirectory` and falls back to a text field
/// when it is absent.
fn ready_script(base_url: &str) -> String {
    format!(
        "window.__ANOMALY_LAB__ = {{ \
           apiBaseUrl: {}, \
           pickDirectory: () => window.__TAURI_INTERNALS__.invoke('pick_directory'), \
           revealPath: (path) => \
             window.__TAURI_INTERNALS__.invoke('reveal_path', {{ path }}) \
         }};",
        serde_json::Value::String(base_url.to_owned())
    )
}

/// The globals a shell with no backend injects instead.
///
/// The window is built either way, and this is why: on macOS the setup hook runs inside
/// `did_finish_launching`, an Objective-C callback an unwind may not cross, so Tauri's
/// `panic!` on a setup `Err` becomes `abort()` — a crash report and no window at all. The
/// page reads this and paints the reason instead of the workbench, which is the only way a
/// packaged app can say anything: its stderr is written to nothing.
fn failure_script(failure: &StartupError) -> String {
    let payload = serde_json::json!({
        "message": failure.message,
        "detail": failure.detail,
    });
    format!("window.__ANOMALY_LAB__ = {{ startupError: {payload} }};")
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![pick_directory, reveal_path])
        .setup(|app| {
            // Start the backend first: the window is built only once the sidecar has
            // announced its port, so the UI never renders against a URL that does not
            // exist yet and needs no retry-on-boot logic.
            //
            // Nothing in here may return `Err`. See `failure_script`.
            let script = match Sidecar::spawn(&repo_root()) {
                Ok(sidecar) => {
                    let script = ready_script(&sidecar.base_url);
                    // Managed before the window exists, so that a window that fails to
                    // build still leaves the sidecar reachable by the teardown handler.
                    app.manage(SidecarState(Mutex::new(Some(sidecar))));
                    script
                }
                Err(failure) => {
                    eprintln!("[shell] {}\n{}", failure.message, failure.detail);
                    failure_script(&failure)
                }
            };

            if let Err(error) = WebviewWindowBuilder::new(app, "main", WebviewUrl::default())
                .title("visual-anomaly-lab")
                .inner_size(1280.0, 860.0)
                .initialization_script(&script)
                .build()
            {
                // There is nothing left to show a message with, so exit rather than
                // propagate: an `Err` from here aborts the process instead of ending it.
                eprintln!("[shell] could not create the window: {error}");
                app.handle().exit(1);
            }

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the application");

    app.run(|app_handle, event| match event {
        // macOS keeps an application running after its last window closes. This is a
        // single-window tool, so that convention would leave a sidecar serving a window
        // that no longer exists — closing the window has to mean quitting the app.
        RunEvent::WindowEvent {
            event: WindowEvent::Destroyed,
            ..
        } => {
            app_handle.exit(0);
        }

        // `Exit` is the last event before the process ends and fires for every graceful
        // path: window closed, Cmd-Q, menu quit. The sidecar's own watchdog covers the
        // paths that run no handler at all — SIGKILL, force-quit, or a crash in here —
        // because macOS has no PDEATHSIG equivalent to lean on.
        RunEvent::Exit => {
            if let Some(state) = app_handle.try_state::<SidecarState>() {
                if let Some(mut sidecar) = state.0.lock().ok().and_then(|mut guard| guard.take()) {
                    sidecar.shutdown();
                }
            }
        }

        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A malformed script is another black window: the page would die on a syntax error
    /// before it could read the very thing that explains why it has no backend.
    #[test]
    fn a_failure_script_survives_quotes_newlines_and_backslashes() {
        let failure = StartupError {
            message: "`uv` was not found, so the backend could not be started.".to_owned(),
            detail: "Searched:\n  C:\\uv\n  \"/usr/bin/uv\"\n</script>".to_owned(),
        };

        let script = failure_script(&failure);
        let payload = script
            .strip_prefix("window.__ANOMALY_LAB__ = { startupError: ")
            .and_then(|rest| rest.strip_suffix(" };"))
            .expect("the script assigns exactly one object");

        let parsed: serde_json::Value =
            serde_json::from_str(payload).expect("the payload is valid JSON");
        assert_eq!(parsed["message"], failure.message.as_str());
        assert_eq!(parsed["detail"], failure.detail.as_str());
        assert!(
            !script.contains('\n'),
            "a raw newline would break the script"
        );
    }

    #[test]
    fn a_ready_script_quotes_the_base_url() {
        let script = ready_script("http://127.0.0.1:54321");
        assert!(
            script.contains("apiBaseUrl: \"http://127.0.0.1:54321\""),
            "{script}"
        );
        assert!(script.contains("pickDirectory"));
        assert!(!script.contains("startupError"));
    }
}
