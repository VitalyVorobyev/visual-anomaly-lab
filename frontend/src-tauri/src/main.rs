// Hide the console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod sidecar;

use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};

use crate::sidecar::Sidecar;

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

fn main() {
    let app = tauri::Builder::default()
        .setup(|app| {
            // Start the backend first: the window is built only once the sidecar has
            // announced its port, so the UI never renders against a URL that does not
            // exist yet and needs no retry-on-boot logic.
            let sidecar = Sidecar::spawn(&repo_root())?;

            // Handing the URL over as an injected global rather than through a Tauri
            // command is what keeps `frontend/src/` free of any Tauri import, so the
            // same bundle runs in a plain browser (§2).
            let script = format!(
                "window.__ANOMALY_LAB__ = {{ apiBaseUrl: {} }};",
                serde_json::to_string(&sidecar.base_url)?
            );

            WebviewWindowBuilder::new(app, "main", WebviewUrl::default())
                .title("visual-anomaly-lab")
                .inner_size(1280.0, 860.0)
                .initialization_script(&script)
                .build()?;

            app.manage(SidecarState(Mutex::new(Some(sidecar))));
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
