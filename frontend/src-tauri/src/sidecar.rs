//! Sidecar process lifecycle.
//!
//! The shell's entire job (ADR-0003). Three things here are load-bearing and easy to get
//! subtly wrong on macOS, which ADR-0003 flags as an accepted cost of the two-process design:
//!
//! 1. **The child chooses the port.** We pass `ANOMALY_LAB_PORT=0` and read the port back
//!    from the ready line. Allocating a port here instead would mean binding, closing, and
//!    hoping nothing else takes it before the child re-binds.
//!
//! 2. **Both pipes are drained forever.** A child whose stdout or stderr pipe fills up
//!    blocks on write and hangs. Reading only until the ready line would deadlock the
//!    sidecar the moment it logged enough.
//!
//! 3. **The child is killed by process group.** `uv run` sits between this process and the
//!    Python interpreter, so signalling the direct child alone can leave the interpreter
//!    running. The sidecar's own parent-pid watchdog is the backstop for the case no
//!    signal is delivered at all — a force-quit or a crash of this process.

use std::io::{BufRead, BufReader};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde::Deserialize;

/// How long to wait for the sidecar to announce its port before giving up.
const READY_TIMEOUT: Duration = Duration::from_secs(60);

/// How long the sidecar gets to exit after SIGTERM before SIGKILL.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// The first line the sidecar writes to stdout, in the ADR-0009 event envelope.
#[derive(Debug, Deserialize)]
struct ReadyEvent {
    ev: String,
    port: u16,
    pid: u32,
}

pub struct Sidecar {
    child: Child,
    /// The pid the sidecar reported for itself, which is not necessarily `child.id()`
    /// when an intermediate process such as `uv run` is in between.
    sidecar_pid: u32,
    pub base_url: String,
}

impl Sidecar {
    /// Spawn the backend and block until it reports the port it bound.
    pub fn spawn(repo_root: &std::path::Path) -> Result<Self, String> {
        let mut command = build_command(repo_root);

        command
            .env("ANOMALY_LAB_PORT", "0")
            .env("ANOMALY_LAB_PARENT_PID", std::process::id().to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Put the child in its own process group so the whole tree — including any
        // intermediate `uv run` — can be signalled as a unit on teardown.
        #[cfg(unix)]
        command.process_group(0);

        let mut child = command
            .spawn()
            .map_err(|error| format!("Could not start the backend: {error}"))?;

        let stdout = child
            .stdout
            .take()
            .ok_or("The backend has no stdout pipe.")?;
        let stderr = child
            .stderr
            .take()
            .ok_or("The backend has no stderr pipe.")?;

        // stderr carries the sidecar's logs; drain it for the life of the process.
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                eprintln!("[sidecar] {line}");
            }
        });

        // stdout carries structured events. Report the first one, then keep draining.
        let (sender, receiver) = mpsc::channel::<ReadyEvent>();
        std::thread::spawn(move || {
            let mut sender = Some(sender);
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                match serde_json::from_str::<ReadyEvent>(&line) {
                    Ok(event) if event.ev == "ready" => {
                        if let Some(sender) = sender.take() {
                            let _ = sender.send(event);
                        }
                    }
                    // Anything else is diagnostic output, not an error: ADR-0009 requires
                    // this parser to tolerate non-JSON lines rather than choke on them.
                    _ => eprintln!("[sidecar] {line}"),
                }
            }
        });

        let ready = receiver.recv_timeout(READY_TIMEOUT).map_err(|_| {
            let _ = child.kill();
            format!(
                "The backend did not report readiness within {} seconds.",
                READY_TIMEOUT.as_secs()
            )
        })?;

        let base_url = format!("http://127.0.0.1:{}", ready.port);
        eprintln!("[shell] sidecar ready on {base_url} (pid {})", ready.pid);

        Ok(Self {
            child,
            sidecar_pid: ready.pid,
            base_url,
        })
    }

    /// SIGTERM, grace period, then SIGKILL — for the whole process group (ADR-0009).
    pub fn shutdown(&mut self) {
        #[cfg(unix)]
        {
            let group = -(self.child.id() as i32);
            unsafe {
                libc::kill(group, libc::SIGTERM);
                libc::kill(self.sidecar_pid as i32, libc::SIGTERM);
            }

            let deadline = std::time::Instant::now() + SHUTDOWN_GRACE;
            while std::time::Instant::now() < deadline {
                match self.child.try_wait() {
                    Ok(Some(_)) => {
                        eprintln!("[shell] sidecar exited cleanly");
                        return;
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(100)),
                    Err(_) => break,
                }
            }

            eprintln!("[shell] sidecar did not exit in time; sending SIGKILL");
            unsafe {
                libc::kill(group, libc::SIGKILL);
                libc::kill(self.sidecar_pid as i32, libc::SIGKILL);
            }
        }

        #[cfg(not(unix))]
        {
            let _ = self.child.kill();
        }

        let _ = self.child.wait();
    }
}

/// How to start the backend.
///
/// Overridable via `ANOMALY_LAB_SIDECAR_CMD` so that packaging (M7) can substitute a
/// bundled interpreter without touching any of the lifecycle logic above.
fn build_command(repo_root: &std::path::Path) -> Command {
    if let Ok(custom) = std::env::var("ANOMALY_LAB_SIDECAR_CMD") {
        let mut parts = custom.split_whitespace();
        let program = parts.next().unwrap_or("uv");
        let mut command = Command::new(program);
        command.args(parts);
        return command;
    }

    let mut command = Command::new("uv");
    command.args([
        "run",
        "--directory",
        &repo_root.join("backend").to_string_lossy(),
        "python",
        "-m",
        "anomaly_lab.serve",
    ]);
    command
}
