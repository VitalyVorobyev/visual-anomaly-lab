//! Sidecar process lifecycle.
//!
//! The shell's entire job (ADR-0003). Four things here are load-bearing and easy to get
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
//!
//! 4. **`uv` is found by absolute path, not by `PATH`.** An app launched from Finder is
//!    started by launchd, and `launchctl getenv PATH` is empty on a stock machine: the
//!    process inherits `/usr/bin:/bin:/usr/sbin:/sbin` and nothing else. `uv` installs to
//!    `~/.local/bin`, which is on no such list — so a build that worked from a terminal
//!    could not start its backend at all once it was in `/Applications`.
//!
//! Every failure here is a [`StartupError`] rather than a message on a stream nobody reads:
//! a packaged app's stderr goes nowhere, so what went wrong has to reach the window.

use std::collections::VecDeque;
use std::ffi::OsStr;
use std::io::{BufRead, BufReader};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;

/// How long to wait for the sidecar to announce its port before giving up.
const READY_TIMEOUT: Duration = Duration::from_secs(60);

/// How long the sidecar gets to exit after SIGTERM before SIGKILL.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// How much of the backend's own output to keep for a failure message.
const STDERR_TAIL_LINES: usize = 40;

/// Where `uv` installs itself, in the order worth trying. `~/.local/bin` is uv's own
/// default; the rest are Homebrew (Apple silicon), Homebrew (Intel) and MacPorts.
const UV_FALLBACK_DIRS: [&str; 3] = ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"];

/// The first line the sidecar writes to stdout, in the ADR-0009 event envelope.
#[derive(Debug, Deserialize)]
struct ReadyEvent {
    ev: String,
    port: u16,
    pid: u32,
}

/// Why the backend did not start, in a shape a window can show.
///
/// `message` is the one sentence that names the cause; `detail` is everything needed to act
/// on it — the paths searched, the command attempted, the backend's own last output. A
/// startup failure is the one moment this app cannot explain itself any other way.
#[derive(Debug)]
pub struct StartupError {
    pub message: String,
    pub detail: String,
}

impl StartupError {
    fn new(message: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            detail: detail.into(),
        }
    }
}

/// Lines the backend wrote to stderr, capped and shared with the drain thread.
type StderrTail = Arc<Mutex<VecDeque<String>>>;

pub struct Sidecar {
    child: Child,
    /// The pid the sidecar reported for itself, which is not necessarily `child.id()`
    /// when an intermediate process such as `uv run` is in between.
    sidecar_pid: u32,
    pub base_url: String,
}

impl Sidecar {
    /// Spawn the backend and block until it reports the port it bound.
    pub fn spawn(repo_root: &Path) -> Result<Self, StartupError> {
        let mut command = build_command(repo_root)?;

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

        let described = describe(&command);
        let mut child = command.spawn().map_err(|error| {
            StartupError::new(
                format!("Could not start the backend: {error}"),
                format!("Command:\n  {described}"),
            )
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            StartupError::new("The backend has no stdout pipe.", described.clone())
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            StartupError::new("The backend has no stderr pipe.", described.clone())
        })?;

        // stderr carries the sidecar's logs; drain it for the life of the process. The tail
        // is kept because a packaged app's own stderr is written to nothing at all, and the
        // backend's last words are exactly what a failed start needs to show.
        let tail: StderrTail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL_LINES)));
        let collected = Arc::clone(&tail);
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                eprintln!("[sidecar] {line}");
                if let Ok(mut collected) = collected.lock() {
                    if collected.len() == STDERR_TAIL_LINES {
                        collected.pop_front();
                    }
                    collected.push_back(line);
                }
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

        let ready = match receiver.recv_timeout(READY_TIMEOUT) {
            Ok(event) => event,
            // `Disconnected` is not the same failure as `Timeout` and must not be reported
            // as one: the drain thread drops its sender at EOF, so a backend that dies on
            // startup lands here immediately rather than after the full minute.
            Err(reason) => return Err(never_ready(reason, &mut child, &tail)),
        };

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

/// Turn "no ready line" into a message that says which of the two failures it was.
fn never_ready(reason: RecvTimeoutError, child: &mut Child, tail: &StderrTail) -> StartupError {
    // Read the status *before* killing, so an exit code the backend chose is the one
    // reported rather than the signal we would send over it.
    let status = child.try_wait().ok().flatten();
    kill_now(child);
    let _ = child.wait();

    // The drain thread is not joined: `uv run` leaves a Python interpreter holding the
    // write end of that pipe, and a grandchild that ignores the kill would hang the join
    // forever. A moment is enough for the lines already in flight.
    std::thread::sleep(Duration::from_millis(100));

    let message = match (reason, status) {
        (RecvTimeoutError::Timeout, _) => format!(
            "The backend did not report readiness within {} seconds.",
            READY_TIMEOUT.as_secs()
        ),
        (RecvTimeoutError::Disconnected, Some(status)) => {
            format!("The backend exited before reporting readiness ({status}).")
        }
        (RecvTimeoutError::Disconnected, None) => {
            "The backend closed its output before reporting readiness.".to_owned()
        }
    };

    StartupError::new(message, format!("Its last output:\n{}", tail_text(tail)))
}

fn tail_text(tail: &StderrTail) -> String {
    let lines = match tail.lock() {
        Ok(lines) => lines.iter().cloned().collect::<Vec<_>>(),
        Err(_) => Vec::new(),
    };
    if lines.is_empty() {
        "  (it printed nothing)".to_owned()
    } else {
        lines
            .iter()
            .map(|line| format!("  {line}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Kill the child's whole process group, since `uv run` is not the process that serves.
fn kill_now(child: &mut Child) {
    #[cfg(unix)]
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
}

/// The command line, for a message that has to be actionable without a terminal.
fn describe(command: &Command) -> String {
    let program = command.get_program().to_string_lossy().into_owned();
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ");
    if args.is_empty() {
        program
    } else {
        format!("{program} {args}")
    }
}

/// How to start the backend.
///
/// Overridable via `ANOMALY_LAB_SIDECAR_CMD` so that a different interpreter, or a backend
/// installed anywhere else, can be substituted without touching any of the lifecycle logic
/// above. Otherwise: the backend runs from the checkout this binary was built from, through
/// `uv` resolved to an absolute path.
fn build_command(repo_root: &Path) -> Result<Command, StartupError> {
    if let Ok(custom) = std::env::var("ANOMALY_LAB_SIDECAR_CMD") {
        let mut parts = custom.split_whitespace();
        let program = parts.next().unwrap_or("uv");
        let mut command = Command::new(program);
        command.args(parts);
        return Ok(command);
    }

    // The path to the checkout is baked in at compile time, so an installed binary points
    // at wherever it was built. Say so plainly rather than letting `uv` fail obscurely.
    let backend = repo_root.join("backend");
    if !backend.join("pyproject.toml").is_file() {
        return Err(StartupError::new(
            "The backend source is not where this build expects it.",
            format!(
                "Expected a checkout at:\n  {}\n\n\
                 That path is recorded when the shell is compiled, so a moved or deleted \
                 checkout leaves it stale. Set ANOMALY_LAB_REPO_ROOT to the current \
                 checkout, or ANOMALY_LAB_SIDECAR_CMD to the full command that starts the \
                 backend.",
                repo_root.display()
            ),
        ));
    }

    let uv = resolve_uv(
        std::env::var_os("PATH").as_deref(),
        std::env::var_os("HOME").map(PathBuf::from).as_deref(),
        &is_executable,
    )
    .map_err(|searched| {
        StartupError::new(
            "`uv` was not found, so the backend could not be started.",
            format!(
                "An app launched from Finder inherits almost no PATH — launchd hands it \
                 /usr/bin:/bin:/usr/sbin:/sbin, which is why a build that starts from a \
                 terminal can fail from /Applications.\n\nSearched:\n{}\n\n\
                 Install uv somewhere on this list, or set ANOMALY_LAB_SIDECAR_CMD to the \
                 full command that starts the backend.",
                searched
                    .iter()
                    .map(|path| format!("  {}", path.display()))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        )
    })?;

    let mut command = Command::new(uv);
    command.args([
        "run",
        "--directory",
        &backend.to_string_lossy(),
        "python",
        "-m",
        "anomaly_lab.serve",
    ]);
    Ok(command)
}

/// Every place `uv` could be, in order, with no duplicates.
fn uv_candidates(path_var: Option<&OsStr>, home: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    let mut push = |candidate: PathBuf| {
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    };

    if let Some(path_var) = path_var {
        for directory in std::env::split_paths(path_var) {
            push(directory.join("uv"));
        }
    }
    if let Some(home) = home {
        push(home.join(".local").join("bin").join("uv"));
    }
    for directory in UV_FALLBACK_DIRS {
        push(Path::new(directory).join("uv"));
    }

    candidates
}

/// The first candidate that is actually runnable, or every path that was tried.
///
/// `runnable` is a parameter so this is testable without putting a binary on disk.
fn resolve_uv(
    path_var: Option<&OsStr>,
    home: Option<&Path>,
    runnable: &dyn Fn(&Path) -> bool,
) -> Result<PathBuf, Vec<PathBuf>> {
    let candidates = uv_candidates(path_var, home);
    match candidates.iter().find(|candidate| runnable(candidate)) {
        Some(found) => Ok(found.clone()),
        None => Err(candidates),
    }
}

fn is_executable(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    #[cfg(unix)]
    {
        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        metadata.is_file()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    /// What launchd hands a Finder-launched app on a stock machine.
    const LAUNCHD_PATH: &str = "/usr/bin:/bin:/usr/sbin:/sbin";

    fn only(runnable: &'static str) -> impl Fn(&Path) -> bool {
        move |path: &Path| path == Path::new(runnable)
    }

    #[test]
    fn prefers_path_over_the_fallback_directories() {
        let path = OsString::from("/opt/homebrew/bin:/somewhere/else");
        let resolved = resolve_uv(
            Some(&path),
            Some(Path::new("/Users/someone")),
            &only("/somewhere/else/uv"),
        );
        assert_eq!(resolved, Ok(PathBuf::from("/somewhere/else/uv")));
    }

    #[test]
    fn finds_the_uv_default_install_when_path_is_the_launchd_one() {
        let path = OsString::from(LAUNCHD_PATH);
        let resolved = resolve_uv(
            Some(&path),
            Some(Path::new("/Users/someone")),
            &only("/Users/someone/.local/bin/uv"),
        );
        assert_eq!(resolved, Ok(PathBuf::from("/Users/someone/.local/bin/uv")));
    }

    #[test]
    fn falls_back_to_homebrew_without_a_home() {
        let path = OsString::from(LAUNCHD_PATH);
        let resolved = resolve_uv(Some(&path), None, &only("/opt/homebrew/bin/uv"));
        assert_eq!(resolved, Ok(PathBuf::from("/opt/homebrew/bin/uv")));
    }

    #[test]
    fn a_file_that_is_not_runnable_is_not_a_candidate() {
        let path = OsString::from(LAUNCHD_PATH);
        let resolved = resolve_uv(Some(&path), Some(Path::new("/Users/someone")), &|_| false);
        let searched = resolved.expect_err("nothing is runnable, so nothing can resolve");
        assert!(searched.contains(&PathBuf::from("/usr/bin/uv")));
        assert!(searched.contains(&PathBuf::from("/Users/someone/.local/bin/uv")));
        assert!(searched.contains(&PathBuf::from("/opt/homebrew/bin/uv")));
    }

    #[test]
    fn the_searched_list_names_every_place_tried_exactly_once() {
        let path = OsString::from("/opt/homebrew/bin:/opt/homebrew/bin:/usr/bin");
        let candidates = uv_candidates(Some(&path), Some(Path::new("/Users/someone")));
        let unique: std::collections::HashSet<_> = candidates.iter().collect();
        assert_eq!(unique.len(), candidates.len(), "{candidates:?}");
        assert_eq!(
            candidates.first(),
            Some(&PathBuf::from("/opt/homebrew/bin/uv"))
        );
    }
}
