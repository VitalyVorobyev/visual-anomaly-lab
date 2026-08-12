use std::path::PathBuf;

use anomaly_lab_runner::{Bundle, write_f32};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "anomaly-lab-runner", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate hashes and run the bundle's deterministic parity fixture.
    Verify {
        /// Directory containing manifest.json and its declared payloads.
        bundle: PathBuf,
    },
    /// Run one prepared little-endian NCHW float32 tensor.
    Infer {
        bundle: PathBuf,
        #[arg(long)]
        input: PathBuf,
        /// Optional little-endian float32 anomaly-map output.
        #[arg(long)]
        map_output: Option<PathBuf>,
    },
}

fn main() {
    if let Err(error) = run() {
        let payload = serde_json::json!({
            "status": "error",
            "error": format!("{error:#}"),
        });
        eprintln!(
            "{}",
            serde_json::to_string(&payload).expect("serialize error report")
        );
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Verify { bundle } => {
            let report = Bundle::open(bundle)?.verify_fixture()?;
            println!("{}", serde_json::to_string(&report)?);
        }
        Command::Infer {
            bundle,
            input,
            map_output,
        } => {
            let bundle = Bundle::open(bundle)?;
            let (report, map) = bundle.infer_file(input)?;
            if let Some(path) = map_output {
                write_f32(&path, &map)
                    .with_context(|| format!("write anomaly map {}", path.display()))?;
            }
            println!("{}", serde_json::to_string(&report)?);
        }
    }
    Ok(())
}
