//! Verified reader and ONNX Runtime executor for deployment bundle versions 1 and 2.

use std::{
    collections::HashSet,
    fs,
    path::{Component, Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use ort::{
    inputs,
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MIN_FORMAT_VERSION: u32 = 1;
const MAX_FORMAT_VERSION: u32 = 2;

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub portable_format: String,
    pub graph_path: String,
    pub input: PixelInput,
    pub anomaly_map: MapOutput,
    pub score: ScoreContract,
    pub operating_point: Option<OperatingPoint>,
    pub files: Vec<FileDigest>,
    pub parity: ParityFixture,
}

#[derive(Debug, Deserialize)]
pub struct PixelInput {
    pub coordinate_frame: String,
    pub color: String,
    pub value_min: f32,
    pub value_max: f32,
    pub tensor: TensorSpec,
}

#[derive(Debug, Deserialize)]
pub struct MapOutput {
    pub coordinate_frame: String,
    pub higher_is_more_anomalous: bool,
    pub tensor: TensorSpec,
}

#[derive(Debug, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: String,
    pub layout: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ScoreContract {
    pub kind: String,
    pub percentile: Option<f64>,
    pub top_k: Option<usize>,
    pub tensor: Option<ScalarTensorSpec>,
}

#[derive(Debug, Deserialize)]
pub struct ScalarTensorSpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct OperatingPoint {
    pub rule: String,
    pub value: f32,
    pub subset: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct FileDigest {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
pub struct ParityFixture {
    pub input_path: String,
    pub expected_map_path: String,
    pub expected_score: f32,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
}

#[derive(Debug)]
pub struct Bundle {
    root: PathBuf,
    pub manifest: Manifest,
}

#[derive(Debug, Serialize)]
pub struct InferenceReport {
    pub status: &'static str,
    pub provider: &'static str,
    pub score: f32,
    pub operating_point: Option<f32>,
    pub predicted_defect: Option<bool>,
    pub inference_ms: f64,
    pub map_elements: usize,
    pub map_min: f32,
    pub map_max: f32,
}

#[derive(Debug, Serialize)]
pub struct VerificationReport {
    pub status: &'static str,
    pub format_version: u32,
    pub provider: &'static str,
    pub checked_files: usize,
    pub map_max_absolute_error: f32,
    pub score_absolute_error: f32,
    pub inference_ms: f64,
}

impl Bundle {
    /// Read a bundle, validate its version and semantics, and verify every payload hash.
    ///
    /// # Errors
    ///
    /// Returns an error for an unreadable or unsupported manifest, unsafe payload path,
    /// symlink, missing file, byte-count mismatch, or SHA-256 mismatch.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        ensure!(
            root.is_dir(),
            "bundle root is not a directory: {}",
            root.display()
        );
        let root = root
            .canonicalize()
            .with_context(|| format!("canonicalize bundle root {}", root.display()))?;
        let manifest_path = root.join("manifest.json");
        reject_symlink(&manifest_path)?;
        let manifest: Manifest = serde_json::from_slice(
            &fs::read(&manifest_path)
                .with_context(|| format!("read {}", manifest_path.display()))?,
        )
        .context("parse manifest.json")?;
        validate_manifest(&manifest)?;

        let mut seen = HashSet::new();
        for item in &manifest.files {
            ensure!(
                seen.insert(&item.path),
                "duplicate payload path {}",
                item.path
            );
            let path = resolve_payload(&root, &item.path)?;
            reject_symlink(&path)?;
            let content = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            ensure!(
                u64::try_from(content.len())? == item.bytes,
                "byte count mismatch for {}: manifest {}, actual {}",
                item.path,
                item.bytes,
                content.len()
            );
            ensure!(
                sha256(&content) == item.sha256,
                "SHA-256 mismatch for {}",
                item.path
            );
        }
        for required in [
            &manifest.graph_path,
            &manifest.parity.input_path,
            &manifest.parity.expected_map_path,
        ] {
            ensure!(
                seen.contains(required),
                "required payload {required} is not checksummed"
            );
        }
        Ok(Self { root, manifest })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Execute one already-prepared little-endian NCHW float32 tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the input cannot be read or violates the manifest, or
    /// when ONNX Runtime cannot load or execute the graph.
    pub fn infer_file(&self, input_path: impl AsRef<Path>) -> Result<(InferenceReport, Vec<f32>)> {
        let input = read_f32(
            input_path.as_ref(),
            element_count(&self.manifest.input.tensor.shape)?,
        )?;
        self.infer(&input)
    }

    /// Execute one already-prepared NCHW float32 tensor held in memory.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input values or shape, graph/runtime failures,
    /// output-contract violations, or an unsupported score reducer.
    pub fn infer(&self, input: &[f32]) -> Result<(InferenceReport, Vec<f32>)> {
        let expected_input = element_count(&self.manifest.input.tensor.shape)?;
        ensure!(
            input.len() == expected_input,
            "input has {} float32 values; expected {expected_input}",
            input.len()
        );
        ensure!(
            input.iter().all(|value| value.is_finite()),
            "input contains a non-finite value"
        );
        let shape: Vec<i64> = self
            .manifest
            .input
            .tensor
            .shape
            .iter()
            .map(|&dimension| i64::try_from(dimension))
            .collect::<std::result::Result<_, _>>()?;
        let graph = resolve_payload(&self.root, &self.manifest.graph_path)?;
        let mut session = Session::builder()?
            // `ort`'s pre-2.0 builder error owns the non-Send builder. Convert it to
            // text at this boundary instead of trying to store it in anyhow's error.
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .commit_from_file(graph)
            .context("load ONNX graph")?;
        let tensor = TensorRef::from_array_view((shape, input))?;
        let started = Instant::now();
        let outputs = session.run(inputs![self.manifest.input.tensor.name.as_str() => tensor])?;
        let output = outputs
            .get(self.manifest.anomaly_map.tensor.name.as_str())
            .context("graph did not return the declared anomaly-map tensor")?;
        let (actual_shape, map) = output.try_extract_tensor::<f32>()?;
        let expected_output_shape: Vec<i64> = self
            .manifest
            .anomaly_map
            .tensor
            .shape
            .iter()
            .map(|&dimension| i64::try_from(dimension))
            .collect::<std::result::Result<_, _>>()?;
        ensure!(
            actual_shape.as_ref() == expected_output_shape,
            "anomaly-map shape {actual_shape:?} does not match manifest {:?}",
            self.manifest.anomaly_map.tensor.shape
        );
        ensure!(!map.is_empty(), "anomaly map is empty");
        ensure!(
            map.iter().all(|value| value.is_finite()),
            "anomaly map contains a non-finite value"
        );
        let map = map.to_vec();
        let score = self.resolve_score(&outputs, &map)?;
        let operating_point = self
            .manifest
            .operating_point
            .as_ref()
            .map(|point| point.value);
        let report = InferenceReport {
            status: "ok",
            provider: "CPUExecutionProvider",
            score,
            operating_point,
            predicted_defect: operating_point.map(|threshold| score >= threshold),
            inference_ms: started.elapsed().as_secs_f64() * 1_000.0,
            map_elements: map.len(),
            map_min: map.iter().copied().fold(f32::INFINITY, f32::min),
            map_max: map.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        };
        Ok((report, map))
    }

    fn resolve_score(&self, outputs: &SessionOutputs<'_>, map: &[f32]) -> Result<f32> {
        match self.manifest.score.kind.as_str() {
            "percentile_linear" => percentile_linear(
                map,
                self.manifest
                    .score
                    .percentile
                    .context("percentile score contract has no percentile")?,
            ),
            "max" => Ok(map.iter().copied().fold(f32::NEG_INFINITY, f32::max)),
            "top_k_mean" => top_k_mean(
                map,
                self.manifest
                    .score
                    .top_k
                    .context("top-k score contract has no top_k")?,
            ),
            "tensor" => {
                let tensor = self
                    .manifest
                    .score
                    .tensor
                    .as_ref()
                    .context("tensor score contract has no tensor")?;
                let output = outputs
                    .get(tensor.name.as_str())
                    .context("graph did not return the declared score tensor")?;
                let (shape, values) = output.try_extract_tensor::<f32>()?;
                let expected: Vec<i64> = tensor
                    .shape
                    .iter()
                    .map(|&dimension| i64::try_from(dimension))
                    .collect::<std::result::Result<_, _>>()?;
                ensure!(
                    shape.as_ref() == expected,
                    "score shape {shape:?} does not match manifest {:?}",
                    tensor.shape
                );
                ensure!(values.len() == 1, "score tensor must contain one value");
                let score = values[0];
                ensure!(
                    score.is_finite(),
                    "score tensor contains a non-finite value"
                );
                Ok(score)
            }
            kind => bail!("unsupported score contract {kind}"),
        }
    }

    /// Run the bundled deterministic fixture and enforce its published tolerances.
    ///
    /// # Errors
    ///
    /// Returns an error when inference fails or any anomaly-map or score value exceeds
    /// the manifest's parity tolerance.
    pub fn verify_fixture(&self) -> Result<VerificationReport> {
        let input = resolve_payload(&self.root, &self.manifest.parity.input_path)?;
        let expected_map_path =
            resolve_payload(&self.root, &self.manifest.parity.expected_map_path)?;
        let expected_map = read_f32(
            &expected_map_path,
            element_count(&self.manifest.anomaly_map.tensor.shape)?,
        )?;
        let (report, actual_map) = self.infer_file(input)?;
        let map_max_absolute_error = actual_map
            .iter()
            .zip(&expected_map)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        for (index, (actual, expected)) in actual_map.iter().zip(&expected_map).enumerate() {
            let allowed = self.manifest.parity.absolute_tolerance
                + self.manifest.parity.relative_tolerance * expected.abs();
            ensure!(
                (actual - expected).abs() <= allowed,
                "fixture map parity failed at element {index}: actual {actual}, expected {expected}, allowed {allowed}"
            );
        }
        let score_absolute_error = (report.score - self.manifest.parity.expected_score).abs();
        ensure!(
            score_absolute_error <= self.manifest.parity.absolute_tolerance,
            "fixture score parity failed: actual {}, expected {}, tolerance {}",
            report.score,
            self.manifest.parity.expected_score,
            self.manifest.parity.absolute_tolerance
        );
        Ok(VerificationReport {
            status: "ok",
            format_version: self.manifest.format_version,
            provider: report.provider,
            checked_files: self.manifest.files.len(),
            map_max_absolute_error,
            score_absolute_error,
            inference_ms: report.inference_ms,
        })
    }
}

/// Write contiguous little-endian float32 values.
///
/// # Errors
///
/// Returns an error when the destination cannot be written.
pub fn write_f32(path: impl AsRef<Path>, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path.as_ref(), bytes).with_context(|| format!("write {}", path.as_ref().display()))
}

#[allow(clippy::float_cmp)] // The bundle contract names the exact canonical range [0, 1].
fn validate_manifest(manifest: &Manifest) -> Result<()> {
    ensure!(
        (MIN_FORMAT_VERSION..=MAX_FORMAT_VERSION).contains(&manifest.format_version),
        "unsupported bundle format version {}; runner supports {} through {}",
        manifest.format_version,
        MIN_FORMAT_VERSION,
        MAX_FORMAT_VERSION
    );
    ensure!(
        manifest.portable_format == "onnx",
        "portable_format must be onnx"
    );
    for (name, tensor) in [
        ("input", &manifest.input.tensor),
        ("anomaly_map", &manifest.anomaly_map.tensor),
    ] {
        ensure!(tensor.dtype == "float32", "{name} dtype must be float32");
        ensure!(tensor.layout == "NCHW", "{name} layout must be NCHW");
        ensure!(tensor.shape.len() == 4, "{name} must have four dimensions");
        ensure!(
            tensor.shape.iter().all(|&dimension| dimension > 0),
            "{name} has a zero dimension"
        );
        ensure!(tensor.shape[0] == 1, "{name} batch dimension must be one");
        ensure!(!tensor.name.is_empty(), "{name} tensor name is empty");
    }
    ensure!(
        manifest.anomaly_map.tensor.shape[1] == 1,
        "anomaly map must have one plane"
    );
    ensure!(
        manifest.input.tensor.shape[2..] == manifest.anomaly_map.tensor.shape[2..],
        "input and anomaly map spatial dimensions differ"
    );
    ensure!(
        manifest.input.coordinate_frame == "prepared",
        "input frame must be prepared"
    );
    ensure!(
        manifest.anomaly_map.coordinate_frame == "prepared",
        "anomaly-map frame must be prepared"
    );
    ensure!(
        manifest.anomaly_map.higher_is_more_anomalous,
        "unsupported score direction"
    );
    ensure!(
        manifest.input.value_min == 0.0 && manifest.input.value_max == 1.0,
        "input range must be [0, 1]"
    );
    let expected_channels = match manifest.input.color.as_str() {
        "rgb" => 3,
        "grayscale" => 1,
        color => bail!("unsupported input color mode {color}"),
    };
    ensure!(
        manifest.input.tensor.shape[1] == expected_channels,
        "input color and channel count disagree"
    );
    match manifest.score.kind.as_str() {
        "percentile_linear" => ensure!(
            (0.0..=100.0).contains(
                &manifest
                    .score
                    .percentile
                    .context("percentile score contract has no percentile")?
            ),
            "score percentile is outside [0, 100]"
        ),
        "max" => {}
        "top_k_mean" => ensure!(
            manifest
                .score
                .top_k
                .context("top-k score contract has no top_k")?
                > 0,
            "score top_k must be positive"
        ),
        "tensor" => {
            let tensor = manifest
                .score
                .tensor
                .as_ref()
                .context("tensor score contract has no tensor")?;
            ensure!(tensor.dtype == "float32", "score dtype must be float32");
            ensure!(tensor.shape == [1], "score tensor shape must be [1]");
            ensure!(!tensor.name.is_empty(), "score tensor name is empty");
        }
        kind => bail!("unsupported score contract {kind}"),
    }
    ensure!(
        manifest.parity.absolute_tolerance > 0.0,
        "absolute tolerance must be positive"
    );
    ensure!(
        manifest.parity.relative_tolerance >= 0.0,
        "relative tolerance must be non-negative"
    );
    Ok(())
}

fn resolve_payload(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    ensure!(!relative.is_empty(), "bundle payload path is empty");
    ensure!(
        relative_path
            .components()
            .all(|component| matches!(component, Component::Normal(_))),
        "unsafe bundle payload path {relative}"
    );
    Ok(root.join(relative_path))
}

fn reject_symlink(path: &Path) -> Result<()> {
    let metadata =
        fs::symlink_metadata(path).with_context(|| format!("inspect {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file(),
        "payload is not a regular file: {}",
        path.display()
    );
    ensure!(
        !metadata.file_type().is_symlink(),
        "payload is a symlink: {}",
        path.display()
    );
    Ok(())
}

fn sha256(content: &[u8]) -> String {
    format!("{:x}", Sha256::digest(content))
}

fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .context("tensor element count overflows usize")
    })
}

fn read_f32(path: &Path, expected: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let expected_bytes = expected
        .checked_mul(size_of::<f32>())
        .context("tensor byte count overflows usize")?;
    ensure!(
        bytes.len() == expected_bytes,
        "{} has {} bytes; expected {}",
        path.display(),
        bytes.len(),
        expected_bytes
    );
    Ok(bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn percentile_linear(values: &[f32], percentile: f64) -> Result<f32> {
    ensure!(!values.is_empty(), "cannot reduce an empty anomaly map");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "cannot reduce non-finite values"
    );
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let position = (sorted.len() - 1) as f64 * percentile / 100.0;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = (position - lower as f64) as f32;
    Ok(sorted[lower] + (sorted[upper] - sorted[lower]) * fraction)
}

#[allow(clippy::cast_precision_loss)]
fn top_k_mean(values: &[f32], top_k: usize) -> Result<f32> {
    ensure!(!values.is_empty(), "cannot reduce an empty anomaly map");
    ensure!(top_k > 0, "top_k must be positive");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "cannot reduce non-finite values"
    );
    let count = top_k.min(values.len());
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    Ok(sorted[sorted.len() - count..].iter().sum::<f32>() / count as f32)
}

#[cfg(test)]
mod tests {
    use super::{percentile_linear, resolve_payload, top_k_mean};
    use std::path::Path;

    #[test]
    fn linear_percentile_matches_numpy_definition() {
        let values = [0.0, 10.0, 20.0, 30.0];
        for (percentile, expected) in [(0.0, 0.0), (50.0, 15.0), (95.0, 28.5), (100.0, 30.0)] {
            let actual = percentile_linear(&values, percentile).unwrap();
            assert!((actual - expected).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn bundle_paths_are_posix_relative() {
        let root = Path::new("bundle");
        assert!(resolve_payload(root, "fixture/input.f32").is_ok());
        assert!(resolve_payload(root, "../secret").is_err());
        assert!(resolve_payload(root, "/tmp/model.onnx").is_err());
    }

    #[test]
    fn top_k_mean_caps_at_the_map_size() {
        assert!((top_k_mean(&[1.0, 4.0, 2.0, 3.0], 2).unwrap() - 3.5).abs() < f32::EPSILON);
        assert!((top_k_mean(&[1.0, 3.0], 20).unwrap() - 2.0).abs() < f32::EPSILON);
        assert!(top_k_mean(&[1.0], 0).is_err());
    }
}
