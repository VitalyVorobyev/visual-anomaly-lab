/**
 * How a byte count reads on screen, everywhere.
 *
 * Four screens used to carry their own copy — two printing `KB`, one `KiB`, one rounding to
 * whole numbers — so the same artifact directory was "64.7 MiB" on Prepare and "67.8 MB"
 * one screen over, both computed by dividing by 1024. The units are binary because the
 * divisor is, and one function means one answer.
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`;
}
