/**
 * The Prepare tab's URL state: the open revision, and the job it is following.
 *
 * `jobProfile` is the revision the job was *started* for, which is not always the open one
 * — a reader can start a build and then look at another revision — and a preview result is
 * only drawn over the revision it describes.
 */

export type PrepareJobMode = "preview" | "build" | "asset";

export interface PrepareState {
  profile?: number;
  job?: number;
  jobProfile?: number;
  mode?: PrepareJobMode;
}

const MODES: readonly PrepareJobMode[] = ["preview", "build", "asset"];

export function readPrepareState(params: URLSearchParams): PrepareState {
  const mode = MODES.find((value) => value === params.get("mode"));
  const job = positive(params.get("job"));
  return {
    profile: positive(params.get("profile")),
    // A job without its mode cannot be drawn correctly, so the pair is read together.
    job: mode === undefined ? undefined : job,
    jobProfile: mode === undefined ? undefined : positive(params.get("jobProfile")),
    mode: job === undefined ? undefined : mode,
  };
}

export function writePrepareState(state: PrepareState): URLSearchParams {
  const params = new URLSearchParams();
  if (state.profile !== undefined) params.set("profile", String(state.profile));
  if (state.job !== undefined && state.mode !== undefined) {
    params.set("job", String(state.job));
    params.set("mode", state.mode);
    if (state.jobProfile !== undefined) params.set("jobProfile", String(state.jobProfile));
  }
  return params;
}

function positive(raw: string | null): number | undefined {
  const value = Number(raw);
  return raw !== null && Number.isInteger(value) && value > 0 ? value : undefined;
}
