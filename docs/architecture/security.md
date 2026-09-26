# Security and privacy posture

This is a single-user, local research tool, and the security model is stated plainly:

- **Local-only binding.** The FastAPI sidecar binds `127.0.0.1` exclusively — never `0.0.0.0`. It is not
  reachable from the network, and the port is ephemeral in packaged builds.
- **No authentication, by design.** Tokens on a loopback-only single-user process would add ceremony
  without changing the threat model (ADR-0003). The API must not be exposed beyond loopback without
  revisiting this.
- **Private data contract (ADR-0022).** Source images never enter the repository or leave the machine:
  - they live **outside the working tree**, so `git add -A` cannot reach them;
  - `*.bmp` / `*.BMP`, `data/` and `privatedata/` are gitignored;
  - images are **referenced in place by absolute path**, never copied into a tracked directory;
  - `scripts/check-repo-safety.sh` is a pre-push guard that fails if anything private is staged;
  - nothing in the backend uploads, telemeters or phones home; model weights are downloaded from public
    sources only, and only on explicit user action.
- **Path handling.** Import roots come from the user via a native picker. Media endpoints serve files only
  by `image_id` through the database — never by a client-supplied path — so no request can read an
  arbitrary file.
- **Destructive path handling.** Experiment deletion removes artifacts only when the stored path is exactly
  `data/artifacts/exp-<id>`; region profile deletion applies the same rule to
  `data/region-profiles/profile-<id>`, every size's build beneath it. Dataset deletion composes those directories with exact image-id
  thumbnail keys and its accepted `data/manifests/dataset-<id>-*.json`; a corrupt or edited path, or a
  symlink, is refused with 409. Neither preview nor cleanup follows symlinks, and API tests keep an external
  synthetic source sentinel intact across the full row-and-storage cascade.
- **No sandboxing claims.** Job workers run with the user's own privileges. Datasets and model
  configurations are trusted local input; the tool is not designed to run untrusted content.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
