# ADR-0023: Raw values are served beside the rendered picture, for reading and not for drawing

**Status:** Accepted (2026-08-08)

Amends **ADR-0019**, which ruled out "returning raw `.npy` for the client to decode". That reasoning
still holds and is not reversed: the colormap and the display range stay server-side, in one
language, and no picture is ever drawn in the browser from these bytes. What changes is that
*reading a number* turns out not to be *drawing a picture*, and ADR-0019 conflated them because at
the time nothing needed the number.

## Context

The result viewer draws a photograph with an anomaly map over it and prints the map's range beneath.
Everything a reader can learn from it is qualitative: this region is hotter than that one. The
obvious next question at any pixel — *how hot, in the units the metrics are computed in* — had no
answer anywhere in the application.

Three ways to answer it were considered.

**Invert the rendered PNG.** Sample the canvas under the cursor and map the colour back to a value.
This is the option the codebase's own discipline forbids: ADR-0007 says nothing about colormap,
normalization or blending is baked into stored data, and reading numbers out of a picture is that
rule run backwards. It is also not a function. The LUT is 256 quantized entries interpolated from
twelve anchors (`media/overlay.py`), values clip at both ends so every value above the range maps to
one colour, the PNG is bilinearly resampled to the source grid when `native=true`
(`api/routers/images.py`), and alpha is a gamma of the normalized value. The inverse is multi-valued,
lossy, and reports a *display* quantity rather than the model's number.

**A point-query endpoint.** `GET .../value?x=&y=`. One HTTP round trip per `pointermove`, which needs
debouncing, gives a readout that lags the cursor, and cannot show anything at all while panning.

**Fetch the plane once and index it in the browser.** At the default 256×256 an anomaly map is 256 KB
of float32 — one fetch per image, revalidated by ETag exactly as the PNG already is, and then every
readout is an array index with no network at all.

The third is chosen. What made ADR-0019 refuse it was the prospect of a numpy `.npy` parser in
TypeScript and a colormap living in two languages. Neither follows from serving the numbers.

## Decision

**A `map`-kind array is available as a fixed-header float32 blob, for a numeric readout only.**

- **The format is not `.npy`.** A 24-byte header — ASCII `VAM1`, then `width`, `height`, `stride`,
  `channels` and one reserved word as little-endian `uint32` — followed by `channels` plane-major
  blocks of `width × height` little-endian float32. There is no dtype string, no shape tuple, no
  pickle path, and therefore no numpy reader in TypeScript: the decoder is a `DataView` and a loop.
- **`channels` is in the header, so nothing has to know it in advance.** A preprocessed source is
  `(H, W, C)` where `C` follows the experiment's colour mode. A client that had to know `C` before
  asking would either encode it in the UI — which the channel-count rule forbids — or discover it by
  requesting planes until one 404s. One request returns them all and says how many there were.
- **It is not a rendering path, and the API says so.** The colormap, the alpha rule and the run-wide
  display range stay entirely server-side. Nothing in the frontend may draw a picture from these
  bytes; a second renderer would be exactly the two-languages problem ADR-0019 refused.
- **Bounded by an integer `stride` reported in the header.** Above roughly 4 MB the plane is decimated
  by an integer factor and the header says which, so a reader can tell an exact value from a sampled
  one. 256×256 — the default, and EfficientAD's design point — is always stride 1.
- **Three planes are served, by the routes that already own their addressing.** The stored anomaly map
  (`GET /api/images/{id}/anomaly-map/values`, resolved through `image_result` like its PNG); any
  diagnostic array (`format=raw` on the existing payload route, resolved through the index, so
  ADR-0019's "path traversal is impossible by construction" is inherited rather than restated); and
  the **preprocessed source** the model actually saw
  (`GET /api/experiments/{id}/images/{image_id}/source-values`).
- **The source planes are the preprocessed array, not the display image.** They are what
  `load_array` produced for the model, at the experiment's own size and colour mode, so a
  per-channel readout is the number the method consumed rather than an 8-bit preview of it. The ETag
  covers the image's `sha256` and the frozen preprocessing config, so it is fetched once per image
  and never again.
- **`graph` and `table` stay refused**, with the same 400 as before. Their payload is inline.
- **The payload ETag now covers the display range.** It hashed size and mtime only, so anything that
  changed a key's range left every already-fetched PNG cached at the old scale. This was reachable
  before and is more reachable now.

**Ruled out:** inverting the rendered colour, for the reasons above; a point-query endpoint (a
readout that lags the pointer and vanishes while panning); and serving the *display* tier's pixels as
the "source" values, which would report what the browser is showing rather than what the model read.

## Consequences

The hover readout generalizes for free. Because a diagnostic array is addressed the same way, the
per-branch panes inherit the same readout with no code written per method — which is the property
ADR-0018 exists to produce, arriving here without being asked for.

Negative consequences, accepted honestly:

- **A second wire format.** Small and versioned by its magic, but it is a format, and the next person
  to add a plane has to know it exists. It is one function on each side.
- **256 KB per plane.** A sample with three acquisition channels and two per-branch maps is a handful
  of requests to be fully readable, and an RGB source is three planes inside one of them. On a
  loopback interface to a local sidecar this is cheap; over anything else it would not be, and
  nothing in the design would notice.
- **The stride cap makes some readouts approximate**, and a reader who does not look at the header
  will not know. The header carries it and the UI prints it; that is the whole mitigation.
- **The browser now holds a copy of the numbers**, so a future change to how a map is written must
  invalidate these responses as well as the PNG. Both hang off the same file's size and mtime, which
  is why the ETag fix above belongs in this record rather than beside it.
