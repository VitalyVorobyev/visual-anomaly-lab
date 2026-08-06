/**
 * The typed HTTP client.
 *
 * `generated.ts` is produced from the backend's OpenAPI schema by
 * `scripts/gen-api-types.sh` and committed, so `tsc` and CI work without a running
 * backend. Regenerate it whenever a route or schema changes; the diff *is* the API
 * contract changing.
 */

import createClient from "openapi-fetch";

import { resolveApiBaseUrl } from "./baseUrl";
import type { paths } from "./generated";

export const apiBaseUrl = resolveApiBaseUrl();

export const api = createClient<paths>({ baseUrl: apiBaseUrl });

export type HealthResponse = paths["/api/health"]["get"]["responses"][200]["content"]["application/json"];
