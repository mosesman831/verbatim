/**
 * @verbatim/memory-client — thin client for the Verbatim `/v2/memory` API.
 *
 * Zero dependencies: global `fetch` only (Node >= 18 / browsers). Every
 * method maps one route from docs/v6_contracts.md §4; every result type
 * mirrors the JSON shape the service serializes (enums arrive as their
 * `.value` strings; unknown/extra fields are preserved by the service,
 * not by these interfaces).
 *
 *   const mem = new MemoryClient({ baseUrl: "http://127.0.0.1:8390", token });
 *   const add = await mem.add("the release tag is v6.2");
 *   const ready = await mem.waitReady(add.receipt_id);
 *   const res = await mem.search("what is the release tag?");
 */

/** Options accepted by {@link MemoryClient}'s constructor. */
export interface MemoryClientOptions {
  /** Base URL of the service, e.g. "http://127.0.0.1:8390". */
  baseUrl: string;
  /** Bearer token provisioned for this principal. */
  token: string;
  /** Per-request timeout in milliseconds (default 5000). */
  timeoutMs?: number;
}

/** Error thrown for non-2xx responses — carries the service error shape. */
export class MemoryApiError extends Error {
  status: number;
  code: string;
  retryable: boolean;

  constructor(message: string, status: number, code: string, retryable: boolean) {
    super(message);
    this.name = "MemoryApiError";
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }
}

/** Advisory possible_update on an add — nothing mutates until replaces=. */
export interface UpdateCandidate {
  schema: string;
  /** Serialized MemoryRef of the prior record. */
  ref: string;
  /** contradicts | newer_value | negates | refines */
  relation: string;
  reason: string;
  score: number;
}

/** POST /v2/memory/add response. */
export interface AddResult {
  schema: string;
  memory_id: string;
  /** Serialized MemoryRef — the handle inspect/forget accept. */
  ref: string;
  source_revision: number;
  /** Pass to waitReady / the readiness route. */
  receipt_id: string;
  /** accepted | replayed (idempotent replay returns the stored result). */
  acceptance: string;
  replayed: boolean;
  /** Per-capability readiness snapshot at return time. */
  readiness: Record<string, string>;
  warnings: string[];
  possible_updates: UpdateCandidate[];
  /** not_requested | queued | deferred | unavailable */
  inference: string;
}

/** One recalled item inside {@link SearchResult.items}. */
export interface Hit {
  schema: string;
  memory_id: string;
  /** MemoryRef for source-backed hits. */
  ref: string;
  /** Exact claim/view object ref. */
  object_ref: string;
  /** source | claim | view | card */
  kind: string;
  /** Byte-exact evidence quote. */
  quote: string;
  score: number;
  score_family: string;
  /** active | superseded | corrected | archived | … */
  lifecycle: string;
  /** supported | contested | unassessed | … */
  support_status: string;
  /** supporting | contrary | context */
  role: string;
  type: string;
  valid_time: string | null;
  recorded_time: string | null;
  collapsed_duplicates: number;
  corroboration: number;
  warnings: string[];
  score_detail: Record<string, number>;
}

/** POST /v2/memory/search response. */
export interface SearchResult {
  schema: string;
  /** ready | partial | pending | blocked | unavailable */
  status: string;
  items: Hit[];
  warnings: string[];
  readiness: Record<string, unknown>;
  coverage: Record<string, unknown>;
  /** Pass back as `after` to continue the causal session. */
  causal_token: string;
}

/** POST /v2/memory/inspect response — provenance for one ref. */
export interface InspectResult {
  schema: string;
  ref: string;
  found: boolean;
  detail: string;
  provenance: Record<string, unknown>;
  revisions: Record<string, unknown>[];
  lifecycle: Record<string, unknown>;
  evidence: Record<string, unknown>[];
  enrichment: Record<string, unknown>;
  warnings: string[];
}

/** POST /v2/memory/forget response. */
export interface ForgetResult {
  schema: string;
  /** preview | operation */
  mode: string;
  mutated: boolean;
  /** Preview mode: pass back as confirm_token to execute. */
  confirmation_token: string;
  selection: string[];
  suppression_state: string;
  closure_state: string;
  receipt_id: string;
  warnings: string[];
}

/** GET /v2/memory/readiness/{receipt_id} response. */
export interface ReadinessResult {
  schema: string;
  receipt_id: string;
  /** ready | partial | pending | blocked | unavailable */
  state: string;
  capabilities: Record<string, string>;
  causal_satisfied: boolean;
  waited_ms: number;
}

/** GET /v2/memory/status response — honest operational snapshot. */
export interface StatusResult {
  schema: string;
  profile: string;
  store_tag: string;
  namespace: string;
  caller: string;
  worker: Record<string, unknown>;
  encoder: string;
  cache: Record<string, unknown>;
  readiness_counts: Record<string, number>;
  capabilities: Record<string, string>;
  warnings: string[];
}

/** GET /v2/memory/capabilities response. */
export interface CapabilitiesResult {
  [capability: string]: unknown;
}

/** Options for {@link MemoryClient.add} (request body fields only). */
export interface AddOptions {
  infer?: boolean;
  metadata?: Record<string, unknown>;
  /** Serialized MemoryRef this record replaces. */
  replaces?: string;
  idempotency_key?: string;
}

/** Options for {@link MemoryClient.search} (request body fields only). */
export interface SearchOptions {
  limit?: number;
  /** session | eventual | … */
  consistency?: string;
  timeout_ms?: number;
  ready_timeout_ms?: number;
  /** causal_token from a prior SearchResult — continues the session. */
  after?: string;
}

/** Options for {@link MemoryClient.inspect}. */
export interface InspectOptions {
  /** evidence | … — detail level the service honors. */
  detail?: string;
}

/** Options for {@link MemoryClient.forget}. */
export interface ForgetOptions {
  /** Token from a preview ForgetResult — executes the pinned selection. */
  confirm_token?: string;
  /** Ask for a preview instead of executing. */
  preview?: boolean;
}

const DEFAULT_TIMEOUT_MS = 5000;
const DEFAULT_READY_TIMEOUT_MS = 2000;
const DEFAULT_POLL_MS = 100;
const TERMINAL_READINESS = new Set(["ready", "partial", "blocked", "unavailable"]);

/**
 * Bearer-authenticated client for the `/v2/memory` routes. Holds no state
 * beyond the constructor options; safe to share across concurrent calls.
 */
export class MemoryClient {
  private readonly baseUrl: string;
  private readonly token: string;
  private readonly timeoutMs: number;

  constructor(opts: MemoryClientOptions) {
    if (!opts || typeof opts.baseUrl !== "string" || !opts.baseUrl) {
      throw new TypeError("MemoryClient requires a baseUrl");
    }
    if (typeof opts.token !== "string" || !opts.token) {
      throw new TypeError("MemoryClient requires a bearer token");
    }
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.token = opts.token;
    this.timeoutMs =
      typeof opts.timeoutMs === "number" && opts.timeoutMs > 0
        ? opts.timeoutMs
        : DEFAULT_TIMEOUT_MS;
  }

  /**
   * Evidence-preserving add. Returns a receipt — the record is durable at
   * return; readiness of derived lanes is reported via waitReady.
   */
  add(text: string, opts?: AddOptions): Promise<AddResult> {
    const body: Record<string, unknown> = { text };
    if (opts) {
      for (const key of ["infer", "metadata", "replaces", "idempotency_key"] as const) {
        if (opts[key] !== undefined) body[key] = opts[key];
      }
    }
    return this.request("POST", "/v2/memory/add", body);
  }

  /** Recall with provenance. `after` continues a causal session. */
  search(query: string, opts?: SearchOptions): Promise<SearchResult> {
    const body: Record<string, unknown> = { query };
    if (opts) {
      for (const key of [
        "limit",
        "consistency",
        "timeout_ms",
        "ready_timeout_ms",
        "after",
      ] as const) {
        if (opts[key] !== undefined) body[key] = opts[key];
      }
    }
    return this.request("POST", "/v2/memory/search", body);
  }

  /** Explain one memory: provenance, revisions, evidence spans. */
  inspect(ref: string, opts?: InspectOptions): Promise<InspectResult> {
    const body: Record<string, unknown> = { ref };
    if (opts?.detail !== undefined) body.detail = opts.detail;
    return this.request("POST", "/v2/memory/inspect", body);
  }

  /**
   * Targeted forget. Pass `{ preview: true }` for a confirmation-bound
   * preview, then `{ confirm_token }` to execute exactly that selection.
   */
  forget(ref: string, opts?: ForgetOptions): Promise<ForgetResult> {
    const body: Record<string, unknown> = { ref };
    if (opts?.confirm_token !== undefined) body.confirm_token = opts.confirm_token;
    if (opts?.preview !== undefined) body.preview = opts.preview;
    return this.request("POST", "/v2/memory/forget", body);
  }

  /** Honest operational snapshot. */
  status(): Promise<StatusResult> {
    return this.request("GET", "/v2/memory/status");
  }

  /** One readiness snapshot for a receipt. */
  readiness(receiptId: string): Promise<ReadinessResult> {
    return this.request(
      "GET",
      `/v2/memory/readiness/${encodeURIComponent(receiptId)}`,
    );
  }

  /** Honest capability dict — deferred/unavailable lanes say so. */
  capabilities(): Promise<CapabilitiesResult> {
    return this.request("GET", "/v2/memory/capabilities");
  }

  /**
   * Bounded readiness wait: polls the readiness route until the receipt
   * leaves `pending`, then returns the final snapshot. A deadline expiry
   * returns the last observed snapshot (still `pending`) rather than
   * throwing — the service stays the authority on state.
   */
  async waitReady(
    receiptId: string,
    timeoutMs: number = DEFAULT_READY_TIMEOUT_MS,
    pollIntervalMs: number = DEFAULT_POLL_MS,
  ): Promise<ReadinessResult> {
    const deadline = Date.now() + timeoutMs;
    let last: ReadinessResult | undefined;
    for (;;) {
      last = await this.readiness(receiptId);
      if (TERMINAL_READINESS.has(last.state) || Date.now() >= deadline) {
        return last;
      }
      await new Promise((resolve) =>
        setTimeout(resolve, Math.min(pollIntervalMs, Math.max(1, deadline - Date.now()))),
      );
    }
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const init: RequestInit = {
      method,
      headers: {
        authorization: `Bearer ${this.token}`,
        accept: "application/json",
      },
      signal: AbortSignal.timeout(this.timeoutMs),
    };
    if (body !== undefined) {
      (init.headers as Record<string, string>)["content-type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    const resp = await fetch(`${this.baseUrl}${path}`, init);
    const text = await resp.text();
    let parsed: unknown;
    try {
      parsed = text ? JSON.parse(text) : undefined;
    } catch {
      parsed = undefined;
    }
    if (!resp.ok) {
      const err = (parsed ?? {}) as Record<string, unknown>;
      throw new MemoryApiError(
        typeof err.error === "string"
          ? err.error
          : `HTTP ${resp.status} from ${method} ${path}`,
        resp.status,
        typeof err.code === "string" ? err.code : "http_error",
        err.retryable === true,
      );
    }
    return parsed as T;
  }
}

export default MemoryClient;
