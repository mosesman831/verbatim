# @verbatim/memory-client

Thin, zero-dependency TypeScript client for the Verbatim `/v2/memory` HTTP API
(`docs/v6_contracts.md` §4). Global `fetch` only — Node ≥ 18 or any modern
browser. Every result type mirrors the service's JSON shapes verbatim.

## Usage

```ts
import { MemoryClient } from "@verbatim/memory-client";

const mem = new MemoryClient({
  baseUrl: "http://127.0.0.1:8390",
  token: process.env.VERBATIM_TOKEN!,
  timeoutMs: 5000,
});

const added = await mem.add("the release tag is v6.2", {
  metadata: { topic: "release" },
});
const ready = await mem.waitReady(added.receipt_id, 2000);

const res = await mem.search("what is the release tag?", { limit: 5 });
for (const hit of res.items) console.log(hit.quote, hit.ref);

const detail = await mem.inspect(res.items[0].ref);
const preview = await mem.forget(res.items[0].ref, { preview: true });
await mem.forget(res.items[0].ref, { confirm_token: preview.confirmation_token });

const status = await mem.status();
await mem.capabilities();
```

## API

| Method | Route |
| --- | --- |
| `add(text, opts?)` | `POST /v2/memory/add` |
| `search(query, opts?)` | `POST /v2/memory/search` |
| `inspect(ref, opts?)` | `POST /v2/memory/inspect` |
| `forget(ref, opts?)` | `POST /v2/memory/forget` |
| `status()` | `GET /v2/memory/status` |
| `readiness(receiptId)` | `GET /v2/memory/readiness/{receipt_id}` |
| `capabilities()` | `GET /v2/memory/capabilities` |
| `waitReady(receiptId, timeoutMs?, pollMs?)` | polls readiness until non-`pending` |

Non-2xx responses throw `MemoryApiError` carrying the service's
`{error, code, retryable}` shape. `waitReady` returns the last snapshot on
deadline — the service stays the authority on state.

## Serving the API

```bash
verbatim-service --path store.vdb --user me --token "$VERBATIM_TOKEN"
```

## Build

`main`/`types` point at `src/index.ts` directly (Node ≥ 22.6 strips the
types natively; bundlers and `ts-node` handle it too). `npm run build`
emits compiled JS + `.d.ts` to `dist/` when `tsc` is available.
