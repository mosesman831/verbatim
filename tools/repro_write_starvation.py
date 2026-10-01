"""Write-starvation repro: managed worker draining vs foreground adds.

Memory(worker="managed"), 64 heavy source adds enqueue a deep backlog;
a foreground thread then runs an add/search burst while the worker
drains. Before the fix this produced "begin transaction: database is
locked" on the facade (3 lock failures in one measured run of this
script; 12 in a heavier instrumented probe) because a stale prescan
fell back to fused link_near/update scans *inside* the worker's write
transaction (219-560 ms under the WAL write lock vs the 250 ms client
busy cap).

Run modes:
  REPRO_BEFORE=1  — restore pre-fix behavior: no stale-replan retries
                    (_PLAN_STALE_RETRIES=0 => fused fallback like the
                    old code) and no store-level writer interleave
                    (between-job drain yield stays, as it pre-existed).
  default         — fixed code paths.

Env: REPRO_SEED (64), REPRO_BURST_S (16), REPRO_RUNS (1).
"""
import os, sys, tempfile, threading, time, collections, contextlib, functools

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from verbatim.storage.store import Store
import verbatim.storage.store as st
from verbatim import ingest as _ingest_mod
from verbatim.jobs import source_jobs as sj
from verbatim.dedup import links as dlinks
from verbatim.querying import updates as upd
from verbatim.core.types import VerbatimError
from verbatim.memory.facade import Memory

BEFORE = os.environ.get("REPRO_BEFORE") == "1"

# ---------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------
TXLOG = collections.defaultdict(list)      # id(store) -> [hold/acq rows]
BEGIN_MARKS = collections.Counter()        # path -> stalled-BEGIN refreshes
WW_TRUE = [0]                              # writers_waiting()==True checks
YIELD_SLEEPS = [0]                         # store-level pre-BEGIN yields
PLAN = collections.Counter()               # near_plan / fused_near / fused_updates
TIMES = collections.defaultdict(list)      # named stage timings
STALE_ABORTS = [0]                         # _PlanStale raised (rolled-back commits)


_orig_write_tx = Store._write_tx

@contextlib.contextmanager
def _spy_write_tx(self, *, exclusive, deadline_us=None, budget_ms=None):
    key = id(self)
    t0 = time.monotonic()
    with _orig_write_tx(
        self, exclusive=exclusive, deadline_us=deadline_us, budget_ms=budget_ms
    ) as conn:
        t1 = time.monotonic()
        try:
            yield conn
        finally:
            t2 = time.monotonic()
            TXLOG[key].append(
                {"acq_wait_ms": (t1 - t0) * 1e3, "hold_ms": (t2 - t1) * 1e3}
            )

Store._write_tx = _spy_write_tx


_real_note = st.note_writer_wait
def _note(path):
    BEGIN_MARKS[path] += 1
    return _real_note(path)
st.note_writer_wait = _note


_real_ww = st.writers_waiting
def _ww_spy(path, horizon_s=st._WRITER_WAIT_HORIZON_S):
    r = _real_ww(path, horizon_s)
    if r:
        WW_TRUE[0] += 1
    return r
# drain_report resolves through ingest's module globals
_ingest_mod.writers_waiting = _ww_spy


# store-level pre-BEGIN yield: count the 5ms sleeps issued under the
# _yield_to_blocked_writers flag by timing around the flag check is
# invasive; instead wrap time.sleep inside _write_tx? Simpler: wrap the
# yield branch via a flag-aware writers_waiting is enough — count actual
# sleeps by patching store.time.sleep selectively is fragile. We instead
# monkeypatch Store-level marker: read flag + wrap sleep when in store.
_orig_sleep = st.time.sleep
def _sleep_spy(s):
    if s == st._WRITER_YIELD_S:
        YIELD_SLEEPS[0] += 1
    return _orig_sleep(s)


def timed(name, mod, attr):
    fn = getattr(mod, attr)
    @functools.wraps(fn)
    def w(*a, **kw):
        t = time.monotonic()
        try:
            return fn(*a, **kw)
        finally:
            TIMES[name].append((time.monotonic() - t) * 1e3)
    setattr(mod, attr, w)
    return fn


timed("link_near", dlinks, "link_near")
timed("commit_near", dlinks, "commit_near")
timed("detect_update_candidates", upd, "detect_update_candidates")
timed("_deps_fingerprint", sj, "_deps_fingerprint")
timed("_prescan", sj, "_prescan")

_orig_run_dedup = sj._run_dedup
def rd(conn, **kw):
    PLAN["near_plan" if kw.get("near_plan") is not None else "fused_near"] += 1
    return _orig_run_dedup(conn, **kw)
sj._run_dedup = rd

_orig_detect = sj._detect_updates
def du(conn, **kw):
    PLAN["fused_updates"] += 1
    return _orig_detect(conn, **kw)
sj._detect_updates = du


class _CountedStale(sj._PlanStale):
    def __init__(self, *a):
        STALE_ABORTS[0] += 1
        super().__init__(*a)


if BEFORE:
    # Pre-fix behavior: prescan still runs but a stale fingerprint takes
    # the fused in-transaction fallback (0 retries), and the managed
    # worker's store performs no pre-BEGIN interleave. The pre-existing
    # between-job drain yield remains (ingest's writers_waiting stays
    # real) so this reproduces exactly the shipped starvation mode.
    sj._PLAN_STALE_RETRIES = 0
    st.writers_waiting = lambda path, horizon_s=0.4: False
else:
    sj._PlanStale = _CountedStale
    st.writers_waiting = _ww_spy
    st.time.sleep = _sleep_spy


WORDS = ("deploy runbook falcon rollback checklist wiki service tier "
         "database cache queue worker lease fencing commit wal sqlite "
         "schema migration index projection token window release ").split()

def big_text(i, n=900):
    return " ".join(WORDS[(i * 7 + j) % len(WORDS)] for j in range(n))


def pct(v, q):
    if not v:
        return 0.0
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))]


def summarize_tx(key, name):
    rows = TXLOG.get(key) or []
    if not rows:
        print(f"  {name}: no txs")
        return
    holds = sorted(r["hold_ms"] for r in rows)
    waits = sorted(r["acq_wait_ms"] for r in rows)
    print(
        f"  {name}: n={len(rows)} "
        f"hold_ms p50={pct(holds,.5):.1f} p90={pct(holds,.9):.1f} "
        f"max={holds[-1]:.1f} | acq_wait p90={pct(waits,.9):.1f} max={waits[-1]:.1f}"
    )


def run(tag=""):
    TXLOG.clear(); BEGIN_MARKS.clear(); PLAN.clear(); TIMES.clear()
    WW_TRUE[0] = 0; YIELD_SLEEPS[0] = 0; STALE_ABORTS[0] = 0

    d = tempfile.mkdtemp(prefix="vbrepro-")
    path = os.path.join(d, "mem.db")
    m = Memory(path=path, worker="managed")
    worker = m._worker_handle.worker
    fid, wid = id(m._store), id(worker._store)

    seed = int(os.environ.get("REPRO_SEED", "64"))
    fails = []
    for i in range(seed):
        try:
            m.add(f"seed {i}: " + big_text(i))
        except Exception as e:
            fails.append(("seed", i, str(e)[:90]))

    ops = {"add": 0, "search": 0}
    add_ms = []
    stop = threading.Event()

    def client():
        i = 0
        while not stop.is_set():
            try:
                if i % 3 == 2:
                    m.search("falcon deploy runbook", limit=4)
                    ops["search"] += 1
                else:
                    t = time.monotonic()
                    m.add(f"burst {i} {time.monotonic()} " + big_text(i, 60))
                    add_ms.append((time.monotonic() - t) * 1e3)
                    ops["add"] += 1
            except VerbatimError as e:
                fails.append(("burst", i, str(e.code), str(e)[:90]))
            except Exception as e:
                fails.append(("burst", i, type(e).__name__, str(e)[:90]))
            i += 1

    th = threading.Thread(target=client, daemon=True)
    th.start()
    time.sleep(float(os.environ.get("REPRO_BURST_S", "16")))
    stop.set(); th.join()

    lock_fails = [f for f in fails
                  if "database is locked" in str(f) or "STORE_BUSY" in str(f)]
    other_fails = [f for f in fails if f not in lock_fails]

    print(f"=== {tag} (mode={'BEFORE' if BEFORE else 'FIXED'}) ===")
    print(f"  ops: adds={ops['add']} searches={ops['search']} "
          f"FAILURES={len(fails)} (lock={len(lock_fails)} other={len(other_fails)})")
    for f in fails[:10]:
        print("   FAIL", f)
    if add_ms:
        print(f"  add ms p50={pct(add_ms,.5):.1f} p90={pct(add_ms,.9):.1f} max={max(add_ms):.1f}")
    summarize_tx(fid, "facade store txs")
    summarize_tx(wid, "worker store txs")
    print(f"  writer-wait marks (stalled BEGIN refreshes): {sum(BEGIN_MARKS.values())}")
    print(f"  writers_waiting==True checks (drain+store): {WW_TRUE[0]}  "
          f"5ms interleave sleeps (drain+pre-BEGIN): {YIELD_SLEEPS[0]}")
    print(f"  plan usage: {dict(PLAN)}  stale_aborts={STALE_ABORTS[0]} "
          f"prescans={len(TIMES.get('_prescan', []))}")
    for k in sorted(TIMES, key=lambda k: -sum(TIMES[k])):
        v = TIMES[k]
        print(f"    {k:26s} n={len(v):4d} p50={pct(v,.5):8.1f} "
              f"p90={pct(v,.9):8.1f} max={max(v):8.1f} ms")
    try:
        dw = worker._store.diagnostics()["write_tx"]
        df = m._store.diagnostics()["write_tx"]
        print(f"  worker write_tx hold_max_ms={dw['hold_us_max']/1e3:.1f} "
              f"facade busy_errors={df['busy_errors']}")
    except Exception:
        pass
    try:
        m.close(timeout_ms=8000)
    except Exception as e:
        print("  close:", e)
    return len(lock_fails)


if __name__ == "__main__":
    n = int(os.environ.get("REPRO_RUNS", sys.argv[1] if len(sys.argv) > 1 else "1"))
    total = 0
    for r in range(n):
        total += run(tag=f"run {r}")
    print(f"\nTOTAL LOCK FAILURES: {total}")
    sys.exit(1 if total else 0)
