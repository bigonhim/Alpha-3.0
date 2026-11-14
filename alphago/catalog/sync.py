"""Download every market's data catalogue at once (alpha-harness catalog/sync.py, on AlphaGo's job queue and SQLite).

A market is one (region, delay, universe); BRAIN scopes `/data-fields` exactly that way, since a field that exists in
USA D1 may not exist in EUR D0. `GET /data-fields` at version 3.0 with all four scope parameters returns a whole
market in one response, so a full catalogue is one request per market plus the dataset records, which are read a
region at a time and shared by that region's markets.

Every market runs its own pipeline: its fields first (with dataset rows derived from them, so it is browsable at
once), then BRAIN's full dataset records. Eight markets hold a payload at once (BRAIN throttles to ~34 requests a
minute serially but ~67 at eight in flight). USA D1 TOP3000 goes first and alone, so research can start while the rest
arrive. A market that fails is retried three times per stage, then noted and skipped rather than ending the sync.
Region ALL is its own, much slower download (fifty fields per request, dataset by dataset).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import delete, func, select, tuple_, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..brain import brain
from ..brain.catalog_api import catalog_api
from ..brain.platform import REGION_AGNOSTIC
from ..db import db
from ..db.models import DataField, Dataset
from ..db.models_catalog import CatalogSyncRun
from ..events import bus
from ..jobs.queue import JobContext, jobs
from . import search

log = logging.getLogger(__name__)

INSTRUMENT_TYPE = "EQUITY"
#: Markets downloading (and holding a full field payload in memory) at once.
ALL_CONCURRENCY = 8
#: How long a market waits for the shared per-region dataset read before fetching its own.
BULK_DATASETS_WAIT = 12.0
#: Regions read at once for the shared dataset index: BRAIN allows about a request a second.
REGION_DATASETS_CONCURRENCY = 4
#: Datasets paged at once when a market is read dataset by dataset (region ALL).
PAGED_FIELDS_CONCURRENCY = 4
#: Tries per market for each stage; the client already retries throttling and server errors per request.
SCOPE_ATTEMPTS = 3
#: Seconds before the n-th retry (times n): 2 s, then 4 s.
RETRY_WAIT = 2.0
#: Rows per write transaction, so other writers interleave with a big market's upsert.
WRITE_CHUNK = 4000
#: Progress steps per market state: its fields arrive (1), its details arrive (2); failed or fields-only is finished.
STEPS = {"waiting": 0, "fetching": 0, "details": 1, "fields": 2, "done": 2, "failed": 2}
STATE_LABELS = {"waiting": "Not Synced", "fetching": "Fetching Fields", "fields": "Fields Ready",
                "details": "Filling Details", "done": "Synced", "failed": "Failed"}
#: The labs read MATRIX, VECTOR and GROUP; where BRAIN lists one id under two types, the usable one is kept.
TYPE_RANK = {"MATRIX": 0, "VECTOR": 1, "GROUP": 2}
JOB_KIND = "catalog_sync_all"


@dataclass(frozen=True)
class Market:
    region: str
    delay: int
    universe: str

    @property
    def label(self) -> str:
        return f"{self.region}/D{self.delay}/{self.universe}"

    @property
    def params(self) -> dict[str, Any]:
        return {"instrumentType": INSTRUMENT_TYPE, "region": self.region, "delay": self.delay,
                "universe": self.universe}


#: The market a first sync downloads before any other: the Dashboard's, so a first task can run meanwhile.
STARTER = Market("USA", 1, "TOP3000")


class SyncCancelled(RuntimeError):
    """The user stopped a running sync."""


def offered_markets(options: Any = None) -> list[Market]:
    """Every EQUITY market the account can simulate, read off BRAIN's own settings (`OPTIONS /simulations`), so a
    new region or universe is picked up without a code change."""
    opts = options or brain.options
    return [Market(region, int(delay), universe) for region in opts.regions for delay in opts.delays(region)
            for universe in opts.universes(region)]


def now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- live progress

class Progress:
    """A running sync's per-market states, read by the Sync matrix and the Dashboard."""

    def __init__(self, run_id: int, kind: str, targets: list[Market], job_id: int | None) -> None:
        self.run_id, self.kind, self.job_id = run_id, kind, job_id
        self.started_at = now()
        self.finished_at: datetime | None = None
        self.status = "RUNNING"
        self.error: str | None = None
        self.failed: list[str] = []
        self.fields = self.datasets = 0
        self.markets: dict[str, dict[str, Any]] = {
            t.label: {"region": t.region, "delay": t.delay, "universe": t.universe, "state": "waiting",
                      "fields": None} for t in targets}
        self.version = 0

    def payload(self) -> dict[str, Any]:
        markets = list(self.markets.values())
        arrived = sum(m["state"] not in ("waiting", "fetching") for m in markets)
        return {"id": self.run_id, "kind": self.kind, "status": self.status, "job_id": self.job_id,
                "stage": "fields" if arrived < len(markets) else "details", "scopes_done": arrived,
                "scopes_total": len(markets), "markets": markets,
                "fraction": sum(STEPS[m["state"]] for m in markets) / (2 * len(markets)) if markets else 1.0,
                "fields_synced": self.fields, "datasets_synced": self.datasets, "error": failures(self.failed),
                "started_at": self.started_at, "finished_at": self.finished_at}


live: Progress | None = None


def failures(failed: list[str]) -> str | None:
    """Failed markets as one line; None while nothing has failed."""
    if not failed:
        return None
    more = f"; and {len(failed) - 5} more" if len(failed) > 5 else ""
    return f"{len(failed)} failed: {'; '.join(failed[:5])}{more}"


def _reason(exc: BaseException) -> str:
    return getattr(exc, "message", None) or str(exc) or type(exc).__name__


async def latest_run(kind: str | None = None) -> dict[str, Any] | None:
    """The live run if one is going, else the last stored one (of `kind`)."""
    if live is not None and live.status == "RUNNING" and (kind is None or live.kind == kind):
        return live.payload()
    stmt = select(CatalogSyncRun).order_by(CatalogSyncRun.id.desc()).limit(1)
    if kind:
        stmt = stmt.where(CatalogSyncRun.kind == kind)
    async with db.session() as s:
        run = await s.scalar(stmt)
    if run is None:
        return None
    status = run.status
    if status == "RUNNING" and not (live is not None and live.run_id == run.id):
        status = "FAILED"                 # left RUNNING by a closed app
    markets = list(run.markets or [])
    return {"id": run.id, "kind": run.kind, "status": status, "job_id": run.job_id, "stage": "details",
            "scopes_done": len(markets), "scopes_total": len(markets), "markets": markets, "fraction": 1.0,
            "fields_synced": run.fields_synced, "datasets_synced": run.datasets_synced,
            "error": run.error or ("The app closed during this sync." if status != run.status else None),
            "started_at": run.started_at, "finished_at": run.finished_at}


def running() -> bool:
    return live is not None and live.status == "RUNNING"


async def start(*, region_agnostic: bool = False) -> int:
    """Queue the download as a background job; refuses a second one while one runs."""
    if running():
        raise RuntimeError("A catalogue sync is already running. Cancel it first, or wait for it to finish.")
    title = "Downloading region-agnostic Data Fields" if region_agnostic else "Downloading BRAIN Data Fields"
    return await jobs.submit(JOB_KIND, title, {"region_agnostic": region_agnostic})


async def cancel() -> bool:
    """Stop the running sync. Markets that already arrived stay in the catalogue."""
    if not running() or live is None or live.job_id is None:
        return False
    await jobs.cancel(live.job_id)
    return True


# ---------------------------------------------------------------- row builders

def _ref(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, dict):
        rid = value.get("id")
        return (str(rid) if rid is not None else None), value.get("name")
    if value:
        return str(value), str(value)
    return None, None


def _day(value: Any) -> str | None:
    return str(value)[:10] if value else None


def field_rows(raw: list[dict[str, Any]], market: Market, stamp: datetime) -> list[dict[str, Any]]:
    """One row per field id: BRAIN repeats an id under two types in a few markets, and the usable type wins."""
    by_id: dict[str, dict[str, Any]] = {}
    ranks: dict[str, int] = {}
    for item in raw:
        fid = item.get("id")
        if not fid:
            continue
        fid = str(fid)
        rank = TYPE_RANK.get(str(item.get("type") or ""), len(TYPE_RANK))
        if fid in ranks and ranks[fid] <= rank:
            continue
        ranks[fid] = rank
        ds_id, _ = _ref(item.get("dataset"))
        cat_id, cat_name = _ref(item.get("category"))
        sub_id, sub_name = _ref(item.get("subcategory"))
        by_id[fid] = dict(
            id=fid, dataset_id=ds_id or "", region=market.region, delay=market.delay, universe=market.universe,
            type=str(item.get("type") or "MATRIX"), description=item.get("description"), category=cat_name,
            category_id=cat_id, subcategory_id=sub_id, subcategory=sub_name, coverage=item.get("coverage"),
            date_coverage=item.get("dateCoverage"), user_count=item.get("userCount"),
            alpha_count=item.get("alphaCount"), pyramid_multiplier=item.get("pyramidMultiplier"),
            themes=item.get("themes") or None, date_created=_day(item.get("dateCreated")),
            region_coverage=item.get("regionCoverage"), synced_at=stamp)
    return list(by_id.values())


def derived_dataset_rows(rows: list[dict[str, Any]], market: Market, raw: list[dict[str, Any]],
                         stamp: datetime) -> list[dict[str, Any]]:
    """Dataset rows built from the fields alone (every field names its dataset, category and subcategory), so a
    market is browsable before its details arrive. The details step overwrites them with BRAIN's full records."""
    names = {}
    for item in raw:
        ds = item.get("dataset")
        if isinstance(ds, dict) and ds.get("id") is not None:
            names.setdefault(str(ds["id"]), ds.get("name"))
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        ds_id = r["dataset_id"]
        if not ds_id:
            continue
        entry = out.setdefault(ds_id, dict(
            id=ds_id, region=market.region, delay=market.delay, universe=market.universe, name=names.get(ds_id),
            category_id=r["category_id"], category_name=r["category"], subcategory=r["subcategory"], field_count=0,
            synced_at=stamp, fields_synced_at=stamp))
        entry["field_count"] += 1
    return list(out.values())


def dataset_row(d: dict[str, Any], market: Market, stamp: datetime) -> dict[str, Any]:
    cat_id, cat_name = _ref(d.get("category"))
    _sub_id, sub_name = _ref(d.get("subcategory"))
    return dict(id=str(d.get("id")), region=market.region, delay=market.delay, universe=market.universe,
                name=d.get("name"), description=d.get("description"), category_id=cat_id, category_name=cat_name,
                subcategory=sub_name, coverage=d.get("coverage"), value_score=d.get("valueScore"),
                user_count=d.get("userCount"), alpha_count=d.get("alphaCount"), field_count=d.get("fieldCount"),
                pyramid_multiplier=d.get("pyramidMultiplier"), themes=d.get("themes") or None, synced_at=stamp)


# ---------------------------------------------------------------- writes

_FIELD_KEY = ("id", "region", "delay", "universe")


async def store_fields(market: Market, rows: list[dict[str, Any]], datasets: list[dict[str, Any]],
                       stamp: datetime) -> None:
    """Replace one market's fields: upsert every row (chunked, so other writers get the lock between chunks), then
    drop the fields BRAIN no longer lists there. Derived dataset rows only fill gaps: a re-sync must not blank the
    Value Score, Coverage and counts an earlier details step stored."""
    table = DataField.__table__
    for i in range(0, len(rows), WRITE_CHUNK):
        chunk = rows[i:i + WRITE_CHUNK]
        stmt = sqlite_insert(table)
        stmt = stmt.on_conflict_do_update(index_elements=list(_FIELD_KEY), set_={
            k: getattr(stmt.excluded, k) for k in chunk[0] if k not in _FIELD_KEY})
        async with db.write() as s:
            await s.execute(stmt, chunk)
    ds_table = Dataset.__table__
    async with db.write() as s:
        await s.execute(delete(DataField).where(
            DataField.region == market.region, DataField.delay == market.delay, DataField.universe == market.universe,
            DataField.synced_at < stamp))
        if datasets:
            stmt = sqlite_insert(ds_table)
            keep = ("name", "category_id", "category_name", "subcategory")
            stmt = stmt.on_conflict_do_update(index_elements=list(_FIELD_KEY), set_={
                **{k: func.coalesce(ds_table.c[k], getattr(stmt.excluded, k)) for k in keep},
                "field_count": stmt.excluded.field_count, "fields_synced_at": stmt.excluded.fields_synced_at})
            await s.execute(stmt, datasets)
    await db.kv_set(f"last_sync:fields:{market.region}:{market.delay}:{market.universe}", stamp.isoformat())


async def store_datasets(market: Market, rows: list[dict[str, Any]]) -> None:
    """BRAIN's full dataset records for a market whose fields landed."""
    if rows:
        stmt = sqlite_insert(Dataset.__table__)
        stmt = stmt.on_conflict_do_update(index_elements=list(_FIELD_KEY), set_={
            k: getattr(stmt.excluded, k) for k in rows[0] if k not in _FIELD_KEY})
        async with db.write() as s:
            await s.execute(stmt, rows)
    await db.kv_set(f"last_sync:datasets:{market.region}:{market.delay}:{market.universe}", now().isoformat())


async def prune(targets: list[Market], *, region_agnostic: bool) -> int:
    """Drop markets BRAIN no longer offers: left, they stay browsable and a lab could spend simulations on them."""
    keep = {(t.region, t.delay, t.universe) for t in targets}
    async with db.session() as s:
        held = (await s.execute(select(DataField.region, DataField.delay, DataField.universe).distinct())).all()
    gone = [tuple(h) for h in held if tuple(h) not in keep and ((h[0] == REGION_AGNOSTIC) == region_agnostic)]
    if not gone:
        return 0
    async with db.write() as s:
        result = await s.execute(delete(DataField).where(
            tuple_(DataField.region, DataField.delay, DataField.universe).in_(gone)))
        await s.execute(delete(Dataset).where(tuple_(Dataset.region, Dataset.delay, Dataset.universe).in_(gone)))
    return int(result.rowcount or 0)


# ---------------------------------------------------------------- the download

async def _retry(work: Callable[[], Awaitable[Any]], label: str) -> Any:
    """One market's step up to SCOPE_ATTEMPTS times, waiting 2 s then 4 s between tries."""
    for attempt in range(1, SCOPE_ATTEMPTS + 1):
        try:
            return await work()
        except (asyncio.CancelledError, SyncCancelled):
            raise
        except Exception as exc:
            if attempt == SCOPE_ATTEMPTS:
                raise
            log.warning("catalog sync %s: try %d failed: %s", label, attempt, _reason(exc))
            await asyncio.sleep(RETRY_WAIT * attempt)
    return None


async def _all_datasets(targets: list[Market]) -> dict[tuple[str, int, str], list[dict[str, Any]]] | None:
    """Every market's datasets, read a region at a time (one unscoped read is answered with a 504 at BRAIN's
    thirty-second gateway limit; one region takes eight to sixteen seconds). A region that fails sends only its own
    markets to their own requests."""
    api = catalog_api()
    regions = sorted({t.region for t in targets})
    gate = asyncio.Semaphore(REGION_DATASETS_CONCURRENCY)

    async def one(region: str) -> list[dict[str, Any]]:
        async with gate:
            return await api.data_sets_all(instrumentType=INSTRUMENT_TYPE, region=region)

    results = await asyncio.gather(*(one(r) for r in regions), return_exceptions=True)
    indexed: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    failed = 0
    for region, result in zip(regions, results):
        if isinstance(result, BaseException):
            failed += 1
            log.warning("catalog sync: datasets of %s failed: %s", region, _reason(result))
            continue
        for row in result:
            if row.get("region") is None or row.get("delay") is None or row.get("universe") is None:
                continue
            indexed.setdefault((str(row["region"]), int(row["delay"]), str(row["universe"])), []).append(row)
    return None if failed == len(regions) else indexed


async def _shared(task: asyncio.Task, market: Market) -> list[dict[str, Any]] | None:
    """This market's slice of the shared read, or None to fetch its own. Time-boxed: a slow read must never hold
    forty markets still; `asyncio.wait` gives up on the task without cancelling it, so a later market still gets it."""
    if not task.done():
        await asyncio.wait([task], timeout=BULK_DATASETS_WAIT)
    if not task.done() or task.cancelled() or task.exception() is not None:
        return None
    indexed = task.result()
    return None if indexed is None else indexed.get((market.region, market.delay, market.universe))


async def _fields(market: Market, bulk: asyncio.Task, state: dict[str, Any], progress: Progress,
                  emit: Callable[[], Awaitable[None]]) -> list[dict[str, Any]]:
    """One market's fields, however BRAIN is willing to hand them over."""
    api = catalog_api()
    if market.region != REGION_AGNOSTIC:
        return await api.data_fields_all(market.region, market.delay, market.universe, INSTRUMENT_TYPE)
    # region ALL: dataset by dataset, fifty at a time — hundreds of requests, hence the running count
    sets = await _shared(bulk, market)
    if sets is None:
        sets = await api.data_sets_all(**market.params)
    gate = asyncio.Semaphore(PAGED_FIELDS_CONCURRENCY)
    out: list[dict[str, Any]] = []
    lost: list[str] = []

    async def one(dataset_id: str) -> None:
        async with gate:
            try:
                found = await api.data_fields_paged(dataset_id, market.region, market.delay, market.universe,
                                                    INSTRUMENT_TYPE)
            except asyncio.CancelledError:
                raise
            except Exception as exc:         # one dataset must not discard the hundred already downloaded
                lost.append(f"{dataset_id}: {_reason(exc)}")
                return
            out.extend(found)
            state["fields"] = len(out)
            await emit()

    await asyncio.gather(*(one(str(d.get("id"))) for d in sets if d.get("id")))
    if lost:
        progress.failed.append(f"{market.label}: {len(lost)} of {len(sets)} datasets could not be read "
                               f"({'; '.join(lost[:3])})")
    return out


async def crawl(progress: Progress, targets: list[Market], first: Market | None,
                emit: Callable[[], Awaitable[None]]) -> list[Market]:
    """Every market's pipeline at once, bounded by the gate; returns the markets whose fields landed."""
    gate = asyncio.Semaphore(ALL_CONCURRENCY)
    lead = asyncio.Event()
    if first not in targets:
        lead.set()
    synced: list[Market] = []

    async def all_datasets() -> dict[tuple[str, int, str], list[dict[str, Any]]] | None:
        await lead.wait()
        return await _all_datasets(targets)

    bulk = asyncio.create_task(all_datasets(), name="catalog-datasets")

    async def fetch(market: Market) -> bool:
        state = progress.markets[market.label]
        async with gate:
            state["state"] = "fetching"
            await emit()

            async def store() -> tuple[int, int]:
                started = time.perf_counter()
                raw = await _fields(market, bulk, state, progress, emit)
                if not raw:
                    raise RuntimeError("BRAIN returned no data fields")
                stamp = now()
                rows = await asyncio.to_thread(field_rows, raw, market, stamp)
                derived = derived_dataset_rows(rows, market, raw, stamp)
                del raw
                await store_fields(market, rows, derived, stamp)
                log.info("catalog sync %s: %d fields in %.1f s", market.label, len(rows),
                         time.perf_counter() - started)
                return len(rows), len(derived)

            try:
                n_fields, n_sets = await _retry(store, market.label)
            except asyncio.CancelledError:
                raise
            except Exception as exc:              # one market failing must not end the whole sync
                state["state"] = "failed"
                progress.failed.append(f"{market.label}: {_reason(exc)}")
                await emit()
                return False
        progress.fields += n_fields
        progress.datasets += n_sets
        state.update(state="details", fields=n_fields)
        synced.append(market)
        await emit()
        return True

    async def fill(market: Market) -> None:
        state = progress.markets[market.label]

        async def details() -> None:
            sets = await _shared(bulk, market)
            if sets is None:
                sets = await catalog_api().data_sets_all(**market.params)
            stamp = now()
            await store_datasets(market, [dataset_row(d, market, stamp) for d in sets if d.get("id") is not None])

        try:
            async with gate:                       # dataset paging is many requests per market: inside the gate too
                await _retry(details, market.label)
            state["state"] = "done"
        except asyncio.CancelledError:
            raise
        except Exception as exc:                   # the market stays browsable on its derived rows
            state["state"] = "fields"
            progress.failed.append(f"{market.label} details: {_reason(exc)}")
        await emit()

    async def pipeline(market: Market) -> None:
        if market != first:
            await lead.wait()
        try:
            landed = await fetch(market)
        finally:
            lead.set()                             # landed or not, nothing waits on the first market any longer
        if landed:
            await fill(market)

    try:
        await asyncio.gather(*(pipeline(t) for t in targets))
    finally:
        if not bulk.done():
            bulk.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await bulk
    return synced


async def _save(progress: Progress) -> None:
    async with db.write() as s:
        await s.execute(update(CatalogSyncRun).where(CatalogSyncRun.id == progress.run_id).values(
            status=progress.status, fields_synced=progress.fields, datasets_synced=progress.datasets,
            markets=list(progress.markets.values()), error=failures(progress.failed) or progress.error,
            finished_at=progress.finished_at))


async def run_sync(ctx: JobContext | None, params: dict[str, Any]) -> str:
    """The job: every ordinary market (or, with `region_agnostic`, every universe of region ALL)."""
    global live
    agnostic = bool(params.get("region_agnostic"))
    offered = [Market(**m) for m in params["markets"]] if params.get("markets") else offered_markets()
    targets = [t for t in offered if (t.region == REGION_AGNOSTIC) == agnostic]
    if not targets:
        raise RuntimeError("This account cannot run region-agnostic simulations, so BRAIN offers no all-regions "
                           "market to download." if agnostic else
                           "BRAIN's settings list no markets to download. Sign in again and retry.")
    kind = "agnostic" if agnostic else "all"
    async with db.write() as s:
        run = CatalogSyncRun(kind=kind, status="RUNNING", job_id=getattr(ctx, "job_id", None),
                             markets=[{"region": t.region, "delay": t.delay, "universe": t.universe,
                                       "state": "waiting", "fields": None} for t in targets])
        s.add(run)
        await s.flush()
        run_id = run.id
    progress = Progress(run_id, kind, targets, getattr(ctx, "job_id", None))
    live = progress
    last = {"t": 0.0}

    async def emit(force: bool = False) -> None:
        progress.version += 1
        if ctx is None or (not force and time.monotonic() - last["t"] < 1.0):
            return
        last["t"] = time.monotonic()
        p = progress.payload()
        await ctx.update(p["fraction"], f"{p['scopes_done']} of {p['scopes_total']} markets · "
                                        f"{progress.fields:,} fields")

    first = STARTER if STARTER in targets else None
    try:
        synced = await crawl(progress, targets, first, emit)
    except asyncio.CancelledError:
        progress.status, progress.finished_at = "CANCELLED", now()
        progress.error = "Cancelled. Markets that already arrived stay in the catalogue; press Sync to finish the rest."
        with contextlib.suppress(Exception):
            await asyncio.shield(_save(progress))
        bus.publish("sync", "Catalogue sync cancelled. Markets that already arrived stay in the catalogue.", "warning")
        raise
    except Exception as exc:
        progress.status, progress.finished_at, progress.error = "FAILED", now(), _reason(exc)
        await _save(progress)
        raise
    if brain.options.live and synced and not params.get("markets"):
        try:
            removed = await prune(targets, region_agnostic=agnostic)
            if removed:
                log.info("catalog sync: pruned %d fields of markets BRAIN no longer offers", removed)
        except Exception:
            log.warning("catalog sync: prune failed", exc_info=True)
    try:
        await search.rebuild()               # new fields are unsearchable until indexed, and the index is cheap
    except Exception:
        log.warning("catalog sync: search index rebuild failed", exc_info=True)
    from ..engine import catalogs
    catalogs.invalidate()
    progress.status = "COMPLETE" if synced else "FAILED"
    progress.finished_at = now()
    await _save(progress)
    await emit(force=True)
    note = failures(progress.failed)
    summary = (f"Synced {len(synced)} of {len(targets)} markets: {progress.fields:,} fields, "
               f"{progress.datasets:,} datasets" + (f". {note} — press Sync to try them again." if note else "."))
    bus.publish("sync", summary, "warning" if note else "success")
    if not synced:
        raise RuntimeError(note or "No market could be downloaded.")
    return summary


async def rebuild_index(_ctx: JobContext | None = None, _params: dict[str, Any] | None = None) -> str:
    n = await search.rebuild()
    return f"{n:,} fields indexed for search"


def register() -> None:
    jobs.register(JOB_KIND)(run_sync)
    jobs.register("catalog_search_index")(rebuild_index)
