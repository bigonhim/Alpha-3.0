"""Cost and quota planner: what 100 simulations have bought you so far, by dataset and strategy.

From your own history (first-pass simulations only; a near miss that the repair loop made submittable counts for the
idea it came from): expected promising, strong, BRAIN-check-passed and submitted alphas per 100 simulations, with
80% intervals. The suggested split of a quota spends most where submittable alphas came from, keeps a share for
exploring strategies with little data, and can be turned into research tasks that still wait for your approval.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import select

from ..db import db
from ..db.models import Alpha, Candidate, OptimizationAttempt
from .datasets import NOT_FIRST_PASS, survival_weight
from .tournament import wilson

STRATEGY_LABELS = {"search": "Search Lab", "pair": "Double dataset · Pair", "interaction": "Interaction",
                   "llm": "AI ideas", "template": "Templates", "mutation": "Mutations", "genius": "Genius AI",
                   "spc": "SPC", "manual": "Manual", "matrix": "Simulation Matrix"}
TASK_KIND = {"search": "single", "pair": "pair", "interaction": "interaction"}
RANK = {"promising": 1, "strong": 2, "ready": 3, "submitted": 4}


async def yield_table(region: str, delay: int, min_sims: int = 1) -> list[dict[str, Any]]:
    async with db.session() as s:
        rows = (await s.execute(select(Candidate.dataset_id, Candidate.origin, Candidate.alpha_id, Alpha.verdict,
                                       Alpha.robust_state)
                                .outerjoin(Alpha, Alpha.id == Candidate.alpha_id)
                                .where(Candidate.region == region, Candidate.delay == delay,
                                       Candidate.status == "done", Candidate.origin.notin_(NOT_FIRST_PASS)))).all()
        roots = [aid for _, _, aid, _, _ in rows if aid]
        best_child: dict[str, str] = {}
        for i in range(0, len(roots), 400):
            for root, verdict in (await s.execute(
                    select(OptimizationAttempt.root_alpha_id, Alpha.verdict)
                    .join(Alpha, Alpha.id == OptimizationAttempt.child_alpha_id)
                    .where(OptimizationAttempt.root_alpha_id.in_(roots[i:i + 400])))).all():
                if RANK.get(verdict or "", 0) > RANK.get(best_child.get(root, ""), 0):
                    best_child[root] = verdict
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for ds_key, origin, aid, verdict, robust in rows:
        v = verdict if RANK.get(verdict or "", 0) >= RANK.get(best_child.get(aid or "", ""), 0) \
            else best_child.get(aid or "")
        r = RANK.get(v or "", 0)
        for ds in str(ds_key or "—").split(",") or ["—"]:
            a = agg[(ds or "—", origin)]
            a["sims"] += 1
            a["promising"] += r >= 1
            a["strong"] += r >= 2
            a["passed"] += r >= 3
            a["submitted"] += r >= 4
            a["weighted"] += survival_weight(v, robust)
    out = []
    for (ds, origin), a in agg.items():
        n = int(a["sims"])
        if n < min_sims:
            continue
        lo, hi = wilson(int(a["passed"]), n)
        out.append({"dataset": ds, "strategy": origin, "strategy_label": STRATEGY_LABELS.get(origin, origin),
                    "sims": n, "promising_per_100": round(100 * a["promising"] / n, 1),
                    "strong_per_100": round(100 * a["strong"] / n, 1),
                    "passed_per_100": round(100 * a["passed"] / n, 2),
                    "passed_lo": round(100 * lo, 2), "passed_hi": round(100 * hi, 2),
                    "submitted": int(a["submitted"]), "task_kind": TASK_KIND.get(origin)})
    return sorted(out, key=lambda r: (-(r["passed_per_100"]), -r["strong_per_100"], -r["sims"]))


def allocate(rows: list[dict[str, Any]], total: int, *, explore_share: float = 0.2,
             min_data: int = 50) -> list[dict[str, Any]]:
    """Split `total` simulations: (1 - explore_share) in proportion to the posterior mean pass rate
    (Beta(1 + passed, 1 + sims − passed)), explore_share evenly over strategies with fewer than `min_data`
    simulations. Rows that cannot become research tasks are still listed (run them as AI batches)."""
    if not rows or total <= 0:
        return []
    means = {i: (1 + r["passed_per_100"] * r["sims"] / 100) / (2 + r["sims"]) for i, r in enumerate(rows)}
    explore = [i for i, r in enumerate(rows) if r["sims"] < min_data]
    exploit_total = total - (int(total * explore_share) if explore else 0)
    s = sum(means.values()) or 1.0
    plan = {i: exploit_total * m / s for i, m in means.items()}
    for i in explore:
        plan[i] += int(total * explore_share) / len(explore)
    out = []
    for i, n in sorted(plan.items(), key=lambda kv: -kv[1]):
        if round(n) >= 10:
            r = rows[i]
            out.append({**r, "allocate": int(round(n)),
                        "expected_passed": round(int(round(n)) * means[i], 2)})
    return out
