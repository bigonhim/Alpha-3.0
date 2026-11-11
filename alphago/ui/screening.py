"""The "Screening before simulating" card: every filter that decides which generated ideas are worth a simulation.

Shown on Tasks and Alpha Autopilot. Each switch is on by default (settings.research / settings.llm); changes save at once
and apply to the next wave of every task. "Reset to defaults" turns every screen back on with its default values.
"""

from __future__ import annotations

from nicegui import ui

from ..config import LLMSettings, ResearchSettings, settings
from .components import card

DEFAULTS = ResearchSettings()


def screening_card(expanded: bool = False) -> None:
    r, llm_cfg = settings.research, settings.llm
    with ui.expansion("Screening before simulating", icon="filter_alt", value=expanded).classes("w-full"):
        with card("w-full"):
            ui.label("These screens spend fewer simulations on ideas that cannot pass. All of them are on by default; "
                     "a change applies to the next wave of every running task.").classes("ag-muted text-xs")
            with ui.row().classes("gap-6 items-start flex-wrap w-full"):
                with ui.column().classes("gap-1 w-[360px]"):
                    fs = ui.switch("Field screening (search tasks)", value=r.field_screening).tooltip(
                        "One probe simulation per field first; the rest of the budget goes to the fields whose probe "
                        "showed a signal. Negative probes are used sign-flipped. Needs 6+ fields in the dataset.")
                    with ui.row().classes("gap-2 items-end"):
                        min_sh = ui.number("Keep fields with |Sharpe| ≥", value=r.screen_min_sharpe, min=0, max=3,
                                           step=0.1, format="%.2f").classes("w-44")
                        share = ui.number("Keep at most (share)", value=r.screen_keep_share, min=0.05, max=1,
                                          step=0.05, format="%.2f").classes("w-40")
                        keep_min = ui.number("…but at least", value=r.screen_keep_min, min=1, max=50).classes("w-28")
                with ui.column().classes("gap-1 w-[300px]"):
                    dup = ui.switch("Leave out copies of submitted alphas", value=r.duplicate_filter).tooltip(
                        "A generated idea with the same shape and fields as one of your submitted alphas would fail "
                        "self-correlation; it is not simulated.")
                    cap_on = ui.switch("Daily cap per economic idea", value=r.cluster_share > 0).tooltip(
                        "No single idea (momentum, reversal, value, …) may take more than this share of the day's "
                        "simulations, so the day's quota covers several ideas.")
                    with ui.row().classes("gap-2 items-end"):
                        c_share = ui.number("Max share", value=r.cluster_share or DEFAULTS.cluster_share, min=0.05,
                                            max=1, step=0.05, format="%.2f").classes("w-28")
                        c_min = ui.number("Never below (sims)", value=r.cluster_min, min=0, step=10).classes("w-36")
                with ui.column().classes("gap-1 w-[300px]"):
                    sc = ui.switch("AI self-check of each batch", value=llm_cfg.self_check).tooltip(
                        "One extra AI call per batch: the AI critiques its own ideas and drops the worst (at most 25%).")
                    fam = ui.number("Variants per expression shape (per batch)", value=r.family_cap_per_batch, min=1,
                                    max=20).classes("w-64")
            state = ui.label().classes("ag-muted text-xs")

            def paint_state() -> None:
                on = sum([r.field_screening, r.duplicate_filter, r.cluster_share > 0, llm_cfg.self_check])
                state.set_text(f"{on} of 4 screens on.")
                for el, enabled in ((min_sh, r.field_screening), (share, r.field_screening),
                                    (keep_min, r.field_screening), (c_share, r.cluster_share > 0),
                                    (c_min, r.cluster_share > 0)):
                    el.enable() if enabled else el.disable()

            def save() -> None:
                r.field_screening = bool(fs.value)
                r.screen_min_sharpe = float(min_sh.value if min_sh.value is not None else DEFAULTS.screen_min_sharpe)
                r.screen_keep_share = float(share.value or DEFAULTS.screen_keep_share)
                r.screen_keep_min = int(keep_min.value or DEFAULTS.screen_keep_min)
                r.duplicate_filter = bool(dup.value)
                r.cluster_share = float(c_share.value or DEFAULTS.cluster_share) if cap_on.value else 0.0
                r.cluster_min = int(c_min.value if c_min.value is not None else DEFAULTS.cluster_min)
                r.family_cap_per_batch = int(fam.value or DEFAULTS.family_cap_per_batch)
                llm_cfg.self_check = bool(sc.value)
                settings.save()
                paint_state()

            def reset() -> None:
                fs.value, dup.value, cap_on.value, sc.value = True, True, True, LLMSettings().self_check
                min_sh.value, share.value, keep_min.value = (DEFAULTS.screen_min_sharpe, DEFAULTS.screen_keep_share,
                                                             DEFAULTS.screen_keep_min)
                c_share.value, c_min.value = DEFAULTS.cluster_share, DEFAULTS.cluster_min
                fam.value = DEFAULTS.family_cap_per_batch
                save()
                ui.notify("Every screen is back on with its default values.", type="positive")

            for el in (fs, min_sh, share, keep_min, dup, cap_on, c_share, c_min, sc, fam):
                el.on_value_change(lambda _: save())
            ui.button("Reset to defaults (all on)", icon="restart_alt", on_click=reset).props("flat dense")
            paint_state()
